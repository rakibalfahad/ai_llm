"""
Redis-based distributed LLM slot queue.

Design: BLPOP / RPUSH token-bucket
───────────────────────────────────
The queue is implemented as a Redis LIST of N slot tokens:

    llm:slot_pool = ["slot:0", "slot:1", "slot:2"]   (N = llm_parallel_requests)

  acquire() → BLPOP llm:slot_pool <timeout>
    • Atomically pops a token (blocks server-side if list is empty — zero polling).
    • Redis BLPOP serves waiting clients in FIFO order → fair queuing.
    • Returns the token string, or None if timeout expires.

  release(token) → RPUSH llm:slot_pool <token>
    • Returns the token to the pool.
    • Immediately wakes the longest-waiting BLPOP caller (Redis does this atomically).

  init() → DEL + RPUSH  (called once on startup, protected by a Redis lock)
    • Resets the slot pool to exactly N tokens.
    • The lock prevents two replicas from double-initialising simultaneously.

Why this beats asyncio.Semaphore for multi-replica setups:
    • asyncio.Semaphore lives inside one process's memory. If you run
      `docker compose up --scale rag-api=3`, each replica has its own
      semaphore and thinks it can use all N slots → the LLM server gets
      3×N concurrent requests → GPU OOM.
    • Redis is external to all replicas → single source of truth.

Redis key reference:
    llm:slot_pool       LIST   — available slot tokens
    llm:waiting_count   STRING — INCR/DECR counter of blocked acquire() calls
    llm:init_lock       STRING — NX lock used during pool initialisation
"""
import asyncio
import logging
from typing import Optional

import redis.asyncio as aioredis
from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

POOL_KEY    = "llm:slot_pool"
WAITING_KEY = "llm:waiting_count"
INIT_LOCK   = "llm:init_lock"
INIT_LOCK_TTL = 10   # seconds — long enough for the init pipeline to complete


class LLMSlotQueue:
    """
    Distributed FIFO slot queue backed by Redis.
    All methods are coroutines and safe to call concurrently.
    """

    def __init__(self, redis: Redis, slots: int, queue_timeout: int) -> None:
        self._redis = redis
        self._slots = slots
        self._queue_timeout = queue_timeout

    # ── Startup ────────────────────────────────────────────────────────────────

    async def init(self) -> None:
        """
        Reset the slot pool to exactly `slots` tokens.

        Uses a short-lived NX lock so that when multiple replicas start at the
        same time, only one performs the reset. Others wait briefly and then
        verify the pool size is correct.
        """
        # Acquire a non-expiring-once-set NX lock (TTL = INIT_LOCK_TTL seconds)
        acquired = await self._redis.set(
            INIT_LOCK, "1", nx=True, ex=INIT_LOCK_TTL
        )

        if acquired:
            logger.info("Initialising Redis slot pool: %d slots", self._slots)
            async with self._redis.pipeline(transaction=True) as pipe:
                pipe.delete(POOL_KEY)
                pipe.delete(WAITING_KEY)
                for i in range(self._slots):
                    pipe.rpush(POOL_KEY, f"slot:{i}")
                await pipe.execute()
            logger.info("Redis slot pool ready.")
        else:
            # Another replica is initialising; wait for it to finish
            logger.info("Redis slot pool init in progress on another replica — waiting…")
            for _ in range(20):
                await asyncio.sleep(0.5)
                if not await self._redis.exists(INIT_LOCK):
                    break
            pool_len = await self._redis.llen(POOL_KEY)
            logger.info("Redis slot pool has %d/%d tokens after wait.", pool_len, self._slots)

    # ── Acquire ────────────────────────────────────────────────────────────────

    async def acquire(self) -> Optional[str]:
        """
        Wait for a free LLM slot.

        Blocks up to `queue_timeout` seconds.
        Returns the slot token string (e.g. "slot:0") if a slot was acquired,
        or None if the timeout expired before any slot became available.

        While blocked, the asyncio event loop is free to serve other requests
        (health checks, RAG searches, other users' streams).
        """
        await self._redis.incr(WAITING_KEY)
        try:
            # BLPOP blocks on the Redis server side — not in Python.
            # timeout=0 means block forever; we always pass a positive value.
            result = await self._redis.blpop(
                [POOL_KEY],
                timeout=self._queue_timeout,
            )
        except RedisError as exc:
            logger.error("Redis error during acquire: %s", exc)
            raise
        finally:
            await self._redis.decr(WAITING_KEY)

        if result is None:
            # Timeout expired — no slot became available
            return None

        _key, token = result
        return token.decode() if isinstance(token, bytes) else token

    # ── Release ────────────────────────────────────────────────────────────────

    async def release(self, token: str) -> None:
        """
        Return a slot token to the pool.

        Redis atomically wakes the oldest BLPOP waiter (FIFO).
        This is safe to call from a streaming generator's finally block —
        even if the client disconnected mid-stream.
        """
        try:
            await self._redis.rpush(POOL_KEY, token)
        except RedisError as exc:
            # Log but don't raise — if release fails the slot is permanently lost.
            # The server restart / init() will recover it.
            logger.error("Redis error during release of %s: %s", token, exc)

    # ── Monitoring ────────────────────────────────────────────────────────────

    async def active_count(self) -> int:
        """Number of slots currently checked out (running inference)."""
        try:
            available = await self._redis.llen(POOL_KEY)
            return max(0, self._slots - available)
        except RedisError:
            return -1

    async def queue_depth(self) -> int:
        """Number of requests currently blocked waiting for a slot."""
        try:
            val = await self._redis.get(WAITING_KEY)
            return int(val) if val else 0
        except RedisError:
            return -1

    async def health_check(self) -> bool:
        """Returns True if Redis is reachable."""
        try:
            return await self._redis.ping()
        except RedisError:
            return False
