"""
╔══════════════════════════════════════════════════════════════════════╗
║  Tutorial 01 — Tokenization & Byte-Pair Encoding (BPE)              ║
║  pytorch_llm/tutorials/01_tokenizer_bpe.py                          ║
╚══════════════════════════════════════════════════════════════════════╝

WHAT YOU WILL LEARN
───────────────────
  1. Why tokenization exists and what it does
  2. Character-level tokenizer (simplest possible tokenizer)
  3. Byte-Pair Encoding (BPE) — the algorithm used by GPT, LLaMA, etc.
  4. How to train BPE on raw text
  5. How to compare your tokenizer with HuggingFace's real one

RUN IN DOCKER
─────────────
  cd deeplearning/
  docker run --rm --gpus all \\
    -v $(pwd)/data:/workspace/data \\
    -v $(pwd)/../pytorch_llm:/workspace/pytorch_llm \\
    deeplearning:v100-llm \\
    python3 /workspace/pytorch_llm/tutorials/01_tokenizer_bpe.py
"""

import re
import json
import os
from collections import Counter, defaultdict


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Why Tokenization?
# ══════════════════════════════════════════════════════════════════════════════
#
# Neural networks work with numbers, not text. We need a function:
#
#     encode : str  → List[int]    "Hello" → [15496, 11]
#     decode : List[int] → str     [15496, 11] → "Hello"
#
# The integers are called TOKEN IDs. The mapping from token → ID is the
# VOCABULARY. Vocabulary size is a key hyperparameter:
#   - Too small (char-level, ~256): sequences are very long, model struggles
#     to learn word-level patterns.
#   - Too large (word-level, ~50k+): rare words are never seen during training.
#   - BPE (~32k–128k): best of both worlds — common words are one token,
#     rare words are split into subwords.
#
# LLaMA 2 uses 32,000 tokens. LLaMA 3 uses 128,256 tokens.


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Character-Level Tokenizer
# ══════════════════════════════════════════════════════════════════════════════

class CharTokenizer:
    """
    Simplest possible tokenizer: every unique character is one token.
    Vocabulary size = number of unique characters in the training text.

    Pros : works out of the box, no training needed
    Cons : long sequences, no knowledge of word structure
    """

    def __init__(self):
        self.char_to_id: dict[str, int] = {}
        self.id_to_char: dict[int, str] = {}

    def train(self, text: str) -> None:
        """Build vocabulary from all unique chars in text."""
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        print(f"[CharTokenizer] vocab size = {len(self.char_to_id)}")

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Byte-Pair Encoding (BPE)
# ══════════════════════════════════════════════════════════════════════════════
#
# BPE algorithm (Sennrich et al. 2016):
#
#   1. Start: vocabulary = all individual characters (bytes)
#   2. Repeat N times:
#      a. Count all adjacent token pairs in the corpus
#      b. Find the most frequent pair (A, B)
#      c. Merge A B → AB everywhere in the corpus
#      d. Add AB to vocabulary as a new token
#   3. Stop when vocabulary size reaches the target
#
# Example with vocab_size=5 on "aababc":
#   Start  : a a b a b c         pairs: (a,a)=1  (a,b)=2  (b,a)=1  (b,c)=1
#   Merge 1: aa b aa b c → merge (a,b)  → "ab"
#            a ab a ab c          pairs: (a,ab)=2  (ab,a)=1  (a,ab)=1  (ab,c)=1
#   Merge 2: merge (a,ab) → "aab"
#   ...


class BPETokenizer:
    """
    BPE tokenizer trained from scratch on raw text.
    Simplified implementation — teaches the core algorithm clearly.
    """

    def __init__(self, vocab_size: int = 500):
        self.vocab_size = vocab_size
        self.merges: list[tuple[str, str]] = []   # ordered merge rules
        self.vocab: dict[str, int] = {}            # token → id
        self.id_to_token: dict[int, str] = {}      # id → token

    # ── Training ──────────────────────────────────────────────────────────────

    def _get_vocab_from_text(self, text: str) -> dict[tuple, int]:
        """
        Split text into words, then represent each word as a tuple of chars
        with a special end-of-word marker '</w>'.

        "hello world" → {('h','e','l','l','o','</w>'): 1,
                          ('w','o','r','l','d','</w>'): 1}

        The </w> marker lets the model distinguish "low" at end of word
        (low</w>) from "low" in the middle of "lower" (l o w e r</w>).
        """
        words = re.findall(r"\S+", text.lower())
        word_freq: dict[tuple, int] = Counter()
        for word in words:
            chars = tuple(list(word) + ["</w>"])
            word_freq[chars] += 1
        return dict(word_freq)

    def _get_pairs(self, vocab: dict[tuple, int]) -> dict[tuple[str,str], int]:
        """Count frequency of every adjacent pair across all words."""
        pairs: dict[tuple[str,str], int] = Counter()
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i+1])] += freq
        return pairs

    def _merge_vocab(self,
                     pair: tuple[str, str],
                     vocab: dict[tuple, int]) -> dict[tuple, int]:
        """Apply one merge rule: replace all occurrences of `pair` with merged token."""
        merged = "".join(pair)
        new_vocab: dict[tuple, int] = {}
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i+1]) == pair:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def train(self, text: str) -> None:
        """Run BPE training on raw text."""
        print(f"\n[BPETokenizer] Training on {len(text):,} characters...")

        # Step 1: start with character vocabulary
        vocab = self._get_vocab_from_text(text)

        # Base vocabulary: all unique chars + special tokens
        base_chars = set()
        for word in vocab:
            base_chars.update(word)
        base_tokens = ["<unk>", "<pad>", "<bos>", "<eos>"] + sorted(base_chars)

        self.vocab = {tok: i for i, tok in enumerate(base_tokens)}
        self.merges = []

        # Step 2: iteratively merge most frequent pairs
        num_merges = self.vocab_size - len(self.vocab)
        print(f"[BPETokenizer] Base chars: {len(base_tokens)} | Merges to learn: {num_merges}")

        for merge_idx in range(num_merges):
            pairs = self._get_pairs(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)

            new_token = "".join(best_pair)
            self.vocab[new_token] = len(self.vocab)

            if merge_idx % 100 == 0:
                print(f"  merge {merge_idx:4d}: {best_pair[0]!r:10s} + "
                      f"{best_pair[1]!r:10s} → {new_token!r:12s}  "
                      f"(freq={pairs[best_pair]})")

        self.id_to_token = {i: tok for tok, i in self.vocab.items()}
        print(f"[BPETokenizer] Final vocab size: {len(self.vocab)}")

    # ── Encode ────────────────────────────────────────────────────────────────

    def _tokenize_word(self, word: str) -> list[str]:
        """Apply learned merge rules to a single word."""
        tokens = list(word) + ["</w>"]

        for merge in self.merges:
            merged = "".join(merge)
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == merge:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text to a list of token IDs."""
        words = re.findall(r"\S+|\s+", text.lower())
        ids = []
        for word in words:
            word = word.strip()
            if not word:
                continue
            tokens = self._tokenize_word(word)
            for tok in tokens:
                ids.append(self.vocab.get(tok, self.vocab["<unk>"]))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        tokens = [self.id_to_token.get(i, "<unk>") for i in ids]
        text = " ".join(tokens)
        text = text.replace(" </w>", " ").replace("</w>", " ")
        # Reconstruct subwords that were split
        text = re.sub(r" (?=[^A-Z\s])", "", text)  # rejoin subwords
        return text.strip()

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"vocab": self.vocab, "merges": self.merges}, f, indent=2)
        print(f"[BPETokenizer] Saved to {path}")

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        self.vocab = data["vocab"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.id_to_token = {i: tok for tok, i in self.vocab.items()}
        print(f"[BPETokenizer] Loaded {len(self.vocab)} tokens from {path}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Vocabulary Analysis Helpers
# ══════════════════════════════════════════════════════════════════════════════

def analyze_tokenization(tokenizer, text: str, label="Tokenizer") -> None:
    """Show how a tokenizer splits example sentences."""
    print(f"\n── {label} Analysis ──")
    examples = [
        "The quick brown fox jumps over the lazy dog.",
        "Transformers use self-attention to process tokens.",
        "unbelievable",
        "GPU parallelization",
        "PyTorch is great for deep learning!",
    ]
    for sentence in examples:
        ids = tokenizer.encode(sentence)
        # Show tokens for char tokenizer
        if hasattr(tokenizer, 'char_to_id'):
            tokens = [sentence[i] for i in range(min(len(sentence), 20))]
        else:
            tokens = [tokenizer.id_to_token.get(i, "?") for i in ids[:15]]
        print(f"  Input   : {sentence[:50]}")
        print(f"  Tokens  : {tokens[:12]}")
        print(f"  IDs     : {ids[:12]}")
        print(f"  Length  : {len(ids)} tokens")
        print()


def compare_compression(char_tok, bpe_tok, text: str) -> None:
    """Show how BPE compresses sequences vs character-level."""
    print("\n── Compression Comparison ──")
    sample = text[:500]
    char_ids = char_tok.encode(sample)
    bpe_ids  = bpe_tok.encode(sample)
    print(f"  Characters      : {len(sample)}")
    print(f"  Char tokens     : {len(char_ids)}  (ratio 1.0x)")
    print(f"  BPE tokens      : {len(bpe_ids)}   (ratio {len(char_ids)/max(1,len(bpe_ids)):.2f}x more efficient)")
    print(f"  Compression gain: {(1 - len(bpe_ids)/len(char_ids))*100:.1f}%")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — HuggingFace Tokenizer Comparison
# ══════════════════════════════════════════════════════════════════════════════

def hf_tokenizer_demo() -> None:
    """
    Compare our BPE with the real LLaMA tokenizer from HuggingFace.
    Uses TinyLlama (free, no auth needed).
    """
    print("\n── HuggingFace Tokenizer (TinyLlama) ──")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        print(f"  Vocab size     : {tok.vocab_size:,}")

        examples = [
            "The quick brown fox",
            "GPU parallelization",
            "unbelievable",
        ]
        for text in examples:
            ids    = tok.encode(text)
            tokens = tok.convert_ids_to_tokens(ids)
            decoded = tok.decode(ids)
            print(f"\n  Input   : {text!r}")
            print(f"  Tokens  : {tokens}")
            print(f"  IDs     : {ids}")
            print(f"  Decoded : {decoded!r}")

    except Exception as e:
        print(f"  (Skipped — HF not available or network issue: {e})")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def get_training_text() -> str:
    """Use a small built-in text or a file if available."""
    data_path = "/workspace/data/shakespeare.txt"
    if os.path.exists(data_path):
        with open(data_path) as f:
            text = f.read()
        print(f"Loaded Shakespeare from {data_path} ({len(text):,} chars)")
        return text

    # Built-in fallback: excerpt from Shakespeare
    text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream—ay, there's the rub:
    For in that sleep of death what dreams may come
    When we have shuffled off this mortal coil
    Must give us pause—there's the respect
    That makes calamity of so long life.
    All the world's a stage, and all the men and women merely players;
    They have their exits and their entrances, and one man in his time plays many parts.
    What a piece of work is a man! How noble in reason, how infinite in faculty!
    In form and moving how express and admirable! In action how like an angel!
    In apprehension how like a god! The beauty of the world. The paragon of animals.
    Friends, Romans, countrymen, lend me your ears; I come to bury Caesar, not to praise him.
    The evil that men do lives after them; the good is oft interred with their bones.
    """ * 30  # repeat to give BPE enough data to learn patterns
    return text


def main():
    print("=" * 70)
    print("  Tutorial 01 — Tokenization & Byte-Pair Encoding")
    print("=" * 70)

    text = get_training_text()

    # ── Part A: Character Tokenizer ──────────────────────────────────────────
    print("\n" + "─" * 70)
    print("PART A — Character-Level Tokenizer")
    print("─" * 70)
    char_tok = CharTokenizer()
    char_tok.train(text)
    analyze_tokenization(char_tok, text, label="Character Tokenizer")

    # Verify round-trip
    sample = "Hello, world!"
    encoded = char_tok.encode(sample)
    decoded = char_tok.decode(encoded)
    print(f"  Round-trip test: {sample!r} → {encoded} → {decoded!r}")
    assert sample == decoded, "Round-trip failed!"
    print("  Round-trip: PASS\n")

    # ── Part B: BPE Tokenizer ────────────────────────────────────────────────
    print("─" * 70)
    print("PART B — Byte-Pair Encoding Tokenizer (vocab_size=300)")
    print("─" * 70)
    bpe_tok = BPETokenizer(vocab_size=300)
    bpe_tok.train(text)

    analyze_tokenization(bpe_tok, text, label="BPE Tokenizer")

    # ── Part C: Compression comparison ──────────────────────────────────────
    print("─" * 70)
    print("PART C — Compression: Char vs BPE")
    print("─" * 70)
    compare_compression(char_tok, bpe_tok, text)

    # ── Part D: Save / Load ──────────────────────────────────────────────────
    save_path = "/workspace/data/bpe_tokenizer.json"
    bpe_tok.save(save_path)

    loaded_tok = BPETokenizer()
    loaded_tok.load(save_path)

    ids_orig   = bpe_tok.encode("deep learning is powerful")
    ids_loaded = loaded_tok.encode("deep learning is powerful")
    assert ids_orig == ids_loaded, "Save/Load mismatch!"
    print("  Save/Load round-trip: PASS")

    # ── Part E: HuggingFace comparison ──────────────────────────────────────
    print("\n" + "─" * 70)
    print("PART D — Real LLaMA Tokenizer (HuggingFace)")
    print("─" * 70)
    hf_tokenizer_demo()

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("""
  Character tokenizer:
    + Zero training needed
    + Perfect round-trip
    - Huge sequences (1 char = 1 token)
    - No subword structure

  BPE tokenizer (what GPT/LLaMA uses):
    + Compact sequences (~4 chars/token on average)
    + Handles rare words by decomposing into subwords
    + Vocabulary is learned from data
    - Requires training
    - Encoding is slightly slower

  Key numbers to remember:
    LLaMA-2 vocab size : 32,000 tokens
    LLaMA-3 vocab size : 128,256 tokens (better multilingual coverage)
    GPT-4  vocab size  : ~100,000 tokens

  Next → Tutorial 02: Attention Mechanism
    python3 /workspace/pytorch_llm/tutorials/02_attention_mechanism.py
""")


if __name__ == "__main__":
    main()
