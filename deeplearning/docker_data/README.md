# Docker Image Loading Issue Resolution

## Overview

This document describes the resolution of a **"no space left on
device"** error encountered while loading a large Docker image (TensorRT
LLM) on a CentOS system.

## Environment

-   **Server:** JF5300-B11A371T\
-   **OS:** CentOS\
-   **Working Directory:**
    `/data/home/ralfahad/projects/deeplearning/docker_data`\
-   **Image:** `trtllm.tar` (TensorRT LLM release 0.18.0)

## Problem Description

### Initial Error

``` bash
cd docker_data/
sudo docker load -i trtllm.tar
```

**Error Message:**

    failed to ingest "blobs/sha256/573ed973781f112a78ec51d9cf4a8ef2851613078afd97d276f9d5731fa0a996": failed to copy: failed to send write: write /var/lib/containerd/io.containerd.content.v1.content/ingest/9fde8cbc2b0032ce0d76aab15da633eff0f13142d17588068517a35e8b06f9c5/data: no space left on device

## Root Cause Analysis

-   Root partition (`/`) was **87% full** with only **8.8G** total
    space.\
-   Docker's containerd stores temporary image ingestion data under
    `/var/lib/containerd`, located on the root partition.\
-   The Docker image size (96.9GB) required far more space than
    available.

## Storage Layout Before Fix

    NAME   FSTYPE  FSAVAIL  FSUSE%  MOUNTPOINTS
    sda    xfs     886.9G       1%   /home
    sdb4   xfs       8.8G      87%   /
    sdb5   xfs     664.6G       1%   /storage

------------------------------------------------------------------------

## Solution

### 1. Stop Docker Services

``` bash
sudo systemctl stop docker containerd
```

### 2. Relocate Containerd Data Directory

``` bash
# Create new containerd directory on /home partition
sudo mkdir -p /home/containerd

# Move existing containerd data to backup location
sudo mv /var/lib/containerd /home/containerd-backup 2>/dev/null || true

# Create symbolic link from original location to new location
sudo ln -sf /home/containerd /var/lib/containerd
```

### 3. Restart Services

``` bash
sudo systemctl start containerd docker
```

### 4. Load Docker Image

``` bash
sudo docker load -i trtllm.tar
```

------------------------------------------------------------------------

## Results

### Success Confirmation

``` bash
docker images
```

**Output:**

    IMAGE ID          DISK USAGE   CONTENT SIZE
    b015ee56a403      96.9GB       48.2GB   tensorrt_llm/release:0.18.0

### Verification Commands

``` bash
docker images
docker ps -a
```

## Storage Layout After Fix

-   **Docker Root Dir:** `/home/docker` (886.9G available)\
-   **Containerd Data:** `/home/containerd` (symlinked from
    `/var/lib/containerd`)\
-   **Root Partition:** No longer constrained by temporary image
    ingestion data

------------------------------------------------------------------------

## Key Learnings

### Technical Insights

-   Containerd uses `/var/lib/containerd` for temporary image extraction
    storage.
-   Large Docker images require substantial temporary disk space.
-   Symbolic links allow transparent relocation without modifying Docker
    configs.

### Best Practices

-   Monitor root partition disk usage frequently.
-   Store Docker and containerd data on large-capacity partitions.
-   Assess your storage layout using `df -h` and `lsblk -f` before
    loading big images.

------------------------------------------------------------------------

## Troubleshooting Commands

### Check Disk Space

``` bash
df -h
lsblk -f
du -sh /var/lib/docker
```

### Docker Information

``` bash
docker info | grep "Docker Root Dir"
docker system df
docker images
```

------------------------------------------------------------------------

## Alternative Solutions

If the symbolic linking method does not work, consider: - **Docker
Daemon Configuration:** Modify `/etc/docker/daemon.json` to set
`data-root` - **Containerd Configuration:** Update
`/etc/containerd/config.toml` to change root directory - **Temporary
Directory Override:** Set `TMPDIR` for Docker service

------------------------------------------------------------------------

## Status

✅ **Resolved**

## Date

\[Current Date\]

## Author

System Administrator
