from __future__ import annotations

from contextlib import nullcontext

import torch
import triton
import triton.language as tl


def device_guard(tensor: torch.Tensor):
    """Returns context manager to ensure that the Triton kernel launches on the correct device."""
    if tensor.is_cuda:  # NVIDIA or AMD/ROCm
        return torch.cuda.device_of(tensor)
    elif hasattr(tensor, "is_xpu") and tensor.is_xpu:  # Intel GPUs
        return torch.xpu.device_of(tensor)
    else:  # CPU or other back-ends
        return nullcontext()


@triton.heuristics(
    {
        "BLOCK_M": lambda x: 128 if x["D"] <= 384 else 64,
        "BLOCK_N": lambda x: 128 if x["D"] <= 384 else 64,
        "BLOCK_K": lambda x: 16 if x["D"] <= 32 or x["D"] > 384 else 32,
        "GROUP_SIZE_M": lambda x: 8 if x["D"] <= 32 else 16,
        "num_warps": lambda x: 4,
    }
)
@triton.jit
def _kmeans_kernel(
    x_ptr,
    x_norm_ptr,
    c_ptr,
    c_norm_ptr,
    best_dist_ptr,
    best_idx_ptr,
    B,
    C,
    D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Map flat CTA id to (pid_m, pid_n) in “grouped” launch order
    pid = tl.program_id(axis=0)

    num_pid_m = tl.cdiv(B, BLOCK_M)  # row-tiles
    num_pid_n = tl.cdiv(C, BLOCK_N)  # centroid-tiles

    # Super-group into GROUP_SIZE_M blocks to minimize loading from global memory
    num_pid_in_grp = GROUP_SIZE_M * num_pid_n
    first_pid_m = (pid // num_pid_in_grp) * GROUP_SIZE_M
    group_rows = min(num_pid_m - first_pid_m, GROUP_SIZE_M)

    pid_m = first_pid_m + ((pid % num_pid_in_grp) % group_rows)  # row-tile index
    pid_n = (pid % num_pid_in_grp) // group_rows  # centroid-tile index

    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    rows = row_start + tl.arange(0, BLOCK_M)
    cols = col_start + tl.arange(0, BLOCK_N)

    row_mask = rows < B
    col_mask = cols < C

    # load norms
    x_n = tl.load(x_norm_ptr + rows, mask=row_mask, other=0.0)  # [BM]
    c_n = tl.load(c_norm_ptr + cols, mask=col_mask, other=0.0)  # [BN]

    # pipelined K‑loop, will hold partial dot‑products in registers
    dot_acc = tl.zeros([BLOCK_M, BLOCK_N], tl.float32)

    # compute matmul tiled across SMs
    for k0 in range(0, D, BLOCK_K):
        k_range = k0 + tl.arange(0, BLOCK_K)

        # load X slice
        x_ptrs = x_ptr + rows[:, None] * D + k_range[None, :]
        xk = tl.load(x_ptrs, mask=row_mask[:, None]).to(tl.float16)

        # load C slice
        c_ptrs = c_ptr + cols[:, None] * D + k_range[None, :]
        ck = tl.load(c_ptrs, mask=col_mask[:, None]).to(tl.float16)

        # accumulate
        dot_acc += tl.dot(xk, tl.trans(ck), out_dtype=tl.float32)

    # finish distance formula
    dist = tl.fma(dot_acc, -2.0, x_n[:, None] + c_n[None, :])  # [BM, BN]

    # set large value for invalid distances
    dist = tl.where(col_mask[None, :], dist, 1e38)

    # local arg‑min (inside this tile)
    tile_min, tile_idx = tl.min(dist, axis=1, return_indices=True)

    # compete with global best using atomics
    prev = tl.atomic_min(best_dist_ptr + rows, tile_min, mask=row_mask)
    improved = tile_min < prev

    # update best_ids
    tl.store(best_idx_ptr + rows, tl.where(improved, col_start + tile_idx, tl.load(best_idx_ptr + rows)), mask=row_mask)


def triton_kmeans(
    data_chunk: torch.Tensor,
    data_chunk_norms: torch.Tensor,
    centroids: torch.Tensor,
    centroids_sqnorm: torch.Tensor,
    best_ids: torch.Tensor,
):
    B, D = data_chunk.shape
    C = centroids.shape[0]
    best_dist = torch.full((B,), 1e38, device=data_chunk.device, dtype=torch.float32)

    def grid(meta):
        return (triton.cdiv(B, meta["BLOCK_M"]) * triton.cdiv(C, meta["BLOCK_N"]),)  # 1D grid

    # Without this Triton always tries to launch from device:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)
    with device_guard(data_chunk):
        _kmeans_kernel[grid](
            data_chunk,
            data_chunk_norms,
            centroids,
            centroids_sqnorm,
            best_dist,
            best_ids,
            B,
            C,
            D,
        )
