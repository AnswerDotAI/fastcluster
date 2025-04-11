import torch
import triton
import triton.language as tl


@triton.heuristics(
    {
        "BLOCK_M": lambda a: 128 if a["D"] <= 128 else (64 if a["D"] <= 512 else 32),
        "BLOCK_N": lambda a: 64 if a["D"] <= 128 else 32,
        "num_warps": lambda a: 4 if a["D"] <= 64 else 8,
        "num_stages": lambda a: 2 if a["D"] < 64 else 1,
    }
)
@triton.jit
def _chunked_kmeans_kernel(
    data_ptr,  # [B, D], row-major
    x_norm_ptr,  # [B], precomputed L2 norms of data
    centroids_ptr,  # [C, D], row-major
    centroids_sqnorm_ptr,  # [C], precomputed L2 norms of centroids
    best_ids_ptr,  # [B], int32 (to store best centroid indices)
    B,  # number of data points
    C,  # number of centroids
    D: tl.constexpr,  # dimension, or number of features
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Each Triton block processes BLOCK_M rows of data. The kernel:
      1) loads those rows (and their precomputed norms from x_norm_ptr),
      2) loops over all centroids in chunks of BLOCK_N,
      3) computes distances, finds the best centroid,
      4) writes out the best centroid index for each data point.
    """
    # 1) Identify which data rows this block handles
    block_id = tl.program_id(axis=0)
    row_start = block_id * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    mask = rows < B

    # 2) Load data rows and precomputed x_norm: shape [BLOCK_M, D]
    row_offsets = rows[:, None] * D + tl.arange(0, D)
    x = tl.load(data_ptr + row_offsets, mask=mask[:, None], other=0.0)

    # shape: [BLOCK_M]
    x_norm = tl.load(x_norm_ptr + rows, mask=mask, other=0.0)

    # Prepare "best distance" + "best index"
    best_dist = tl.full([BLOCK_M], 1e38, dtype=tl.float32)
    best_idx = tl.zeros([BLOCK_M], dtype=tl.int64)

    # 3) Iterate over the centroids in chunks of BLOCK_N
    for chunk in range(0, C, BLOCK_N):
        cids = chunk + tl.arange(0, BLOCK_N)
        c_mask = cids < C

        # Load sub-block of centroids: shape [BLOCK_N, D]
        c_offsets = cids[:, None] * D + tl.arange(0, D)
        cvals = tl.load(centroids_ptr + c_offsets, mask=c_mask[:, None], other=0.0).to(x.dtype)

        # Load centroid norms: shape [BLOCK_N]
        c_sqnorm = tl.load(centroids_sqnorm_ptr + cids, mask=c_mask, other=0.0).to(x.dtype)

        # Compute distance = x_norm + c_sqnorm - 2 * dot(x, c)
        dots = tl.dot(x, tl.trans(cvals))  # shape [BLOCK_M, BLOCK_N]
        dist_chunk = tl.fma(dots, -2.0, x_norm[:, None] + c_sqnorm[None, :])

        # Find the argmin along the BLOCK_N dimension
        local_min_vals, local_min_idx = tl.min(dist_chunk, axis=1, return_indices=True)

        improved = local_min_vals < best_dist
        best_dist = tl.where(improved, local_min_vals, best_dist)
        best_idx = tl.where(improved, chunk + local_min_idx, best_idx)

    # 4) Write out the best centroid indices
    tl.store(best_ids_ptr + rows, best_idx, mask=mask)


def chunked_kmeans_kernel(
    data_chunk: torch.Tensor,
    data_chunk_norms: torch.Tensor,
    centroids: torch.Tensor,
    centroids_sqnorm: torch.Tensor,
    best_ids: torch.Tensor,
):
    """
    Launches the Triton kernel to assign each point to its nearest centroid in one pass.

    best_ids: pre-allocated [B] (int32) to store the best centroid ID of each point.
    """
    B, D = data_chunk.shape
    C = centroids.shape[0]

    def grid(meta):
        return (triton.cdiv(B, meta["BLOCK_M"]),)

    _chunked_kmeans_kernel[grid](
        data_chunk,
        data_chunk_norms,
        centroids,
        centroids_sqnorm,
        best_ids,
        B,  # num_points
        C,  # num_centroids
        D,  # dimension, or number of features
    )
