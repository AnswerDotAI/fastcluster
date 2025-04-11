import time

import torch
import numpy as np

from fastkmeans.triton_kernels import chunked_kmeans_kernel

def _get_device(preset: str | int | torch.device | None = None):
    if isinstance(preset, torch.device):
        return preset
    if isinstance(preset, str):
        return torch.device(preset)
    if torch.cuda.is_available(): # cuda currently handles both AMD and NVIDIA GPUs
        return torch.device(f"cuda:{preset if isinstance(preset, int) and preset < torch.cuda.device_count() else 0}")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return torch.device(f'xpu:{preset if isinstance(preset, int) and preset < torch.xpu.device_count() else 0}')
    return torch.device('cpu')


def _is_bfloat16_supported(device:torch.device):
    if device.type == 'cuda':
        return torch.cuda.is_bf16_supported()
    elif device.type == 'xpu' and hasattr(torch.xpu, 'is_bf16_supported'):
        return torch.xpu.is_bf16_supported()
    else:
        return False


@torch.inference_mode()
def _kmeans_torch_double_chunked(
    data: torch.Tensor,
    data_norms: torch.Tensor,
    k: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
    max_iters: int = 25,
    tol: float = 1e-8,
    chunk_size_data: int = 50_000,
    chunk_size_centroids: int = 10_000,
    max_points_per_centroid: int = 256,
    verbose: bool = False,
    use_triton: bool | None = None,
):
    """
    An efficient kmeans implementation that minimises OOM risks on modern hardware by using conversative double chunking.

    Returns
    -------
    centroids_cpu : torch.Tensor, shape (k, n_features), float32
    labels_cpu    : torch.Tensor, shape (n_samples_used,), long
        Where n_samples_used can be smaller than the original if subsampling occurred.
    """

    if dtype is None:
        dtype = torch.float16 if device.type in ['cuda', 'xpu'] else torch.float32

    n_samples_original, n_features = data.shape
    n_samples = n_samples_original

    if max_points_per_centroid is not None and n_samples > k * max_points_per_centroid:
        target_n_samples = k * max_points_per_centroid
        perm = torch.randperm(n_samples, device=data.device)
        indices = perm[:target_n_samples]
        data = data[indices]
        data_norms = data_norms[indices]
        n_samples = target_n_samples
        del perm, indices

    if n_samples < k:
        raise ValueError(f"Number of training points ({n_samples}) is less than k ({k}).")

    # centroid init -- random is the only supported init
    rand_indices = torch.randperm(n_samples)[:k]
    centroids = data[rand_indices].clone().to(device=device, dtype=dtype)
    prev_centroids = centroids.clone()

    labels = torch.empty(n_samples, dtype=torch.int64, device='cpu')  # Keep labels on CPU

    for iteration in range(max_iters):
        iteration_start_time = time.time()

        centroid_norms = (centroids ** 2).sum(dim=1)
        cluster_sums = torch.zeros((k, n_features), device=device, dtype=torch.float32)
        cluster_counts = torch.zeros((k,), device=device, dtype=torch.float32)

        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + chunk_size_data, n_samples)

            data_chunk = data[start_idx:end_idx].to(device=device, dtype=dtype, non_blocking=True)
            data_chunk_norms = data_norms[start_idx:end_idx].to(device=device, dtype=dtype, non_blocking=True)
            batch_size = data_chunk.size(0)
            best_ids = torch.zeros((batch_size,), device=device, dtype=torch.int64)

            if use_triton:
                chunked_kmeans_kernel(
                    data_chunk=data_chunk,
                    data_chunk_norms=data_chunk_norms,
                    centroids=centroids,
                    centroids_sqnorm=centroid_norms,
                    best_ids=best_ids,
                )
            else:
                best_dist = torch.full((batch_size,), float('inf'), device=device, dtype=dtype)
                c_start = 0
                while c_start < k:
                    c_end = min(c_start + chunk_size_centroids, k)
                    centroid_chunk = centroids[c_start:c_end]
                    centroid_chunk_norms = centroid_norms[c_start:c_end]

                    dist_chunk = data_chunk_norms.unsqueeze(1) + centroid_chunk_norms.unsqueeze(0)
                    dist_chunk = dist_chunk.addmm_(data_chunk, centroid_chunk.t(), alpha=-2.0, beta=1.0)

                    local_min_vals, local_min_ids = torch.min(dist_chunk, dim=1)
                    improved_mask = local_min_vals < best_dist
                    best_dist[improved_mask] = local_min_vals[improved_mask]
                    best_ids[improved_mask] = (c_start + local_min_ids[improved_mask])

                    c_start = c_end

            cluster_sums.index_add_(0, best_ids, data_chunk.float())
            cluster_counts.index_add_(0, best_ids, torch.ones_like(best_ids, device=device, dtype=torch.float32))

            labels[start_idx:end_idx] = best_ids.to('cpu', non_blocking=True)
            start_idx = end_idx

        new_centroids = torch.zeros_like(centroids, device=device, dtype=dtype)
        non_empty = (cluster_counts > 0)
        new_centroids[non_empty] = (cluster_sums[non_empty] / cluster_counts[non_empty].unsqueeze(1)).to(dtype=dtype)

        empty_ids = (~non_empty).nonzero(as_tuple=True)[0]
        if len(empty_ids) > 0:
            reinit_indices = torch.randint(0, n_samples, (len(empty_ids),), device='cpu')
            random_data = data[reinit_indices].to(device=device, dtype=dtype, non_blocking=True)
            new_centroids[empty_ids] = random_data

        shift = torch.norm(new_centroids - prev_centroids.to(new_centroids.device), dim=1).sum().item()
        centroids = new_centroids

        prev_centroids = centroids.clone()

        iteration_time = time.time() - iteration_start_time
        if verbose:
            print(f"Iteration {iteration+1}/{max_iters} took {iteration_time:.4f}s, total time: {time.time() - iteration_start_time + iteration_time:.4f}s, shift: {shift:.6f}")

        if shift < tol:
            if verbose:
                print(f"Converged after {iteration+1} iterations (shift: {shift:.6f} < tol: {tol})")
            break

    centroids_cpu = centroids.to('cpu', dtype=torch.float32)
    return centroids_cpu, labels


class FastKMeans:
    """
    A drop-in replacement for Faiss's Kmeans API, implemented with PyTorch
    double-chunked KMeans under the hood.

    Parameters
    ----------
    d  : int
        Dimensionality of the input features (n_features).
    k  : int
        Number of clusters.
    niter : int, default=20
        Maximum number of iterations.
    tol : float, default=1e-4
        Stopping threshold for centroid movement.
    gpu : bool, default=True
        Whether to force GPU usage if available. If False, CPU is used.
    seed : int, default=0
        Random seed for centroid initialization and (if needed) subsampling.
    max_points_per_centroid : int, optional, default=1_000_000_000
        If n_samples > k * max_points_per_centroid, the data will be subsampled to exactly
        k * max_points_per_centroid points before clustering.
    chunk_size_data : int, default=50_000
        Chunk size along the data dimension for assignment/update steps.
    chunk_size_centroids : int, default=10_000
        Chunk size along the centroid dimension for assignment/update steps.
    use_triton : bool | None, default=None
       Use the fast Triton backend for the assignment/update steps.
       If None, the Triton backend will be enabled for modern GPUs.
    """

    def __init__(
        self,
        d: int,
        k: int,
        niter: int = 25,
        tol: float = 1e-8,
        gpu: bool = True,
        seed: int = 0,
        max_points_per_centroid: int = 256,
        chunk_size_data: int = 50_000,
        chunk_size_centroids: int = 10_000,
        device: str | int | torch.device | None = None,
        dtype: torch.dtype = None,
        pin_gpu_memory: bool = True,
        verbose: bool = False,
        nredo: int = 1, # for compatibility only
        use_triton: bool | None = None,
    ):
        self.d = d
        self.k = k
        self.niter = niter
        self.tol = tol
        self.seed = seed
        self.max_points_per_centroid = max_points_per_centroid
        self.chunk_size_data = chunk_size_data
        self.chunk_size_centroids = chunk_size_centroids
        self.device = _get_device("cpu" if gpu is False else device)
        self.centroids = None
        self.dtype = dtype
        self.pin_gpu_memory = pin_gpu_memory
        self.verbose = verbose
        if use_triton is not False:
            use_triton = _is_bfloat16_supported(self.device) # assume triton is supported if GPU supports bfloat16
        self.use_triton = use_triton
        if nredo != 1:
            raise ValueError("nredo must be 1, redos not currently supported")

    def train(self, data: np.ndarray):
        """
        Trains (fits) the KMeans model on the given data and sets `self.centroids`. Designed to mimic faiss's `train()` method.

        Parameters
        ----------
        data : np.ndarray of shape (n_samples, d), float32
        """
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        np.random.seed(self.seed)

        # Move data to PyTorch CPU Tensor
        data_torch = torch.from_numpy(data)
        data_norms_torch = (data_torch ** 2).sum(dim=1)

        device = _get_device(self.device)
        if device == 'cuda' and self.pin_gpu_memory:
            data_torch = data_torch.pin_memory()
            data_norms_torch = data_norms_torch.pin_memory()

        centroids, _ = _kmeans_torch_double_chunked(
            data_torch,
            data_norms_torch,
            k=self.k,
            max_iters=self.niter,
            tol=self.tol,
            device=device,
            dtype=self.dtype,
            chunk_size_data=self.chunk_size_data,
            chunk_size_centroids=self.chunk_size_centroids,
            max_points_per_centroid=self.max_points_per_centroid,
            verbose=self.verbose,
            use_triton=self.use_triton,
        )
        self.centroids = centroids.numpy()

    def fit(self, data: np.ndarray):
        """
        Same as train(), included for interface similarity with scikit-learn's `fit()`.
        """
        self.train(data)
        return self

    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Assigns each data point to the nearest centroid for even more compatibility with scikit-learn's `predict()`, which is what cool libraries do.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,), int64
        """
        if self.centroids is None:
            raise RuntimeError("Must call train() or fit() before predict().")

        data_torch = torch.from_numpy(data)
        data_norms_torch = (data_torch ** 2).sum(dim=1)

        # We'll do a chunked assignment pass, similar to the main loop, but no centroid updates
        centroids_torch = torch.from_numpy(self.centroids)
        centroids_torch = centroids_torch.to(device=self.device, dtype=torch.float32)
        centroid_norms = (centroids_torch ** 2).sum(dim=1)

        n_samples = data_torch.shape[0]
        labels = torch.empty(n_samples, dtype=torch.long, device='cpu')

        start_idx = 0
        while start_idx < n_samples:
            end_idx = min(start_idx + self.chunk_size_data, n_samples)

            data_chunk = data_torch[start_idx:end_idx].to(device=self.device, dtype=torch.float32, non_blocking=True)
            data_chunk_norms = data_norms_torch[start_idx:end_idx].to(device=self.device, dtype=torch.float32, non_blocking=True)
            batch_size = data_chunk.size(0)
            best_ids = torch.zeros((batch_size,), device=self.device, dtype=torch.long)

            if self.use_triton:
                chunked_kmeans_kernel(
                    data_chunk,
                    data_chunk_norms,
                    centroids_torch,
                    centroid_norms,
                    best_ids,
                )
            else:
                best_dist = torch.full((batch_size,), float('inf'), device=self.device, dtype=torch.float32)
                c_start = 0
                k = centroids_torch.shape[0]
                while c_start < k:
                    c_end = min(c_start + self.chunk_size_centroids, k)
                    centroid_chunk = centroids_torch[c_start:c_end]
                    centroid_chunk_norms = centroid_norms[c_start:c_end]

                    dist_chunk = data_chunk_norms.unsqueeze(1) + centroid_chunk_norms.unsqueeze(0)
                    dist_chunk = dist_chunk.addmm_(data_chunk, centroid_chunk.t(), alpha=-2.0, beta=1.0)

                    local_min_vals, local_min_ids = torch.min(dist_chunk, dim=1)
                    improved_mask = local_min_vals < best_dist
                    best_dist[improved_mask] = local_min_vals[improved_mask]
                    best_ids[improved_mask] = (c_start + local_min_ids[improved_mask])
                    c_start = c_end

            labels[start_idx:end_idx] = best_ids.to('cpu')
            start_idx = end_idx

        return labels.numpy()

    def fit_predict(self, data: np.ndarray) -> np.ndarray:
        """
        Chains fit and predict, once again inspired by the great scikit-learn.
        """
        self.fit(data)
        return self.predict(data)
