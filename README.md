# fastkmeans

![Python Versions](https://img.shields.io/badge/Python-3.9_3.10_3.11_3.12_3.13-blue)
[![Twitter Follow](https://img.shields.io/twitter/follow/bclavie?style=social)](https://twitter.com/bclavie)
[![Twitter Follow](https://img.shields.io/twitter/follow/benjamin_warner?style=social)](https://twitter.com/benjamin_warner)
<!-- [![Downloads](https://static.pepy.tech/badge/fastkmeans/month)](https://pepy.tech/project/fastkmeans) -->

_A fast and efficient k-means implementation for PyTorch, with support for GPU and CPU._

---

Welcome to `fastkmeans`! This is an extremely tiny library, meant to be slotted-in anywhere you need "fast-enough" PyTorch native k-means clustering. It's compatible with any PyTorch-compatible CPU or GPU, matching or outperforming `faiss` by ~4-5× on a single GPU, and is without install woes, relying on just two dependencies you already have installed: `torch` and `numpy`.

###  Get started

```sh
[uv] pip install fastkmeans
```

... and that's all you need to do! `FastKMeans` is now ready to use.

###  So what does this do?

There's very, very little to this library. It provides a single interface, `FastKMeans`, which you can use by importing it from `fastkmeans`. This interface has been designed to be a slot-in replacement for existing FAISS implementations, while being *mostly* sklearn-compliant as well. Effectively, this means that three methods are exposed:

- train(): mimics the FAISS API, training the model on a dataset.
- fit(): mimics the sklearn API, training the model on a dataset.
- predict(): mimics the sklearn API, use the trained clusters to predict where new points belong.
- fit_predict(): mimics the sklearn API, chaining the two calls above.

#### Behaviour

Whenever possible, the library attempts to mimic the FAISS API, albeit with a bit more flexibility. We encourage you to check out the [API docstring](https://github.com/AnswerDotAI/fastkmeans/blob/main/fastkmeans/kmeans.py#L170) to see what the arguments are, as they are straightforward.

The default behaviour of the library mostly follows faiss's, including downsampling data to a maximum of 256 points per centroid to speed up calculations, which can be freely modified and/or disabled.  The only major difference is that, by default, the library does adopt an early stopping mechanism based on a `tol` parameter, which stops the algorithm when the centroids don't move more than `tol` between iterations. This is unlike faiss', whose default behaviour is to run for a fixed number of iterations no matter what -- you can restore this behaviour by setting `tol` to -1.

#### Chunking

The algorith is implemented with a double-chunking logics, where both the data points and the centroids are split into moderately-sized chunks, avoiding the risks of OOMs. The defaults allow you to cluster 26_214_400 128-dimensional points into 262_144 clusters with ~11GB memory usage (including storing the data in fp32). You can check out the available arguments [here](https://github.com/AnswerDotAI/fastkmeans/blob/17c2a1b4cabc84c7bb0cb392fd2d2e3ec4d1b825/fastkmeans/kmeans.py#L132) to see how to tweak these. As a rule of thumb, increasing chunk sizes will speed up computations, at the cost of memory usage, and decreasing it will have the reverse effect.

> Note: See the Triton section below for triton specific chunking details.

###  Why `fastkmeans`?

The main motivation behind `fastkmeans` is having a considerably easier way to package late-interaction models, such as ColBERT, in both its Stanford implementation, its PyLate implentation, and for the RAGatouille high-level API. The use of clustering for ColBERT is perhaps somewhat peculiar, as it relies on **large numbers of clusters** for relatively few data points (~100 per cluster centre). This has been a major problem in getting ColBERT to be more usable, as the existing alternatives, while great in their own merit, have flaws for this particular use:

- `faiss` is highly-optimized and is the original library used by the ColBERT authors and most implementations nowadays. However, it is rather difficult to install as there are no "perfect" PyPi wheels, with many segfault issues reported, as the official install is only supported via conda or from source. It can also be finnicky, causing issues with PyTorch if not installed via conda too. Finally, and this is a major problem for maintaining libraries such as ragatouille: it requires different packages and different install methods for its `cpu` and `gpu` variant, meaning additional friction for users and the inability to provide a nice default.
- `fast-pytorch-kmeans` is a great library which provides lightning fast kmeans implementation in PyTorch. However, it relies on highly vectorized operations which are exceedingly memory hungry, and consistently OOMs on consumer hardware when trying to index even a moderate number of colbert documents (or produces suboptimal clusters with minibatching).
- `scikit-learn`, while being the ML giant whose shoulders we all stand on, only supports CPU. This becomes unusably slow when indexing larger volumes of documents, especially as there'll (almost) always be a GPU available in these situations.

There are some libraries (such as NVidia's own implementations), but they again require more dependencies than we'd like for nimble packaging, and/or are less flexible in terms of hardware.

### Triton Kernel

`fastkmeans`'s Triton kmeans kernel is ~4-5 times faster than single-GPU `faiss` or `fastkmeans`'s PyTorch backend. On a modern GPU (Ampere or newer), the Triton backend is enabled by default.

While the Triton kernel uses significantly less memory than the PyTorch implementation, increasing the chunk size above 512K can result in slower performance.

> Note: Use PyTorch 2.7.1 or newer if using the Triton backend on AMD or Intel GPUs.

### Speed

Below is `fastkmeans` benchmarked against `faiss` on a single RTX 4090 GPU, with 128-dimensional data points at various data scales that will commonly be used in ColBERT-style indexing (8192, 16384, 32768, 65536, and 131072 clusters, each with w/ cluster_size*100 data points).

![fastkmeans benchmark](./benchmark_plots/4090_benchmark.png)

#### Benchmarking

To benchmark `fastkmeans` against `faiss` on your own machine, install faiss and PyTorch 2.5 via the `bench_env.yaml` Conda environment:

```bash
conda env create -f bench_env.yaml
conda activate fastkmeans
pip install fastkmeans
```

Then, run the benchmark script:

```bash
CUDA_VISIBLE_DEVICES=0 python speedbench.py --do-faiss --do-fastkmeans --do-fastkmeans-triton --do-evals
```

If you don't need to benchmark faiss or do not have a Nvidia GPU, you can install `fastkmeans` with the minimalbenchmark dependencies:

```bash
[uv] pip install fastkmeans[bench]
```
and run the benchmark script:

```bash
python speedbench.py --do-fastkmeans  --do-evals
```

###  Limitations

- On a few toy datasets & MNIST, `fastkmeans` reaches roughly the same NMI as `faiss` and `scikit-learn`, indicating that it creates at least somewhat coherent clusters. However, it's not extensively tested, especially in non-ColBERT uses, so your mileage may vary.
- The "chunking" defaults to avoid OOMs is rather simplistic, and you might need to tweak the numbers depending on your hardware and dataset size.
- The library currently assumes you'll be using it either on a CPU or a single GPU. Multiple GPUs don't currently provide a major speed-up, this might change in the future, though we expect users wanting to index 10M+ documents to likely have the more robust `faiss` available on their machine anyway.

### Citation

If you use fastmeans and want to/need to cite it in your work, please feel free to use the citation below:

```bibtex
@misc{fastkmeans2025,
  author = {Benjamin Clavié and Benjamin Warner},
  title = {fastkmeans: Accelerated KMeans Clustering in PyTorch and Triton},
  year = {2025},
  howpublished = {\url{https://github.com/AnswerDotAI/fastkmeans/}}
}
```