# Hyperspace vs HNSW benchmark

Reproduce: `dotnet run -c Release --project Src/Hyperspace.Benchmark -- --n 20000 --dim 128 --queries 500 --out results.md`
(`--micro` times one distance call per storage mode; a build with `-p:DefineConstants=DIAG` adds `--evals`, which counts distance evaluations per query.)

Setup: 20,000 synthetic unit-norm vectors of 128 dimensions drawn around 100 Gaussian cluster centres, 500 queries from the same
distribution, exact ground truth by brute force, cosine distance, `M = 16`, `efConstruction = 100`, k = 10. One run on a shared 4-core
Linux VM (.NET 10). **Single run: QPS differences under roughly 20-30 % are within noise** (the Float32 row moved between 14k and 21k QPS
at ef 64 across four runs of unchanged code). Synthetic clustered data is not your embeddings; use the harness on your own vectors before
drawing conclusions.

How to read the columns:
* **Heap** is `GC.GetTotalMemory` after a forced compacting GC, before and after the build. HNSW holds the caller's vectors by reference, so the
  raw vectors (9.8 MiB) are added to its figure; Hyperspace stores its own copy, so its figure is complete. HNSW's construction-time distance
  cache is freed with `DisableDistanceCache()` first; both libraries have their layer 0 flattened (`OptimizeIfNeeded` / `FreezeFlatLayer0`).
* **Build** is wall-clock for the whole build (Hyperspace encodes in parallel and links sequentially; for PQ it includes codebook training).
* **QPS** counts complete single queries (`Search(query, k)`, which allocates its result array); the multi-thread column runs queries in parallel on 4 cores.
* `+ rerank x4` keeps the original float32 vectors (so it costs their memory) and re-scores 4k candidates exactly.
* Hyperspace rows use its defaults (`LevelProbability = 1/M`); the `p = 0.5` row is HyperspaceDB's value.

| Index | Build | Heap | B/vec | ef | Recall@10 | QPS 1 thread | QPS 4 threads |
|---|---:|---:|---:|---:|---:|---:|---:|
| HNSW.Net (SelectSimple) | 1.9 s | 19.0 MiB | 996 | 16 | 0.5216 | 39,856 | 122,365 |
|  |  |  |  | 32 | 0.5262 | 33,920 | 95,719 |
|  |  |  |  | 64 | 0.5274 | 21,754 | 73,523 |
|  |  |  |  | 128 | 0.5276 | 14,967 | 50,093 |
|  |  |  |  | 256 | 0.5276 | 12,826 | 42,052 |
| HNSW.Net (SelectHeuristic) | 2.5 s | 19.0 MiB | 996 | 16 | 0.9426 | 33,996 | 95,175 |
|  |  |  |  | 32 | 0.9924 | 21,735 | 68,317 |
|  |  |  |  | 64 | 0.9994 | 13,598 | 53,941 |
|  |  |  |  | 128 | 1.0000 | 11,210 | 34,735 |
|  |  |  |  | 256 | 1.0000 | 4,429 | 17,879 |
| Hyperspace Float32 | 2.3 s | 14.6 MiB | 767 | 16 | 0.9554 | 35,202 | 119,113 |
|  |  |  |  | 32 | 0.9938 | 22,806 | 86,513 |
|  |  |  |  | 64 | 0.9996 | 15,292 | 58,465 |
|  |  |  |  | 128 | 1.0000 | 13,433 | 42,875 |
|  |  |  |  | 256 | 1.0000 | 4,904 | 20,377 |
| Hyperspace Float32 (level p=0.5, as HyperspaceDB) | 3.2 s | 16.7 MiB | 875 | 16 | 0.9530 | 32,815 | 104,906 |
|  |  |  |  | 32 | 0.9934 | 21,414 | 74,404 |
|  |  |  |  | 64 | 0.9996 | 15,291 | 49,539 |
|  |  |  |  | 128 | 1.0000 | 9,372 | 35,381 |
|  |  |  |  | 256 | 1.0000 | 4,546 | 17,342 |
| Hyperspace Float64 (default) | 3.5 s | 24.2 MiB | 1271 | 16 | 0.9554 | 20,625 | 67,771 |
|  |  |  |  | 32 | 0.9938 | 14,689 | 58,933 |
|  |  |  |  | 64 | 0.9996 | 10,319 | 40,018 |
|  |  |  |  | 128 | 1.0000 | 6,377 | 28,995 |
|  |  |  |  | 256 | 1.0000 | 4,272 | 13,223 |
| Hyperspace Int8 | 2.0 s | 7.3 MiB | 383 | 16 | 0.9480 | 49,920 | 121,513 |
|  |  |  |  | 32 | 0.9804 | 26,451 | 88,128 |
|  |  |  |  | 64 | 0.9854 | 18,832 | 62,265 |
|  |  |  |  | 128 | 0.9860 | 11,152 | 44,840 |
|  |  |  |  | 256 | 0.9860 | 5,575 | 20,797 |
| Hyperspace Int4 | 8.0 s | 6.8 MiB | 354 | 16 | 0.8352 | 18,931 | 63,420 |
|  |  |  |  | 32 | 0.8504 | 15,628 | 47,909 |
|  |  |  |  | 64 | 0.8516 | 10,475 | 35,222 |
|  |  |  |  | 128 | 0.8518 | 7,593 | 26,302 |
|  |  |  |  | 256 | 0.8518 | 3,220 | 12,788 |
| Hyperspace Turbo | 6.9 s | 6.4 MiB | 335 | 16 | 0.7286 | 23,720 | 78,617 |
|  |  |  |  | 32 | 0.7378 | 18,358 | 64,752 |
|  |  |  |  | 64 | 0.7388 | 15,367 | 50,877 |
|  |  |  |  | 128 | 0.7390 | 10,385 | 34,634 |
|  |  |  |  | 256 | 0.7390 | 4,488 | 16,556 |
| Hyperspace Ternary | 4.1 s | 4.4 MiB | 229 | 16 | 0.2106 | 32,761 | 113,322 |
|  |  |  |  | 32 | 0.2874 | 23,483 | 74,168 |
|  |  |  |  | 64 | 0.3720 | 15,732 | 53,933 |
|  |  |  |  | 128 | 0.4288 | 11,298 | 37,795 |
|  |  |  |  | 256 | 0.4516 | 6,935 | 24,885 |
| Hyperspace Ternary + rerank x4 | 4.0 s | 14.7 MiB | 770 | 16 | 0.5778 | 17,439 | 59,978 |
|  |  |  |  | 32 | 0.5778 | 17,239 | 58,669 |
|  |  |  |  | 64 | 0.6948 | 12,452 | 47,909 |
|  |  |  |  | 128 | 0.8078 | 10,191 | 36,402 |
|  |  |  |  | 256 | 0.8516 | 6,919 | 24,019 |
| Hyperspace Binary | 7.3 s | 5.7 MiB | 300 | 16 | 0.3100 | 44,066 | 140,590 |
|  |  |  |  | 32 | 0.3132 | 31,247 | 103,349 |
|  |  |  |  | 64 | 0.3128 | 23,303 | 74,501 |
|  |  |  |  | 128 | 0.3128 | 15,533 | 49,445 |
|  |  |  |  | 256 | 0.3128 | 6,829 | 24,137 |
| Hyperspace Binary + rerank x4 | 7.4 s | 16.2 MiB | 849 | 16 | 0.6980 | 23,927 | 83,141 |
|  |  |  |  | 32 | 0.6980 | 23,091 | 80,201 |
|  |  |  |  | 64 | 0.6984 | 19,593 | 69,206 |
|  |  |  |  | 128 | 0.6984 | 12,501 | 45,804 |
|  |  |  |  | 256 | 0.6984 | 6,228 | 23,099 |
| Hyperspace PQ | 3.2 s | 3.8 MiB | 199 | 16 | 0.0290 | 34,638 | 119,439 |
|  |  |  |  | 32 | 0.0400 | 34,755 | 106,877 |
|  |  |  |  | 64 | 0.0548 | 26,886 | 80,416 |
|  |  |  |  | 128 | 0.0772 | 17,117 | 58,324 |
|  |  |  |  | 256 | 0.0954 | 11,692 | 40,628 |
| Hyperspace PQ + rerank x4 | 2.7 s | 14.1 MiB | 739 | 16 | 0.1010 | 26,698 | 80,051 |
|  |  |  |  | 32 | 0.1010 | 25,562 | 77,378 |
|  |  |  |  | 64 | 0.1194 | 21,633 | 70,652 |
|  |  |  |  | 128 | 0.1686 | 15,546 | 54,098 |
|  |  |  |  | 256 | 0.2198 | 10,820 | 38,504 |
| Hyperspace OPQ | 2.9 s | 3.6 MiB | 190 | 16 | 0.0364 | 34,270 | 112,744 |
|  |  |  |  | 32 | 0.0504 | 32,481 | 99,557 |
|  |  |  |  | 64 | 0.0642 | 25,647 | 78,762 |
|  |  |  |  | 128 | 0.0800 | 19,333 | 59,859 |
|  |  |  |  | 256 | 0.0964 | 11,834 | 40,350 |
| Hyperspace Int8 + rerank x4 | 2.3 s | 17.8 MiB | 932 | 16 | 0.9966 | 25,839 | 79,129 |
|  |  |  |  | 32 | 0.9966 | 21,341 | 81,621 |
|  |  |  |  | 64 | 0.9994 | 14,779 | 53,508 |
|  |  |  |  | 128 | 1.0000 | 13,556 | 37,145 |
|  |  |  |  | 256 | 1.0000 | 4,951 | 18,847 |

## What the numbers say

* **Like for like (HNSW `SelectHeuristic` vs Hyperspace `Float32`) the two are on par for search**: the same recall at every ef (0.95 at ef 16, 0.99 at 32,
  1.00 from 64) and single-thread QPS within noise of each other. Hyperspace now **builds as fast** (2.3 s vs 2.5 s) and its footprint is **23 % smaller**
  (14.6 MiB vs 19.0 MiB, 760 vs 996 bytes per vector).
* **HNSW's default neighbour selection (`SelectSimple`) stalls at about 0.5 recall on this clustered data** whatever the ef, because links stay
  inside clusters; `SelectHeuristic` (what Hyperspace always uses) does not. This depends on the data but is worth knowing.
* **`Float64`, Hyperspace's default storage, is about two thirds as fast and 1.7x larger than `Float32`** with identical recall. For ordinary Euclidean /
  cosine embeddings set `Quantization = Float32`; keep float64 for hyperbolic data near the ball boundary.
* **Quantized storage now pays off.** `Int8` is 2.5x smaller than `Float32` (7.3 vs 14.6 MiB), builds as fast and, at ef 16, searches faster (50k vs 35k QPS)
  for a recall of 0.985 instead of 1.0; with `+ rerank x4` it reaches 0.9994. TurboQuant and 4-bit keep 0.74 / 0.85 recall at 6.4 / 6.8 MiB. The
  aggressive codecs are fast but, on this tightly clustered data, inaccurate: ternary reaches 0.37 single-pass and 0.69 re-ranked, binary 0.31 and 0.70, and
  product quantization about 0.06 and 0.12 (8 sub-vectors of 16 dimensions against clusters whose members differ by less than the quantization error).
* **The graph is about 400 bytes per vector at 128 dimensions** (links and .NET object overhead), against 512 for a float32 vector, so even 8-byte PQ codes only
  reach about 190 B/vector. Quantization pays off more at higher dimensions.

## What changed since the first run of this benchmark, and what is and is not established

| | first run | now |
|---|---:|---:|
| Float32 build / heap | 4.4 s / 17.5 MiB | 2.3 s / 14.6 MiB |
| Int8 build / heap / QPS (ef 16, 1 thread) | 10.1 s / 10.2 MiB / 14.9k | 2.0 s / 7.3 MiB / 49.9k |
| Turbo build / QPS (ef 64) | 20.2 s / 5.1k | 6.9 s / 15.4k |
| Binary build / QPS (ef 64) | 32.2 s / 2.8k | 7.3 s / 23.3k |

Three changes are involved; they were not all ablated:

1. **`LevelProbability` defaulting to `1/M` instead of HyperspaceDB's 0.5** (measured on its own, Float32): build 4.1 s to 2.1 s, heap 16.8 to 14.5 MiB, same recall, queries about
   10 % faster (within noise). It stacks about 4 layers for 20,000 vectors instead of about 14.
2. **Freezing layer 0 no longer keeps a second copy** (measured: -0.7 MiB on 20,000 x 128; the first run's Float32 row held layer 0 twice).
3. **The quantized codecs no longer allocate on every query or insertion.** An unfiltered, non-hybrid `Search(query, Span<SearchResult>)` allocates 0 bytes once the
   calling thread is warm, for every storage mode (bytes per call: 0, against 424 for the array overload, which only allocates its result array); a prepared query, its rotation
   buffers and its 16-512 KB lookup tables are reused per thread. The vectorized / table-driven distance kernels were committed one step earlier and *by themselves changed
   nothing end to end* (previous run: Int4, Turbo, Ternary, Binary within noise, Int8 +16 %); the large gains appeared only with the allocation removal. That is consistent
   with per-query table allocation having cancelled the kernel gains, but I did not run a separate ablation to prove it.

The kernels themselves are fast. `--micro` (cache-resident vectors, one call): at 128 dimensions Float32 29 ns, Int8 37 ns, Binary 17 ns, Turbo 59 ns, Ternary 78 ns, Int4 114 ns.

## Not explained yet

* A Float32 query costs about 250 ns per distance evaluation end to end (`--evals`: 251 evaluations in 64 us at ef 64) against a 29 ns kernel, so most of a query is
  traversal and bookkeeping, not distance computation. A priority-queue push + pop is only about 19 ns, so that is not the main cost. This has not been profiled.
* Ternary and product-quantization recall dropped a little (0.41 to 0.37, 0.064 to 0.055 at ef 64) when the level default changed: their graphs are built from
  noisy distances and vary with the level draw; HNSW's `SelectSimple` moved 0.514 to 0.527 between runs of unchanged code, so differences of this size are not conclusive.
