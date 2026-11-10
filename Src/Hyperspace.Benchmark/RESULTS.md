# Hyperspace vs HNSW benchmark

Reproduce: `dotnet run -c Release --project Src/Hyperspace.Benchmark -- --n 20000 --dim 128 --queries 500 --out results.md`
(`--micro` times one distance call per storage mode; a build with `-p:DefineConstants=DIAG` adds `--evals`, which counts distance evaluations per query.)

Setup: 20,000 synthetic unit-norm vectors of 128 dimensions drawn around 100 Gaussian cluster centres, 500 queries from the same
distribution, exact ground truth by brute force, cosine distance, `M = 16`, `efConstruction = 100`, k = 10. One run on a shared 4-core
Linux VM (.NET 10). **Single run: QPS differences under roughly 20-30 % are within noise** (the Float32 row moved between 13k and 21k QPS at ef 64 across five runs, and HNSW's own rows, on code that never changed, moved by up to 30 %). Synthetic clustered data is not your embeddings; use the harness on your own vectors before
drawing conclusions.

How to read the columns:
* **Heap** is `GC.GetTotalMemory` after a forced compacting GC, before and after the build. HNSW holds the caller's vectors by reference, so the
  raw vectors (9.8 MiB) are added to its figure; Hyperspace stores its own copy, so its figure is complete. HNSW's construction-time distance
  cache is freed with `DisableDistanceCache()` first; both libraries have their layer 0 flattened (`OptimizeIfNeeded` / `FreezeFlatLayer0`).
* **Build** is wall-clock for the whole build (Hyperspace encodes in parallel and links sequentially; for PQ it includes codebook training).
* **QPS** counts complete single queries (`Search(query, k)`, which allocates its result array); the multi-thread column runs queries in parallel on 4 cores.
* `+ rerank` keeps the original float32 vectors (so it costs their memory) and re-scores the codec's default pool of candidates exactly (4k for Int8, 25k for ternary, binary and product quantization).
* Hyperspace rows use its defaults (`LevelProbability = 1/M`); the `p = 0.5` row is HyperspaceDB's value.

| Index | Build | Heap | B/vec | ef | Recall@10 | QPS 1 thread | QPS 4 threads |
|---|---:|---:|---:|---:|---:|---:|---:|
| HNSW.Net (SelectSimple) | 1.9 s | 19.0 MiB | 996 | 16 | 0.5460 | 37,865 | 113,816 |
|  |  |  |  | 32 | 0.5512 | 24,866 | 79,485 |
|  |  |  |  | 64 | 0.5528 | 18,933 | 58,716 |
|  |  |  |  | 128 | 0.5532 | 16,590 | 42,423 |
|  |  |  |  | 256 | 0.5532 | 11,028 | 33,760 |
| HNSW.Net (SelectHeuristic) | 2.6 s | 19.0 MiB | 996 | 16 | 0.9456 | 32,998 | 93,191 |
|  |  |  |  | 32 | 0.9922 | 22,245 | 66,562 |
|  |  |  |  | 64 | 0.9994 | 13,935 | 49,525 |
|  |  |  |  | 128 | 1.0000 | 8,297 | 35,306 |
|  |  |  |  | 256 | 1.0000 | 4,296 | 18,076 |
| Hyperspace Float32 | 2.4 s | 14.6 MiB | 767 | 16 | 0.9554 | 34,321 | 129,271 |
|  |  |  |  | 32 | 0.9938 | 25,579 | 80,844 |
|  |  |  |  | 64 | 0.9996 | 16,897 | 57,680 |
|  |  |  |  | 128 | 1.0000 | 12,722 | 38,950 |
|  |  |  |  | 256 | 1.0000 | 4,854 | 19,370 |
| Hyperspace Float32 (level p=0.5, as HyperspaceDB) | 4.1 s | 16.7 MiB | 875 | 16 | 0.9530 | 30,883 | 100,651 |
|  |  |  |  | 32 | 0.9934 | 22,867 | 71,910 |
|  |  |  |  | 64 | 0.9996 | 15,241 | 52,613 |
|  |  |  |  | 128 | 1.0000 | 9,857 | 37,264 |
|  |  |  |  | 256 | 1.0000 | 5,386 | 19,704 |
| Hyperspace Float64 (default) | 2.7 s | 24.2 MiB | 1271 | 16 | 0.9554 | 31,901 | 91,210 |
|  |  |  |  | 32 | 0.9938 | 17,654 | 66,807 |
|  |  |  |  | 64 | 0.9996 | 15,239 | 43,517 |
|  |  |  |  | 128 | 1.0000 | 8,077 | 30,975 |
|  |  |  |  | 256 | 1.0000 | 3,880 | 14,423 |
| Hyperspace Int8 | 2.1 s | 7.3 MiB | 383 | 16 | 0.9480 | 47,687 | 159,419 |
|  |  |  |  | 32 | 0.9804 | 36,742 | 109,920 |
|  |  |  |  | 64 | 0.9854 | 21,461 | 75,728 |
|  |  |  |  | 128 | 0.9860 | 14,303 | 48,744 |
|  |  |  |  | 256 | 0.9860 | 6,536 | 23,992 |
| Hyperspace Int4 | 3.6 s | 6.8 MiB | 354 | 16 | 0.8352 | 24,675 | 76,612 |
|  |  |  |  | 32 | 0.8504 | 18,028 | 53,282 |
|  |  |  |  | 64 | 0.8516 | 11,898 | 46,538 |
|  |  |  |  | 128 | 0.8518 | 9,100 | 33,905 |
|  |  |  |  | 256 | 0.8518 | 4,560 | 17,365 |
| Hyperspace Turbo | 3.8 s | 6.4 MiB | 335 | 16 | 0.7286 | 26,697 | 83,880 |
|  |  |  |  | 32 | 0.7378 | 22,191 | 67,512 |
|  |  |  |  | 64 | 0.7388 | 16,088 | 51,220 |
|  |  |  |  | 128 | 0.7390 | 11,066 | 38,945 |
|  |  |  |  | 256 | 0.7390 | 5,705 | 19,574 |
| Hyperspace Ternary | 2.7 s | 5.9 MiB | 309 | 16 | 0.4488 | 24,247 | 77,418 |
|  |  |  |  | 32 | 0.4534 | 20,001 | 62,625 |
|  |  |  |  | 64 | 0.4532 | 15,666 | 47,714 |
|  |  |  |  | 128 | 0.4532 | 10,719 | 36,127 |
|  |  |  |  | 256 | 0.4532 | 4,653 | 17,185 |
| Hyperspace Ternary + rerank | 2.8 s | 16.4 MiB | 858 | 16 | 1.0000 | 4,495 | 16,181 |
|  |  |  |  | 32 | 1.0000 | 4,530 | 16,307 |
|  |  |  |  | 64 | 1.0000 | 4,355 | 15,500 |
|  |  |  |  | 128 | 1.0000 | 3,847 | 15,397 |
|  |  |  |  | 256 | 1.0000 | 3,976 | 14,795 |
| Hyperspace Binary | 1.6 s | 5.8 MiB | 301 | 16 | 0.3102 | 46,519 | 132,531 |
|  |  |  |  | 32 | 0.3132 | 36,654 | 108,816 |
|  |  |  |  | 64 | 0.3128 | 21,432 | 79,748 |
|  |  |  |  | 128 | 0.3128 | 15,892 | 54,453 |
|  |  |  |  | 256 | 0.3128 | 7,295 | 26,454 |
| Hyperspace Binary + rerank | 1.7 s | 16.2 MiB | 850 | 16 | 1.0000 | 5,364 | 21,170 |
|  |  |  |  | 32 | 1.0000 | 5,999 | 20,534 |
|  |  |  |  | 64 | 1.0000 | 5,470 | 22,594 |
|  |  |  |  | 128 | 1.0000 | 5,769 | 22,092 |
|  |  |  |  | 256 | 1.0000 | 5,491 | 20,853 |
| Hyperspace PQ | 2.8 s | 3.8 MiB | 199 | 16 | 0.0290 | 39,006 | 117,538 |
|  |  |  |  | 32 | 0.0400 | 33,459 | 105,539 |
|  |  |  |  | 64 | 0.0548 | 26,969 | 84,352 |
|  |  |  |  | 128 | 0.0772 | 20,139 | 61,366 |
|  |  |  |  | 256 | 0.0954 | 12,805 | 42,418 |
| Hyperspace PQ + rerank | 2.7 s | 14.1 MiB | 739 | 16 | 0.3104 | 7,746 | 31,096 |
|  |  |  |  | 32 | 0.3104 | 7,968 | 29,937 |
|  |  |  |  | 64 | 0.3104 | 8,227 | 30,253 |
|  |  |  |  | 128 | 0.3104 | 8,396 | 30,228 |
|  |  |  |  | 256 | 0.3124 | 7,324 | 29,229 |
| Hyperspace OPQ | 2.8 s | 3.6 MiB | 190 | 16 | 0.0364 | 30,193 | 112,200 |
|  |  |  |  | 32 | 0.0504 | 32,826 | 79,488 |
|  |  |  |  | 64 | 0.0642 | 24,087 | 72,368 |
|  |  |  |  | 128 | 0.0800 | 19,600 | 60,675 |
|  |  |  |  | 256 | 0.0964 | 11,896 | 40,647 |
| Hyperspace Int8 + rerank | 2.1 s | 17.8 MiB | 932 | 16 | 0.9966 | 23,627 | 76,985 |
|  |  |  |  | 32 | 0.9966 | 23,753 | 75,363 |
|  |  |  |  | 64 | 0.9994 | 19,271 | 62,821 |
|  |  |  |  | 128 | 1.0000 | 12,647 | 43,617 |
|  |  |  |  | 256 | 1.0000 | 5,747 | 21,774 |

## What the numbers say

* **Like for like (HNSW `SelectHeuristic` vs Hyperspace `Float32`) the two are on par for search**: the same recall at every ef (0.95 at ef 16, 0.99 at 32,
  1.00 from 64) and single-thread QPS within noise of each other. Hyperspace now **builds about as fast** (2.5 s vs 2.9 s in this run) and its footprint is **23 % smaller**
  (14.6 MiB vs 19.0 MiB, 760 vs 996 bytes per vector).
* **HNSW's default neighbour selection (`SelectSimple`) stalls at about 0.5 recall on this clustered data** whatever the ef, because links stay
  inside clusters; `SelectHeuristic` (what Hyperspace always uses) does not. This depends on the data but is worth knowing.
* **`Float64`, Hyperspace's default storage, is about two thirds as fast and 1.7x larger than `Float32`** with identical recall. For ordinary Euclidean /
  cosine embeddings set `Quantization = Float32`; keep float64 for hyperbolic data near the ball boundary.
* **Quantized storage now pays off.** `Int8` is 2x smaller than `Float32` (7.3 vs 14.6 MiB), builds as fast and, at ef 16, searches faster (48k vs 34k QPS in this run)
  for a recall of 0.95-0.985 instead of 1.0; with `+ rerank` it reaches 0.9966-1.0. TurboQuant and 4-bit keep 0.74 / 0.85 recall at 6.4 / 6.8 MiB.
* **The aggressive codecs need re-ranking, with a big enough pool.** On this tightly clustered data ternary reaches 0.45 and binary 0.31 on their own, but **1.0000 with `+ rerank`**
  (the default pool is 25x the requested count), at roughly 4.5k-5.5k QPS and the memory of the extra float32 copy. Product quantization (8 bytes) reaches only 0.31 re-ranked: its
  codes rank this data too coarsely for a 25x pool. The pool is the lever, not the graph: in the `--rerank` experiment (128 dims, 100 clusters of 200, ef 256) recall at pools of
  4x / 10x / 25x / 50x was 0.45 / 0.79 / 0.99 / 1.00 for ternary, 0.39 / 0.74 / 1.00 / 1.00 for binary and 0.34 / 0.69 / 1.00 / 1.00 for product quantization, and the default of 4x used before
  recovered under half of the true neighbours. How big a pool a codec needs depends on the data and the dimension: tune `RerankOversample` on your own vectors.
* **The graph is about 400 bytes per vector at 128 dimensions** (links and .NET object overhead), against 512 for a float32 vector, so even 8-byte PQ codes only
  reach about 190 B/vector. Quantization pays off more at higher dimensions.

## What changed since the first run of this benchmark, and what is and is not established

| | first run | now |
|---|---:|---:|
| Float32 build / heap | 4.4 s / 17.5 MiB | 2.4 s / 14.6 MiB |
| Int8 build / heap / QPS (ef 16, 1 thread) | 10.1 s / 10.2 MiB / 14.9k | 1.9 s / 7.3 MiB / 51.2k |
| Turbo build / QPS (ef 64) | 20.2 s / 5.1k | 6.9 s / 15.2k |
| Binary build / QPS (ef 64) | 32.2 s / 2.8k | 7.5 s / 23.1k |

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

## The quantized codecs (graph construction, re-ranking)

Graph construction compares stored vectors with each other, and for the quantized codecs that had been decoding both to floats and running a float metric. `--micro` (128 dims, ns per call):

| mode | query to vector | vector to vector, before | vector to vector, now |
|---|---:|---:|---:|
| Float32 | 29 | 23 | 23 |
| Int8 | 37 | 44 | 25 |
| 4-bit | 78 (was 125) | 419 | 104 |
| TurboQuant | 47 | 549 | 133 |
| Ternary | 76 | 1,362 | 17 |
| Binary | 16 | 1,098 | 6 |
| PQ / OPQ | 13 / 7 | 212 / 171 | 41 / 20 |

The new kernels compare the codes directly (popcount of the XOR for binary, popcounts of the bit planes for ternary, table lookups per nibble pair for TurboQuant and 4-bit, a
symmetric centroid table for PQ) and are exact: recall was bit-identical before and after on every row. Build times (20,000 x 128): binary 7.5 s to 1.6 s, 4-bit 7.0 s to 3.6 s,
TurboQuant 6.9 s to 3.8 s, 8-bit 2.1 s to 1.8 s (its pair comparison, one integer dot product of the codes plus a stored sum of squares, went from 44 to 25 ns at 128 dimensions and from 200 to 135 ns at 768; it was never the main cost of an 8-bit build). The PQ symmetric table costs 256 KiB per sub-vector, so it is dropped when the index is frozen (it added 2 MiB to the 20,000-vector PQ index until then).

Two findings from the recall work:

* **A real bug, found by an experiment that was meant to do something else.** Ternary with the cosine metric compared stored vectors by the squared L2 of the decoded codes (shorter than unit length) while
  scoring queries by `2 (1 - cos)`; the neighbour-selection heuristic weighs one against the other, so the graph was sparse and poor (recall 0.43 at a 25x pool against 0.99+). It surfaced because building
  the graph from the original vectors "fixed" it. Ternary's graph is now denser (+1.5 MiB here) and its recall rose (0.37 to 0.45 single-pass).
* **Building the graph from the original vectors (`BuildFromOriginals`, tried and removed) changed nothing for any other codec.**

## Optimization experiments after that, with what held up

Single-shot timings on this shared VM are not trustworthy: the same code measured 94, 126 and 162 us per query on three runs, and one early "prefetch halves the cost" result was
pure noise. What follows was measured by **interleaving the variants in one process** on the same index (`--ab`: 25 rounds each, minimum and median), which cancels the drift.

| Experiment | Result | Kept |
|---|---|---|
| Prefetch the vectors of a node's unvisited neighbours before computing their distances (x86 only) | **-8 % per query at 20,000 vectors, -12 % at 100,000** (Float32, ef 64), consistent in min and median | yes |
| Where the time goes (`--evals`, Float32, ef 64) | cost per distance evaluation grows with the dataset: 120 ns at 1,500 vectors, 259 ns at 20,000, 366 ns at 100,000 (cache misses), on top of a roughly 90 ns fixed part | - |
| A purpose-built binary heap instead of `PriorityQueue` | push + pop 18 ns vs 17 ns: **no gain** | no, reverted |
| Four independent accumulators in the float L2 / dot / L1 kernels (the single accumulator is a dependency chain) | 32-43 ns vs 29-31 ns for a 128-d distance: **no gain**, my explanation for the kernel's cost was wrong | no, reverted |

## Not explained yet

* Even with every vector in cache, a Float32 query costs about 100-140 ns per distance evaluation end to end against a 29 ns call to `Distance` (`--micro`), and the
  priority queue is only about 18 ns of it. The rest (visited-set checks, bounds checks, dispatch around each distance, the upper-layer descent) has not been
  profiled; there is no sampling profiler on this machine.
* Ternary and product-quantization recall dropped a little (0.41 to 0.37, 0.064 to 0.055 at ef 64) when the level default changed: their graphs are built from
  noisy distances and vary with the level draw; HNSW's `SelectSimple` moved 0.514 to 0.527 between runs of unchanged code, so differences of this size are not conclusive.

## Product quantization: sub-vector count

The default was one byte per 16 dimensions (8 sub-vectors at 128 dimensions), which loses almost all of the ranking. It is now one byte per 4 dimensions. 20,000 x 128, cosine, M=16, efConstruction=100, recall@10, single-threaded QPS at ef 16 / ef 256:

| Sub-vectors | Bytes per vector | Build | Recall at ef 256 | Recall with re-ranking | QPS (ef 16 / 256) |
|---|---|---|---|---|---|
| 8 (old default) | 8 | 3.9 s | 0.0954 | 0.3124 | 27,376 / 9,851 |
| 16 | 16 | 4.1 s | 0.3218 | not measured | 24,765 / 9,874 |
| 32 (new default) | 32 | 5.4 s | 0.5696 | 1.0000 | 16,186 / 6,818 |
| 64 | 64 | 8.7 s | 0.8508 | not measured | 13,849 / 4,581 |

Re-ranking 32-sub-vector codes against the kept originals reaches recall 1.0000 at every ef, at 15.6 MiB (the codes plus the originals). Without originals, PQ stays the lossy option; pass a `ProductQuantizer` with more sub-vectors when recall matters more than size.
