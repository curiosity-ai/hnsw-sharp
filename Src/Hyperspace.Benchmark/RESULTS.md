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
* `+ rerank x4` keeps the original float32 vectors (so it costs their memory) and re-scores 4k candidates exactly.
* Hyperspace rows use its defaults (`LevelProbability = 1/M`); the `p = 0.5` row is HyperspaceDB's value.

| Index | Build | Heap | B/vec | ef | Recall@10 | QPS 1 thread | QPS 4 threads |
|---|---:|---:|---:|---:|---:|---:|---:|
| HNSW.Net (SelectSimple) | 2.1 s | 19.0 MiB | 996 | 16 | 0.5498 | 39,838 | 120,712 |
|  |  |  |  | 32 | 0.5556 | 30,975 | 100,285 |
|  |  |  |  | 64 | 0.5572 | 18,460 | 62,553 |
|  |  |  |  | 128 | 0.5578 | 12,239 | 36,426 |
|  |  |  |  | 256 | 0.5578 | 11,181 | 34,782 |
| HNSW.Net (SelectHeuristic) | 2.9 s | 19.0 MiB | 996 | 16 | 0.9396 | 26,095 | 75,993 |
|  |  |  |  | 32 | 0.9926 | 18,124 | 59,486 |
|  |  |  |  | 64 | 0.9994 | 18,051 | 47,583 |
|  |  |  |  | 128 | 1.0000 | 7,661 | 25,744 |
|  |  |  |  | 256 | 1.0000 | 5,800 | 18,636 |
| Hyperspace Float32 | 2.5 s | 14.6 MiB | 767 | 16 | 0.9554 | 30,642 | 116,966 |
|  |  |  |  | 32 | 0.9938 | 29,660 | 75,595 |
|  |  |  |  | 64 | 0.9996 | 13,141 | 53,271 |
|  |  |  |  | 128 | 1.0000 | 8,573 | 37,753 |
|  |  |  |  | 256 | 1.0000 | 6,170 | 18,595 |
| Hyperspace Float32 (level p=0.5, as HyperspaceDB) | 4.5 s | 16.7 MiB | 875 | 16 | 0.9530 | 23,609 | 103,814 |
|  |  |  |  | 32 | 0.9934 | 29,920 | 83,577 |
|  |  |  |  | 64 | 0.9996 | 20,190 | 58,048 |
|  |  |  |  | 128 | 1.0000 | 13,370 | 38,574 |
|  |  |  |  | 256 | 1.0000 | 4,665 | 15,877 |
| Hyperspace Float64 (default) | 3.6 s | 24.2 MiB | 1271 | 16 | 0.9554 | 22,130 | 78,720 |
|  |  |  |  | 32 | 0.9938 | 14,720 | 63,369 |
|  |  |  |  | 64 | 0.9996 | 10,703 | 43,991 |
|  |  |  |  | 128 | 1.0000 | 8,290 | 36,480 |
|  |  |  |  | 256 | 1.0000 | 3,296 | 15,158 |
| Hyperspace Int8 | 1.9 s | 7.3 MiB | 383 | 16 | 0.9480 | 51,221 | 146,545 |
|  |  |  |  | 32 | 0.9804 | 38,508 | 113,988 |
|  |  |  |  | 64 | 0.9854 | 25,404 | 75,124 |
|  |  |  |  | 128 | 0.9860 | 16,485 | 50,033 |
|  |  |  |  | 256 | 0.9860 | 5,758 | 22,482 |
| Hyperspace Int4 | 7.0 s | 6.8 MiB | 354 | 16 | 0.8352 | 21,681 | 67,396 |
|  |  |  |  | 32 | 0.8504 | 16,749 | 49,905 |
|  |  |  |  | 64 | 0.8516 | 10,467 | 40,474 |
|  |  |  |  | 128 | 0.8518 | 7,979 | 30,895 |
|  |  |  |  | 256 | 0.8518 | 3,654 | 13,755 |
| Hyperspace Turbo | 6.9 s | 6.4 MiB | 335 | 16 | 0.7286 | 29,690 | 87,117 |
|  |  |  |  | 32 | 0.7378 | 22,408 | 68,156 |
|  |  |  |  | 64 | 0.7388 | 15,185 | 52,783 |
|  |  |  |  | 128 | 0.7390 | 12,312 | 40,278 |
|  |  |  |  | 256 | 0.7390 | 5,990 | 20,209 |
| Hyperspace Ternary | 3.7 s | 4.4 MiB | 229 | 16 | 0.2106 | 40,524 | 119,918 |
|  |  |  |  | 32 | 0.2874 | 24,845 | 79,985 |
|  |  |  |  | 64 | 0.3720 | 15,542 | 58,586 |
|  |  |  |  | 128 | 0.4288 | 13,021 | 44,307 |
|  |  |  |  | 256 | 0.4516 | 8,300 | 28,321 |
| Hyperspace Ternary + rerank x4 | 4.2 s | 14.8 MiB | 778 | 16 | 0.5778 | 19,109 | 63,070 |
|  |  |  |  | 32 | 0.5778 | 18,216 | 63,493 |
|  |  |  |  | 64 | 0.6948 | 14,849 | 47,637 |
|  |  |  |  | 128 | 0.8078 | 9,711 | 39,168 |
|  |  |  |  | 256 | 0.8516 | 6,838 | 26,737 |
| Hyperspace Binary | 7.5 s | 5.7 MiB | 300 | 16 | 0.3100 | 43,147 | 145,190 |
|  |  |  |  | 32 | 0.3132 | 31,711 | 112,115 |
|  |  |  |  | 64 | 0.3128 | 23,108 | 78,155 |
|  |  |  |  | 128 | 0.3128 | 16,817 | 58,502 |
|  |  |  |  | 256 | 0.3128 | 7,395 | 26,438 |
| Hyperspace Binary + rerank x4 | 7.1 s | 16.1 MiB | 842 | 16 | 0.6980 | 29,154 | 80,392 |
|  |  |  |  | 32 | 0.6980 | 20,368 | 78,314 |
|  |  |  |  | 64 | 0.6984 | 19,053 | 59,822 |
|  |  |  |  | 128 | 0.6984 | 13,186 | 50,291 |
|  |  |  |  | 256 | 0.6984 | 6,585 | 23,951 |
| Hyperspace PQ | 3.4 s | 3.8 MiB | 199 | 16 | 0.0290 | 37,999 | 118,479 |
|  |  |  |  | 32 | 0.0400 | 35,472 | 108,231 |
|  |  |  |  | 64 | 0.0548 | 26,541 | 81,626 |
|  |  |  |  | 128 | 0.0772 | 20,173 | 62,775 |
|  |  |  |  | 256 | 0.0954 | 12,623 | 40,721 |
| Hyperspace PQ + rerank x4 | 3.2 s | 14.1 MiB | 739 | 16 | 0.1010 | 25,774 | 83,240 |
|  |  |  |  | 32 | 0.1010 | 27,588 | 81,946 |
|  |  |  |  | 64 | 0.1194 | 24,311 | 73,533 |
|  |  |  |  | 128 | 0.1686 | 15,535 | 52,387 |
|  |  |  |  | 256 | 0.2198 | 10,324 | 37,416 |
| Hyperspace OPQ | 2.9 s | 3.6 MiB | 190 | 16 | 0.0364 | 35,855 | 109,988 |
|  |  |  |  | 32 | 0.0504 | 33,906 | 99,949 |
|  |  |  |  | 64 | 0.0642 | 28,028 | 80,046 |
|  |  |  |  | 128 | 0.0800 | 18,852 | 62,293 |
|  |  |  |  | 256 | 0.0964 | 12,612 | 43,068 |
| Hyperspace Int8 + rerank x4 | 1.9 s | 17.8 MiB | 932 | 16 | 0.9966 | 27,471 | 86,731 |
|  |  |  |  | 32 | 0.9966 | 29,327 | 80,911 |
|  |  |  |  | 64 | 0.9994 | 21,649 | 68,031 |
|  |  |  |  | 128 | 1.0000 | 12,826 | 44,280 |
|  |  |  |  | 256 | 1.0000 | 5,331 | 22,538 |

## What the numbers say

* **Like for like (HNSW `SelectHeuristic` vs Hyperspace `Float32`) the two are on par for search**: the same recall at every ef (0.95 at ef 16, 0.99 at 32,
  1.00 from 64) and single-thread QPS within noise of each other. Hyperspace now **builds about as fast** (2.5 s vs 2.9 s in this run) and its footprint is **23 % smaller**
  (14.6 MiB vs 19.0 MiB, 760 vs 996 bytes per vector).
* **HNSW's default neighbour selection (`SelectSimple`) stalls at about 0.5 recall on this clustered data** whatever the ef, because links stay
  inside clusters; `SelectHeuristic` (what Hyperspace always uses) does not. This depends on the data but is worth knowing.
* **`Float64`, Hyperspace's default storage, is about two thirds as fast and 1.7x larger than `Float32`** with identical recall. For ordinary Euclidean /
  cosine embeddings set `Quantization = Float32`; keep float64 for hyperbolic data near the ball boundary.
* **Quantized storage now pays off.** `Int8` is 2x smaller than `Float32` (7.3 vs 14.6 MiB), builds as fast and, at ef 16, searches about as fast or faster (51k vs 31k QPS in this run)
  for a recall of 0.985 instead of 1.0; with `+ rerank x4` it reaches 0.9994. TurboQuant and 4-bit keep 0.74 / 0.85 recall at 6.4 / 6.8 MiB. The
  aggressive codecs are fast but, on this tightly clustered data, inaccurate: ternary reaches 0.37 single-pass and 0.69 re-ranked, binary 0.31 and 0.70, and
  product quantization about 0.06 and 0.12 (8 sub-vectors of 16 dimensions against clusters whose members differ by less than the quantization error).
* **The graph is about 400 bytes per vector at 128 dimensions** (links and .NET object overhead), against 512 for a float32 vector, so even 8-byte PQ codes only
  reach about 190 B/vector. Quantization pays off more at higher dimensions.

## What changed since the first run of this benchmark, and what is and is not established

| | first run | now |
|---|---:|---:|
| Float32 build / heap | 4.4 s / 17.5 MiB | 2.5 s / 14.6 MiB |
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
