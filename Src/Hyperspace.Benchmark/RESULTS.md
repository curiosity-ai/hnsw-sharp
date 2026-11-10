# Hyperspace vs HNSW benchmark

Reproduce: `dotnet run -c Release --project Src/Hyperspace.Benchmark -- --n 20000 --dim 128 --queries 500 --out results.md`

Setup: 20,000 synthetic unit-norm vectors of 128 dimensions drawn around 100 Gaussian cluster centres, 500 queries from the same
distribution, exact ground truth by brute force, cosine distance, `M = 16`, `efConstruction = 100`, k = 10. One run on a shared 4-core
Linux VM (.NET 10). **Single run: differences under roughly 20-30 % in QPS are within noise** (note the irregular ef = 256 rows).
Synthetic clustered data is not your embeddings; use the harness on your own vectors before drawing conclusions.

How to read the columns:
* **Heap** is `GC.GetTotalMemory` after a forced compacting GC, before and after the build. HNSW holds the caller's vectors by reference, so the
  raw vectors (9.8 MiB) are added to its figure; Hyperspace stores its own copy, so its figure is complete. HNSW's construction-time distance
  cache is freed with `DisableDistanceCache()` first; both libraries have their layer 0 flattened (`OptimizeIfNeeded` / `FreezeFlatLayer0`).
* **Build** is wall-clock for the whole build (Hyperspace encodes in parallel and links sequentially; for PQ it includes codebook training).
* **QPS** counts complete single queries; the multi-thread column runs queries in parallel on 4 cores.
* `+ rerank x4` keeps the original float32 vectors (so it costs their memory) and re-scores 4k candidates exactly.

| Index | Build | Heap | B/vec | ef | Recall@10 | QPS 1 thread | QPS 4 threads |
|---|---:|---:|---:|---:|---:|---:|---:|
| HNSW.Net (SelectSimple) | 2.0 s | 19.0 MiB | 996 | 16 | 0.5064 | 42,793 | 122,172 |
|  |  |  |  | 32 | 0.5128 | 30,295 | 96,032 |
|  |  |  |  | 64 | 0.5138 | 22,627 | 67,902 |
|  |  |  |  | 128 | 0.5138 | 15,946 | 48,808 |
|  |  |  |  | 256 | 0.5138 | 12,739 | 40,315 |
| HNSW.Net (SelectHeuristic) | 2.4 s | 19.0 MiB | 996 | 16 | 0.9432 | 34,489 | 98,117 |
|  |  |  |  | 32 | 0.9920 | 23,829 | 79,184 |
|  |  |  |  | 64 | 0.9994 | 15,176 | 55,952 |
|  |  |  |  | 128 | 1.0000 | 9,935 | 36,430 |
|  |  |  |  | 256 | 1.0000 | 4,398 | 18,988 |
| Hyperspace Float32 | 4.4 s | 17.5 MiB | 918 | 16 | 0.9530 | 34,249 | 123,110 |
|  |  |  |  | 32 | 0.9934 | 24,493 | 85,493 |
|  |  |  |  | 64 | 0.9996 | 20,802 | 63,710 |
|  |  |  |  | 128 | 1.0000 | 14,613 | 40,519 |
|  |  |  |  | 256 | 1.0000 | 5,508 | 19,759 |
| Hyperspace Float64 (default) | 6.9 s | 27.1 MiB | 1422 | 16 | 0.9530 | 19,305 | 63,329 |
|  |  |  |  | 32 | 0.9934 | 13,320 | 48,961 |
|  |  |  |  | 64 | 0.9996 | 9,813 | 35,396 |
|  |  |  |  | 128 | 1.0000 | 7,121 | 27,416 |
|  |  |  |  | 256 | 1.0000 | 3,607 | 12,406 |
| Hyperspace Int8 | 10.1 s | 10.2 MiB | 535 | 16 | 0.9452 | 14,856 | 49,034 |
|  |  |  |  | 32 | 0.9804 | 11,921 | 40,353 |
|  |  |  |  | 64 | 0.9854 | 8,939 | 31,169 |
|  |  |  |  | 128 | 0.9860 | 6,526 | 22,249 |
|  |  |  |  | 256 | 0.9860 | 2,963 | 11,284 |
| Hyperspace Int4 | 21.6 s | 9.7 MiB | 509 | 16 | 0.8358 | 10,515 | 34,794 |
|  |  |  |  | 32 | 0.8502 | 8,077 | 26,099 |
|  |  |  |  | 64 | 0.8516 | 6,265 | 20,820 |
|  |  |  |  | 128 | 0.8518 | 4,667 | 14,998 |
|  |  |  |  | 256 | 0.8518 | 2,284 | 7,656 |
| Hyperspace Turbo | 20.2 s | 9.6 MiB | 502 | 16 | 0.7292 | 7,666 | 27,060 |
|  |  |  |  | 32 | 0.7378 | 6,160 | 22,446 |
|  |  |  |  | 64 | 0.7388 | 5,130 | 18,028 |
|  |  |  |  | 128 | 0.7390 | 3,571 | 13,494 |
|  |  |  |  | 256 | 0.7390 | 1,638 | 6,412 |
| Hyperspace Ternary | 15.6 s | 5.4 MiB | 286 | 16 | 0.2846 | 7,238 | 25,829 |
|  |  |  |  | 32 | 0.3452 | 4,991 | 18,722 |
|  |  |  |  | 64 | 0.4082 | 3,729 | 13,703 |
|  |  |  |  | 128 | 0.4374 | 2,860 | 10,916 |
|  |  |  |  | 256 | 0.4456 | 1,745 | 6,420 |
| Hyperspace Ternary + rerank x4 | 15.1 s | 15.9 MiB | 835 | 16 | 0.6670 | 4,196 | 15,414 |
|  |  |  |  | 32 | 0.6670 | 4,063 | 16,030 |
|  |  |  |  | 64 | 0.7636 | 3,196 | 12,813 |
|  |  |  |  | 128 | 0.8248 | 2,806 | 10,175 |
|  |  |  |  | 256 | 0.8420 | 1,522 | 5,910 |
| Hyperspace Binary | 32.2 s | 8.9 MiB | 469 | 16 | 0.3110 | 4,072 | 15,063 |
|  |  |  |  | 32 | 0.3132 | 3,332 | 12,902 |
|  |  |  |  | 64 | 0.3128 | 2,802 | 10,839 |
|  |  |  |  | 128 | 0.3128 | 2,209 | 8,133 |
|  |  |  |  | 256 | 0.3128 | 885 | 3,426 |
| Hyperspace Binary + rerank x4 | 31.4 s | 19.6 MiB | 1028 | 16 | 0.6980 | 2,996 | 11,406 |
|  |  |  |  | 32 | 0.6980 | 3,027 | 11,309 |
|  |  |  |  | 64 | 0.6984 | 2,672 | 10,239 |
|  |  |  |  | 128 | 0.6984 | 1,894 | 7,745 |
|  |  |  |  | 256 | 0.6984 | 826 | 3,089 |
| Hyperspace PQ | 4.2 s | 4.5 MiB | 234 | 16 | 0.0458 | 32,797 | 90,978 |
|  |  |  |  | 32 | 0.0488 | 28,824 | 87,444 |
|  |  |  |  | 64 | 0.0636 | 24,491 | 74,935 |
|  |  |  |  | 128 | 0.0758 | 19,777 | 62,610 |
|  |  |  |  | 256 | 0.0930 | 12,488 | 41,601 |
| Hyperspace PQ + rerank x4 | 4.2 s | 14.7 MiB | 769 | 16 | 0.1496 | 23,273 | 71,200 |
|  |  |  |  | 32 | 0.1496 | 22,949 | 73,032 |
|  |  |  |  | 64 | 0.1670 | 19,813 | 65,478 |
|  |  |  |  | 128 | 0.1962 | 14,549 | 54,548 |
|  |  |  |  | 256 | 0.2384 | 10,069 | 35,652 |
| Hyperspace OPQ | 4.0 s | 4.3 MiB | 226 | 16 | 0.0544 | 33,348 | 97,654 |
|  |  |  |  | 32 | 0.0604 | 26,501 | 87,102 |
|  |  |  |  | 64 | 0.0730 | 22,320 | 78,472 |
|  |  |  |  | 128 | 0.0846 | 19,211 | 61,219 |
|  |  |  |  | 256 | 0.1012 | 12,956 | 42,424 |
| Hyperspace Int8 + rerank x4 | 10.2 s | 20.5 MiB | 1073 | 16 | 0.9966 | 8,540 | 33,220 |
|  |  |  |  | 32 | 0.9966 | 9,260 | 33,652 |
|  |  |  |  | 64 | 0.9994 | 7,252 | 29,328 |
|  |  |  |  | 128 | 1.0000 | 5,893 | 21,202 |
|  |  |  |  | 256 | 1.0000 | 2,548 | 9,723 |

## What the numbers say

* **Like for like (HNSW `SelectHeuristic` vs Hyperspace `Float32`), the two are equivalent**: the same recall at every ef (0.94-0.95 at ef 16,
  0.99 at 32, 1.00 from 64), single-thread QPS within noise of each other, and Hyperspace's footprint 8 % smaller (918 vs 996 B/vector).
  Hyperspace builds about 1.8x slower (4.4 s vs 2.4 s).
* **HNSW's default neighbour selection (`SelectSimple`) stalls at 0.51 recall on this clustered data** whatever the ef, because
  links stay inside clusters; `SelectHeuristic` (what Hyperspace always does) does not. This depends on the data but is worth knowing.
* **`Float64`, Hyperspace's default storage, is about half as fast and 1.5x larger than `Float32`** with identical recall. For ordinary Euclidean / cosine
  embeddings set `Quantization = Float32`; keep float64 for hyperbolic data near the ball boundary.
* **Quantization saves memory but costs speed in this implementation.** The codecs are scalar code (no SIMD), so `Int8` runs 2-3x slower than
  `Float32` and the 4-bit, TurboQuant, ternary and binary codecs 3-8x slower, and they build 2-7x slower. `Int8` is a reasonable trade
  (10.2 MiB vs 17.5, recall 0.985 against 1.0); with `+ rerank x4` it reaches 0.997-1.0 but holds both copies (20.5 MiB).
* **The aggressive codecs are weak on this data**: ternary reaches 0.45 single-pass and 0.84 with re-ranking, binary 0.31 and 0.70, product
  quantization 0.09 and 0.24 (8 sub-vectors of 16 dimensions against tight clusters whose members differ by less than the quantization
  error). Raise the oversample, or use them where the data is less clustered and the dimension larger.
* **At 128 dimensions the graph dominates the footprint**: about 400 bytes per vector for links and .NET object overhead, against 512 for a
  float32 vector, so even 8-byte PQ codes only get to 226 B/vector. Quantization pays off at higher dimensions.

## Known inefficiencies this exposed

* Freezing layer 0 copies its links into flat arrays but leaves the per-node arrays in place, so a frozen index holds layer 0 twice.
* Graph links are one `int[]` per node and layer; a flat or pooled layout would cut the per-vector overhead substantially.
* The quantized distance functions are scalar; `Vector<T>` versions (as for float L2/dot) would close most of the QPS gap.
* Build time of the rotating codecs is dominated by pair-distance decoding during graph construction.
