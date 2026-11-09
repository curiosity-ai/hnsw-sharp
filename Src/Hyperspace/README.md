# Hyperspace

A managed embedding database engine for .NET: a hierarchical navigable small-world (HNSW) index over optionally quantized vectors, with
twelve distance metrics (including hyperbolic geometry), metadata and geometric filters, BM25 hybrid search, a property graph, fuzzy-logic
queries, Riemannian math and flat binary persistence.

It is a C# port of the algorithms of [HyperspaceDB](https://github.com/YARlabs/hyperspace-db) (the Rust embedding database), built on the
conventions of the [HNSW](../../README.md) package of this repository: no native dependencies, no external packages, an API centred on one
index class, and a flat graph format that is read and written as spans.

What is **not** here is the database *server*: no gRPC / REST API, WAL or tiered storage, replication, billing, plugins or dashboard. The
package is the engine you embed.

```csharp
var index = new HyperspaceIndex(new HyperspaceOptions
{
    Dimension = 768,
    Metric = DistanceMetric.Cosine,
    Quantization = QuantizationMode.ScalarInt8,
});

int id = index.Add(embedding, new Dictionary<string, string> { ["title"] = "Intro", ["year"] = "2024" });
index.AddRange(items);                                    // encodes in parallel, links in order

SearchResult[] hits = index.Search(query, k: 10, new SearchOptions
{
    Filter = FilterExpr.And(FilterExpr.Match("lang", "en"), FilterExpr.Range("year", 2020, null)),
    IncludeMetadata = true,
});
```

## Metrics

| `DistanceMetric` | Distance |
| --- | --- |
| `L2` | squared Euclidean |
| `Cosine` | squared L2 of unit vectors (`2 - 2cos`); vectors are normalized on the way in unless `NormalizeVectors = false` |
| `Dot` | `1 - a.b` |
| `L1`, `Linf` | Manhattan, Chebyshev |
| `Angular` | `acos(cos) / pi`, obeys the triangle inequality |
| `Jaccard` | continuous Tanimoto `1 - a.b / (a.a + b.b - a.b)` |
| `Hamming` | fraction of coordinates whose sign differs |
| `Poincare` | hyperbolic geodesic distance in the Poincare ball (vectors must be inside the unit ball) |
| `Lorentz` | hyperbolic geodesic distance on the hyperboloid (first coordinate is time; `-t^2 + |x|^2 = -1`) |
| `Hybrid` | Lorentz distance of the first 33 dimensions plus squared L2 of the rest |
| `Wasserstein` | 1-D earth mover's distance between the vectors as histograms |

`Metrics.Distance(...)` evaluates any of them directly over `float` or `double` spans; `Metrics.Parse` accepts the HyperspaceDB names and
aliases (`"euclidean"`, `"ip"`, `"chebyshev"`, `"emd"`...). Hyperbolic metrics are what to use for hierarchical data (taxonomies, code, org
charts): a 32-64 dimensional hyperbolic space can hold a tree that needs hundreds of Euclidean dimensions.

## Quantization

`HyperspaceOptions.Quantization` chooses how vectors are stored. Bytes per vector at 1024 dimensions, and how well the distance to the
codes tracks the true distance (Spearman rank correlation on unit-norm Gaussian embeddings, measured with this library):

| Mode | Bytes | vs float32 | Rank correlation | Notes |
| --- | ---: | ---: | ---: | --- |
| `None` | 8192 | 0.5x | 1.00 | float64, the default; exact hyperbolic distances near the ball boundary |
| `Float32` | 4096 | 1x | 1.00 | |
| `ScalarInt8` | 1032 | 4x | 1.00 | per-vector scale; fixed scale and optional anisotropic refinement for Poincare |
| `ScalarInt4` | 772 | 5.3x | 0.99 | 4 bits in blocks of 16 with a block scale |
| `Turbo` | 520 | 7.9x | 0.99 | TurboQuant: rotation + norm + 4-bit Lloyd-Max centroids |
| `Ternary` | 264 | 15.5x | 0.90 | BitNet `{-1, 0, +1}`, 1.58 bits, popcount-friendly bit planes |
| `Binary` | 132 | 31x | 0.79 | sign bits of the rotated vector, asymmetric search |
| `ProductQuantization` / `OPQ` | 64 | 64x | 0.5-0.65 | one byte per 16 dimensions, asymmetric lookup tables, squared L2 |
| `AsymmetricHybrid` | | | | float32 head (e.g. the 33 Lorentz dimensions) + 8-bit tail |

The aggressive codecs (`Ternary`, `Binary`, `PQ`) have low single-pass recall by design. Keep the original vectors and re-rank for
near-exact results (HyperspaceDB calls this *two-pass search*):

```csharp
var options = new HyperspaceOptions { Dimension = 1024, Quantization = QuantizationMode.Ternary, KeepOriginalVectors = true, RerankOversample = 4 };
...
index.Search(query, 10);                                                       // 40 candidates from the codes, re-scored exactly
index.Search(query, 10, new SearchOptions { RerankOversample = 1 });           // first pass only
```

Product quantization needs codebooks: call `index.TrainProductQuantizer(samples)` while the index is empty (or pass a trained
`ProductQuantizer` in the options). Binary, ternary, TurboQuant and OPQ rotate vectors with `TurboRotation`, an `O(D log D)` randomized
Walsh-Hadamard transform that is exactly orthogonal.

## Filters, hybrid search, fuzzy queries

* **Filters** (`FilterExpr`): `Match`, `Prefix`, `Range`, `And`, `Or`, `Not`, plus geometric `InBox`, `InBall` (hyperbolic for 33-dimensional
  Lorentz centres) and `InCone` (ConE angular sectors). Metadata is indexed on insert, so they compile to id bitmaps (`IdSet`); a
  filter that leaves few candidates is answered by an exact scan instead of walking a graph it would have disconnected.
  Geometric filters test the *stored* coordinates, i.e. after normalization for `Cosine`.
* **Hybrid search**: `SearchOptions.HybridQuery = "text"` fuses the vector ranking with BM25 over every metadata value, by reciprocal rank
  fusion (`rrf`, `HybridAlpha` is the constant, default 60) or weighted normalized scores (`weighted`, `HybridAlpha` is the vector weight).
  BM25 variants: Robertson, Lucene, ATIRE, BM25L, BM25+. The `Tokenizer` is pluggable (default: English stop words and a light stemmer).
* **Fuzzy logic**: `SearchFuzzy(FuzzyQuery.And(a, b, TNorm.Product), k)` with t-norms (`Min`, `Product`, `Lukasiewicz`), t-conorms and negation.
* **Momentum**: `SearchMomentum(past, current, steps, k)` extrapolates a trajectory in the geometry of the metric and searches ahead of it.

## Property graph

`index.Link(source, target, relation, weight)` stores typed, weighted edges between item ids (12 bytes each, `CompactEdge`), with
bounded multi-hop BFS (`SearchOptions.Graph = new GraphFilter(seeds, hops, relations)`) that restricts a vector search to what is reachable,
compaction to CSR (`CsrGraph`, in the same `HYPR_CSR` file layout as HyperspaceDB) and declarative edge induction from metadata
(`path_hierarchy`, `foreign_key`, `exact_match` via `InduceGraphFromMetadata`).

## Math toolkit

`Riemannian` (Mobius addition, exp / log maps, parallel transport, Frechet mean, Poincare / Lorentz conversion, spatial entropy,
Lyapunov convergence, Koopman extrapolation, context resonance), `MemoryReconsolidator` (Riemannian SGD "sleep mode"),
`KeyedOrthogonalProjection` (a keyed, exactly isometric shield against embedding-inversion), `GromovHyperbolicity` (which metric suits
your data), `KMeans` (k-means++, parallel Lloyd, out-of-core mini-batch) and `MetaRouter` (IVF-style routing to segments, two-level
beyond 32 segments).

## Persistence

Two formats, both flat little-endian and read without building an intermediate object graph, in the spirit of the `HNSW2` format of the
HNSW package:

```csharp
// 1. The graph only - the caller keeps the items, exactly like SmallWorld.SerializeGraph / DeserializeGraph.
index.SerializeGraph(stream);                      // or an IBufferWriter<byte>
var (copy, itemsNotInGraph) = HyperspaceIndex.DeserializeGraph(items, stream);
copy.AddRange(itemsNotInGraph);                    // vectors a crash left outside the persisted graph

// 2. A complete snapshot: graph + encoded vectors + metadata + property graph (+ original vectors).
index.Save("index.hsp");                           // atomic: temp file, then rename
var loaded = HyperspaceIndex.Load("index.hsp");
```

The graph section stores each node's links as one run of `int`s (layer start offsets, total, neighbour ids) - the representation the
search uses - and validates every record on read, so a corrupt file is an `InvalidDataException`, not an out-of-range exception hours
later. The parameters (metric, quantization, `M`, BM25 settings, product-quantizer codebooks...) travel with the graph; a custom
`Tokenizer` is a delegate and does not, so pass it to `Load` / `DeserializeGraph`. See `HyperspaceIndex.Serialization.cs` for the byte layout.
After loading, layer 0 is frozen into contiguous arrays (`FreezeFlatLayer0`) unless `FreezeOnLoad = false`.

`ProductQuantizer.Save` writes the `PQCB` codebook file and `CsrGraph.Write` the `HYPR_CSR` graph file with the layouts HyperspaceDB
uses, and `TurboRotation` uses its sign stream; these were taken from its source but have **not** been cross-checked by running the Rust
implementation.

## Differences from HyperspaceDB

* The `Cosine` metric normalizes vectors and queries on the way in (HyperspaceDB expects normalized input and is otherwise a plain L2).
* Approximate codecs report the `Cosine` metric as `2(1 - cos)`, the same scale as the exact distance.
* `ScalarInt8` uses a per-vector scale except for Poincare (where it keeps HyperspaceDB's fixed scale), so unnormalized embeddings do not clamp.
* `Binary` compares the rotated query to the sign bits asymmetrically for every metric (an angular approximation).
* Graph construction runs under a reader-writer lock instead of HyperspaceDB's lock-free node updates: searches are parallel, writes are exclusive.
* Fuzzy queries, momentum search and graph-restricted search are index methods here; in HyperspaceDB they live in the server layer.
* The lexical index is rebuilt on demand rather than persisted, and metadata indexes are rebuilt on load.
* `float` input is accepted for every metric; a float32 Lorentz vector is allowed a precision-scaled tolerance around the hyperboloid.

Not ported: the server and its storage tiering (WAL, LSM chunks, S3), replication and gossip, billing, plugins (`hs-embed`, `hs-git-ast`),
GPU kernels, zonal quantization, the `AsymmetricHybridLowBit` / `AsymmetricHybridExtreme` 801-dimension layouts, the SuperKMeans
GEMM / PDX / ADSampling machinery (replaced by a plain k-means), the spatial AABB filter index (geometric filters scan) and
hand-written per-CPU SIMD kernels (`Vector<T>` is used for the L2, dot and L1 hot paths).

## Package name

`Hyperspace` was not published on nuget.org when checked: both the flat-container and the registration endpoints return 404 for it (and for
`HyperspaceDB`, `Hyperspace.Core` and `HyperspaceSharp`), while they return 200 for an existing package such as `HNSW`. The API cannot show
reserved ID prefixes, so confirm when you first push. Change `PackageId` in `Hyperspace.csproj` if you would rather use another name.
