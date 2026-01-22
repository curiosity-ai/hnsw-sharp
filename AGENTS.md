# AGENTS Guide for HNSW.Net

## Library Overview
HNSW.Net is a .NET library that implements the Hierarchical Navigable Small World (HNSW) algorithm for fast approximate nearest-neighbor (ANN) search in high-dimensional vector spaces. The core API centers around building a graph from input vectors, querying for the *k* nearest neighbors, and serializing/deserializing the graph for reuse.

## Key Concepts
- **HNSW graph**: A multi-layer small-world graph that enables efficient ANN search by navigating from higher, sparser layers to lower, denser layers.
- **Parameters**:
  - `M`: Controls the maximum number of connections per node.
  - `LevelLambda`: Affects level generation probability (higher values yield fewer high-level nodes).
- **Distance function**: Users supply a distance metric (e.g., cosine distance). HNSW.Net ships several cosine distance variants optimized for different scenarios.

## Usage Styles
1. **Build a graph**
   - Instantiate `SmallWorld<TItem, TDistance>` with a distance function.
   - Provide `SmallWorld.Parameters` (or `SmallWorldParameters`) to configure graph construction.
   - Call `BuildGraph` with items and a random generator.
2. **Query for nearest neighbors**
   - Call `KNNSearch` with a query item and `k` to retrieve nearest neighbors.
   - Optionally filter items via a predicate or cancel via a `CancellationToken`.
3. **Persist and reload graphs**
   - Use `SerializeGraph` to save a graph to a stream.
   - Use `DeserializeGraph` to rebuild the graph from serialized data and the original items.

## Key Methods & Functionality
- `SmallWorld<TItem, TDistance>.BuildGraph(...)`: Constructs the HNSW graph from input items.
- `SmallWorld<TItem, TDistance>.KNNSearch(...)`: Runs the ANN search and returns a list of results with distances.
- `SmallWorld<TItem, TDistance>.SerializeGraph(...)`: Serializes graph metadata and edges.
- `SmallWorld<TItem, TDistance>.DeserializeGraph(...)`: Rehydrates a graph from serialized data.
- `CosineDistance.*`: Built-in distance functions, including generic and SIMD-optimized implementations.

## Helpful Notes for Contributors
- The library is source-distributed; updates should be made directly in `Src/HNSW.Net`.
- Keep public API changes documented in README snippets and ensure tests in `Src/HNSW.Net.Tests` remain green when possible.
