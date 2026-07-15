[![Build Status](https://dev.azure.com/curiosity-ai/mosaik/_apis/build/status/hnsw-sharp?branchName=master)](https://dev.azure.com/curiosity-ai/mosaik/_build/latest?definitionId=7&branchName=master)

<a href="https://curiosity.ai"><img src="https://curiosity.ai/media/cat.color.square.svg" width="100" height="100" align="right" /></a>


# HNSW.Net
.Net library for fast approximate nearest neighbours search.

Exact _k_ nearest neighbours search algorithms tend to perform poorly in high-dimensional spaces. To overcome curse of dimensionality the ANN algorithms come in place. This library implements one of such algorithms described in the ["Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"](https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf) article. It provides simple API for building nearest neighbours graphs, (de)serializing them and running k-NN search queries.

## Usage
Check out the following code snippets once you've added the library reference to your project.
##### How to build a graph?
```c#
var parameters = new SmallWorld<float[], float>.Parameters()
{
  M = 15,
  LevelLambda = 1 / Math.Log(15),
  EfSearch = 50,
};

float[] vectors = GetFloatVectors();
var graph = new SmallWorld<float[], float>(CosineDistance.NonOptimized);
graph.BuildGraph(vectors, new Random(42), parameters);
```
##### How to run k-NN search?
```c#
SmallWorld<float[], float> graph = GetGraph();

float[] query = Enumerable.Repeat(1f, 100).ToArray();
var best20 = graph.KNNSearch(query, 20);
var best1 = best20.OrderBy(r => r.Distance).First();
```
##### How to speed up search with early termination?
KNN search can stop traversing the graph once the result set has stopped improving, trading a little recall for fewer distance computations (see the [Manticore write-up](https://manticoresearch.com/blog/knn-early-termination/) and "Patience in Proximity", Teofili &amp; Lin, ECIR 2025). The savings grow with `k` and `EfSearch`. It is disabled by default.
```c#
var parameters = new SmallWorld<float[], float>.Parameters()
{
  EfSearch = 100,
  EnableEarlyTermination = true,             // turn the optimization on
  EarlyTerminationSaturationThreshold = 0.95, // optional: fraction of top-k left unchanged to count a hop as "non-improving"
  EarlyTerminationPatience = 0,               // optional: consecutive non-improving hops before stopping (0 = adaptive, scales with EfSearch)
};
```
##### How to (de)serialize the graph?
```c#
SmallWorld<float[], float> graph = GetGraph();
byte[] buffer = graph.SerializeGraph(); // buffer stores information about parameters and graph edges

// distance function must be the same as the one which was used for building the original graph
var copy = new SmallWorld<float[], float>(CosineDistance.NonOptimized);
copy.DeserializeGraph(vectors, buffer); // the original vectors to attach to the "copy" vertices
```
##### Distance functions
The only one distance function supplied by the library is the cosine distance. But there are 4 versions to address universality/performance tradeoff.
```c#
CosineDistance.NonOptimized // most generic version works for all cases
CosineDistance.ForUnits     // gives correct result only when arguments are "unit" vectors
CosineDistance.SIMD         // uses SIMD instructions to optimize calculations
CosineDistance.SIMDForUnits // uses SIMD and requires arguments to be "units"
```
But the API allows to inject any custom distance function tailored specifically for your needs.

##### Performance options
For `float[]` vectors these opt-in parameters (all off by default, so existing behaviour is unchanged) reduce per-comparison overhead on the hot path:
```c#
var parameters = new SmallWorldParameters()
{
  // Compute distances with a built-in, inlined SIMD inner product over a single contiguous
  // backing buffer instead of routing every comparison through the distance delegate + jagged
  // array. Vectors are assumed unit length, so distance = 1 - dot (cosine distance). This is the
  // single biggest win: fastest build and query with identical recall.
  UseBuiltInUnitInnerProduct = true,

  // Remove the optional pairwise distance cache entirely (never allocated; the hot path skips the
  // cache-lookup branch and the distance-calculation counter). Matches how hnswlib / Lucene work,
  // which keep no distance cache at all.
  RemoveDistanceCache = true,

  // Track visited nodes with a packed bit set (1 bit/node) instead of the default epoch-tagged
  // int[] (4 bytes/node). Denser in cache, but the int[] resets in O(1) by bumping the epoch while
  // the bit set must be cleared each search - only worth it for very large graphs.
  UseBitSetVisited = false,
};
```

## Contributing
Your contributions and suggestions are very welcome! 
Please note that this project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

The contributions to this project are [released](https://help.github.com/articles/github-terms-of-service/#6-contributions-under-repository-license) to the public under the [project's open source license](LICENSE). Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.

### How to contribute
If you've found a bug or have a feature request then please open an issue with detailed description.
We will be glad to see your pull requests as well.

1. Prepare workspace.
```
git clone https://github.com/Microsoft/HNSW.Net.git
cd HNSW.Net
git checkout -b [username]/[feature]
```
2. Update the library and add tests if needed.
3. Build and test the changes.
```
cd Src
dotnet build
dotnet test
```
4. Send the pull request from `[username]/[feature]` to `master` branch.
5. Get approve and merge the changes.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

### Releasing
The library is distributed as a bundle of sources.
We are working on enabling CI and creating Nuget package for the project.
