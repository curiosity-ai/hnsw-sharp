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

