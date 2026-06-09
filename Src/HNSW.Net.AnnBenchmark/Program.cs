using BenchmarkDotNet.Running;

namespace HNSW.Net.AnnBenchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            // One benchmark per ANN-Benchmarks distance type, each using the smallest dataset of that type:
            //   Euclidean -> Fashion-MNIST (784d), Angular -> GloVe-25, Jaccard -> Kosarak.
            // Pass e.g. --filter *AngularBenchmark* to run a single one; with no args BenchmarkSwitcher prompts.
            BenchmarkSwitcher
                .FromTypes(new[]
                {
                    typeof(EuclideanBenchmark),
                    typeof(AngularBenchmark),
                    typeof(JaccardBenchmark),
                })
                .Run(args);
        }
    }
}
