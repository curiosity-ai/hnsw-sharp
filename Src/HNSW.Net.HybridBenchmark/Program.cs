using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace HNSW.Net.HybridBenchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            var config = DefaultConfig.Instance;
            BenchmarkRunner.Run<HybridBenchmark>(config, args);
        }
    }
}
