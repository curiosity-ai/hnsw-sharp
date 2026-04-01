using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Running;

namespace HNSW.Net.SiftBenchmark
{
    class Program
    {
        static void Main(string[] args)
        {
            var a = new SiftBenchmarkTurboQuant() { M = 32, EfSearch = 200, EfConstruction = 200, QuickTest = true };
            a.Setup();
            a.Cleanup();

            var b = new SiftBenchmark() { M = 32, EfSearch = 200, EfConstruction = 200 , QuickTest = true};
            b.Setup();
            b.Cleanup();

            return;

            var config = DefaultConfig.Instance;
            BenchmarkRunner.Run<SiftBenchmarkTurboQuant>(config, args);
            BenchmarkRunner.Run<SiftBenchmark>(config, args);
        }
    }
}
