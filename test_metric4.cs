using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Engines;
using BenchmarkDotNet.Running;

public class Program {
    public static void Main() {
        BenchmarkRunner.Run<MyBench>();
    }
}

public class MyBench {
    public double Recall { get; set; } = 0.5;

    [Benchmark]
    public void Test() {}

    [IterationCleanup]
    public void Cleanup() {
        Recall = 0.99;
    }
}
