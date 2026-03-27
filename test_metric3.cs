using System;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Running;
using BenchmarkDotNet.Configs;

public class Program {
    public static void Main() {
        var summary = BenchmarkRunner.Run<MyBench>();
    }
}

public class MyBench {
    [GlobalSetup]
    public void Setup() {}

    [Benchmark]
    public void Test() {}

    [IterationCleanup]
    public void Cleanup() {

    }
}
