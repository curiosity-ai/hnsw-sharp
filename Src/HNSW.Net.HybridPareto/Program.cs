using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using HNSW.Net;

namespace HNSW.Net.HybridPareto
{
    public struct Item
    {
        public int Id;
        public float[] Vector;
        public int Attribute;
    }

    public class Result
    {
        public string Method { get; set; }
        public int M { get; set; }
        public int EfConstruction { get; set; }
        public int EfSearch { get; set; }
        public int Gamma { get; set; }
        public int Mb { get; set; }
        public double Qps { get; set; }
        public double RecallAt10 { get; set; }
    }

    class Program
    {
        static void Main(string[] args)
        {
            double percentage = 0.01;
            if (args.Length > 0)
            {
                if (double.TryParse(args[0], out double parsedPercentage))
                {
                    percentage = parsedPercentage;
                }
                else
                {
                    Console.WriteLine($"Invalid percentage argument '{args[0]}', defaulting to {percentage}");
                }
            }

            Console.WriteLine($"Using {percentage:P} of the SIFT dataset.");

            string workingDir = Path.Combine(Path.GetTempPath(), "hnsw-bench");
            if (!Directory.Exists(workingDir)) Directory.CreateDirectory(workingDir);

            Dataset.DownloadAndExtractAsync(workingDir).GetAwaiter().GetResult();

            string siftDir = Path.Combine(workingDir, "sift");
            Console.WriteLine("Reading dataset...");
            var baseVectorsFull = Dataset.ReadFvecs(Path.Combine(siftDir, "sift_base.fvecs"));
            var queryVectorsFull = Dataset.ReadFvecs(Path.Combine(siftDir, "sift_query.fvecs"));

            int keepBaseVectors = (int)(baseVectorsFull.Length * percentage);
            int keepQueryVectors = (int)(queryVectorsFull.Length * percentage);
            // Ensure we keep at least 1 query vector to avoid zero queries
            if (keepQueryVectors == 0 && queryVectorsFull.Length > 0) keepQueryVectors = 1;

            var baseVectors = baseVectorsFull.Take(keepBaseVectors).ToArray();
            var queryVectors = queryVectorsFull.Take(keepQueryVectors).ToArray();

            var baseAttributes = Dataset.GenerateRandomAttributes(baseVectors.Length, seed: 42);
            var queryAttributes = Dataset.GenerateRandomAttributes(queryVectors.Length, seed: 43);

            var baseItems = new Item[baseVectors.Length];
            for (int i = 0; i < baseVectors.Length; i++)
            {
                baseItems[i] = new Item { Id = i, Vector = baseVectors[i], Attribute = baseAttributes[i] };
            }

            var queryItems = new Item[queryVectors.Length];
            for (int i = 0; i < queryVectors.Length; i++)
            {
                queryItems[i] = new Item { Id = i, Vector = queryVectors[i], Attribute = queryAttributes[i] };
            }

            Console.WriteLine($"Subset: {baseVectors.Length}/{baseVectorsFull.Length} base vectors");
            Console.WriteLine($"Subset: {queryVectors.Length}/{queryVectorsFull.Length} query vectors");

            Console.WriteLine("Computing hybrid ground truth for subset...");
            var groundTruth = Dataset.ComputeHybridGroundTruth(baseVectors, baseAttributes, queryVectors, queryAttributes, 10);

            var results = new List<Result>();

            int[] mValues = { 2, 4, 8, 16, 32 };
            int[] efConstructionValues = { 10, 20, 50, 100, 200, 400 };
            int[] efSearchValues = { 10, 20, 50, 100, 200, 400 };

            float DistanceFunc(Item a, Item b) => L2Distance.SIMD(a.Vector, b.Vector);

            foreach (var m in mValues)
            {
                foreach (var efConstruction in efConstructionValues)
                {
                    // Define configurations:
                    // (MethodName, OptimizeForFiltering, Gamma, Mb)
                    var configurations = new[]
                    {
                        ("HNSW", false, 0, 0),
                        ("ACORN-1", true, 1, m),
                        ("ACORN-Gamma12", true, 12, m),
                        ("ACORN-Gamma24", true, 24, m)
                    };

                    foreach (var (method, optimizeForFiltering, gamma, mb) in configurations)
                    {
                        var parameters = new SmallWorldParameters
                        {
                            M = m,
                            LevelLambda = 1 / Math.Log(m),
                            ConstructionPruning = efConstruction,
                            EnableDistanceCacheForConstruction = true,
                            OptimizeForFiltering = optimizeForFiltering,
                            Gamma = gamma,
                            Mb = mb
                        };

                        Console.WriteLine($"\nBuilding {method} graph with M={m}, EfConstruction={efConstruction}...");
                        var swBuild = Stopwatch.StartNew();
                        var graph = new SmallWorld<Item, float>(DistanceFunc, DefaultRandomGenerator.Instance, parameters);
                        graph.AddItems(baseItems, new ConsoleProgress());
                        swBuild.Stop();
                        Console.WriteLine($"Graph built in {swBuild.Elapsed.TotalSeconds:N2}s.");

                        foreach (var efSearch in efSearchValues)
                        {
                            graph.Parameters.EfSearch = efSearch;

                            Console.WriteLine($"Searching {method} with EfSearch={efSearch}...");

                            // Warmup
                            for (int i = 0; i < Math.Min(100, queryItems.Length); i++)
                            {
                                var q = queryItems[i];
                                graph.KNNSearch(q, 10, item => item.Attribute == q.Attribute);
                            }

                            int correct10 = 0;
                            var swSearch = Stopwatch.StartNew();
                            for (int i = 0; i < queryItems.Length; i++)
                            {
                                var queryItem = queryItems[i];
                                var searchResults = graph.KNNSearch(queryItem, 10, item => item.Attribute == queryItem.Attribute);

                                if (searchResults.Any(r => r.Item.Id == groundTruth[i][0]))
                                {
                                    correct10++;
                                }
                            }
                            swSearch.Stop();

                            double recall10 = (double)correct10 / queryItems.Length;
                            double qps = queryItems.Length / swSearch.Elapsed.TotalSeconds;

                            Console.WriteLine($"  -> Recall@10: {recall10:P2}, QPS: {qps:N2}");

                            results.Add(new Result
                            {
                                Method = method,
                                M = m,
                                EfConstruction = efConstruction,
                                EfSearch = efSearch,
                                Gamma = gamma,
                                Mb = mb,
                                Qps = qps,
                                RecallAt10 = recall10
                            });
                        }
                    }
                }
            }

            Console.WriteLine("\n--- Pareto Sets ---");

            var groupedResults = results.GroupBy(r => r.Method);

            foreach (var group in groupedResults)
            {
                Console.WriteLine($"\nMethod: {group.Key}");
                Console.WriteLine($"{"M",-5} {"EfC",-5} {"EfS",-5} {"Gamma",-6} {"Mb",-4} {"QPS",-10} {"Recall@10",-10}");

                var methodResults = group.ToList();
                var paretoFront = new List<Result>();

                foreach (var r in methodResults)
                {
                    bool isDominated = false;
                    foreach (var other in methodResults)
                    {
                        if (other == r) continue;

                        bool otherIsBetterOrEqual = other.Qps >= r.Qps && other.RecallAt10 >= r.RecallAt10;
                        bool otherIsStrictlyBetter = other.Qps > r.Qps || other.RecallAt10 > r.RecallAt10;

                        if (otherIsBetterOrEqual && otherIsStrictlyBetter)
                        {
                            isDominated = true;
                            break;
                        }
                    }

                    if (!isDominated)
                    {
                        paretoFront.Add(r);
                    }
                }

                // Sort Pareto front by Recall@10
                paretoFront = paretoFront.OrderBy(r => r.RecallAt10).ToList();

                foreach (var p in paretoFront)
                {
                    Console.WriteLine($"{p.M,-5} {p.EfConstruction,-5} {p.EfSearch,-5} {p.Gamma,-6} {p.Mb,-4} {p.Qps,-10:F2} {p.RecallAt10,-10:P2}");
                }
            }
        }
    }

    internal class ConsoleProgress : IProgressReporter
    {
        private Stopwatch _sw;
        public void Progress(int current, int total)
        {
            if (_sw is null) _sw = Stopwatch.StartNew();
            if (_sw.Elapsed.TotalSeconds > 5)
            {
                Console.WriteLine($"At {current:n0} of {total:n0} or {100f * current / total}%");
                _sw.Restart();
            }
        }
    }
}
