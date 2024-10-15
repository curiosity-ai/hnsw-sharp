// <copyright file="Program.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net.Demo
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.IO;
    using System.Linq;
    using System.Numerics;
    using System.Runtime.CompilerServices;
    using System.Runtime.Intrinsics;
    using System.Runtime.Serialization.Formatters.Binary;
    using System.Text.Json;
    using System.Threading;
    using System.Threading.Tasks;

    using Parameters = SmallWorld<float[], float>.Parameters;

    public static partial class Program
    {
        private const int SampleSize = 500;
        private const int SampleIncrSize = 100;
        private const int TestSize = 10 * SampleSize;
        private const int Dimensionality = 128;
        private const string VectorsPathSuffix = "vectors.hnsw";
        private const string GraphPathSuffix = "graph.hnsw";

        public static async Task Main()
        {
            await MultithreadAddAndReadAsync();
            BuildAndSave("random");
            LoadAndSearch("random");
        }

        private static async Task MultithreadAddAndReadAsync()
        {
            var world = new SmallWorld<float[], float>(CosineDistance.SIMDForUnits, DefaultRandomGenerator.Instance, new Parameters() { EnableDistanceCacheForConstruction  = true, InitialDistanceCacheSize = SampleSize, NeighbourHeuristic = NeighbourSelectionHeuristic.SelectHeuristic, KeepPrunedConnections = true, ExpandBestSelection = true}, threadSafe : false);

            var cts = new CancellationTokenSource();

            var taskAdd = Task.Run(() =>
            {
                while (!cts.IsCancellationRequested)
                {
                    Console.Write($"Generating {SampleSize} sample vectors... ");
                    var clock = Stopwatch.StartNew();
                    var sampleVectors = RandomVectors(Dimensionality, SampleSize);
                    Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");

                    Console.WriteLine("Building HNSW graph... ");

                    using (var listener = new MetricsEventListener(EventSources.GraphBuildEventSource.Instance))
                    {
                        clock = Stopwatch.StartNew();
                        for (int i = 0; i < (SampleSize / SampleIncrSize); i++)
                        {
                            world.AddItems(sampleVectors.Skip(i * SampleIncrSize).Take(SampleIncrSize).ToArray());
                            Console.WriteLine($"\nAt {i + 1} of {SampleSize / SampleIncrSize}  Elapsed: {clock.ElapsedMilliseconds} ms.\n");
                        }
                        Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");
                    }
                }
            });

            var taskSearch = Task.Run(async () =>
            {
                while (!cts.IsCancellationRequested)
                {
                    try
                    {
                        var searchVectors = RandomVectors(Dimensionality, SampleSize);
                        Console.WriteLine("Running search agains the graph... ");
                        using (var listener = new MetricsEventListener(EventSources.GraphSearchEventSource.Instance))
                        {
                            var clock = Stopwatch.StartNew();
                            await Parallel.ForEachAsync(searchVectors, (vector, ct) =>
                            {
                                world.KNNSearch(vector, 10);
                                Console.Write('.');
                                return default;
                            });
                            Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");
                        }
                    }
                    catch (Exception E)
                    {
                        throw;
                    }
                }
            });


            cts.CancelAfter(TimeSpan.FromMinutes(15));

            await Task.WhenAll(taskAdd, taskSearch);
        }

        private static void BuildAndSave(string pathPrefix)
        {
            var world = new SmallWorld<float[], float>(CosineDistance.SIMDForUnits, DefaultRandomGenerator.Instance, new Parameters() { EnableDistanceCacheForConstruction  = true, InitialDistanceCacheSize = SampleSize, NeighbourHeuristic = NeighbourSelectionHeuristic.SelectHeuristic, KeepPrunedConnections = true, ExpandBestSelection = true});

            Console.Write($"Generating {SampleSize} sample vectors... ");
            var clock = Stopwatch.StartNew();
            var sampleVectors = RandomVectors(Dimensionality, SampleSize);
            Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");

            Console.WriteLine("Building HNSW graph... ");
            using (var listener = new MetricsEventListener(EventSources.GraphBuildEventSource.Instance))
            {
                clock = Stopwatch.StartNew();
                for(int i = 0; i < (SampleSize / SampleIncrSize); i++)
                {
                    world.AddItems(sampleVectors.Skip(i * SampleIncrSize).Take(SampleIncrSize).ToArray());
                    Console.WriteLine($"\nAt {i+1} of {SampleSize / SampleIncrSize}  Elapsed: {clock.ElapsedMilliseconds} ms.\n");
                }
                Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");
            }

            Console.Write($"Saving HNSW graph to '${Path.Combine(Directory.GetCurrentDirectory(), pathPrefix)}'... ");
            clock = Stopwatch.StartNew();
            MemoryStream sampleVectorsStream = new MemoryStream();
            JsonSerializer.Serialize(sampleVectorsStream, sampleVectors);
            File.WriteAllBytes($"{pathPrefix}.{VectorsPathSuffix}", sampleVectorsStream.ToArray());


            using (var f = File.Open($"{pathPrefix}.{GraphPathSuffix}", FileMode.Create))
            {
                world.SerializeGraph(f);
            }

            Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");
        }

        private static void LoadAndSearch(string pathPrefix)
        {
            Stopwatch clock;

            Console.Write("Loading HNSW graph... ");
            clock = Stopwatch.StartNew();
            var sampleVectors = JsonSerializer.Deserialize<List<float[]>>(File.ReadAllText($"{pathPrefix}.{VectorsPathSuffix}"));
            SmallWorld<float[], float> world;
            using (var f = File.OpenRead($"{pathPrefix}.{GraphPathSuffix}"))
            {
                world = SmallWorld<float[], float>.DeserializeGraph(sampleVectors, CosineDistance.SIMDForUnits, DefaultRandomGenerator.Instance, f);
            }
            Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");

            Console.Write($"Generating {TestSize} test vectos... ");
            clock = Stopwatch.StartNew();
            var vectors = RandomVectors(Dimensionality, TestSize);
            Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");

            Console.WriteLine("Running search agains the graph... ");
            using (var listener = new MetricsEventListener(EventSources.GraphSearchEventSource.Instance))
            {
                clock = Stopwatch.StartNew();
                Parallel.ForEach(vectors, (vector) =>
                {
                    world.KNNSearch(vector, 10);
                });
                Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");
            }
        }

        private static List<float[]> RandomVectors(int vectorSize, int vectorsCount)
        {
            var vectors = new List<float[]>();

            for (int i = 0; i < vectorsCount; i++)
            {
                var vector = new float[vectorSize];
                DefaultRandomGenerator.Instance.NextFloats(vector);
                VectorUtils.NormalizeSIMD(vector);
                vectors.Add(vector);
            }

            return vectors;
        }
    }
}
