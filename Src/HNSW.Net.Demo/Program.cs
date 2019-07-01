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
    using System.Runtime.Serialization.Formatters.Binary;
    using System.Threading.Tasks;

    using Parameters = SmallWorld<float[], float>.Parameters;

    public static partial class Program
    {
        private const int SampleSize = 5_000;
        private const int SampleIncrSize = 1000;
        private const int TestSize = 10 * SampleSize;
        private const int Dimensionality = 128;
        private const string VectorsPathSuffix = "vectors.hnsw";
        private const string GraphPathSuffix = "graph.hnsw";

        public static void Main()
        {
            BuildAndSave("random");
            LoadAndSearch("random");
        }

        private static void BuildAndSave(string pathPrefix)
        {
            var world = new SmallWorld<float[], float>(CosineDistance.SIMDForUnits, DefaultRandomGenerator.Instance, new Parameters() { EnableDistanceCacheForConstruction  = false});

            Console.Write($"Generating {SampleSize} sample vectos... ");
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
            BinaryFormatter formatter = new BinaryFormatter();
            MemoryStream sampleVectorsStream = new MemoryStream();
            formatter.Serialize(sampleVectorsStream, sampleVectors);
            File.WriteAllBytes($"{pathPrefix}.{VectorsPathSuffix}", sampleVectorsStream.ToArray());

            using (var f = File.OpenWrite($"{pathPrefix}.{GraphPathSuffix}"))
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
            BinaryFormatter formatter = new BinaryFormatter();
            var sampleVectors = (List<float[]>)formatter.Deserialize(new MemoryStream(File.ReadAllBytes($"{pathPrefix}.{VectorsPathSuffix}")));
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
