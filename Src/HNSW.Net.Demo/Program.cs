// <copyright file="Program.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net.Demo
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Diagnostics.Tracing;
    using System.IO;
    using System.Linq;
    using System.Numerics;
    using System.Runtime.CompilerServices;
    using System.Runtime.Serialization.Formatters.Binary;
    using System.Threading.Tasks;

    using Parameters = SmallWorld<float[], float>.Parameters;

    /// <summary>
    /// The demo program.
    /// </summary>
    public static class Program
    {
        private const int SampleSize = 50_000;
        private const int TestSize = 10 * SampleSize;
        private const int Dimensionality = 256;
        private const string VectorsPathSuffix = "vectors.hnsw";
        private const string GraphPathSuffix = "graph.hnsw";

        /// <summary>
        /// Entry point.
        /// </summary>
        public static void Main()
        {
            BuildAndSave("random");
            LoadAndSearch("random");
        }

        private static void BuildAndSave(string pathPrefix)
        {
            Stopwatch clock;
            List<float[]> sampleVectors;

            var parameters = new Parameters();
            parameters.EnableDistanceCacheForConstruction = false;

            var world = new SmallWorld<float[], float>(CosineDistance.SIMDForUnits);

            Console.Write($"Generating {SampleSize} sample vectos... ");
            clock = Stopwatch.StartNew();
            sampleVectors = RandomVectors(Dimensionality, SampleSize);
            Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");

            Console.WriteLine("Building HNSW graph... ");
            using (var listener = new MetricsEventListener(EventSources.GraphBuildEventSource.Instance))
            {
                clock = Stopwatch.StartNew();
                world.BuildGraph(sampleVectors, new Random(42), parameters);
                Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");
            }

            Console.Write($"Saving HNSW graph to '${Path.Combine(Directory.GetCurrentDirectory(), pathPrefix)}'... ");
            clock = Stopwatch.StartNew();
            BinaryFormatter formatter = new BinaryFormatter();
            MemoryStream sampleVectorsStream = new MemoryStream();
            formatter.Serialize(sampleVectorsStream, sampleVectors);
            File.WriteAllBytes($"{pathPrefix}.{VectorsPathSuffix}", sampleVectorsStream.ToArray());
            File.WriteAllBytes($"{pathPrefix}.{GraphPathSuffix}", world.SerializeGraph());
            Console.WriteLine($"Done in {clock.ElapsedMilliseconds} ms.");
        }

        private static void LoadAndSearch(string pathPrefix)
        {
            Stopwatch clock;
            var world = new SmallWorld<float[], float>(CosineDistance.NonOptimized);

            Console.Write("Loading HNSW graph... ");
            clock = Stopwatch.StartNew();
            BinaryFormatter formatter = new BinaryFormatter();
            var sampleVectors = (List<float[]>)formatter.Deserialize(new MemoryStream(File.ReadAllBytes($"{pathPrefix}.{VectorsPathSuffix}")));
            world.DeserializeGraph(sampleVectors, File.ReadAllBytes($"{pathPrefix}.{GraphPathSuffix}"));
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
            var random = new Random(42);
            var vectors = new List<float[]>();

            for (int i = 0; i < vectorsCount; i++)
            {
                var vector = new float[vectorSize];
                for (int j = 0; j < vectorSize; j++)
                {
                    vector[j] = (float)random.NextDouble();
                }

                Normalize(ref vector);

                vectors.Add(vector);
            }

            return vectors;
        }

        public static void Normalize(ref float[] vector)
        {
            var f = DotProduct(ref vector, ref vector);
            float f1 = (float)(1 / Math.Sqrt(f));
            Debug.Assert(!float.IsNaN(f1));
            Multiply(ref vector, f1);
        }


        private static readonly int _vs1 = Vector<float>.Count;
        private static readonly int _vs2 = 2 * Vector<float>.Count;
        private static readonly int _vs3 = 3 * Vector<float>.Count;
        private static readonly int _vs4 = 4 * Vector<float>.Count;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float DotProduct(ref float[] lhs, ref float[] rhs)
        {
            float result = 0f;

            var count = lhs.Length;
            var offset = 0;

            while (count >= _vs4)
            {
                result += Vector.Dot(new Vector<float>(lhs, offset), new Vector<float>(rhs, offset));
                result += Vector.Dot(new Vector<float>(lhs, offset + _vs1), new Vector<float>(rhs, offset + _vs1));
                result += Vector.Dot(new Vector<float>(lhs, offset + _vs2), new Vector<float>(rhs, offset + _vs2));
                result += Vector.Dot(new Vector<float>(lhs, offset + _vs3), new Vector<float>(rhs, offset + _vs3));
                if (count == _vs4) return result;
                count -= _vs4;
                offset += _vs4;
            }

            if (count >= _vs2)
            {
                result += Vector.Dot(new Vector<float>(lhs, offset), new Vector<float>(rhs, offset));
                result += Vector.Dot(new Vector<float>(lhs, offset + _vs1), new Vector<float>(rhs, offset + _vs1));
                if (count == _vs2) return result;
                count -= _vs2;
                offset += _vs2;
            }
            if (count >= _vs1)
            {
                result += Vector.Dot(new Vector<float>(lhs, offset), new Vector<float>(rhs, offset));
                if (count == _vs1) return result;
                count -= _vs1;
                offset += _vs1;
            }
            if (count > 0)
            {
                while (count > 0)
                {
                    result += lhs[offset] * rhs[offset];
                    offset++; count--;
                }
            }
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static void Multiply(ref float[] lhs, float f)
        {
            var count = lhs.Length;
            var offset = 0;

            while (count >= _vs4)
            {
                (new Vector<float>(lhs, offset) * f).CopyTo(lhs, offset);
                (new Vector<float>(lhs, offset + _vs1) * f).CopyTo(lhs, offset + _vs1);
                (new Vector<float>(lhs, offset + _vs2) * f).CopyTo(lhs, offset + _vs2);
                (new Vector<float>(lhs, offset + _vs3) * f).CopyTo(lhs, offset + _vs3);
                if (count == _vs4) return;
                count -= _vs4;
                offset += _vs4;
            }

            if (count >= _vs2)
            {
                (new Vector<float>(lhs, offset) * f).CopyTo(lhs, offset);
                (new Vector<float>(lhs, offset + _vs1) * f).CopyTo(lhs, offset + _vs1);
                if (count == _vs2) return;
                count -= _vs2;
                offset += _vs2;
            }
            if (count >= _vs1)
            {
                (new Vector<float>(lhs, offset) * f).CopyTo(lhs, offset);
                if (count == _vs1) return;
                count -= _vs1;
                offset += _vs1;
            }
            if (count > 0)
            {
                while (count > 0)
                {
                    lhs[offset] *= f;
                    offset++; count--;
                }
            }
        }

        private class MetricsEventListener : EventListener
        {
            private readonly EventSource eventSource;

            public MetricsEventListener(EventSource eventSource)
            {
                this.eventSource = eventSource;
                this.EnableEvents(this.eventSource, EventLevel.LogAlways, EventKeywords.All, new Dictionary<string, string> { { "EventCounterIntervalSec", "1" } });
            }

            public override void Dispose()
            {
                this.DisableEvents(this.eventSource);
                base.Dispose();
            }

            protected override void OnEventWritten(EventWrittenEventArgs eventData)
            {
                var counterData = eventData.Payload?.FirstOrDefault() as IDictionary<string, object>;
                if (counterData?.Count == 0)
                {
                    return;
                }

                Console.WriteLine($"[{counterData["Name"]}]: Avg={counterData["Mean"]}; SD={counterData["StandardDeviation"]}; Count={counterData["Count"]}");
            }
        }
    }
}
