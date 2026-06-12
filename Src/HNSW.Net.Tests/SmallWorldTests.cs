// <copyright file="SmallWorldTests.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    /// <summary>
    /// Tests for <see cref="SmallWorld{TItem, TDistance}"/>
    /// </summary>
    [TestClass]
    public class SmallWorldTests
    {
        // Set floating point error to 5.96 * 10^-7
        // For cosine distance error can be bigger in theory but for test data it's not the case.
        private const float FloatError = 0.000000596f;

        private IReadOnlyList<float[]> vectors;

        /// <summary>
        /// Initializes test resources.
        /// </summary>
        [TestInitialize]
        public void TestInitialize()
        {
            var data = File.ReadAllLines(@"vectors.txt");
            vectors = data.Select(r => Array.ConvertAll(r.Split('\t'), x => float.Parse(x, CultureInfo.CurrentCulture))).ToList();
        }

        /// <summary>
        /// Basic test for knn search - this test might fail sometimes, as the construction of the graph does not guarantee an exact answer
        /// </summary>
        [TestMethod]
        public void KNNSearchTest()
        {
            var parameters = new SmallWorldParameters();
            var graph = new SmallWorld<float[], float>(CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, parameters);
            graph.AddItems(vectors);

            int bestWrong = 0;
            float maxError = float.MinValue;

            for (int i = 0; i < vectors.Count; ++i)
            {
                var result = graph.KNNSearch(vectors[i], 20);
                var best = result.OrderBy(r => r.Distance).First();
                Assert.AreEqual(20, result.Count);
                if (best.Id != i)
                {
                    bestWrong++;
                }
                maxError = Math.Max(maxError, best.Distance);
            }
            Assert.AreEqual(0, bestWrong);
            Assert.AreEqual(0, maxError, FloatError);
        }

        /// <summary>
        /// Basic test for knn search - this test might fail sometimes, as the construction of the graph does not guarantee an exact answer
        /// </summary>
        [TestMethod]
        public void KNNSearchWithFilterTest()
        {
            var parameters = new SmallWorldParameters();
            var graph = new SmallWorld<float[], float>(CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, parameters);
            graph.AddItems(vectors);

            for (int i = 0; i < vectors.Count; ++i)
            {
                var result = graph.KNNSearch(vectors[i], 20, filterItem: v => false);
                Assert.AreEqual(0, result.Count);
            }
        }

        /// <summary>
        /// Basic test for knn search - this test might fail sometimes, as the construction of the graph does not guarantee an exact answer
        /// </summary>
        [DataTestMethod]
        [DataRow(false,false)]
        [DataRow(false,true)]
        [DataRow(true, false)]
        [DataRow(true, true)]
        public void KNNSearchTestAlgorithm4(bool expandBestSelection, bool keepPrunedConnections)
        {
            var parameters = new SmallWorldParameters() { NeighbourHeuristic = NeighbourSelectionHeuristic.SelectHeuristic, ExpandBestSelection = expandBestSelection, KeepPrunedConnections = keepPrunedConnections };
            var graph = new SmallWorld<float[], float>(CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, parameters);
            graph.AddItems(vectors);

            int bestWrong = 0;
            float maxError = float.MinValue;

            for (int i = 0; i < vectors.Count; ++i)
            {
                var result = graph.KNNSearch(vectors[i], 20);
                var best = result.OrderBy(r => r.Distance).First();
                Assert.AreEqual(20, result.Count);
                if (best.Id != i)
                {
                    bestWrong++;
                }
                maxError = Math.Max(maxError, best.Distance);
            }
            Assert.AreEqual(0, 100f * bestWrong / vectors.Count); //Percentage of failed cases
            Assert.AreEqual(0, maxError, FloatError);
        }

        /// <summary>
        /// Verifies that enabling early termination preserves a high recall against the exhaustive search baseline.
        /// </summary>
        [DataTestMethod]
        [DataRow(0)]   // adaptive patience
        [DataRow(4)]   // explicit (aggressive) patience
        [DataRow(16)]  // explicit (conservative) patience
        public void KNNSearchEarlyTerminationRecallTest(int patience)
        {
            const int k = 20;

            var baselineParams = new SmallWorldParameters();
            var baseline = new SmallWorld<float[], float>(CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, baselineParams);
            baseline.AddItems(vectors);

            var earlyParams = new SmallWorldParameters
            {
                EnableEarlyTermination = true,
                EarlyTerminationPatience = patience,
            };
            var early = new SmallWorld<float[], float>(CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, earlyParams);
            early.AddItems(vectors);

            int totalOverlap = 0;
            for (int i = 0; i < vectors.Count; ++i)
            {
                var expected = baseline.KNNSearch(vectors[i], k).Select(r => r.Id).ToHashSet();
                var actual = early.KNNSearch(vectors[i], k).Select(r => r.Id).ToList();

                Assert.AreEqual(k, actual.Count);
                totalOverlap += actual.Count(expected.Contains);
            }

            double recall = (double)totalOverlap / (vectors.Count * k);
            Assert.IsTrue(recall >= 0.85, $"Recall {recall:p2} with early termination (patience {patience}) is below the 85% threshold.");
        }

        /// <summary>
        /// Serialization deserialization tests.
        /// </summary>
        [TestMethod]
        public void SerializeDeserializeTest()
        {
            byte[] buffer;
            string original;

            // restrict scope of original graph
            var stream = new MemoryStream();
            {
                var parameters = new SmallWorldParameters()
                {
                    M = 15,
                    LevelLambda = 1 / Math.Log(15),
                };

                var graph = new SmallWorld<float[], float>(CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, parameters);
                graph.AddItems(vectors);

                graph.SerializeGraph(stream);
                original = graph.Print();
            }
            stream.Position = 0;

            var copy = SmallWorld<float[], float>.DeserializeGraph(vectors, CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, stream);

            Assert.AreEqual(original, copy.Graph.Print());
        }

        /// <summary>
        /// Serialization deserialization tests.
        /// </summary>
        [TestMethod]
        public void SerializeDeserializeWithRemainingItemsTest()
        {
            byte[] buffer;
            string original;
            int rng_state;
            var rng = new RewindableRandomNumberGenerator();

            // restrict scope of original graph
            var stream = new MemoryStream();
            {
                var parameters = new SmallWorldParameters()
                {
                    M = 15,
                    LevelLambda = 1 / Math.Log(15),
                };
                int itemsToLeaveBehindOnSerialization = (int)(vectors.Count / 2);
                
                var graph = new SmallWorld<float[], float>(CosineDistance.NonOptimized, rng, parameters);
                graph.AddItems(vectors.Take(itemsToLeaveBehindOnSerialization).ToArray());

                graph.SerializeGraph(stream);
                rng_state = rng.GetState();
                graph.AddItems(vectors.Skip(itemsToLeaveBehindOnSerialization).ToArray());
                rng.RewindTo(rng_state);

                original = graph.Print();
            }
            stream.Position = 0;

            var copy = SmallWorld<float[], float>.DeserializeGraph(vectors, CosineDistance.NonOptimized, rng, stream);

            copy.Graph.AddItems(copy.ItemsNotInGraph);

            Assert.AreEqual(original, copy.Graph.Print());
        }

        /// <summary>
        /// Regression test: searching while items are being added concurrently (non thread-safe mode, relying on
        /// the version/retry mechanism) must not throw. A previous optimization had the search alias the live
        /// connection lists, so a concurrent insertion that reallocated a node's backing array could hand the
        /// searching thread an out-of-bounds span, surfacing as an ArgumentOutOfRangeException while computing
        /// the distance to a "neighbour" whose id was garbage read past the end of the array.
        /// </summary>
        [TestMethod]
        public void ConcurrentAddAndSearchDoesNotThrowTest()
        {
            // Build a graph, then round-trip it through serialization so its nodes are stored in the flattened
            // (cached) form. Adding more items afterwards lazily re-hydrates the touched cached nodes into fresh
            // List<int> connection lists that grow - and therefore reallocate their backing array - as new
            // neighbours are connected. A concurrent search that aliased such a list could observe a span whose
            // length (the just-incremented count) exceeded the old, smaller backing array and read a garbage id
            // past its end, blowing up with ArgumentOutOfRangeException at Items[garbageId] inside RuntimeDistance.
            // The 224-vector fixture is too small to keep the graph mutating long enough to expose the race,
            // so generate a larger synthetic set (deterministically, for reproducibility).
            var rngData = new Random(12345);
            var data = Enumerable.Range(0, 4000)
                .Select(_ => Enumerable.Range(0, 16).Select(__ => (float)rngData.NextDouble()).ToArray())
                .ToArray();

            int seed = 1500;

            var stream = new MemoryStream();
            var builder = new SmallWorld<float[], float>(CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, new SmallWorldParameters());
            builder.AddItems(data.Take(seed).ToArray());
            builder.SerializeGraph(stream);
            stream.Position = 0;

            var (graph, _) = SmallWorld<float[], float>.DeserializeGraph(data.Take(seed).ToArray(), CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, stream, threadSafe: false);

            var remaining = data.Skip(seed).ToArray();

            Exception failure = null;
            using var stop = new System.Threading.CancellationTokenSource();

            var readers = Enumerable.Range(0, Math.Max(3, Environment.ProcessorCount)).Select(_ => System.Threading.Tasks.Task.Run(() =>
            {
                try
                {
                    var rnd = new Random();
                    while (!stop.IsCancellationRequested)
                    {
                        graph.KNNSearch(data[rnd.Next(seed)], 10);
                    }
                }
                catch (Exception ex)
                {
                    System.Threading.Volatile.Write(ref failure, ex);
                    stop.Cancel();
                }
            })).ToArray();

            try
            {
                // add the remaining items one at a time to maximise the number of connection-list re-hydrations
                // (and therefore reallocations) happening concurrently with the searches
                for (int i = 0; i < remaining.Length && !stop.IsCancellationRequested; ++i)
                {
                    graph.AddItems(new[] { remaining[i] });
                }
            }
            catch (Exception ex)
            {
                System.Threading.Volatile.Write(ref failure, ex);
            }
            finally
            {
                stop.Cancel();
            }

            System.Threading.Tasks.Task.WaitAll(readers);

            Assert.IsNull(failure, $"Concurrent add/search threw: {failure}");
        }
    }
}