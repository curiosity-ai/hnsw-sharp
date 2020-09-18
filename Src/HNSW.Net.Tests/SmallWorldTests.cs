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
            var parameters = new SmallWorld<float[], float>.Parameters();
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
        [DataTestMethod]
        [DataRow(false,false)]
        [DataRow(false,true)]
        [DataRow(true, false)]
        [DataRow(true, true)]
        public void KNNSearchTestAlgorithm4(bool expandBestSelection, bool keepPrunedConnections )
        {
            var parameters = new SmallWorld<float[], float>.Parameters() { NeighbourHeuristic = NeighbourSelectionHeuristic.SelectHeuristic, ExpandBestSelection = expandBestSelection, KeepPrunedConnections = keepPrunedConnections };
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
                var parameters = new SmallWorld<float[], float>.Parameters()
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

            Assert.AreEqual(original, copy.Print());
        }
    }
}