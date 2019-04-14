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
            this.vectors = data.Select(r => Array.ConvertAll(r.Split('\t'), x => float.Parse(x, CultureInfo.CurrentCulture))).ToList();
        }

        /// <summary>
        /// Basic test for knn search
        /// </summary>
        [TestMethod]
        public void KNNSearchTest()
        {
            var parameters = new SmallWorld<float[], float>.Parameters();
            var graph = new SmallWorld<float[], float>(CosineDistance.NonOptimized);
            graph.BuildGraph(this.vectors, new Random(42), parameters);

            for (int i = 0; i < this.vectors.Count; ++i)
            {
                var result = graph.KNNSearch(this.vectors[i], 20);
                var best = result.OrderBy(r => r.Distance).First();
                Assert.AreEqual(20, result.Count);
                Assert.AreEqual(i, best.Id);
                Assert.AreEqual(0, best.Distance, FloatError);
            }
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
            {
                var parameters = new SmallWorld<float[], float>.Parameters()
                {
                    M = 15,
                    LevelLambda = 1 / Math.Log(15),
                };

                var graph = new SmallWorld<float[], float>(CosineDistance.NonOptimized);
                graph.BuildGraph(this.vectors, new Random(42), parameters);

                buffer = graph.SerializeGraph();
                original = graph.Print();
            }

            var copy = new SmallWorld<float[], float>(CosineDistance.NonOptimized);
            copy.DeserializeGraph(this.vectors, buffer);

            Assert.AreEqual(original, copy.Print());
        }
    }
}