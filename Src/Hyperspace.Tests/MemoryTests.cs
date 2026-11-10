namespace Hyperspace.Tests
{
    using System;
    using System.Buffers;
    using System.Collections.Generic;
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    [TestClass]
    public class MemoryTests
    {
        private static long Heap()
        {
            GC.Collect(2, GCCollectionMode.Forced, true, true);
            GC.WaitForPendingFinalizers();
            GC.Collect(2, GCCollectionMode.Forced, true, true);
            return GC.GetTotalMemory(true);
        }

        private static HyperspaceIndex Build(float[][] data, QuantizationMode mode = QuantizationMode.Float32, DistanceMetric metric = DistanceMetric.Cosine, Action<HyperspaceOptions>? configure = null)
        {
            var o = new HyperspaceOptions { Dimension = data[0].Length, Metric = metric, Quantization = mode, Seed = 3, EfConstruction = 60 };
            configure?.Invoke(o);
            var index = new HyperspaceIndex(o);
            if (mode is QuantizationMode.ProductQuantization or QuantizationMode.OPQ)
            {
                index.TrainProductQuantizer(data.Take(1000).ToArray(), 2);
            }

            index.AddRange(data);
            return index;
        }

        // ------------------------------------------------------------------ frozen layer 0

        [TestMethod]
        public void FreezingLayer0ReleasesTheCopyItReplaces()
        {
            var data = TestData.Clustered(8000, 32, 1);
            var index = Build(data);
            long before = Heap();
            index.FreezeFlatLayer0();
            long after = Heap();
            Assert.IsTrue(index.IsLayer0Frozen);
            Assert.IsTrue(after <= before, $"freezing must not add memory: {before} -> {after} bytes (the flat arrays alone are ~{8000 * 32 * 4} bytes)");
            GC.KeepAlive(index);
        }

        [TestMethod]
        public void FrozenAndThawedGraphsAnswerAndSerializeIdentically()
        {
            var data = TestData.Clustered(1500, 24, 2);
            var queries = TestData.Clustered(25, 24, 3);
            var index = Build(data);
            var thawedAnswers = queries.Select(q => index.Search(q, 10).Select(r => (r.Id, r.Distance)).ToArray()).ToArray();
            var neighbours = Enumerable.Range(0, 200).Select(i => index.GetNeighbors(i)).ToArray();
            var thawedBytes = new ArrayBufferWriter<byte>();
            index.SerializeGraph(thawedBytes);

            index.FreezeFlatLayer0();
            Assert.IsTrue(index.IsLayer0Frozen);
            for (int i = 0; i < queries.Length; i++)
            {
                CollectionAssert.AreEqual(thawedAnswers[i], index.Search(queries[i], 10).Select(r => (r.Id, r.Distance)).ToArray());
            }

            for (int i = 0; i < neighbours.Length; i++)
            {
                CollectionAssert.AreEqual(neighbours[i], index.GetNeighbors(i));
            }

            var frozenBytes = new ArrayBufferWriter<byte>();
            index.SerializeGraph(frozenBytes);
            CollectionAssert.AreEqual(thawedBytes.WrittenSpan.ToArray(), frozenBytes.WrittenSpan.ToArray(), "the file does not depend on where layer 0 lives");

            index.Delete(5);
            Assert.IsTrue(index.IsLayer0Frozen, "deleting only hides an item; the links do not change");
            Assert.IsFalse(index.Search(data[5], 5).Any(r => r.Id == 5));
        }

        [TestMethod]
        public void WritesThawLayer0AndKeepTheGraphIntact()
        {
            var data = TestData.Clustered(1200, 16, 4);
            var index = Build(data.Take(1000).ToArray());
            index.FreezeFlatLayer0();

            int added = index.Add(data[1000]);
            Assert.IsFalse(index.IsLayer0Frozen, "a write thaws layer 0");
            Assert.AreEqual(added, index.Search(data[1000], 1)[0].Id);

            index.FreezeFlatLayer0();
            index.Update(7, data[1100]);
            Assert.IsFalse(index.IsLayer0Frozen);
            Assert.AreEqual(7, index.Search(data[1100], 1)[0].Id);

            index.FreezeFlatLayer0();
            index.OptimizeAsSng(1.2);
            Assert.IsFalse(index.IsLayer0Frozen);
            Assert.AreEqual(7, index.Search(data[1100], 1)[0].Id);

            // Freezing twice is harmless, and the graph still finds its own members.
            index.FreezeFlatLayer0();
            index.FreezeFlatLayer0();
            for (int i = 0; i < 1000; i += 97)
            {
                Assert.IsTrue(index.Search(data[i], 5).Any(r => r.Id == i || r.Distance < 1e-6), $"item {i}");
            }
        }

        [TestMethod]
        public void LoadedIndexesHoldLayer0Once()
        {
            var data = TestData.Clustered(3000, 32, 5);
            var index = Build(data, configure: o => o.FreezeOnLoad = false);
            var bytes = new ArrayBufferWriter<byte>();
            index.Save(bytes);
            index = null!;

            long start = Heap();
            var loaded = HyperspaceIndex.Load(bytes.WrittenMemory);
            long frozen = Heap() - start;
            Assert.IsTrue(loaded.IsLayer0Frozen);

            loaded.Add(data[0]);                  // thaws
            Assert.IsFalse(loaded.IsLayer0Frozen);
            long thawed = Heap() - start;
            Assert.IsTrue(frozen <= thawed + (64 * 1024), $"frozen {frozen} B vs thawed {thawed} B");
            GC.KeepAlive(loaded);
        }

        // ------------------------------------------------------------------ allocation-free queries

        [TestMethod]
        [DataRow(QuantizationMode.None, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.Float32, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.Float32, DistanceMetric.L2)]
        [DataRow(QuantizationMode.Float32, DistanceMetric.Dot)]
        [DataRow(QuantizationMode.ScalarInt8, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.ScalarInt4, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.Turbo, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.Ternary, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.Binary, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.ProductQuantization, DistanceMetric.L2)]
        [DataRow(QuantizationMode.OPQ, DistanceMetric.L2)]
        public void SearchingIntoASpanDoesNotAllocate(QuantizationMode mode, DistanceMetric metric)
        {
            var data = TestData.Clustered(1500, 64, 6);
            var queries = TestData.Clustered(60, 64, 7);
            var index = Build(data, mode, metric);
            AssertNoAllocation(index, queries, new SearchOptions { Ef = 64 }, $"{mode}/{metric}");
            index.FreezeFlatLayer0();
            AssertNoAllocation(index, queries, new SearchOptions { Ef = 64 }, $"{mode}/{metric} frozen");
        }

        [TestMethod]
        public void SearchingWithReRankingTruncationAndMetadataDoesNotAllocate()
        {
            var data = TestData.Clustered(1500, 64, 8);
            var queries = TestData.Clustered(60, 64, 9);
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 64, Quantization = QuantizationMode.Binary, KeepOriginalVectors = true, Seed = 1 });
            for (int i = 0; i < data.Length; i++)
            {
                index.Add(data[i], new Dictionary<string, string> { ["n"] = i.ToString() });
            }

            AssertNoAllocation(index, queries, new SearchOptions { Ef = 64, RerankOversample = 6, IncludeMetadata = true }, "rerank + metadata");

            var plain = Build(data, QuantizationMode.ScalarInt8, DistanceMetric.L2);
            AssertNoAllocation(plain, queries, new SearchOptions { Ef = 64, TruncateDimensions = 32 }, "truncated");
        }

        [TestMethod]
        public void HyperbolicQueriesDoNotAllocate()
        {
            var ball = TestData.Poincare(1200, 16, 10);
            AssertNoAllocation(Build(ball, QuantizationMode.None, DistanceMetric.Poincare), TestData.Poincare(40, 16, 11), new SearchOptions { Ef = 64 }, "poincare");
            AssertNoAllocation(Build(ball, QuantizationMode.ScalarInt8, DistanceMetric.Poincare), TestData.Poincare(40, 16, 11), new SearchOptions { Ef = 64 }, "poincare int8");
            var lorentz = TestData.Lorentz(1200, 17, 12);
            AssertNoAllocation(Build(lorentz, QuantizationMode.None, DistanceMetric.Lorentz), TestData.Lorentz(40, 17, 13), new SearchOptions { Ef = 64 }, "lorentz");
        }

        private static void AssertNoAllocation(HyperspaceIndex index, float[][] queries, SearchOptions options, string label)
        {
            var results = new SearchResult[10];
            for (int pass = 0; pass < 3; pass++)
            {
                foreach (var q in queries) index.Search(q, results, options);   // warm up the thread's scratch buffers
            }

            long before = GC.GetAllocatedBytesForCurrentThread();
            int found = 0;
            for (int pass = 0; pass < 3; pass++)
            {
                foreach (var q in queries) found += index.Search(q, results, options);
            }

            long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
            Assert.IsTrue(found > 0, label);
            Assert.AreEqual(0L, allocated, $"{label}: {allocated} bytes allocated by {3 * queries.Length} searches");
        }

        [TestMethod]
        public void SpanAndArraySearchesAgree()
        {
            var data = TestData.Clustered(800, 16, 14);
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 16, Seed = 2 });
            for (int i = 0; i < data.Length; i++)
            {
                index.Add(data[i], new Dictionary<string, string> { ["n"] = i.ToString() });
            }

            var q = data[3];
            var array = index.Search(q, 7, new SearchOptions { IncludeMetadata = true });
            var span = new SearchResult[7];
            Assert.AreEqual(7, index.Search(q, span, new SearchOptions { IncludeMetadata = true }));
            for (int i = 0; i < 7; i++)
            {
                Assert.AreEqual(array[i].Id, span[i].Id);
                Assert.AreEqual(array[i].Distance, span[i].Distance);
                Assert.AreEqual(span[i].Id.ToString(), span[i].Metadata!["n"]);
            }

            // Fewer matches than slots: only the first are written.
            var wide = new SearchResult[20];
            Assert.AreEqual(1, index.Search(q, wide, new SearchOptions { ExactMetadata = new Dictionary<string, string> { ["n"] = "3" } }));
            Assert.AreEqual(3, wide[0].Id);
            Assert.AreEqual(0, index.Search(q, new SearchResult[1], new SearchOptions { Filter = FilterExpr.Match("n", "nope") }));
        }

        [TestMethod]
        public void MetadataHandedOutIsImmutableAndNotCopied()
        {
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 4, Seed = 1 });
            var source = new Dictionary<string, string> { ["k"] = "v" };
            index.Add(new[] { 1f, 0f, 0f, 0f }, source);
            source["k"] = "changed";                       // the index keeps its own copy

            var a = index.GetMetadata(0)!;
            var b = index.Search(new[] { 1f, 0f, 0f, 0f }, 1, new SearchOptions { IncludeMetadata = true })[0].Metadata!;
            Assert.AreEqual("v", a["k"]);
            Assert.AreSame(a, b, "no copy per result");
            Assert.ThrowsExactly<NotSupportedException>(() => ((IDictionary<string, string>)a)["k"] = "x");
        }

        [TestMethod]
        public void IndexesOfDifferentShapesCanShareAThread()
        {
            var small = TestData.Clustered(300, 8, 15);
            var large = TestData.Clustered(300, 96, 16);
            var a = Build(small, QuantizationMode.Float32, DistanceMetric.L2);
            var b = Build(large, QuantizationMode.Ternary, DistanceMetric.Cosine);
            var ra = new SearchResult[3];
            var rb = new SearchResult[3];
            for (int round = 0; round < 20; round++)
            {
                a.Search(small[round], ra);
                b.Search(large[round], rb);
                Assert.AreEqual(round, ra[0].Id);
                Assert.AreEqual(0.0, ra[0].Distance, 1e-6);
                Assert.AreEqual(3, b.Search(large[round], 3).Length);
            }
        }

        // ------------------------------------------------------------------ indexing

        [TestMethod]
        public void AddingMakesLittleGarbageBeyondWhatItStores()
        {
            var data = TestData.Clustered(3000, 128, 17);
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 128, Quantization = QuantizationMode.Float32, Seed = 4, EfConstruction = 60 });
            for (int i = 0; i < 1500; i++) index.Add(data[i]);                // warm up

            long before = GC.GetAllocatedBytesForCurrentThread();
            for (int i = 1500; i < 3000; i++) index.Add(data[i]);
            double perAdd = (GC.GetAllocatedBytesForCurrentThread() - before) / 1500.0;

            // Stored per item: the encoded vector (512 B + header), the node and its link lists (~350 B). Preparing, searching and
            // selecting neighbours must reuse the thread's buffers: a prepared query alone would add > 1.5 KB.
            Assert.IsTrue(perAdd < 1500, $"{perAdd:F0} bytes allocated per Add");
        }

        [TestMethod]
        public void BatchAddsReuseTheirPreparedQueries()
        {
            var data = TestData.Clustered(3000, 128, 18);
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 128, Quantization = QuantizationMode.Float32, Seed = 5, EfConstruction = 60 });
            index.AddRange(data.Take(512).ToArray());                           // warm up (pool threads, scratch)
            long before = GC.GetTotalAllocatedBytes(true);
            index.AddRange(data.Skip(512).ToArray());
            double perItem = (GC.GetTotalAllocatedBytes(true) - before) / (double)(data.Length - 512);
            Assert.IsTrue(perItem < 2200, $"{perItem:F0} bytes allocated per batched item");
        }
    }
}
