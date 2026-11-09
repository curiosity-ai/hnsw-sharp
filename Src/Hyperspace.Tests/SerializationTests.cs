namespace Hyperspace.Tests
{
    using System;
    using System.Buffers;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Text;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    [TestClass]
    public class SerializationTests
    {
        private static HyperspaceOptions Options(int dim, DistanceMetric metric = DistanceMetric.Cosine, QuantizationMode mode = QuantizationMode.None)
            => new HyperspaceOptions { Dimension = dim, Metric = metric, Quantization = mode, Seed = 12 };

        private static IReadOnlyList<HyperspaceItem> Items(float[][] data)
            => data.Select((v, i) => new HyperspaceItem(v, new Dictionary<string, string> { ["n"] = i.ToString(), ["text"] = "doc " + i })).ToArray();

        private static void AssertSameAnswers(HyperspaceIndex a, HyperspaceIndex b, float[][] queries, int k = 10, SearchOptions? options = null)
        {
            foreach (var q in queries)
            {
                var x = a.Search(q, k, options);
                var y = b.Search(q, k, options);
                CollectionAssert.AreEqual(x.Select(r => r.Id).ToArray(), y.Select(r => r.Id).ToArray());
                for (int i = 0; i < x.Length; i++)
                {
                    Assert.AreEqual(x[i].Distance, y[i].Distance, 1e-9);
                }
            }
        }

        [TestMethod]
        public void GraphRoundTripGivesIdenticalSearches()
        {
            var data = TestData.Clustered(500, 24, 1);
            var queries = TestData.Clustered(25, 24, 2);
            var index = new HyperspaceIndex(Options(24));
            index.AddRange(Items(data));
            index.Delete(3);

            using var ms = new MemoryStream();
            index.SerializeGraph(ms);

            ms.Position = 0;
            var (copy, rest) = HyperspaceIndex.DeserializeGraph(Items(data), ms);
            Assert.AreEqual(0, rest.Count);
            Assert.AreEqual(index.Count, copy.Count);
            Assert.IsFalse(copy.Contains(3));
            Assert.IsTrue(copy.IsLayer0Frozen, "loading freezes layer 0");
            AssertSameAnswers(index, copy, queries);
            for (int n = 0; n < data.Length; n += 37)
            {
                CollectionAssert.AreEqual(index.GetNeighbors(n), copy.GetNeighbors(n));
            }

            AssertSameAnswers(index, copy, queries, options: new SearchOptions { Filter = FilterExpr.Range("n", 100, 300) });
        }

        [TestMethod]
        public void GraphCanBeWrittenToABufferWriter()
        {
            var data = TestData.Clustered(200, 8, 3);
            var index = new HyperspaceIndex(Options(8, DistanceMetric.L2));
            index.AddRange(data);

            var writer = new ArrayBufferWriter<byte>();
            index.SerializeGraph(writer);
            using var ms = new MemoryStream();
            index.SerializeGraph(ms);
            CollectionAssert.AreEqual(ms.ToArray(), writer.WrittenSpan.ToArray(), "both paths produce the same bytes");

            var (copy, _) = HyperspaceIndex.DeserializeGraph(HyperspaceItem.FromVectors(data), writer.WrittenMemory);
            AssertSameAnswers(index, copy, data.Take(10).ToArray(), 5);
        }

        [TestMethod]
        public void ItemsBeyondTheGraphComeBack()
        {
            var data = TestData.Clustered(300, 8, 4);
            var index = new HyperspaceIndex(Options(8, DistanceMetric.L2));
            index.AddRange(data.Take(250).ToArray());
            using var ms = new MemoryStream();
            index.SerializeGraph(ms);

            ms.Position = 0;
            var (copy, rest) = HyperspaceIndex.DeserializeGraph(Items(data), ms);
            Assert.AreEqual(250, copy.Count);
            Assert.AreEqual(50, rest.Count);
            CollectionAssert.AreEqual(data[250], rest[0].Vector);
            copy.AddRange(rest);
            Assert.AreEqual(300, copy.Count);
            Assert.AreEqual(299, copy.Search(data[299], 1)[0].Id);
        }

        [TestMethod]
        public void FewerItemsThanNodesIsCorruption()
        {
            var data = TestData.Clustered(50, 4, 5);
            var index = new HyperspaceIndex(Options(4, DistanceMetric.L2));
            index.AddRange(data);
            using var ms = new MemoryStream();
            index.SerializeGraph(ms);
            ms.Position = 0;
            Assert.ThrowsExactly<InvalidDataException>(() => HyperspaceIndex.DeserializeGraph(HyperspaceItem.FromVectors(data.Take(10)), ms));
        }

        [TestMethod]
        public void CorruptGraphsAreRejected()
        {
            var data = TestData.Clustered(60, 4, 6);
            var index = new HyperspaceIndex(Options(4, DistanceMetric.L2));
            index.AddRange(data);
            using var ms = new MemoryStream();
            index.SerializeGraph(ms);
            var good = ms.ToArray();
            var items = HyperspaceItem.FromVectors(data);

            Exception? Try(byte[] bytes)
            {
                try
                {
                    HyperspaceIndex.DeserializeGraph(items, new MemoryStream(bytes));
                    return null;
                }
                catch (Exception e)
                {
                    return e;
                }
            }

            Assert.IsNull(Try(good));

            var badMagic = (byte[])good.Clone();
            badMagic[0] ^= 0xFF;
            Assert.IsInstanceOfType<InvalidDataException>(Try(badMagic));

            Assert.IsNotNull(Try(good.Take(good.Length / 2).ToArray()), "truncated");
            Assert.IsNotNull(Try(good.Take(20).ToArray()), "truncated header");

            // Offset of the first node's neighbour ids: header + parameters, then (nodeCount, entry, maxLevel, total) and the first record header.
            int parametersLength = BitConverter.ToInt32(good, 8);
            int firstRecord = 8 + 4 + parametersLength + 4 + 4 + 4 + 8;
            int layers = BitConverter.ToInt32(good, firstRecord + 4);
            int recordLength = BitConverter.ToInt32(good, firstRecord + 8);
            int ids = firstRecord + 12 + ((layers + 1) * 4);
            Assert.IsTrue(recordLength > layers + 1, "the first node should have neighbours");

            var badNeighbour = (byte[])good.Clone();
            BitConverter.TryWriteBytes(badNeighbour.AsSpan(ids), 9999);
            Assert.IsInstanceOfType<InvalidDataException>(Try(badNeighbour));

            var badId = (byte[])good.Clone();
            BitConverter.TryWriteBytes(badId.AsSpan(firstRecord), 5);
            Assert.IsInstanceOfType<InvalidDataException>(Try(badId));

            var badOffsets = (byte[])good.Clone();
            BitConverter.TryWriteBytes(badOffsets.AsSpan(firstRecord + 12 + (layers * 4)), 12345);
            Assert.IsInstanceOfType<InvalidDataException>(Try(badOffsets));

            var hugeRecord = (byte[])good.Clone();
            BitConverter.TryWriteBytes(hugeRecord.AsSpan(firstRecord + 8), int.MaxValue);
            Assert.IsInstanceOfType<InvalidDataException>(Try(hugeRecord));

            var negativeCount = (byte[])good.Clone();
            BitConverter.TryWriteBytes(negativeCount.AsSpan(8 + 4 + parametersLength), -1);
            Assert.IsInstanceOfType<InvalidDataException>(Try(negativeCount));
        }

        [TestMethod]
        public void GraphLayoutIsTheDocumentedFlatRecord()
        {
            var index = new HyperspaceIndex(Options(2, DistanceMetric.L2));
            index.Add(new[] { 0f, 0f });
            index.Add(new[] { 1f, 0f });
            using var ms = new MemoryStream();
            index.SerializeGraph(ms);
            var b = ms.ToArray();

            Assert.AreEqual("HSPG0001", Encoding.ASCII.GetString(b, 0, 8));
            int p = 8;
            int parametersLength = BitConverter.ToInt32(b, p);
            p += 4 + parametersLength;
            Assert.AreEqual(2, BitConverter.ToInt32(b, p), "node count");
            int entry = BitConverter.ToInt32(b, p + 4);
            Assert.IsTrue(entry is 0 or 1);
            long total = BitConverter.ToInt64(b, p + 12);
            p += 20;
            long sum = 0;
            for (int i = 0; i < 2; i++)
            {
                Assert.AreEqual(i, BitConverter.ToInt32(b, p), "node id");
                int layers = BitConverter.ToInt32(b, p + 4);
                int length = BitConverter.ToInt32(b, p + 8);
                Assert.AreEqual(layers + 1 + index.GetNeighbors(i, 0).Length + Enumerable.Range(1, layers - 1).Sum(l => index.GetNeighbors(i, l).Length), length);
                Assert.AreEqual(0, BitConverter.ToInt32(b, p + 12), "layer 0 starts at 0");
                sum += length;
                p += 12 + (length * 4);
            }

            Assert.AreEqual(total, sum);
            Assert.AreEqual(0, BitConverter.ToInt32(b, p), "no deleted items");
            Assert.AreEqual(b.Length, p + 4);
        }

        [TestMethod]
        [DataRow(DistanceMetric.Cosine, QuantizationMode.None)]
        [DataRow(DistanceMetric.L2, QuantizationMode.Float32)]
        [DataRow(DistanceMetric.Cosine, QuantizationMode.ScalarInt8)]
        [DataRow(DistanceMetric.Cosine, QuantizationMode.ScalarInt4)]
        [DataRow(DistanceMetric.Cosine, QuantizationMode.Turbo)]
        [DataRow(DistanceMetric.Cosine, QuantizationMode.Ternary)]
        [DataRow(DistanceMetric.Cosine, QuantizationMode.Binary)]
        [DataRow(DistanceMetric.L2, QuantizationMode.ProductQuantization)]
        [DataRow(DistanceMetric.L2, QuantizationMode.OPQ)]
        [DataRow(DistanceMetric.Poincare, QuantizationMode.ScalarInt8)]
        public void SnapshotRoundTripPreservesEverything(DistanceMetric metric, QuantizationMode mode)
        {
            var data = metric == DistanceMetric.Poincare ? TestData.Poincare(300, 16, 8) : TestData.Clustered(300, 32, 8);
            int dim = data[0].Length;
            var queries = (metric == DistanceMetric.Poincare ? TestData.Poincare(15, 16, 9) : TestData.Clustered(15, 32, 9));
            var index = new HyperspaceIndex(Options(dim, metric, mode));
            if (mode is QuantizationMode.ProductQuantization or QuantizationMode.OPQ)
            {
                index.TrainProductQuantizer(data, 3);
            }

            index.AddRange(Items(data));
            index.Delete(17);
            index.Link(1, 2, EdgeRelation.Owns, 0.25f);
            index.Link(2, 3, EdgeRelation.References);
            index.Graph.Compact();

            using var ms = new MemoryStream();
            index.Save(ms);
            ms.Position = 0;
            var copy = HyperspaceIndex.Load(ms);

            Assert.AreEqual(index.Count, copy.Count);
            Assert.AreEqual(index.TotalCount, copy.TotalCount);
            Assert.AreEqual(index.DeletedCount, copy.DeletedCount);
            Assert.AreEqual(mode, copy.Quantization);
            Assert.AreEqual(metric, copy.Metric);
            Assert.AreEqual(index.BytesPerVector, copy.BytesPerVector);
            AssertSameAnswers(index, copy, queries);
            Assert.AreEqual("5", copy.GetMetadata(5)!["n"]);
            CollectionAssert.AreEqual(index.GetVector(5), copy.GetVector(5));
            Assert.IsFalse(copy.Contains(17));

            Assert.AreEqual(0.25f, copy.Graph.EdgesFor(1).Single().Weight);
            CollectionAssert.AreEqual(new[] { 1, 2, 3 }, copy.Graph.GetReachabilityMask(new uint[] { 1 }, 2).ToArray());

            AssertSameAnswers(index, copy, queries, options: new SearchOptions { Filter = FilterExpr.Match("n", "40"), IncludeMetadata = true });
            Assert.AreEqual(40, copy.Search(queries[0], 3, new SearchOptions { Filter = FilterExpr.Match("n", "40") }).Single().Id);
        }

        [TestMethod]
        public void HybridSearchWorksAfterReload()
        {
            var data = TestData.Clustered(100, 8, 10);
            var index = new HyperspaceIndex(Options(8));
            index.AddRange(Items(data));
            index.Update(5, data[5], new Dictionary<string, string> { ["text"] = "needle in a haystack" });

            var copy = HyperspaceIndex.Load(SnapshotBytes(index));
            Assert.AreEqual(5, copy.Search(data[50], 3, new SearchOptions { HybridQuery = "needle" })[0].Id);
        }

        private static byte[] SnapshotBytes(HyperspaceIndex index)
        {
            var w = new ArrayBufferWriter<byte>();
            index.Save(w);
            return w.WrittenSpan.ToArray();
        }

        [TestMethod]
        public void AllOptionsSurviveTheRoundTrip()
        {
            var o = new HyperspaceOptions
            {
                Dimension = 40,
                Metric = DistanceMetric.Hybrid,
                Quantization = QuantizationMode.Ternary,
                M = 9,
                EfConstruction = 55,
                EfSearch = 66,
                MaxLayers = 7,
                LevelProbability = 0.25,
                DensityPruning = true,
                AnisotropicRefinement = false,
                TernaryThreshold = 0.5f,
                FusionMethod = "weighted",
                Bm25 = new Bm25Parameters { Method = Bm25Method.Lucene, K1 = 1.5f, B = 0.3f, Delta = 0.1f, NGrams = 2, Language = "english" },
                Graph = new GraphConfig { Enabled = true, StorageLayout = GraphStorageLayout.Csr, MaxHopsLimit = 5, MaxFanoutPerNode = 17 },
            };
            var data = TestData.Hybrid(60, 40, 11);
            var index = new HyperspaceIndex(o);
            index.AddRange(data);

            var copy = HyperspaceIndex.Load(SnapshotBytes(index)).Options;
            Assert.AreEqual(40, copy.Dimension);
            Assert.AreEqual(DistanceMetric.Hybrid, copy.Metric);
            Assert.AreEqual(QuantizationMode.Ternary, copy.Quantization);
            Assert.AreEqual(9, copy.M);
            Assert.AreEqual(55, copy.EfConstruction);
            Assert.AreEqual(66, copy.EfSearch);
            Assert.AreEqual(7, copy.MaxLayers);
            Assert.AreEqual(0.25, copy.LevelProbability);
            Assert.IsTrue(copy.DensityPruning);
            Assert.IsFalse(copy.AnisotropicRefinement);
            Assert.AreEqual(0.5f, copy.TernaryThreshold);
            Assert.AreEqual("weighted", copy.FusionMethod);
            Assert.AreEqual(Bm25Method.Lucene, copy.Bm25.Method);
            Assert.AreEqual(1.5f, copy.Bm25.K1);
            Assert.AreEqual(2, copy.Bm25.NGrams);
            Assert.AreEqual(GraphStorageLayout.Csr, copy.Graph.StorageLayout);
            Assert.AreEqual(17, copy.Graph.MaxFanoutPerNode);
            Assert.AreEqual(33, copy.HeadDimensions);
        }

        [TestMethod]
        public void OriginalVectorsSurviveEveryPersistencePath()
        {
            var data = TestData.Clustered(250, 32, 17);
            var options = Options(32, DistanceMetric.Cosine, QuantizationMode.Binary);
            options.KeepOriginalVectors = true;
            options.RerankOversample = 6;
            var index = new HyperspaceIndex(options);
            index.AddRange(Items(data));
            var queries = TestData.Clustered(10, 32, 18);

            // Snapshot.
            var snapshot = HyperspaceIndex.Load(SnapshotBytes(index));
            Assert.IsTrue(snapshot.Options.KeepOriginalVectors);
            Assert.AreEqual(6, snapshot.Options.RerankOversample);
            CollectionAssert.AreEqual(index.GetVector(9), snapshot.GetVector(9));
            AssertSameAnswers(index, snapshot, queries);

            // Graph + the items the caller keeps.
            using var graph = new MemoryStream();
            index.SerializeGraph(graph);
            graph.Position = 0;
            var (rebuilt, _) = HyperspaceIndex.DeserializeGraph(Items(data), graph);
            CollectionAssert.AreEqual(index.GetVector(9), rebuilt.GetVector(9));
            AssertSameAnswers(index, rebuilt, queries);

            // Compaction.
            index.Delete(0);
            var compact = index.Compact(out var map);
            CollectionAssert.AreEqual(index.GetVector(9), compact.GetVector(map[9]));
            AssertSameAnswers(compact, compact, queries);
        }

        [TestMethod]
        public void EmptyIndexRoundTrips()
        {
            var index = new HyperspaceIndex(Options(5));
            var copy = HyperspaceIndex.Load(SnapshotBytes(index));
            Assert.AreEqual(0, copy.Count);
            Assert.AreEqual(0, copy.Search(new float[5], 3).Length);
            copy.Add(new[] { 1f, 2f, 3f, 4f, 5f });
            Assert.AreEqual(1, copy.Search(new[] { 1f, 2f, 3f, 4f, 5f }, 3).Length);

            var (g, rest) = HyperspaceIndex.DeserializeGraph(Array.Empty<HyperspaceItem>(), CreateGraph(index));
            Assert.AreEqual(0, g.Count);
            Assert.AreEqual(0, rest.Count);
        }

        private static ReadOnlyMemory<byte> CreateGraph(HyperspaceIndex index)
        {
            var w = new ArrayBufferWriter<byte>();
            index.SerializeGraph(w);
            return w.WrittenMemory.ToArray();
        }

        [TestMethod]
        public void SnapshotFilesAreWrittenAtomically()
        {
            var data = TestData.Clustered(100, 8, 13);
            var index = new HyperspaceIndex(Options(8));
            index.AddRange(Items(data));
            string path = Path.Combine(Path.GetTempPath(), "hyperspace-" + Guid.NewGuid().ToString("N") + ".hsp");
            try
            {
                index.Save(path);
                index.Save(path);
                Assert.AreEqual(0, Directory.GetFiles(Path.GetDirectoryName(path)!, Path.GetFileName(path) + ".tmp*").Length);
                var copy = HyperspaceIndex.Load(path);
                AssertSameAnswers(index, copy, data.Take(5).ToArray(), 5);
            }
            finally
            {
                File.Delete(path);
            }
        }

        [TestMethod]
        public void CorruptSnapshotsAreRejected()
        {
            var index = new HyperspaceIndex(Options(8));
            index.AddRange(Items(TestData.Clustered(80, 8, 14)));
            var good = SnapshotBytes(index);

            Assert.ThrowsExactly<InvalidDataException>(() => HyperspaceIndex.Load(good.Skip(1).ToArray()));
            var badMagic = (byte[])good.Clone();
            badMagic[3] = (byte)'?';
            Assert.ThrowsExactly<InvalidDataException>(() => HyperspaceIndex.Load(badMagic));
            Assert.IsNotNull(Record.Exception(() => HyperspaceIndex.Load(good.Take(good.Length - 10).ToArray())));
            Assert.IsNotNull(Record.Exception(() => HyperspaceIndex.Load(good.Take(good.Length / 3).ToArray())));
            HyperspaceIndex.Load(good);
        }

        [TestMethod]
        public void GraphFormatIsCompactComparedToTheSnapshot()
        {
            var data = TestData.Clustered(400, 64, 15);
            var index = new HyperspaceIndex(Options(64, DistanceMetric.Cosine, QuantizationMode.ScalarInt8));
            index.AddRange(data);
            using var graph = new MemoryStream();
            index.SerializeGraph(graph);
            using var snapshot = new MemoryStream();
            index.Save(snapshot);
            Assert.IsTrue(graph.Length < snapshot.Length);
            Assert.IsTrue(snapshot.Length - graph.Length >= 400L * index.BytesPerVector, "the snapshot adds the encoded vectors");
        }

        [TestMethod]
        public void SeparatedProductQuantizerCodebooksTravelWithTheGraph()
        {
            var data = TestData.Clustered(300, 32, 16);
            var index = new HyperspaceIndex(Options(32, DistanceMetric.L2, QuantizationMode.ProductQuantization));
            index.TrainProductQuantizer(data, 4);
            index.AddRange(data);
            using var ms = new MemoryStream();
            index.SerializeGraph(ms);
            ms.Position = 0;
            var (copy, _) = HyperspaceIndex.DeserializeGraph(HyperspaceItem.FromVectors(data), ms);
            AssertSameAnswers(index, copy, data.Take(10).ToArray(), 5);
        }

        private static class Record
        {
            public static Exception? Exception(Action action)
            {
                try
                {
                    action();
                    return null;
                }
                catch (Exception e)
                {
                    return e;
                }
            }
        }
    }
}
