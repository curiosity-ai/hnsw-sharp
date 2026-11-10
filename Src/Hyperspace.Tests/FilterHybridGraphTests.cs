namespace Hyperspace.Tests
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    [TestClass]
    public class FilterTests
    {
        private static HyperspaceIndex Docs(out float[][] data)
        {
            data = TestData.Clustered(600, 16, 77);
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 16, Metric = DistanceMetric.L2, Seed = 2 });
            for (int i = 0; i < data.Length; i++)
            {
                index.Add(data[i], new Dictionary<string, string>
                {
                    ["color"] = i % 3 == 0 ? "red" : i % 3 == 1 ? "green" : "blue",
                    ["year"] = (2000 + (i % 25)).ToString(),
                    ["path"] = "/docs/" + (i % 7) + "/page" + i,
                    ["price"] = (i % 10 + 0.5).ToString(System.Globalization.CultureInfo.InvariantCulture),
                });
            }

            return index;
        }

        private static void AssertAll(HyperspaceIndex index, float[][] data, SearchOptions options, Func<int, bool> predicate, string label)
        {
            options.Ef = 600;
            var hits = index.Search(data[0], 60, options);
            Assert.IsTrue(hits.Length > 0, label);
            foreach (var h in hits)
            {
                Assert.IsTrue(predicate(h.Id), $"{label}: id {h.Id} should have been filtered out");
            }

            // Every item that satisfies the filter is eligible: with a big k the result is exactly the allowed set.
            var all = index.Search(data[0], 600, options);
            var expected = Enumerable.Range(0, data.Length).Where(predicate).ToArray();
            CollectionAssert.AreEquivalent(expected, all.Select(r => r.Id).ToArray(), label);
        }

        [TestMethod]
        public void MatchPrefixRangeAndBooleanLogic()
        {
            var index = Docs(out var data);
            AssertAll(index, data, new SearchOptions { Filter = FilterExpr.Match("color", "red") }, i => i % 3 == 0, "match");
            AssertAll(index, data, new SearchOptions { ExactMetadata = new Dictionary<string, string> { ["color"] = "red", ["year"] = "2003" } }, i => i % 3 == 0 && i % 25 == 3, "exact");
            AssertAll(index, data, new SearchOptions { Filter = FilterExpr.Prefix("path", "/docs/2/") }, i => i % 7 == 2, "prefix");
            AssertAll(index, data, new SearchOptions { Filter = FilterExpr.Range("year", 2010, 2014) }, i => i % 25 >= 10 && i % 25 <= 14, "integer range");
            AssertAll(index, data, new SearchOptions { Filter = FilterExpr.Range("year", 2020, null) }, i => i % 25 >= 20, "open range");
            AssertAll(index, data, new SearchOptions { Filter = FilterExpr.Range("price", 2.0, 4.0) }, i => i % 10 + 0.5 >= 2.0 && i % 10 + 0.5 <= 4.0, "fractional range");
            AssertAll(index, data, new SearchOptions { Filter = FilterExpr.And(FilterExpr.Match("color", "green"), FilterExpr.Range("year", null, 2005)) }, i => i % 3 == 1 && i % 25 <= 5, "and");
            AssertAll(index, data, new SearchOptions { Filter = FilterExpr.Or(FilterExpr.Match("color", "red"), FilterExpr.Match("year", "2001")) }, i => i % 3 == 0 || i % 25 == 1, "or");
            AssertAll(index, data, new SearchOptions { Filter = FilterExpr.Not(FilterExpr.Match("color", "red")) }, i => i % 3 != 0, "not");
            AssertAll(index, data, new SearchOptions { Filter = FilterExpr.And(FilterExpr.Not(FilterExpr.Match("color", "red")), FilterExpr.Or(FilterExpr.Prefix("path", "/docs/1/"), FilterExpr.Match("year", "2024"))) }, i => i % 3 != 0 && (i % 7 == 1 || i % 25 == 24), "nested");
        }

        [TestMethod]
        public void FilterThatMatchesNothingReturnsNothing()
        {
            var index = Docs(out var data);
            Assert.AreEqual(0, index.Search(data[0], 5, new SearchOptions { Filter = FilterExpr.Match("color", "purple") }).Length);
            Assert.AreEqual(0, index.Search(data[0], 5, new SearchOptions { Filter = FilterExpr.Match("nokey", "x") }).Length);
            Assert.AreEqual(0, index.Search(data[0], 5, new SearchOptions { Filter = FilterExpr.Range("year", 3000, 4000) }).Length);
        }

        [TestMethod]
        public void SelectiveFiltersStillReturnTheExactBestMatches()
        {
            // 5000 items, 0.5% satisfy the filter: far below the graph-walk threshold, so this is the exact scan path.
            var data = TestData.Clustered(5000, 16, 78);
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 16, Metric = DistanceMetric.L2, Seed = 4 });
            for (int i = 0; i < data.Length; i++)
            {
                index.Add(data[i], new Dictionary<string, string> { ["rare"] = i % 200 == 0 ? "yes" : "no" });
            }

            var q = data[1];
            var hits = index.Search(q, 5, new SearchOptions { Filter = FilterExpr.Match("rare", "yes") });
            var truth = Enumerable.Range(0, data.Length).Where(i => i % 200 == 0)
                .OrderBy(i => Metrics.Distance(DistanceMetric.L2, (ReadOnlySpan<float>)q, data[i])).Take(5).ToArray();
            CollectionAssert.AreEqual(truth, hits.Select(h => h.Id).ToArray());
        }

        [TestMethod]
        public void DeletedItemsLeaveTheFilters()
        {
            var index = Docs(out var data);
            index.Delete(0);
            var hits = index.Search(data[0], 600, new SearchOptions { Filter = FilterExpr.Match("color", "red"), Ef = 600 });
            Assert.IsFalse(hits.Any(h => h.Id == 0));
            Assert.AreEqual(199, hits.Length);
        }

        [TestMethod]
        public void MetadataIsReturnedOnRequest()
        {
            var index = Docs(out var data);
            Assert.IsNull(index.Search(data[0], 1)[0].Metadata);
            var hit = index.Search(data[0], 1, new SearchOptions { IncludeMetadata = true })[0];
            Assert.AreEqual("red", hit.Metadata!["color"]);
        }

        [TestMethod]
        public void TypedShadowKeysAreUnderstood()
        {
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 4, Seed = 1 });
            var v = new[] { 1f, 2f, 3f, 4f };
            index.Add(v, new Dictionary<string, string> { ["__hs_typed__age"] = "{\"v\": 41}", ["__hs_typed__ok"] = "{\"v\": true}" });
            index.Add(new[] { 4f, 3f, 2f, 1f }, new Dictionary<string, string> { ["__hs_typed__age"] = "{\"v\": 17.5}" });
            Assert.AreEqual(1, index.Search(v, 5, new SearchOptions { Filter = FilterExpr.Range("age", 18, 100) }).Length);
            Assert.AreEqual(1, index.Search(v, 5, new SearchOptions { Filter = FilterExpr.Match("ok", "true") }).Length);
            Assert.AreEqual(1, index.Search(v, 5, new SearchOptions { Filter = FilterExpr.Match("age", "41") }).Length);
        }

        [TestMethod]
        public void GeometricFilters()
        {
            var data = TestData.Clustered(500, 4, 79);
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 4, Metric = DistanceMetric.L2, Seed = 5 });
            index.AddRange(data);

            double[] lo = { -1, -1, -1, -1 }, hi = { 1, 1, 1, 1 };
            AssertAll(index, data, new SearchOptions { Filter = FilterExpr.InBox(lo, hi) }, i => data[i].All(x => x >= -1 && x <= 1), "box");

            double[] centre = { 0, 0, 0, 0 };
            AssertAll(index, data, new SearchOptions { Filter = FilterExpr.InBall(centre, 3.0) }, i => data[i].Sum(x => (double)x * x) <= 9.0, "ball");

            double[] axes = { 0, 0, 0, 0 }, apertures = { 2, 2, 2, 2 };
            var cone = new ConeRegion(axes, apertures, 0.0);
            AssertAll(index, data, new SearchOptions { Filter = FilterExpr.InCone(axes, apertures) }, i => cone.Contains(data[i].Select(x => (double)x).ToArray()), "cone");

            Assert.IsTrue(new BoxRegion(lo, hi).Contains(new double[] { 0, 0, 0, 0 }));
            Assert.IsFalse(new BallRegion(centre, -1).Contains(centre));
        }

        [TestMethod]
        public void LorentzBallIsHyperbolic()
        {
            var centre = Riemannian.LiftToHyperboloid(Enumerable.Repeat(0.0, 32).ToArray());
            var near = Riemannian.LiftToHyperboloid(Enumerable.Range(0, 32).Select(i => i == 0 ? 0.5 : 0.0).ToArray());
            var far = Riemannian.LiftToHyperboloid(Enumerable.Range(0, 32).Select(i => i == 0 ? 5.0 : 0.0).ToArray());
            double dNear = Metrics.Distance(DistanceMetric.Lorentz, centre, near);
            var region = new BallRegion(centre, dNear + 1e-6);
            Assert.IsTrue(region.Contains(near));
            Assert.IsFalse(region.Contains(far));
        }

        [TestMethod]
        public void IdSetAlgebra()
        {
            var a = new IdSet(new[] { 1, 5, 70, 200 });
            var b = new IdSet(new[] { 5, 70, 71 });
            var u = a.Clone();
            u.UnionWith(b);
            CollectionAssert.AreEqual(new[] { 1, 5, 70, 71, 200 }, u.ToArray());
            var i = a.Clone();
            i.IntersectWith(b);
            CollectionAssert.AreEqual(new[] { 5, 70 }, i.ToArray());
            var d = a.Clone();
            d.ExceptWith(b);
            CollectionAssert.AreEqual(new[] { 1, 200 }, d.ToArray());
            Assert.AreEqual(4, a.Count);
            Assert.IsTrue(a.Contains(200));
            Assert.IsFalse(a.Contains(201));
            Assert.IsTrue(a.Remove(200));
            Assert.IsFalse(a.Remove(200));
            CollectionAssert.AreEqual(new[] { 0, 1, 2 }, IdSet.Range(3).ToArray());
            Assert.AreEqual(130, IdSet.Range(130).Count);
            Assert.IsTrue(new IdSet().IsEmpty);
        }
    }

    [TestClass]
    public class HybridSearchTests
    {
        private static HyperspaceIndex Corpus(out float[][] vectors)
        {
            vectors = TestData.Clustered(300, 16, 55);
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 16, Metric = DistanceMetric.Cosine, Seed = 6 });
            for (int i = 0; i < vectors.Length; i++)
            {
                string text = i == 123 ? "the quick brown fox jumps over the lazy dog" : i == 7 ? "foxes are quick and clever animals" : "lorem ipsum dolor sit amet number " + i;
                index.Add(vectors[i], new Dictionary<string, string> { ["text"] = text });
            }

            return index;
        }

        [TestMethod]
        public void KeywordsPullDocumentsIntoTheTop()
        {
            var index = Corpus(out var vectors);
            // A vector far from document 123, but a query text that only it matches well.
            var query = vectors[200];
            var vectorOnly = index.Search(query, 5);
            Assert.IsFalse(vectorOnly.Any(r => r.Id == 123));

            var hybrid = index.Search(query, 5, new SearchOptions { HybridQuery = "quick brown fox" });
            Assert.AreEqual(123, hybrid[0].Id, "the only document with all three words");
            Assert.IsTrue(hybrid.Any(r => r.Id == 7), "'foxes ... quick' stems to the query terms");
            Assert.IsNotNull(hybrid[0].Score);
            Assert.AreEqual(10.0 - hybrid[0].Score!.Value, hybrid[0].Distance, 1e-9);
            for (int i = 1; i < hybrid.Length; i++)
            {
                Assert.IsTrue(hybrid[i - 1].Score >= hybrid[i].Score);
            }
        }

        [TestMethod]
        public void WeightedFusionRespectsTheAlpha()
        {
            var index = Corpus(out var vectors);
            var query = vectors[200];
            var vectorHeavy = index.Search(query, 3, new SearchOptions { HybridQuery = "fox", FusionMethod = "weighted", HybridAlpha = 1.0f });
            Assert.AreEqual(200, vectorHeavy[0].Id);

            var textHeavy = index.Search(query, 3, new SearchOptions { HybridQuery = "quick brown fox", FusionMethod = "weighted", HybridAlpha = 0.0f });
            Assert.AreEqual(123, textHeavy[0].Id);
        }

        [TestMethod]
        public void StopWordsOnlyQueryFallsBackToVectors()
        {
            var index = Corpus(out var vectors);
            var plain = index.Search(vectors[3], 5).Select(r => r.Id).ToArray();
            var hybrid = index.Search(vectors[3], 5, new SearchOptions { HybridQuery = "the and of" }).Select(r => r.Id).ToArray();
            CollectionAssert.AreEqual(plain, hybrid);
        }

        [TestMethod]
        public void LexicalIndexFollowsUpdatesAndDeletes()
        {
            var index = Corpus(out var vectors);
            var q = vectors[200];
            var search = new SearchOptions { HybridQuery = "fox" };

            // Both document 123 and document 7 ("foxes ...", stemmed) contain the term.
            var first = index.Search(q, 3, search)[0].Id;
            Assert.IsTrue(first is 123 or 7, first.ToString());
            index.Delete(first);
            Assert.IsFalse(index.Search(q, 5, search).Any(r => r.Id == first));
            Assert.IsTrue(index.Search(q, 5, search).Any(r => r.Id is 123 or 7), "the other fox document is still found");

            index.UpdateMetadata(50, new Dictionary<string, string> { ["text"] = "a fox appears" });
            Assert.IsTrue(index.Search(q, 3, search).Any(r => r.Id == 50));
            index.UpdateMetadata(50, new Dictionary<string, string> { ["text"] = "nothing to see" });
            Assert.IsFalse(index.Search(q, 5, search).Any(r => r.Id == 50), "the old terms are removed from the index");

            var id = index.Add(vectors[0], new Dictionary<string, string> { ["text"] = "fox fox fox" });
            Assert.IsTrue(index.Search(q, 5, search).Any(r => r.Id == id));
        }

        [TestMethod]
        public void FiltersApplyToTheKeywordSideToo()
        {
            var vectors = TestData.Clustered(100, 8, 3);
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 8, Seed = 1 });
            for (int i = 0; i < vectors.Length; i++)
            {
                index.Add(vectors[i], new Dictionary<string, string> { ["text"] = "common word", ["lang"] = i < 50 ? "en" : "de" });
            }

            var hits = index.Search(vectors[0], 40, new SearchOptions { HybridQuery = "common", Filter = FilterExpr.Match("lang", "de") });
            Assert.IsTrue(hits.Length > 0);
            Assert.IsTrue(hits.All(h => h.Id >= 50));
        }

        [TestMethod]
        public void Bm25MatchesTheReferenceFormulae()
        {
            // N = 10, df = 2, tf = 3, dl = avgdl: the length normalization vanishes, so tf-component = tf (k1 + 1) / (tf + k1).
            float k1 = 1.2f;
            float tf = 3f * (k1 + 1f) / (3f + k1);
            Assert.AreEqual(MathF.Log(11f / 2f), Bm25.Idf(Bm25Method.Bm25Plus, 10, 2), 1e-6);
            Assert.AreEqual(MathF.Log(10f / 2f), Bm25.Idf(Bm25Method.Atire, 10, 2), 1e-6);
            Assert.AreEqual(MathF.Log(1f + (8.5f / 2.5f)), Bm25.Idf(Bm25Method.Lucene, 10, 2), 1e-6);
            Assert.AreEqual(MathF.Log(8.5f / 2.5f), Bm25.Idf(Bm25Method.Robertson, 10, 2), 1e-6);
            Assert.AreEqual(MathF.Log(11f / 2.5f), Bm25.Idf(Bm25Method.Bm25l, 10, 2), 1e-6);

            Assert.AreEqual(tf, Bm25.TermFrequencyComponent(Bm25Method.Atire, 3, 7, 7, k1, 0.75f, 0.5f), 1e-5);
            Assert.AreEqual(tf + 0.5f, Bm25.TermFrequencyComponent(Bm25Method.Bm25Plus, 3, 7, 7, k1, 0.75f, 0.5f), 1e-5);
            Assert.AreEqual(3f / (k1 + 3f), Bm25.TermFrequencyComponent(Bm25Method.Lucene, 3, 7, 7, k1, 0.75f, 0.5f), 1e-5);

            // Longer documents score lower.
            Assert.IsTrue(Bm25.Score(Bm25Method.Lucene, 3, 20, 10, 10, 2, k1, 0.75f, 0) < Bm25.Score(Bm25Method.Lucene, 3, 5, 10, 10, 2, k1, 0.75f, 0));
        }

        [TestMethod]
        public void TokenizerBehaviour()
        {
            var t = Tokenizer.English;
            CollectionAssert.AreEqual(new[] { "quick", "fox" }, t.Tokenize("The Quick, FOXES!").ToArray());
            CollectionAssert.AreEqual(new[] { "box", "church", "class", "glass" }, new[] { "boxes", "churches", "class", "glasses" }.Select(Tokenizer.LightEnglishStem).ToArray());
            CollectionAssert.AreEqual(new[] { "quick", "brown", "quick_brown" }, t.Tokenize("quick brown", 2).ToArray());
            CollectionAssert.AreEqual(new[] { "the", "end" }, Tokenizer.Plain.Tokenize("The END").ToArray());
            CollectionAssert.AreEqual(new[] { "a", "b" }, Tokenizer.FromFunction(s => s.Split(',')).Tokenize("a,b").ToArray());
            Assert.AreEqual("city", Tokenizer.LightEnglishStem("cities"));
            Assert.AreEqual("run", Tokenizer.LightEnglishStem("running"));
            Assert.AreEqual("class", Tokenizer.LightEnglishStem("class"));
            Assert.AreEqual(0, t.Tokenize("").Count);
        }
    }

    [TestClass]
    public class GraphTests
    {
        [TestMethod]
        public void CompactEdgeIsTwelveBytes()
        {
            Assert.AreEqual(12, System.Runtime.InteropServices.Marshal.SizeOf<CompactEdge>());
            var e = new CompactEdge(42, EdgeRelation.DependsOn, 0.95f);
            Assert.IsTrue(e.IsActive);
        }

        [TestMethod]
        public void DynamicInsertDeleteAndReachability()
        {
            var g = new DynamicGraph(10);
            g.InsertEdge(1, 2, EdgeRelation.Owns, 1f);
            g.InsertEdge(2, 3, EdgeRelation.DependsOn, 1f);
            g.InsertEdge(3, 4, EdgeRelation.References, 1f);
            Assert.AreEqual(3, g.TotalEdges);

            var two = g.BfsReachability(new uint[] { 1 }, 2, 64);
            CollectionAssert.AreEqual(new[] { 1, 2, 3 }, two.ToArray());
            CollectionAssert.AreEqual(new[] { 1, 2 }, g.BfsReachability(new uint[] { 1 }, 5, 64, new ushort[] { EdgeRelation.Owns }).ToArray());
            CollectionAssert.AreEqual(new[] { 1 }, g.BfsReachability(new uint[] { 1 }, 0, 64).ToArray());

            Assert.IsTrue(g.DeleteEdge(1, 2, EdgeRelation.Owns));
            Assert.IsFalse(g.DeleteEdge(1, 2, EdgeRelation.Owns));
            CollectionAssert.AreEqual(new[] { 1 }, g.BfsReachability(new uint[] { 1 }, 3, 64).ToArray(), "deletion is visible immediately");

            g.InsertEdge(1, 2, EdgeRelation.Owns, 0.5f);
            Assert.AreEqual(3, g.TotalEdges, "re-inserting revives the edge");
            Assert.AreEqual(0.5f, g.EdgesFor(1).Single().Weight);

            g.InsertEdge(100, 2500, EdgeRelation.Contains, 1f);
            Assert.IsTrue(g.NodeCount >= 2501);
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => g.InsertEdge(DynamicGraph.MaxNodes, 0, 1, 1f));
        }

        [TestMethod]
        public void FanoutLimitCapsHubExpansion()
        {
            var g = new DynamicGraph();
            for (uint i = 1; i <= 100; i++)
            {
                g.InsertEdge(0, i, 1, 1f);
            }

            Assert.AreEqual(1 + 10, g.BfsReachability(new uint[] { 0 }, 1, 10).Count);
            Assert.AreEqual(1 + 10, g.ToCsr().BfsReachability(new uint[] { 0 }, 1, 10).Count);
        }

        [TestMethod]
        public void CsrCompactionAndRoundTrip()
        {
            var g = new DynamicGraph(3);
            g.InsertEdge(0, 1, EdgeRelation.Supersedes, 1f);
            g.InsertEdge(0, 2, EdgeRelation.Amends, 0.8f);
            g.InsertEdge(1, 2, EdgeRelation.Owns, 0.5f);
            g.DeleteEdge(0, 2, EdgeRelation.Amends);

            var csr = g.ToCsr();
            Assert.AreEqual(2, csr.EdgeCount);
            Assert.AreEqual(1, csr.OutDegree(0));
            Assert.AreEqual(0, csr.Neighbors(2).Length);
            Assert.AreEqual(0, csr.Neighbors(99).Length);
            CollectionAssert.AreEqual(new uint[] { 0, 1, 2, 2 }, csr.RowOffsets);

            using var ms = new MemoryStream();
            csr.Write(ms);
            var bytes = ms.ToArray();
            Assert.AreEqual("HYPR_CSR", System.Text.Encoding.ASCII.GetString(bytes, 0, 8));
            Assert.AreEqual(3UL, BitConverter.ToUInt64(bytes, 8));
            Assert.AreEqual(2UL, BitConverter.ToUInt64(bytes, 16));
            Assert.AreEqual(24 + (4 * 4) + (2 * 12), bytes.Length);

            var copy = CsrGraph.Read(new MemoryStream(bytes));
            CollectionAssert.AreEqual(csr.RowOffsets, copy.RowOffsets);
            CollectionAssert.AreEqual(csr.Edges, copy.Edges);

            bytes[0] = (byte)'X';
            Assert.ThrowsExactly<InvalidDataException>(() => CsrGraph.Read(new MemoryStream(bytes)));

            var corrupt = ms.ToArray();
            BitConverter.TryWriteBytes(corrupt.AsSpan(24 + 4), 7u);
            Assert.ThrowsExactly<InvalidDataException>(() => CsrGraph.Read(new MemoryStream(corrupt)));
            Assert.ThrowsExactly<EndOfStreamException>(() => CsrGraph.Read(new MemoryStream(ms.ToArray().AsSpan(0, 30).ToArray())));
        }

        [TestMethod]
        public void EngineFilesAndLayouts()
        {
            var engine = new GraphEngine(new GraphConfig { StorageLayout = GraphStorageLayout.Csr, MaxHopsLimit = 2 });
            for (uint i = 0; i < 6; i++)
            {
                engine.InsertEdge(i, i + 1, EdgeRelation.Contains);
            }

            Assert.AreEqual(1, engine.GetReachabilityMask(new uint[] { 0 }, 5).Count, "the CSR snapshot is stale until compacted");
            engine.Compact();
            CollectionAssert.AreEqual(new[] { 0, 1, 2 }, engine.GetReachabilityMask(new uint[] { 0 }, 5).ToArray(), "hops are capped by the configuration");

            string path = Path.Combine(Path.GetTempPath(), "hyperspace-" + Guid.NewGuid().ToString("N") + ".csr");
            try
            {
                engine.SaveToCsrFile(path);
                var loaded = GraphEngine.LoadFromCsrFile(path, new GraphConfig { MaxHopsLimit = 10 });
                Assert.AreEqual(7, loaded.GetReachabilityMask(new uint[] { 0 }, 10).Count);
                Assert.AreEqual(6, loaded.Dynamic.TotalEdges);
            }
            finally
            {
                File.Delete(path);
            }

            var off = new GraphEngine(new GraphConfig { Enabled = false });
            off.InsertEdge(0, 1, 1);
            Assert.AreEqual(0, off.Dynamic.TotalEdges);
            CollectionAssert.AreEqual(new[] { 0 }, off.GetReachabilityMask(new uint[] { 0 }, 3).ToArray());
        }

        [TestMethod]
        public void InductionRulesBuildEdgesFromMetadata()
        {
            var nodes = new List<(uint, IReadOnlyDictionary<string, string>)>
            {
                (0, new Dictionary<string, string> { ["path"] = "/a", ["sku"] = "x1" }),
                (1, new Dictionary<string, string> { ["path"] = "/a/b", ["sku"] = "x2", ["parent_sku"] = "x1", ["group"] = "g" }),
                (2, new Dictionary<string, string> { ["path"] = "/a/b/c/", ["sku"] = "x3", ["parent_sku"] = "x2", ["group"] = "g" }),
                (3, new Dictionary<string, string> { ["path"] = "/z", ["ref"] = "2", ["group"] = "g" }),
            };

            var engine = new GraphEngine();
            int n = engine.InduceFromMetadata(nodes, new[]
            {
                new MetadataInductionRule("path", "path_hierarchy", "CONTAINS"),
                new MetadataInductionRule("parent_sku", "foreign_key", "depends_on", TargetField: "sku"),
                new MetadataInductionRule("ref", "foreign_key", "REFERENCES"),
                new MetadataInductionRule("group", "exact_match", "7", Weight: 0.5f),
            });

            Assert.AreEqual(2 + 2 + 1 + 6, n);
            Assert.AreEqual(1, engine.EdgesFor(0).Count(e => e.Relation == EdgeRelation.Contains && e.Target == 1), "/a contains /a/b");
            Assert.AreEqual(1, engine.EdgesFor(1).Count(e => e.Relation == EdgeRelation.Contains && e.Target == 2), "trailing slash is ignored");
            Assert.AreEqual(1, engine.EdgesFor(2).Count(e => e.Relation == EdgeRelation.DependsOn && e.Target == 1));
            Assert.AreEqual(1, engine.EdgesFor(3).Count(e => e.Relation == EdgeRelation.References && e.Target == 2));
            Assert.AreEqual(0.5f, engine.EdgesFor(1).First(e => e.Relation == 7).Weight);
            Assert.AreEqual(EdgeRelation.Owns, EdgeRelation.FromLabel("owns"));
            Assert.AreEqual((ushort)42, EdgeRelation.FromLabel("42"));
            Assert.AreEqual(EdgeRelation.Custom, EdgeRelation.FromLabel("whatever"));
            Assert.IsTrue(EdgeRelation.IsHierarchical(EdgeRelation.Contains));
        }

        [TestMethod]
        public void GraphRestrictsVectorSearch()
        {
            var data = TestData.Clustered(300, 8, 12);
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 8, Seed = 8 });
            index.AddRange(data);
            index.Link(0, 10, EdgeRelation.References);
            index.Link(10, 20, EdgeRelation.References);
            index.Link(0, 30, EdgeRelation.Owns);
            index.Link(30, 40, EdgeRelation.Owns);

            var two = index.Search(data[0], 50, new SearchOptions { Graph = new GraphFilter(new uint[] { 0 }, 2) }).Select(r => r.Id).ToArray();
            CollectionAssert.AreEquivalent(new[] { 0, 10, 20, 30, 40 }, two);

            var refs = index.Search(data[0], 50, new SearchOptions { Graph = new GraphFilter(new uint[] { 0 }, 2, new[] { EdgeRelation.References }) }).Select(r => r.Id).ToArray();
            CollectionAssert.AreEquivalent(new[] { 0, 10, 20 }, refs);

            Assert.IsTrue(index.Unlink(10, 20, EdgeRelation.References));
            CollectionAssert.AreEquivalent(new[] { 0, 10 }, index.Search(data[0], 50, new SearchOptions { Graph = new GraphFilter(new uint[] { 0 }, 2, new[] { EdgeRelation.References }) }).Select(r => r.Id).ToArray());

            // Graph restriction composes with metadata filters.
            index.UpdateMetadata(30, new Dictionary<string, string> { ["k"] = "v" });
            CollectionAssert.AreEqual(new[] { 30 }, index.Search(data[0], 5, new SearchOptions { Graph = new GraphFilter(new uint[] { 0 }, 2), Filter = FilterExpr.Match("k", "v") }).Select(r => r.Id).ToArray());
        }

        [TestMethod]
        public void MetadataRulesThroughTheIndex()
        {
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 2, Seed = 1 });
            index.Add(new[] { 0f, 0f }, new Dictionary<string, string> { ["path"] = "/root" });
            index.Add(new[] { 1f, 0f }, new Dictionary<string, string> { ["path"] = "/root/child" });
            index.Add(new[] { 2f, 0f });
            Assert.AreEqual(1, index.InduceGraphFromMetadata(new[] { new MetadataInductionRule("path", "path_hierarchy", "CONTAINS") }));
            Assert.AreEqual(1, index.Graph.EdgesFor(0).Count);
        }

        [TestMethod]
        public void LorentzTreeEmbedding()
        {
            var root = LorentzTreeProjector.ProjectTreeNode(0, 0, 1.0);
            var child = LorentzTreeProjector.ProjectTreeNode(1, 0, 1.0);
            var grandchild = LorentzTreeProjector.ProjectTreeNode(2, 0, 1.0);
            Assert.AreEqual(1.0, LorentzTreeProjector.LorentzDistance(root, child), 1e-9);
            Assert.AreEqual(2.0, LorentzTreeProjector.LorentzDistance(root, grandchild), 1e-9);
            Assert.IsTrue(LorentzTreeProjector.IsInLightCone(root, child, 1.5));
            Assert.IsFalse(LorentzTreeProjector.IsInLightCone(root, grandchild, 1.5));
            Assert.IsFalse(LorentzTreeProjector.IsInLightCone(child, root, 5.0), "an ancestor is in the past cone");
            Assert.AreEqual(0.0, LorentzTreeProjector.ComputeDistortion(new List<(double[], double[], double)> { (root, child, 1.0), (root, grandchild, 2.0) }), 1e-9);
        }
    }

    [TestClass]
    public class ClusteringTests
    {
        private static float[][] Blobs(int perBlob, int blobs, int dim, int seed)
        {
            var r = new Random(seed);
            return Enumerable.Range(0, blobs).SelectMany(b => Enumerable.Range(0, perBlob).Select(_ =>
                Enumerable.Range(0, dim).Select(d => (d == 0 ? b * 20f : 0f) + TestData.Gaussian(r) * 0.5f).ToArray())).ToArray();
        }

        [TestMethod]
        public void KMeansRecoversSeparatedBlobs()
        {
            var data = Blobs(60, 4, 6, 1);
            var result = KMeans.Fit(data, 4, seed: 3);
            Assert.AreEqual(4, result.Centroids.Length);
            for (int b = 0; b < 4; b++)
            {
                var labels = result.Assignments.Skip(b * 60).Take(60).Distinct().ToArray();
                Assert.AreEqual(1, labels.Length, $"blob {b} must be one cluster");
            }

            Assert.AreEqual(4, result.Assignments.Distinct().Count());
            Assert.IsTrue(result.Inertia / data.Length < 6 * 0.25 * 2, "within-cluster variance is about the noise");
            Assert.AreEqual(3, KMeans.Fit(data.Take(3).ToArray(), 10).Centroids.Length, "k is clamped to the number of points");
        }

        [TestMethod]
        public void StreamingKMeansConvergesToTheSameStructure()
        {
            var data = Blobs(200, 3, 4, 2);
            var result = KMeans.FitStreaming(() => data.Chunk(64).Select(c => (IReadOnlyList<float[]>)c), 3, epochs: 3, seed: 1);
            var centreX = result.Centroids.Select(c => c[0]).OrderBy(x => x).ToArray();
            Assert.AreEqual(0f, centreX[0], 2f);
            Assert.AreEqual(20f, centreX[1], 2f);
            Assert.AreEqual(40f, centreX[2], 2f);
        }

        [TestMethod]
        public void RouterPicksTheNearestSegments()
        {
            var router = new MetaRouter();
            for (int i = 0; i < 10; i++)
            {
                router.Register(new SegmentInfo("seg" + i, new[] { i * 10f, 0f }, 100));
            }

            var routes = router.Route(new[] { 31f, 0f }, 2);
            CollectionAssert.AreEqual(new[] { "seg3", "seg4" }, routes.Select(r => r.Segment.Id).ToArray());
            Assert.AreEqual(10, router.Route(new[] { 0f, 0f }, 0).Count, "probe 0 means everything");
            Assert.AreEqual(1000, router.TotalVectorCount);

            router.Register(new SegmentInfo("seg3", new[] { 500f, 0f }, 1));
            Assert.AreEqual("seg3", router.Route(new[] { 499f, 0f }, 1)[0].Segment.Id);
            Assert.IsTrue(router.Unregister("seg3"));
            Assert.IsFalse(router.Unregister("seg3"));
            Assert.AreEqual(9, router.Count);
        }

        [TestMethod]
        public void RouterUsesMacroClustersBeyondThirtyTwoSegments()
        {
            var router = new MetaRouter();
            var r = new Random(4);
            for (int i = 0; i < 200; i++)
            {
                router.Register(new SegmentInfo("s" + i, new[] { i * 3f, (float)r.NextDouble() }, 10));
            }

            long before = router.ProbeOperations;
            var routes = router.Route(new[] { 300f, 0.5f }, 3);
            Assert.AreEqual("s100", routes[0].Segment.Id);
            Assert.IsTrue(router.ProbeOperations - before < 200, "hierarchical routing must not touch every centroid");

            router.ReplaceAll(new[] { new SegmentInfo("only", new[] { 1f, 1f }, 5) });
            Assert.AreEqual(1, router.Count);
        }
    }
}
