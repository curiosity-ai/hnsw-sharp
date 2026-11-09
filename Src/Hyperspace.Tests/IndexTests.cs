namespace Hyperspace.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    [TestClass]
    public class IndexRecallTests
    {
        private const int K = 10;

        private static HyperspaceOptions Options(int dim, DistanceMetric metric, QuantizationMode mode = QuantizationMode.None)
            => new HyperspaceOptions { Dimension = dim, Metric = metric, Quantization = mode, Seed = 7, EfConstruction = 100, EfSearch = 100 };

        [TestMethod]
        [DataRow(DistanceMetric.L2)]
        [DataRow(DistanceMetric.Cosine)]
        [DataRow(DistanceMetric.Dot)]
        [DataRow(DistanceMetric.L1)]
        [DataRow(DistanceMetric.Linf)]
        [DataRow(DistanceMetric.Angular)]
        public void EuclideanFamilyFindsTheTrueNeighbours(DistanceMetric metric)
        {
            var data = TestData.Clustered(1500, 32, 100);
            var queries = TestData.Clustered(40, 32, 101);
            bool normalize = metric == DistanceMetric.Cosine || metric == DistanceMetric.Angular;
            if (metric == DistanceMetric.Dot)
            {
                data = data.Select(v => { var c = (float[])v.Clone(); Metrics.Normalize(c); return c; }).ToArray();
                queries = queries.Select(v => { var c = (float[])v.Clone(); Metrics.Normalize(c); return c; }).ToArray();
            }

            var index = TestData.Build(data, Options(32, metric));
            double recall = TestData.Recall(index, data, queries, K, metric, normalize);
            Assert.IsTrue(recall >= 0.92, $"{metric}: recall {recall}");
        }

        [TestMethod]
        public void PoincareFindsTheTrueNeighbours()
        {
            var data = TestData.Poincare(1500, 16, 200);
            var queries = TestData.Poincare(40, 16, 201);
            var index = TestData.Build(data, Options(16, DistanceMetric.Poincare));
            Assert.IsTrue(TestData.Recall(index, data, queries, K, DistanceMetric.Poincare, false) >= 0.92);

            var fast = Options(16, DistanceMetric.Poincare);
            fast.FastRouting = true;
            var fastIndex = TestData.Build(data, fast);
            Assert.IsTrue(TestData.Recall(fastIndex, data, queries, K, DistanceMetric.Poincare, false) >= 0.88, "Klein routing");
        }

        [TestMethod]
        public void LorentzAndHybridFindTheTrueNeighbours()
        {
            var lor = TestData.Lorentz(1500, 17, 300);
            var lq = TestData.Lorentz(40, 17, 301);
            var index = TestData.Build(lor, Options(17, DistanceMetric.Lorentz));
            Assert.IsTrue(TestData.Recall(index, lor, lq, K, DistanceMetric.Lorentz, false) >= 0.92);

            var hyb = TestData.Hybrid(1200, 65, 310);
            var hq = TestData.Hybrid(30, 65, 311);
            var hindex = TestData.Build(hyb, Options(65, DistanceMetric.Hybrid));
            Assert.IsTrue(TestData.Recall(hindex, hyb, hq, K, DistanceMetric.Hybrid, false) >= 0.9);
        }

        [TestMethod]
        public void RemainingMetricsRunEndToEnd()
        {
            var signed = TestData.Clustered(600, 24, 400);
            var positive = signed.Select(v => v.Select(Math.Abs).ToArray()).ToArray();
            foreach (var (metric, data) in new[] { (DistanceMetric.Jaccard, positive), (DistanceMetric.Hamming, signed), (DistanceMetric.Wasserstein, positive) })
            {
                var index = TestData.Build(data, Options(24, metric));
                var hits = index.Search(data[1], 5);
                Assert.AreEqual(5, hits.Length, metric.ToString());
                for (int i = 1; i < hits.Length; i++)
                {
                    Assert.IsTrue(hits[i - 1].Distance <= hits[i].Distance, metric.ToString());
                }

                // The nearest item is at distance zero from itself (possibly tied with an identical one).
                Assert.AreEqual(0.0, hits[0].Distance, 1e-9, metric.ToString());
            }
        }

        [TestMethod]
        [DataRow(QuantizationMode.Float32, 0.92)]
        [DataRow(QuantizationMode.ScalarInt8, 0.88)]
        [DataRow(QuantizationMode.ScalarInt4, 0.6)]
        [DataRow(QuantizationMode.Turbo, 0.7)]
                [DataRow(QuantizationMode.AsymmetricHybrid, 0.88)]
        public void QuantizedStorageStillFindsNeighbours(QuantizationMode mode, double minRecall)
        {
            var data = TestData.Clustered(1500, 64, 500);
            var queries = TestData.Clustered(40, 64, 501);
            var index = TestData.Build(data, Options(64, DistanceMetric.Cosine, mode));
            double recall = TestData.Recall(index, data, queries, K, DistanceMetric.Cosine, normalize: true);
            Assert.IsTrue(recall >= minRecall, $"{mode}: recall {recall}");
        }

        [TestMethod]
        public void BinaryStorageIsFarBetterThanChance()
        {
            var data = TestData.Clustered(1500, 128, 510);
            var queries = TestData.Clustered(40, 128, 511);
            var index = TestData.Build(data, Options(128, DistanceMetric.Cosine, QuantizationMode.Binary));
            double recall = TestData.Recall(index, data, queries, K, DistanceMetric.Cosine, normalize: true);
            Assert.IsTrue(recall >= 0.08, $"binary recall {recall} (chance is 0.007)");
        }

        [TestMethod]
        public void ProductQuantizationWithTrainedCodebooks()
        {
            var data = TestData.Clustered(1500, 64, 520);
            var queries = TestData.Clustered(40, 64, 521);
            foreach (var mode in new[] { QuantizationMode.ProductQuantization, QuantizationMode.OPQ })
            {
                var index = new HyperspaceIndex(Options(64, DistanceMetric.L2, mode));
                index.TrainProductQuantizer(data, 6);
                index.AddRange(data);
                double recall = TestData.Recall(index, data, queries, K, DistanceMetric.L2, normalize: false);
                Assert.IsTrue(recall >= 0.3, $"{mode}: recall {recall}");
                Assert.ThrowsExactly<InvalidOperationException>(() => index.TrainProductQuantizer(data));
            }

            Assert.ThrowsExactly<InvalidOperationException>(() => new HyperspaceIndex(Options(8, DistanceMetric.L2)).TrainProductQuantizer(data));
        }

        [TestMethod]
        [DataRow(QuantizationMode.Binary, DistanceMetric.Cosine, 0.6)]
        [DataRow(QuantizationMode.Ternary, DistanceMetric.Cosine, 0.8)]
        [DataRow(QuantizationMode.ProductQuantization, DistanceMetric.L2, 0.85)]
        [DataRow(QuantizationMode.OPQ, DistanceMetric.L2, 0.85)]
        [DataRow(QuantizationMode.Turbo, DistanceMetric.Cosine, 0.9)]
        public void TwoPassRerankingRecoversNearExactRecall(QuantizationMode mode, DistanceMetric metric, double minTwoPassRecall)
        {
            var data = TestData.Clustered(1500, 64, 530);
            var queries = TestData.Clustered(40, 64, 531);
            var options = Options(64, metric, mode);
            options.KeepOriginalVectors = true;
            var index = new HyperspaceIndex(options);
            if (mode is QuantizationMode.ProductQuantization or QuantizationMode.OPQ)
            {
                index.TrainProductQuantizer(data, 5);
            }

            index.AddRange(data);
            bool normalize = metric == DistanceMetric.Cosine;
            double single = TestData.Recall(index, data, queries, K, metric, normalize, new SearchOptions { RerankOversample = 1 });
            double twoPass = TestData.Recall(index, data, queries, K, metric, normalize, new SearchOptions { RerankOversample = 12, Ef = 300 });
            Assert.IsTrue(twoPass >= minTwoPassRecall, $"{mode}: single pass {single}, two pass {twoPass}");
            Assert.IsTrue(twoPass > single + 0.1, $"{mode}: re-ranking must help a lot ({single} -> {twoPass})");
            Assert.IsTrue(single > 0.03, $"{mode}: the first pass alone should still beat chance, got {single}");

            // The distances reported after re-ranking are exact ones.
            var hit = index.Search(queries[0], 3, new SearchOptions { RerankOversample = 12 })[0];
            var q = (float[])queries[0].Clone();
            var v = (float[])data[hit.Id].Clone();
            if (normalize)
            {
                Metrics.Normalize(q);
                Metrics.Normalize(v);
            }

            Assert.AreEqual(Metrics.Distance(metric, (ReadOnlySpan<float>)q, v), hit.Distance, 1e-4);
        }

        [TestMethod]
        public void OriginalsAreOnlyKeptOnRequest()
        {
            var data = TestData.Clustered(200, 32, 540);
            var plain = TestData.Build(data, Options(32, DistanceMetric.L2, QuantizationMode.ScalarInt4));
            var kept = Options(32, DistanceMetric.L2, QuantizationMode.ScalarInt4);
            kept.KeepOriginalVectors = true;
            var withOriginals = TestData.Build(data, kept);

            Assert.IsTrue(Enumerable.Range(0, 32).Any(d => plain.GetVector(3)[d] != data[3][d]), "quantized storage is lossy");
            CollectionAssert.AreEqual(data[3], withOriginals.GetVector(3), "originals come back exactly");
            Assert.AreEqual(1, plain.Search(data[3], 1, new SearchOptions { RerankOversample = 10 }).Length, "the oversample is ignored when there is nothing to re-rank against");
            Assert.AreEqual(0.0, withOriginals.Search(data[3], 1)[0].Distance, 1e-6, "re-ranked distance of the item itself is exactly zero");

            Assert.ThrowsExactly<ArgumentException>(() => { kept.RerankOversample = 0; new HyperspaceIndex(kept); });
        }

        [TestMethod]
        public void DiskAnnPruningAndFreezingKeepRecall()
        {
            var data = TestData.Clustered(1500, 32, 600);
            var queries = TestData.Clustered(40, 32, 601);
            var index = TestData.Build(data, Options(32, DistanceMetric.L2));
            double before = TestData.Recall(index, data, queries, K, DistanceMetric.L2, false);

            long Edges(HyperspaceIndex i) => Enumerable.Range(0, data.Length).Sum(n => (long)i.GetNeighbors(n).Length);
            long edgesBefore = Edges(index);
            index.OptimizeAsSng(1.2);
            Assert.IsTrue(Edges(index) < edgesBefore, "RobustPrune must sparsify layer 0");
            double after = TestData.Recall(index, data, queries, K, DistanceMetric.L2, false);
            Assert.IsTrue(after >= 0.8, $"recall {before} -> {after} after Vamana pruning");

            index.FreezeFlatLayer0();
            Assert.IsTrue(index.IsLayer0Frozen);
            var frozen = queries.Select(q => index.Search(q, K).Select(r => r.Id).ToArray()).ToArray();
            index.Add(data[0]);
            Assert.IsFalse(index.IsLayer0Frozen, "a write unfreezes layer 0");
            index.FreezeFlatLayer0();
            for (int i = 0; i < queries.Length; i++)
            {
                // Same graph apart from one extra point, so the result sets must overlap almost completely.
                Assert.IsTrue(frozen[i].Intersect(index.Search(queries[i], K).Select(r => r.Id)).Count() >= K - 2);
            }

            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => index.OptimizeAsSng(0.5));
        }

        [TestMethod]
        public void MatryoshkaTruncationSearchesAPrefix()
        {
            var data = TestData.Clustered(800, 64, 700);
            var index = TestData.Build(data, Options(64, DistanceMetric.L2));
            var hits = index.Search(data[5], 5, new SearchOptions { TruncateDimensions = 16 });
            Assert.AreEqual(5, hits.Length);
            double expected = Metrics.Distance(DistanceMetric.L2, (ReadOnlySpan<float>)data[5].AsSpan(0, 16), data[hits[0].Id].AsSpan(0, 16));
            Assert.AreEqual(expected, hits[0].Distance, 1e-3);

            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => index.Search(data[0], 3, new SearchOptions { TruncateDimensions = 0 }));
            var binary = TestData.Build(data.Take(50).ToArray(), Options(64, DistanceMetric.Cosine, QuantizationMode.Binary));
            Assert.ThrowsExactly<NotSupportedException>(() => binary.Search(data[0], 3, new SearchOptions { TruncateDimensions = 16 }));
        }
    }

    [TestClass]
    public class IndexBehaviourTests
    {
        private static HyperspaceIndex Small(out float[][] data, int n = 400, int dim = 16, DistanceMetric metric = DistanceMetric.L2)
        {
            data = TestData.Clustered(n, dim, 42);
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = dim, Metric = metric, Seed = 1 });
            index.AddRange(data);
            return index;
        }

        [TestMethod]
        public void AnItemIsItsOwnNearestNeighbour()
        {
            var index = Small(out var data);
            for (int i = 0; i < data.Length; i += 17)
            {
                var top = index.Search(data[i], 1)[0];
                Assert.AreEqual(i, top.Id);
                Assert.AreEqual(0.0, top.Distance, 1e-6);
            }
        }

        [TestMethod]
        public void ResultsAreSortedAndBoundedByK()
        {
            var index = Small(out var data);
            var hits = index.Search(data[3], 25);
            Assert.AreEqual(25, hits.Length);
            for (int i = 1; i < hits.Length; i++)
            {
                Assert.IsTrue(hits[i - 1].Distance <= hits[i].Distance);
            }

            Assert.AreEqual(data.Length, index.Search(data[3], 5000, new SearchOptions { Ef = 5000 }).Length);
            Assert.AreEqual(0, new HyperspaceIndex(new HyperspaceOptions { Dimension = 4 }).Search(new float[4], 3).Length);
        }

        [TestMethod]
        public void InvalidInputIsRejected()
        {
            var index = Small(out var data);
            Assert.ThrowsExactly<ArgumentException>(() => index.Add(new float[3]));
            Assert.ThrowsExactly<ArgumentException>(() => index.Search(new float[3], 1));
            Assert.ThrowsExactly<ArgumentException>(() => index.Search(Enumerable.Repeat(float.NaN, 16).ToArray(), 1));
            Assert.ThrowsExactly<ArgumentOutOfRangeException>(() => index.Search(data[0], 0));
            Assert.ThrowsExactly<ArgumentException>(() => new HyperspaceIndex(new HyperspaceOptions { Dimension = 0 }));
            Assert.ThrowsExactly<ArgumentException>(() => new HyperspaceIndex(new HyperspaceOptions { Dimension = 4, Metric = DistanceMetric.Hybrid }));
            Assert.ThrowsExactly<ArgumentException>(() => new HyperspaceIndex(new HyperspaceOptions { Dimension = 4, FastRouting = true }));

            var ball = new HyperspaceIndex(new HyperspaceOptions { Dimension = 2, Metric = DistanceMetric.Poincare });
            Assert.ThrowsExactly<ArgumentException>(() => ball.Add(new[] { 0.9f, 0.9f }));
            ball.Add(new[] { 0.1f, 0.1f });
        }

        [TestMethod]
        public void DeleteHidesItemsAndCompactRenumbers()
        {
            var index = Small(out var data);
            int victim = index.Search(data[10], 1)[0].Id;
            Assert.IsTrue(index.Delete(victim));
            Assert.IsFalse(index.Delete(victim));
            Assert.IsFalse(index.Delete(-1));
            Assert.IsFalse(index.Contains(victim));
            Assert.AreEqual(data.Length - 1, index.Count);
            Assert.AreEqual(data.Length, index.TotalCount);
            Assert.AreEqual(1, index.DeletedCount);
            Assert.IsFalse(index.Search(data[10], 20).Any(r => r.Id == victim));
            Assert.ThrowsExactly<KeyNotFoundException>(() => index.GetVector(victim));

            foreach (int id in Enumerable.Range(0, 100).Where(i => i != victim))
            {
                index.Delete(id);
            }

            var compact = index.Compact(out var map);
            Assert.AreEqual(index.Count, compact.Count);
            Assert.AreEqual(compact.Count, compact.TotalCount);
            Assert.AreEqual(-1, map[victim]);
            Assert.AreEqual(-1, map[5]);
            int survivor = Enumerable.Range(100, data.Length - 100).First(i => i != victim);
            var moved = compact.Search(data[survivor], 1)[0];
            Assert.AreEqual(map[survivor], moved.Id);
        }

        [TestMethod]
        public void UpdateMovesAnItem()
        {
            var index = Small(out var data);
            var target = new float[16];
            Array.Fill(target, 50f);
            Assert.AreNotEqual(7, index.Search(target, 1)[0].Id);
            index.Update(7, target, new Dictionary<string, string> { ["tag"] = "moved" });
            Assert.AreEqual(7, index.Search(target, 1)[0].Id);
            Assert.AreEqual("moved", index.GetMetadata(7)!["tag"]);
            Assert.AreEqual(7, index.Search(target, 1, new SearchOptions { ExactMetadata = new Dictionary<string, string> { ["tag"] = "moved" } })[0].Id);
            Assert.ThrowsExactly<KeyNotFoundException>(() => index.Update(9999, target));

            // Still reachable from other queries, i.e. properly re-linked.
            Assert.IsTrue(index.Search(target.Select(x => x - 0.01f).ToArray(), 3).Any(r => r.Id == 7));
        }

        [TestMethod]
        public void SeededBuildsAreReproducible()
        {
            var data = TestData.Clustered(300, 16, 8);
            HyperspaceIndex Make() { var i = new HyperspaceIndex(new HyperspaceOptions { Dimension = 16, Seed = 99 }); i.AddRange(data); return i; }
            var a = Make();
            var b = Make();
            for (int n = 0; n < data.Length; n += 11)
            {
                CollectionAssert.AreEqual(a.GetNeighbors(n), b.GetNeighbors(n));
            }
        }

        [TestMethod]
        public void ConcurrentReadersAndWriters()
        {
            var index = Small(out var data, n: 300);
            var queries = TestData.Clustered(50, 16, 9);
            var errors = new List<Exception>();
            var stop = false;
            var writer = new Thread(() =>
            {
                try
                {
                    for (int i = 0; i < 200 && !stop; i++)
                    {
                        index.Add(data[i % data.Length]);
                    }
                }
                catch (Exception e)
                {
                    lock (errors) { errors.Add(e); }
                }
            });
            writer.Start();
            System.Threading.Tasks.Parallel.ForEach(queries, q =>
            {
                try
                {
                    for (int r = 0; r < 20; r++)
                    {
                        var hits = index.Search(q, 5);
                        Assert.IsTrue(hits.Length > 0);
                    }
                }
                catch (Exception e)
                {
                    lock (errors) { errors.Add(e); }
                }
            });
            writer.Join();
            Assert.AreEqual(0, errors.Count, errors.FirstOrDefault()?.ToString());
            Assert.AreEqual(500, index.Count);
        }

        [TestMethod]
        public void SearchCanBeCancelled()
        {
            var index = Small(out var data, n: 600);
            using var cts = new CancellationTokenSource();
            cts.Cancel();
            Assert.ThrowsExactly<OperationCanceledException>(() => index.Search(data[0], 5, new SearchOptions { CancellationToken = cts.Token }));
        }

        [TestMethod]
        public void BatchSearchMatchesSingleSearches()
        {
            var index = Small(out var data);
            var qs = data.Take(10).ToArray();
            var batch = index.SearchBatch(qs, 5);
            for (int i = 0; i < qs.Length; i++)
            {
                CollectionAssert.AreEqual(index.Search(qs[i], 5).Select(r => r.Id).ToArray(), batch[i].Select(r => r.Id).ToArray());
            }
        }

        [TestMethod]
        public void MomentumSearchLooksAhead()
        {
            // Points on a line: searching ahead of (0, 1) with 4 more steps lands near x = 5.
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 2, Metric = DistanceMetric.L2, Seed = 1 });
            for (int i = 0; i < 20; i++)
            {
                index.Add(new[] { (float)i, 0f });
            }

            var hits = index.SearchMomentum(new[] { 0f, 0f }, new[] { 1f, 0f }, 4.0, 1);
            Assert.AreEqual(5, hits[0].Id);

            var hyper = new HyperspaceIndex(new HyperspaceOptions { Dimension = 2, Metric = DistanceMetric.Poincare, Seed = 1 });
            for (int i = 0; i < 40; i++)
            {
                hyper.Add(new[] { i * 0.02f, 0f });
            }

            var ahead = hyper.SearchMomentum(new[] { 0.1f, 0f }, new[] { 0.2f, 0f }, 1.0, 1);
            Assert.IsTrue(hyper.GetVector(ahead[0].Id)[0] > 0.25f);
        }

        [TestMethod]
        public void FuzzyQueriesCombineVectors()
        {
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 2, Metric = DistanceMetric.L2, Seed = 3 });
            var left = index.Add(new[] { -3f, 0f });
            var right = index.Add(new[] { 3f, 0f });
            var middle = index.Add(new[] { 0f, 0f });
            for (int i = 0; i < 30; i++)
            {
                index.Add(new[] { (i % 6) - 2.5f, 6f + (i / 6) });
            }

            var a = FuzzyQuery.Vector(new[] { -3f, 0f });
            var b = FuzzyQuery.Vector(new[] { 3f, 0f });

            var either = index.SearchFuzzy(FuzzyQuery.Or(a, b, TConorm.Max), 2);
            CollectionAssert.AreEquivalent(new[] { left, right }, either.Select(r => r.Id).ToArray());
            Assert.AreEqual(1.0, either[0].Score!.Value, 1e-5);

            var both = index.SearchFuzzy(FuzzyQuery.And(a, b, TNorm.Product), 1);
            Assert.AreEqual(middle, both[0].Id, "the point halfway between the two is the best conjunction");

            var notLeft = index.SearchFuzzy(FuzzyQuery.Not(a), 1);
            Assert.AreNotEqual(left, notLeft[0].Id);
            Assert.IsTrue(index.GetVector(notLeft[0].Id)[1] > 5f, "the farthest point from the left one is returned");
        }

        [TestMethod]
        public void FuzzyOperatorsObeyTheirAlgebra()
        {
            foreach (var n in Enum.GetValues<TNorm>())
            {
                Assert.AreEqual(n, n.Dual().Dual());
                foreach (float x in new[] { 0f, 0.3f, 1f })
                {
                    foreach (float y in new[] { 0f, 0.6f, 1f })
                    {
                        Assert.AreEqual(n.Apply(x, y), n.Apply(y, x), 1e-6);
                        Assert.AreEqual(FuzzyOps.Negate(n.Apply(x, y)), n.Dual().Apply(FuzzyOps.Negate(x), FuzzyOps.Negate(y)), 1e-6, "De Morgan");
                        Assert.AreEqual(x, n.Apply(x, 1f), 1e-6);
                        Assert.AreEqual(0f, n.Apply(x, 0f), 1e-6);
                    }
                }
            }

            Assert.AreEqual(0.85f, TNorm.Lukasiewicz.Apply(0.9f, 0.95f), 1e-6);
            Assert.IsTrue(TNorm.Min.Apply(0.9f, 0.95f) >= TNorm.Product.Apply(0.9f, 0.95f));
        }
    }
}
