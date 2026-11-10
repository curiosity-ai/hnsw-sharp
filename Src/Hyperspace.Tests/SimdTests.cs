namespace Hyperspace.Tests
{
    using System;
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    /// <summary>The vectorized / table-driven distance kernels against a plain "decode, then apply the metric" reference.</summary>
    [TestClass]
    public class SimdTests
    {
        [TestMethod]
        public void Int8KernelsMatchTheScalarFormulas()
        {
            var r = new Random(1);
            for (int n = 0; n <= 300; n++)
            {
                var codes = Enumerable.Range(0, n).Select(_ => (sbyte)r.Next(-127, 128)).ToArray();
                var q = Enumerable.Range(0, n).Select(_ => TestData.Gaussian(r)).ToArray();
                float step = 0.013f;

                double dot = 0, l2 = 0;
                for (int i = 0; i < n; i++)
                {
                    dot += codes[i] * (double)q[i];
                    double d = (codes[i] * (double)step) - q[i];
                    l2 += d * d;
                }

                Assert.AreEqual(dot, QuantSimd.DotInt8(codes, q), 1e-3 * Math.Max(1, Math.Abs(dot)), $"dot, n={n}");
                Assert.AreEqual(l2, QuantSimd.L2Int8(codes, q, step), 1e-3 * Math.Max(1, l2), $"l2, n={n}");

                var decoded = new float[n];
                QuantSimd.DecodeInt8(codes, step, decoded);
                for (int i = 0; i < n; i++)
                {
                    Assert.AreEqual(codes[i] * step, decoded[i], 1e-7, $"decode, n={n}, i={i}");
                }
            }
        }

        [TestMethod]
        public void LookupTablesAreTheSumsTheyClaimToBe()
        {
            var r = new Random(2);
            var q = Enumerable.Range(0, 21).Select(_ => TestData.Gaussian(r)).ToArray();

            // Sign table: 8 dimensions per byte, padded with zeros.
            var sign = QuantSimd.BuildSignTable(q, 3);
            for (int b = 0; b < 256; b += 7)
            {
                double expected = 0;
                for (int j = 0; j < 8; j++) expected += ((b >> j) & 1) == 1 ? q[8 + j] : -q[8 + j];
                Assert.AreEqual(expected, sign[256 + b], 1e-5);
            }

            // Nibble-pair table.
            var levels = Enumerable.Range(0, 16).Select(i => (float)(i - 8)).ToArray();
            var pair = QuantSimd.BuildNibblePairTable(q, levels, 11);
            Assert.AreEqual(((5 - 8) * q[6]) + ((12 - 8) * q[7]), pair[(3 * 256) + (5 << 4) + 12], 1e-5);
            Assert.AreEqual((2 - 8) * q[20], pair[(10 * 256) + (2 << 4) + 9], 1e-5, "the dimension past the end counts as zero");

            // Ternary table: only reachable (sign subset of nz) entries are defined.
            var tern = QuantSimd.BuildTernaryTable(q, 6);
            Assert.AreEqual(q[4] - q[6], tern[256 + (0b0101 << 4) + 0b0001], 1e-5);
            Assert.AreEqual(-q[4] - q[6] + q[7], tern[256 + (0b1101 << 4) + 0b1000], 1e-5);
            Assert.AreEqual(0.0, tern[256], 1e-9);

            var codes = new byte[] { 0x12, 0xA0, 0x07 };
            double sum = 0;
            for (int p = 0; p < 3; p++) sum += pair[(p * 256) + codes[p]];
            Assert.AreEqual(sum, QuantSimd.SumTable(codes, pair), 1e-5);
        }

        private static HyperspaceOptions Options(QuantizationMode mode, DistanceMetric metric, int dim, int head = 0)
            => new HyperspaceOptions { Dimension = dim, Metric = metric, Quantization = mode, HeadDimensions = head, NormalizeVectors = false, ValidateVectors = false };

        private static float[] Unit(Random r, int dim, bool ball = false)
        {
            var v = Enumerable.Range(0, dim).Select(_ => TestData.Gaussian(r)).ToArray();
            Metrics.Normalize(v);
            if (ball)
            {
                for (int i = 0; i < v.Length; i++) v[i] *= 0.7f;
            }

            return v;
        }

        [TestMethod]
        [DataRow(QuantizationMode.ScalarInt8, DistanceMetric.L2)]
        [DataRow(QuantizationMode.ScalarInt8, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.ScalarInt8, DistanceMetric.Dot)]
        [DataRow(QuantizationMode.ScalarInt4, DistanceMetric.L2)]
        [DataRow(QuantizationMode.ScalarInt4, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.ScalarInt4, DistanceMetric.Dot)]
        [DataRow(QuantizationMode.Turbo, DistanceMetric.L2)]
        [DataRow(QuantizationMode.Turbo, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.Turbo, DistanceMetric.Dot)]
        [DataRow(QuantizationMode.Ternary, DistanceMetric.L2)]
        [DataRow(QuantizationMode.Ternary, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.Ternary, DistanceMetric.Dot)]
        [DataRow(QuantizationMode.Binary, DistanceMetric.Cosine)]
        public void FastDistancesMatchDecodeThenMetric(QuantizationMode mode, DistanceMetric metric)
        {
            var r = new Random(((int)mode * 31) + (int)metric);
            // Odd sizes, sizes that are not multiples of 8 or 16, and a float head in front of the quantized tail.
            foreach (var (dim, head) in new[] { (3, 0), (8, 0), (16, 0), (37, 0), (64, 0), (100, 0), (129, 0), (129, 33), (65, 33), (200, 33) })
            {
                var codec = VectorCodec.Create(Options(mode, metric, dim, head));
                for (int trial = 0; trial < 12; trial++)
                {
                    var v = Unit(r, dim);
                    var q = Unit(r, dim);
                    var stored = codec.Encode(v);
                    var prepared = codec.Prepare(q);
                    double fast = codec.Distance(stored, prepared);

                    var dec = new float[dim];
                    codec.Decode(stored, dec);
                    double expected = Reference(mode, metric, dec, q);
                    Assert.AreEqual(expected, fast, 2e-3 * Math.Max(1.0, Math.Abs(expected)), $"{mode}/{metric} dim {dim} head {head} trial {trial}");
                }
            }
        }

        [TestMethod]
        [DataRow(QuantizationMode.ScalarInt8, DistanceMetric.L2)]
        [DataRow(QuantizationMode.ScalarInt4, DistanceMetric.L2)]
        [DataRow(QuantizationMode.ScalarInt4, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.ScalarInt4, DistanceMetric.Dot)]
        [DataRow(QuantizationMode.Turbo, DistanceMetric.L2)]
        [DataRow(QuantizationMode.Turbo, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.Turbo, DistanceMetric.Dot)]
        [DataRow(QuantizationMode.Ternary, DistanceMetric.L2)]
        [DataRow(QuantizationMode.Ternary, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.Ternary, DistanceMetric.Dot)]
        [DataRow(QuantizationMode.Binary, DistanceMetric.Cosine)]
        [DataRow(QuantizationMode.ProductQuantization, DistanceMetric.L2)]
        [DataRow(QuantizationMode.OPQ, DistanceMetric.L2)]
        [DataRow(QuantizationMode.ScalarInt4, DistanceMetric.Poincare)]
        [DataRow(QuantizationMode.Ternary, DistanceMetric.Angular)]
        public void PairDistancesMatchDecodeThenMetric(QuantizationMode mode, DistanceMetric metric)
        {
            var r = new Random(((int)mode * 17) + (int)metric);
            foreach (var (dim, head) in new[] { (3, 0), (8, 0), (16, 0), (37, 0), (64, 0), (100, 0), (129, 0), (129, 33), (200, 33), (257, 0) })
            {
                var o = Options(mode, metric, dim, head);
                var codec = VectorCodec.Create(o);
                if (codec is ProductQuantizationCodec pqCodec)
                {
                    pqCodec.Quantizer.Train(Enumerable.Range(0, 400).Select(_ => Unit(r, dim)).ToList(), 2);
                }

                var x = new float[dim];
                var y = new float[dim];
                for (int trial = 0; trial < 15; trial++)
                {
                    var a = Unit(r, dim, ball: metric == DistanceMetric.Poincare);
                    var b = Unit(r, dim, ball: metric == DistanceMetric.Poincare);
                    if (trial == 0)
                    {
                        b = (float[])a.Clone();         // identical codes
                    }
                    else if (trial == 1)
                    {
                        b = new float[dim];             // an all-zero vector: zero scales and norms
                    }

                    var sa = codec.Encode(a);
                    var sb = codec.Encode(b);
                    codec.Decode(sa, x);
                    codec.Decode(sb, y);
                    double expected = Metrics.Distance(codec.PairMetric, (ReadOnlySpan<float>)x, y);
                    double actual = codec.PairDistance(sa, sb);
                    Assert.AreEqual(expected, actual, 3e-3 * Math.Max(1.0, Math.Abs(expected)), $"{mode}/{metric} dim {dim} head {head} trial {trial}");
                    Assert.AreEqual(actual, codec.PairDistance(sb, sa), 1e-5 * Math.Max(1.0, Math.Abs(actual)), "symmetric");
                }
            }
        }

        [TestMethod]
        public void SymmetricProductTableIsRebuiltAfterTraining()
        {
            var r = new Random(5);
            var pq = new ProductQuantizer(32, 4, isOpq: false);
            var codec = new ProductQuantizationCodec(32, DistanceMetric.L2, pq);
            var a = codec.Encode(Unit(r, 32));
            var b = codec.Encode(Unit(r, 32));
            double before = codec.PairDistance(a, b);
            Assert.IsNotNull(pq.SymmetricTable);

            pq.Train(Enumerable.Range(0, 600).Select(_ => Unit(r, 32)).ToList(), 3);
            var a2 = codec.Encode(Unit(r, 32));
            var b2 = codec.Encode(Unit(r, 32));
            var x = new float[32];
            var y = new float[32];
            codec.Decode(a2, x);
            codec.Decode(b2, y);
            Assert.AreEqual(Metrics.Distance(DistanceMetric.L2, (ReadOnlySpan<float>)x, y), codec.PairDistance(a2, b2), 1e-4, "a stale table would answer for the old codebooks");
            Assert.IsTrue(before >= 0);
        }

        [TestMethod]
        public void FreezingReleasesTheSymmetricTableAndInsertingRebuildsIt()
        {
            var data = TestData.Clustered(1500, 32, 3);
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = 32, Metric = DistanceMetric.L2, Quantization = QuantizationMode.ProductQuantization, Seed = 1 });
            index.TrainProductQuantizer(data, 2);
            index.AddRange(data.Take(1000).ToArray());
            Assert.IsTrue(index.Quantizer!.HasSymmetricTable, "inserting builds it");

            long before = GC.GetTotalMemory(true);
            index.FreezeFlatLayer0();
            Assert.IsFalse(index.Quantizer.HasSymmetricTable, "freezing ends the build");
            Assert.IsTrue(GC.GetTotalMemory(true) < before - (index.Quantizer.NumSubvectors * 65536 * 4 / 2), "its memory is actually returned");

            index.AddRange(data.Skip(1000).ToArray());                // thaws, rebuilds the table, links
            Assert.IsTrue(index.Quantizer.HasSymmetricTable);
            Assert.AreEqual(1500, index.Count);
            Assert.AreEqual(1200, index.Search(data[1200], 1, new SearchOptions { Ef = 200 })[0].Id);
        }

        private static double Reference(QuantizationMode mode, DistanceMetric metric, float[] decoded, float[] q)
        {
            double dot = 0, nd = 0, nq = 0, l2 = 0;
            for (int i = 0; i < q.Length; i++)
            {
                dot += (double)decoded[i] * q[i];
                nd += (double)decoded[i] * decoded[i];
                nq += (double)q[i] * q[i];
                double d = decoded[i] - (double)q[i];
                l2 += d * d;
            }

            if (metric == DistanceMetric.Dot)
            {
                return 1.0 - dot;
            }

            switch (mode)
            {
                case QuantizationMode.Binary:
                case QuantizationMode.Ternary when metric == DistanceMetric.Cosine:
                    // Angular approximations: 2 (1 - cos) between the reconstruction and the query.
                    return Math.Max(2.0 * (1.0 - Math.Clamp(dot / Math.Sqrt(nd * nq), -1.0, 1.0)), 0.0);
                case QuantizationMode.Turbo when metric == DistanceMetric.Cosine:
                    return Math.Max(2.0 * (1.0 - dot), 0.0);
                default:
                    return l2;
            }
        }
    }
}
