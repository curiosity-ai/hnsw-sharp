namespace Hyperspace.Tests
{
    using System;
    using System.IO;
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    [TestClass]
    public class QuantizationTests
    {
        private static HyperspaceOptions Options(QuantizationMode mode, DistanceMetric metric = DistanceMetric.L2, int dim = 64)
            => new HyperspaceOptions { Dimension = dim, Metric = metric, Quantization = mode, NormalizeVectors = false, ValidateVectors = false };

        private static float Error(float[] a, float[] b) => MathF.Sqrt(a.Zip(b, (x, y) => (x - y) * (x - y)).Sum());

        private static float Norm(float[] a) => MathF.Sqrt(a.Sum(x => x * x));

        [TestMethod]
        [DataRow(QuantizationMode.None, 1e-6)]
        [DataRow(QuantizationMode.Float32, 1e-6)]
        [DataRow(QuantizationMode.ScalarInt8, 0.02)]
        [DataRow(QuantizationMode.ScalarInt4, 0.2)]
        [DataRow(QuantizationMode.Turbo, 0.25)]
        [DataRow(QuantizationMode.Ternary, 0.8)]
        [DataRow(QuantizationMode.AsymmetricHybrid, 0.02)]
        public void ReconstructionErrorIsBounded(QuantizationMode mode, double relativeError)
        {
            var data = TestData.Clustered(200, 64, 11);
            var index = new HyperspaceIndex(Options(mode));
            index.AddRange(data);
            double worst = 0;
            for (int i = 0; i < data.Length; i++)
            {
                worst = Math.Max(worst, Error(data[i], index.GetVector(i)) / Norm(data[i]));
            }

            Assert.IsTrue(worst <= relativeError, $"{mode}: worst relative error {worst}");
        }

        [TestMethod]
        public void StorageSizesMatchTheAdvertisedCompression()
        {
            int Bytes(QuantizationMode m, int dim = 768, DistanceMetric metric = DistanceMetric.L2) => new HyperspaceIndex(Options(m, metric, dim)).BytesPerVector;
            Assert.AreEqual(768 * 8, Bytes(QuantizationMode.None));
            Assert.AreEqual(768 * 4, Bytes(QuantizationMode.Float32));
            Assert.AreEqual(768 + 8, Bytes(QuantizationMode.ScalarInt8));
            Assert.AreEqual((768 / 8) + 4, Bytes(QuantizationMode.Binary));
            Assert.AreEqual(4 + (768 / 16 * 4) + (768 / 16 * 8), Bytes(QuantizationMode.ScalarInt4));
            Assert.AreEqual(8 + (768 / 2), Bytes(QuantizationMode.Turbo));
            Assert.AreEqual((2 * 96) + 8, Bytes(QuantizationMode.Ternary));
            Assert.AreEqual(768 / 16, Bytes(QuantizationMode.ProductQuantization));
            Assert.AreEqual(768 / 16, Bytes(QuantizationMode.OPQ));
        }

        [TestMethod]
        public void HybridLayoutKeepsTheLorentzHeadExact()
        {
            var data = TestData.Hybrid(100, 129, 21);
            foreach (var mode in new[] { QuantizationMode.ScalarInt4, QuantizationMode.Turbo, QuantizationMode.Ternary, QuantizationMode.AsymmetricHybrid })
            {
                var index = new HyperspaceIndex(Options(mode, DistanceMetric.Hybrid, 129));
                index.AddRange(data);
                for (int i = 0; i < data.Length; i++)
                {
                    var v = index.GetVector(i);
                    for (int d = 0; d < 33; d++)
                    {
                        Assert.AreEqual(data[i][d], v[d], 1e-6, $"{mode} head dimension {d}");
                    }
                }
            }
        }

        [TestMethod]
        public void BinaryCodesKeepTheDirection()
        {
            var data = TestData.Clustered(100, 128, 5);
            var index = new HyperspaceIndex(Options(QuantizationMode.Binary, DistanceMetric.Cosine, 128));
            index.AddRange(data);
            double meanCos = 0;
            for (int i = 0; i < data.Length; i++)
            {
                var v = index.GetVector(i);
                double dot = v.Zip(data[i], (a, b) => (double)a * b).Sum();
                meanCos += dot / (Norm(v) * Norm(data[i]));
            }

            meanCos /= data.Length;
            Assert.IsTrue(meanCos > 0.7, $"sign bits of a rotated Gaussian-like vector should keep cos ~ sqrt(2/pi) = 0.8, got {meanCos}");
        }

        [TestMethod]
        public void AnisotropicRefinementNeverIncreasesTheLoss()
        {
            var data = TestData.Poincare(200, 32, 3);
            double Loss(bool refine)
            {
                var o = Options(QuantizationMode.ScalarInt8, DistanceMetric.Poincare, 32);
                o.AnisotropicRefinement = refine;
                var index = new HyperspaceIndex(o);
                index.AddRange(data);
                double total = 0;
                for (int i = 0; i < data.Length; i++)
                {
                    var v = index.GetVector(i);
                    double nx = data[i].Sum(x => (double)x * x);
                    var e = v.Zip(data[i], (a, b) => (double)a - b).ToArray();
                    double dotErr = e.Zip(data[i], (a, b) => a * b).Sum() / Math.Sqrt(nx);
                    double sq = e.Sum(x => x * x);
                    total += (dotErr * dotErr) + (10 * Math.Max(sq - (dotErr * dotErr), 0));
                }

                return total;
            }

            Assert.IsTrue(Loss(true) <= Loss(false) * 1.0001);
        }

        [TestMethod]
        public void ProductQuantizerTrainsAndLookupMatchesDecode()
        {
            var data = TestData.Clustered(600, 32, 9);
            var pq = new ProductQuantizer(32, 4, isOpq: false);
            var untrained = pq.Encode(data[0]);
            double untrainedErr = Error(data[0], pq.Decode(untrained));
            pq.Train(data, 8);

            double trainedErr = data.Take(100).Average(v => Error(v, pq.Decode(pq.Encode(v))));
            Assert.IsTrue(trainedErr < untrainedErr, $"trained {trainedErr} vs untrained {untrainedErr}");

            // The ADC lookup table is the exact squared distance to the decoded vector.
            var q = data[7];
            var lut = pq.ComputeLookupTable(q);
            foreach (var v in data.Take(20))
            {
                var codes = pq.Encode(v);
                var decoded = pq.Decode(codes);
                double expected = q.Zip(decoded, (a, b) => (double)(a - b) * (a - b)).Sum();
                Assert.AreEqual(expected, lut.Distance(codes), 1e-3 * Math.Max(1, expected));
            }
        }

        [TestMethod]
        public void OpqLookupMatchesDecodeToo()
        {
            var data = TestData.Clustered(300, 64, 2);
            var pq = new ProductQuantizer(64, 8, isOpq: true);
            pq.Train(data, 4);
            var q = data[3];
            var lut = pq.ComputeLookupTable(q);
            foreach (var v in data.Take(10))
            {
                var codes = pq.Encode(v);
                var decoded = pq.Decode(codes);
                double expected = q.Zip(decoded, (a, b) => (double)(a - b) * (a - b)).Sum();
                Assert.AreEqual(expected, lut.Distance(codes), 1e-2 * Math.Max(1, expected));
            }
        }

        [TestMethod]
        public void CodebookFileRoundTripsAndKeepsTheHyperspaceLayout()
        {
            var pq = new ProductQuantizer(32, 4, isOpq: true);
            pq.Train(TestData.Clustered(300, 32, 1), 2);
            using var ms = new MemoryStream();
            pq.Save(ms);
            var bytes = ms.ToArray();

            CollectionAssert.AreEqual(new byte[] { (byte)'P', (byte)'Q', (byte)'C', (byte)'B' }, bytes.Take(4).ToArray());
            Assert.AreEqual(32u, BitConverter.ToUInt32(bytes, 4));
            Assert.AreEqual(4u, BitConverter.ToUInt32(bytes, 8));
            Assert.AreEqual(1u, BitConverter.ToUInt32(bytes, 12));
            Assert.AreEqual(4u * 256 * 8, BitConverter.ToUInt32(bytes, 16));
            Assert.AreEqual(20 + (4 * 256 * 8 * 4), bytes.Length);

            var copy = ProductQuantizer.Load(new MemoryStream(bytes));
            var v = TestData.Clustered(1, 32, 8)[0];
            CollectionAssert.AreEqual(pq.Encode(v), copy.Encode(v));

            bytes[0] = (byte)'X';
            Assert.ThrowsExactly<InvalidDataException>(() => ProductQuantizer.Load(new MemoryStream(bytes)));
        }

        [TestMethod]
        public void DefaultSubvectorCount()
        {
            Assert.AreEqual(48, ProductQuantizer.DefaultSubvectorCount(768));
            Assert.AreEqual(1, ProductQuantizer.DefaultSubvectorCount(16));
            Assert.AreEqual(3, ProductQuantizer.DefaultSubvectorCount(24));
            Assert.AreEqual(7, ProductQuantizer.DefaultSubvectorCount(28));
            Assert.AreEqual(13, ProductQuantizer.DefaultSubvectorCount(13));
        }
    }
}
