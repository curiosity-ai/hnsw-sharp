namespace Hyperspace.Tests
{
    using System;
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    [TestClass]
    public class MetricsTests
    {
        [TestMethod]
        public void KnownValues()
        {
            double[] a = { 1, 0, 0, 0 }, b = { 0, 1, 0, 0 };
            Assert.AreEqual(2.0, Metrics.Distance(DistanceMetric.L2, a, b), 1e-9);
            Assert.AreEqual(2.0, Metrics.Distance(DistanceMetric.Cosine, a, b), 1e-9);
            Assert.AreEqual(1.0, Metrics.Distance(DistanceMetric.Dot, a, b), 1e-9);
            Assert.AreEqual(0.0, Metrics.Distance(DistanceMetric.Dot, a, a), 1e-9);
            Assert.AreEqual(2.0, Metrics.Distance(DistanceMetric.L1, a, b), 1e-9);
            Assert.AreEqual(1.0, Metrics.Distance(DistanceMetric.Linf, a, b), 1e-9);
            Assert.AreEqual(0.5, Metrics.Distance(DistanceMetric.Angular, a, b), 1e-9);
            Assert.AreEqual(1.0, Metrics.Distance(DistanceMetric.Jaccard, a, b), 1e-9);
            Assert.AreEqual(0.0, Metrics.Distance(DistanceMetric.Jaccard, a, a), 1e-9);
            Assert.AreEqual(0.5, Metrics.Distance(DistanceMetric.Hamming, a, b), 1e-9);
        }

        [TestMethod]
        public void FloatAndDoubleAgree()
        {
            var r = new Random(1);
            foreach (var m in new[] { DistanceMetric.L2, DistanceMetric.Dot, DistanceMetric.L1, DistanceMetric.Linf, DistanceMetric.Angular, DistanceMetric.Jaccard, DistanceMetric.Hamming })
            {
                var a = Enumerable.Range(0, 67).Select(_ => TestData.Gaussian(r)).ToArray();
                var b = Enumerable.Range(0, 67).Select(_ => TestData.Gaussian(r)).ToArray();
                double f = Metrics.Distance(m, (ReadOnlySpan<float>)a, b);
                double d = Metrics.Distance(m, a.Select(x => (double)x).ToArray(), b.Select(x => (double)x).ToArray());
                Assert.AreEqual(d, f, 1e-3 * Math.Max(1, Math.Abs(d)), m.ToString());
            }
        }

        [TestMethod]
        public void PoincareMatchesClosedForm()
        {
            // d(0, x) = 2 atanh(|x|) at the origin.
            double[] zero = { 0, 0 }, x = { 0.6, 0 };
            Assert.AreEqual(2 * Math.Atanh(0.6), Metrics.Distance(DistanceMetric.Poincare, zero, x), 1e-9);
            Assert.AreEqual(0.0, Metrics.Distance(DistanceMetric.Poincare, x, x), 1e-6);
        }

        [TestMethod]
        public void LorentzMatchesPoincareThroughTheIsometry()
        {
            var r = new Random(3);
            for (int i = 0; i < 20; i++)
            {
                var p = Enumerable.Range(0, 5).Select(_ => (double)(TestData.Gaussian(r) * 0.2f)).ToArray();
                var q = Enumerable.Range(0, 5).Select(_ => (double)(TestData.Gaussian(r) * 0.2f)).ToArray();
                double dp = Metrics.Distance(DistanceMetric.Poincare, p, q);
                double dl = Metrics.Distance(DistanceMetric.Lorentz, Riemannian.PoincareToLorentz(p), Riemannian.PoincareToLorentz(q));
                Assert.AreEqual(dp, dl, 1e-6);
            }
        }

        [TestMethod]
        public void HybridIsLorentzPlusL2()
        {
            var data = TestData.Hybrid(2, 40, 5);
            double expected = Metrics.Distance(DistanceMetric.Lorentz, data[0].AsSpan(0, 33), data[1].AsSpan(0, 33))
                            + Metrics.Distance(DistanceMetric.L2, data[0].AsSpan(33), data[1].AsSpan(33));
            Assert.AreEqual(expected, Metrics.Distance(DistanceMetric.Hybrid, (ReadOnlySpan<float>)data[0], data[1]), 1e-4);
        }

        [TestMethod]
        public void WassersteinIsTheAreaBetweenCdfs()
        {
            double[] a = { 1, 0, 0 }, b = { 0, 0, 1 };
            Assert.AreEqual(2.0, Metrics.Distance(DistanceMetric.Wasserstein, a, b), 1e-9);
            Assert.AreEqual(0.0, Metrics.Distance(DistanceMetric.Wasserstein, a, a), 1e-9);
            Assert.AreEqual(double.MaxValue, Metrics.Distance(DistanceMetric.Wasserstein, new double[] { 0, 0, 0 }, b));
        }

        [TestMethod]
        public void NamesAndAliases()
        {
            Assert.AreEqual(12, Metrics.SupportedMetrics.Count);
            foreach (var name in Metrics.SupportedMetrics)
            {
                Assert.IsTrue(Metrics.TryParse(name, out var m), name);
                Assert.AreEqual(name, Metrics.GetName(m));
            }

            Assert.AreEqual(DistanceMetric.Dot, Metrics.Parse(" IP "));
            Assert.AreEqual(DistanceMetric.Linf, Metrics.Parse("Chebyshev"));
            Assert.AreEqual(DistanceMetric.Wasserstein, Metrics.Parse("emd"));
            Assert.IsFalse(Metrics.TryParse("nope", out _));
            Assert.AreEqual(2.0, Metrics.Distance("euclidean", new double[] { 1, 0 }, new double[] { 0, 1 }), 1e-9);
        }

        [TestMethod]
        public void Validation()
        {
            Assert.IsNull(Metrics.Validate<float>(DistanceMetric.Poincare, new[] { 0.1f, 0.2f }));
            Assert.IsNotNull(Metrics.Validate<float>(DistanceMetric.Poincare, new[] { 0.9f, 0.9f }));
            Assert.IsNotNull(Metrics.Validate<float>(DistanceMetric.L2, new[] { float.NaN, 1f }));
            Assert.IsNull(Metrics.Validate<double>(DistanceMetric.Lorentz, Riemannian.PoincareToLorentz(new[] { 0.3, 0.1 })));
            Assert.IsNotNull(Metrics.Validate<double>(DistanceMetric.Lorentz, new[] { 1.0, 1.0, 1.0 }));
            Assert.IsNotNull(Metrics.Validate<double>(DistanceMetric.Lorentz, new[] { -1.0, 0.0, 0.0 }));
        }
    }

    [TestClass]
    public class RotationAndMathTests
    {
        [TestMethod]
        public void RotationIsOrthogonalAndInvertible()
        {
            foreach (int dim in new[] { 2, 4, 8, 33, 64, 100, 128, 256, 768, 1024, 1536 })
            {
                var x = Enumerable.Range(0, dim).Select(i => MathF.Sin(i * 0.13f)).ToArray();
                var y = Enumerable.Range(0, dim).Select(i => MathF.Cos(i * 0.29f)).ToArray();
                var rx = TurboRotation.Rotated(x);
                var ry = TurboRotation.Rotated(y);

                double n = x.Sum(v => (double)v * v);
                Assert.AreEqual(n, rx.Sum(v => (double)v * v), 1e-4 * n, $"norm, dim {dim}");
                double dot = x.Zip(y, (a, b) => (double)a * b).Sum();
                double rdot = rx.Zip(ry, (a, b) => (double)a * b).Sum();
                Assert.AreEqual(dot, rdot, 1e-3, $"dot, dim {dim}");

                var back = TurboRotation.Unrotated(rx);
                for (int i = 0; i < dim; i++)
                {
                    Assert.AreEqual(x[i], back[i], 1e-4, $"inverse, dim {dim}, i {i}");
                }
            }
        }

        [TestMethod]
        public void BlocksFollowTheBinaryExpansion()
        {
            CollectionAssert.AreEqual(new[] { (0, 512), (512, 256) }, TurboRotation.DecomposeBlocks(768).ToArray());
            CollectionAssert.AreEqual(new[] { (0, 1024), (1024, 512) }, TurboRotation.DecomposeBlocks(1536).ToArray());
        }

        [TestMethod]
        public void RotationSpreadsAUnitVectorEvenly()
        {
            // A basis vector is mapped to +-1/sqrt(n) everywhere: the signs are pseudo-random, the magnitudes are not.
            var v = new float[8];
            v[0] = 1f;
            TurboRotation.Rotate(v);
            var signs = v.Select(x => x > 0 ? '+' : '-').ToArray();
            Assert.AreEqual(8, signs.Length);
            foreach (var x in v)
            {
                Assert.AreEqual(1f / MathF.Sqrt(8), MathF.Abs(x), 1e-6);
            }
        }

        [TestMethod]
        public void MobiusExpLogAreConsistent()
        {
            double[] x = { 0.1, -0.2, 0.3 }, y = { -0.3, 0.1, 0.2 };
            var v = Riemannian.LogMap(x, y);
            var back = Riemannian.ExpMap(x, v);
            for (int i = 0; i < 3; i++)
            {
                Assert.AreEqual(y[i], back[i], 1e-9);
            }

            // The tangent vector's length (in the metric at x) is the geodesic distance.
            double lambda = 2.0 / (1.0 - x.Sum(t => t * t));
            Assert.AreEqual(Metrics.Distance(DistanceMetric.Poincare, x, y), lambda * Math.Sqrt(v.Sum(t => t * t)), 1e-9);

            var sum = Riemannian.MobiusAdd(x, Array.ConvertAll(x, t => -t));
            foreach (var t in sum)
            {
                Assert.AreEqual(0.0, t, 1e-12);
            }
        }

        [TestMethod]
        public void ParallelTransportPreservesTheNorm()
        {
            double[] x = { 0.1, -0.2, 0.3 }, y = { -0.3, 0.1, 0.2 }, v = { 0.5, 0.2, -0.1 };
            var t = Riemannian.ParallelTransport(x, y, v);
            double lx = 2.0 / (1.0 - x.Sum(a => a * a)), ly = 2.0 / (1.0 - y.Sum(a => a * a));
            Assert.AreEqual(lx * Math.Sqrt(v.Sum(a => a * a)), ly * Math.Sqrt(t.Sum(a => a * a)), 1e-9);
        }

        [TestMethod]
        public void FrechetMeanOfSymmetricPointsIsTheOrigin()
        {
            var points = new System.Collections.Generic.List<double[]> { new[] { 0.5, 0.0 }, new[] { -0.5, 0.0 }, new[] { 0.0, 0.5 }, new[] { 0.0, -0.5 } };
            var mean = Riemannian.FrechetMean(points, 1.0, 100, 1e-12);
            Assert.AreEqual(0.0, mean[0], 1e-6);
            Assert.AreEqual(0.0, mean[1], 1e-6);

            var same = Riemannian.FrechetMean(new[] { new[] { 0.3, 0.1 }, new[] { 0.3, 0.1 } });
            Assert.AreEqual(0.3, same[0], 1e-9);
        }

        [TestMethod]
        public void PoincareLorentzRoundTrip()
        {
            double[] p = { 0.2, -0.4, 0.1 };
            var back = Riemannian.LorentzToPoincare(Riemannian.PoincareToLorentz(p));
            for (int i = 0; i < 3; i++)
            {
                Assert.AreEqual(p[i], back[i], 1e-12);
            }

            var lifted = Riemannian.LiftToHyperboloid(new[] { 0.5, 0.5 });
            Assert.IsNull(Metrics.Validate<double>(DistanceMetric.Lorentz, lifted));
        }

        [TestMethod]
        public void CognitiveHelpers()
        {
            double[] centre = { 0.0, 0.0 };
            var tight = new System.Collections.Generic.List<double[]> { new[] { 0.01, 0.0 }, new[] { 0.0, 0.01 } };
            var wide = new System.Collections.Generic.List<double[]> { new[] { 0.8, 0.0 }, new[] { 0.0, 0.8 } };
            Assert.IsTrue(Riemannian.LocalEntropy(centre, tight) < Riemannian.LocalEntropy(centre, wide));
            Assert.AreEqual(1.0, Riemannian.LocalEntropy(centre, new System.Collections.Generic.List<double[]>()));

            var converging = new System.Collections.Generic.List<double[]> { new[] { 0.7, 0.0 }, new[] { 0.35, 0.0 }, new[] { 0.1, 0.0 }, new[] { 0.02, 0.0 } };
            Assert.IsTrue(Riemannian.LyapunovConvergence(converging) < 0);

            var next = Riemannian.KoopmanExtrapolate(new[] { 0.0, 0.0 }, new[] { 0.1, 0.0 }, 1.0);
            Assert.IsTrue(next[0] > 0.1);
            Assert.AreEqual(0.0, next[1], 1e-9);

            var pulled = Riemannian.ContextResonance(new[] { 0.5, 0.0 }, new[] { -0.5, 0.0 }, 0.5);
            Assert.AreEqual(0.0, pulled[0], 1e-9);
        }

        [TestMethod]
        public void MomentumExtrapolationIsLinearInEuclideanSpaceAndGeodesicInTheBall()
        {
            var lin = Riemannian.ExtrapolateMomentum(DistanceMetric.L2, new[] { 0.0, 0.0 }, new[] { 1.0, 2.0 }, 2.0);
            Assert.AreEqual(3.0, lin[0], 1e-12);
            Assert.AreEqual(6.0, lin[1], 1e-12);

            var hyp = Riemannian.ExtrapolateMomentum(DistanceMetric.Poincare, new[] { 0.0, 0.0 }, new[] { 0.2, 0.0 }, 1.0);
            Assert.IsTrue(hyp[0] > 0.2 && hyp[0] < 1.0);
            Assert.AreEqual(0.0, hyp[1], 1e-9);

            var lor = Riemannian.ExtrapolateMomentum(DistanceMetric.Lorentz, Riemannian.PoincareToLorentz(new[] { 0.0, 0.0 }), Riemannian.PoincareToLorentz(new[] { 0.2, 0.0 }), 1.0);
            Assert.IsNull(Metrics.Validate<double>(DistanceMetric.Lorentz, lor));
            Assert.ThrowsExactly<NotSupportedException>(() => Riemannian.ExtrapolateMomentum(DistanceMetric.Hamming, new[] { 0.0 }, new[] { 1.0 }, 1));
        }

        [TestMethod]
        public void KeyedProjectionIsIsometricInvertibleAndKeyed()
        {
            var shield = KeyedOrthogonalProjection.FromSecret("docs", "secret");
            var r = new Random(7);
            foreach (int dim in new[] { 3, 16, 100, 768 })
            {
                var a = Enumerable.Range(0, dim).Select(_ => (double)TestData.Gaussian(r)).ToArray();
                var b = Enumerable.Range(0, dim).Select(_ => (double)TestData.Gaussian(r)).ToArray();
                var pa = shield.Project(a);
                var pb = shield.Project(b);
                Assert.AreEqual(Metrics.Distance(DistanceMetric.L2, a, b), Metrics.Distance(DistanceMetric.L2, pa, pb), 1e-8);
                Assert.AreEqual(Metrics.Dot<double>(a, b), Metrics.Dot<double>(pa, pb), 1e-8);
                var back = shield.Unproject(pa);
                for (int i = 0; i < dim; i++)
                {
                    Assert.AreEqual(a[i], back[i], 1e-9);
                }
            }

            var v = new double[] { 1, 2, 3, 4, 5, 6, 7, 8 };
            CollectionAssert.AreNotEqual(shield.Project(v), KeyedOrthogonalProjection.FromSecret("docs", "other").Project(v));
            CollectionAssert.AreEqual(shield.Project(v), KeyedOrthogonalProjection.FromSecret("docs", "secret").Project(v));
        }

        [TestMethod]
        public void KeyedProjectionKeepsTheTimeAxisOfHyperboloidPoints()
        {
            var shield = KeyedOrthogonalProjection.FromSecret("c", "k");
            var x = Riemannian.PoincareToLorentz(new[] { 0.2, 0.3, -0.1, 0.05 });
            var p = shield.Project(x, DistanceMetric.Lorentz);
            Assert.AreEqual(x[0], p[0], 1e-12);
            Assert.IsNull(Metrics.Validate<double>(DistanceMetric.Lorentz, p));
        }

        [TestMethod]
        public void ReconsolidationMovesVectorsTowardsTheAttractorInsideTheBall()
        {
            var recon = new MemoryReconsolidator(0.1);
            double[] target = { 0.5, 0.5 };
            var v = new[] { 0.1, -0.1 };
            double before = Metrics.Distance(DistanceMetric.Poincare, v, target);
            var moved = recon.StepTowardsPoincare(v, target);
            Assert.IsTrue(Metrics.Distance(DistanceMetric.Poincare, moved, target) < before);
            Assert.AreEqual(0.1, v[0], 0, "the source is not modified by the copying overload");

            var outside = new[] { 0.99, 0.0 };
            recon.StepTowardsPoincareInPlace(outside, new[] { 5.0, 5.0 });
            Assert.IsTrue(outside.Sum(x => x * x) < 1.0);
        }

        [TestMethod]
        public void GromovRecommendsHyperbolicGeometryForTreeLikeData()
        {
            // A star (tree) metric: points at distance 1 from the root along orthogonal axes.
            var tree = Enumerable.Range(0, 30).Select(i => { var v = new double[30]; v[i] = 1.0; return v; }).ToList();
            var result = GromovHyperbolicity.Analyze(tree, 2000, new Random(1));
            Assert.IsTrue(result.Delta < 0.5, result.Delta.ToString());

            Assert.AreEqual(DistanceMetric.L2, GromovHyperbolicity.Analyze(tree.Take(3).ToList()).RecommendedMetric);
        }
    }
}
