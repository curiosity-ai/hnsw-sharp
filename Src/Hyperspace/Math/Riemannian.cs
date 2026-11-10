// <copyright file="Riemannian.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Riemannian geometry toolkit for the Poincare ball and the Lorentz hyperboloid: Mobius arithmetic, exponential and
    /// logarithmic maps, parallel transport, Frechet means, and the "cognitive" helpers built on them (spatial entropy,
    /// Lyapunov convergence, Koopman extrapolation, context resonance).
    /// </summary>
    public static class Riemannian
    {
        private static double Dot(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
        {
            double sum = 0;
            for (int i = 0; i < a.Length; i++)
            {
                sum += a[i] * b[i];
            }

            return sum;
        }

        private static double NormSq(ReadOnlySpan<double> v) => Dot(v, v);

        private static void Require(bool ok, string message)
        {
            if (!ok)
            {
                throw new ArgumentException(message);
            }
        }

        /// <summary>Radially rescales a point so it is strictly inside the ball of curvature <paramref name="c"/>.</summary>
        public static double[] ProjectToBall(ReadOnlySpan<double> x, double c)
        {
            double n = Math.Sqrt(NormSq(x));
            double maxN = (1.0 / Math.Sqrt(c)) - 1e-9;
            var result = x.ToArray();
            if (n <= maxN || n <= 1e-15)
            {
                return result;
            }

            double s = maxN / n;
            for (int i = 0; i < result.Length; i++)
            {
                result[i] *= s;
            }

            return result;
        }

        /// <summary>Mobius addition <c>x (+) y</c> in the Poincare ball.</summary>
        public static double[] MobiusAdd(ReadOnlySpan<double> x, ReadOnlySpan<double> y, double c = 1.0)
        {
            Require(x.Length == y.Length, "Dimension mismatch");
            Require(c > 0, "Curvature c must be > 0");
            double xy = Dot(x, y), x2 = NormSq(x), y2 = NormSq(y);
            double numLeft = 1.0 + (2.0 * c * xy) + (c * y2);
            double numRight = 1.0 - (c * x2);
            double den = 1.0 + (2.0 * c * xy) + (c * c * x2 * y2);
            if (Math.Abs(den) < 1e-15)
            {
                throw new InvalidOperationException("Mobius addition denominator too small");
            }

            var result = new double[x.Length];
            double inv = 1.0 / den;
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = ((numLeft * x[i]) + (numRight * y[i])) * inv;
            }

            return result;
        }

        /// <summary>Maps the tangent vector <paramref name="v"/> at <paramref name="x"/> onto the manifold.</summary>
        public static double[] ExpMap(ReadOnlySpan<double> x, ReadOnlySpan<double> v, double c = 1.0)
        {
            Require(x.Length == v.Length, "Dimension mismatch");
            Require(c > 0, "Curvature c must be > 0");
            double x2 = NormSq(x);
            double vNormSq = NormSq(v);
            double vNorm = Math.Sqrt(vNormSq);
            if (vNorm < 1e-15)
            {
                return x.ToArray();
            }

            double lambdaX = 2.0 / Math.Max(1.0 - (c * x2), 1e-15);
            double scale = Math.Tanh(Math.Sqrt(c) * lambdaX * vNorm / 2.0) / (Math.Sqrt(c) * vNorm);
            double xy = scale * Dot(x, v);
            double y2 = scale * scale * vNormSq;
            double numLeft = 1.0 + (2.0 * c * xy) + (c * y2);
            double numRight = 1.0 - (c * x2);
            double den = 1.0 + (2.0 * c * xy) + (c * c * x2 * y2);
            if (Math.Abs(den) < 1e-15)
            {
                throw new InvalidOperationException("Mobius addition denominator too small");
            }

            var result = new double[x.Length];
            double inv = 1.0 / den;
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = ((numLeft * x[i]) + (numRight * scale * v[i])) * inv;
            }

            return result;
        }

        /// <summary>Maps the manifold point <paramref name="y"/> back to the tangent space at <paramref name="x"/>.</summary>
        public static double[] LogMap(ReadOnlySpan<double> x, ReadOnlySpan<double> y, double c = 1.0)
        {
            Require(x.Length == y.Length, "Dimension mismatch");
            Require(c > 0, "Curvature c must be > 0");
            var negX = new double[x.Length];
            for (int i = 0; i < negX.Length; i++)
            {
                negX[i] = -x[i];
            }

            var delta = MobiusAdd(negX, y, c);
            double deltaNorm = Math.Sqrt(NormSq(delta));
            if (deltaNorm < 1e-15)
            {
                return new double[x.Length];
            }

            double lambdaX = 2.0 / Math.Max(1.0 - (c * NormSq(x)), 1e-15);
            double arg = Math.Min(Math.Sqrt(c) * deltaNorm, 1.0 - 1e-15);
            double factor = 2.0 / (lambdaX * Math.Sqrt(c)) * Math.Atanh(arg);
            for (int i = 0; i < delta.Length; i++)
            {
                delta[i] = factor * delta[i] / deltaNorm;
            }

            return delta;
        }

        /// <summary>Converts a Euclidean gradient at <paramref name="x"/> into the Riemannian gradient (scaled by the inverse metric).</summary>
        public static double[] RiemannianGradient(ReadOnlySpan<double> x, ReadOnlySpan<double> euclideanGradient, double c = 1.0)
        {
            Require(x.Length == euclideanGradient.Length, "Dimension mismatch");
            Require(c > 0, "Curvature c must be > 0");
            double lambdaX = 2.0 / Math.Max(1.0 - (c * NormSq(x)), 1e-15);
            double scale = 1.0 / (lambdaX * lambdaX);
            var result = new double[x.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = scale * euclideanGradient[i];
            }

            return result;
        }

        private static double[] Gyration(ReadOnlySpan<double> u, ReadOnlySpan<double> v, ReadOnlySpan<double> w, double c)
        {
            var uv = MobiusAdd(u, v, c);
            var vw = MobiusAdd(v, w, c);
            var left = MobiusAdd(u, vw, c);
            for (int i = 0; i < uv.Length; i++)
            {
                uv[i] = -uv[i];
            }

            return MobiusAdd(uv, left, c);
        }

        /// <summary>Parallel-transports the tangent vector <paramref name="v"/> from <paramref name="x"/> to <paramref name="y"/>.</summary>
        public static double[] ParallelTransport(ReadOnlySpan<double> x, ReadOnlySpan<double> y, ReadOnlySpan<double> v, double c = 1.0)
        {
            Require(x.Length == y.Length && x.Length == v.Length, "Dimension mismatch");
            Require(c > 0, "Curvature c must be > 0");
            var negX = new double[x.Length];
            for (int i = 0; i < negX.Length; i++)
            {
                negX[i] = -x[i];
            }

            var gyr = Gyration(y, negX, v, c);
            double lambdaX = 2.0 / Math.Max(1.0 - (c * NormSq(x)), 1e-15);
            double lambdaY = 2.0 / Math.Max(1.0 - (c * NormSq(y)), 1e-15);
            double scale = lambdaX / lambdaY;
            for (int i = 0; i < gyr.Length; i++)
            {
                gyr[i] *= scale;
            }

            return gyr;
        }

        /// <summary>Frechet (Karcher) mean of points in the Poincare ball, by Riemannian gradient descent.</summary>
        public static double[] FrechetMean(IReadOnlyList<double[]> points, double c = 1.0, int maxIterations = 32, double tolerance = 1e-6)
        {
            Require(points.Count > 0, "Points set cannot be empty");
            Require(c > 0, "Curvature c must be > 0");
            int dim = points[0].Length;
            foreach (var p in points)
            {
                Require(p.Length == dim, "Dimension mismatch");
            }

            double inv = 1.0 / points.Count;
            var mu = ProjectToBall(points[0], c);
            for (int iter = 0; iter < Math.Max(1, maxIterations); iter++)
            {
                var grad = new double[dim];
                foreach (var p in points)
                {
                    var lg = LogMap(mu, p, c);
                    for (int i = 0; i < dim; i++)
                    {
                        grad[i] += lg[i];
                    }
                }

                for (int i = 0; i < dim; i++)
                {
                    grad[i] *= inv;
                }

                if (Math.Sqrt(NormSq(grad)) <= Math.Max(tolerance, 1e-15))
                {
                    break;
                }

                mu = ProjectToBall(ExpMap(mu, grad, c), c);
            }

            return mu;
        }

        // ------------------------------------------------------------------ Lorentz model

        /// <summary>Minkowski inner product <c>-u0 v0 + sum(ui vi)</c>.</summary>
        public static double LorentzProduct(ReadOnlySpan<double> u, ReadOnlySpan<double> v)
        {
            if (u.Length == 0 || v.Length == 0)
            {
                return 0.0;
            }

            double dot = -u[0] * v[0];
            int n = Math.Min(u.Length, v.Length);
            for (int i = 1; i < n; i++)
            {
                dot += u[i] * v[i];
            }

            return dot;
        }

        /// <summary>Geodesic distance between two points on the hyperboloid.</summary>
        public static double LorentzDistance(ReadOnlySpan<double> u, ReadOnlySpan<double> v) => Math.Acosh(Math.Max(-LorentzProduct(u, v), 1.0));

        /// <summary>Parallel transport of a tangent vector from <paramref name="x"/> to <paramref name="y"/> on the hyperboloid.</summary>
        public static double[] LorentzParallelTransport(ReadOnlySpan<double> x, ReadOnlySpan<double> y, ReadOnlySpan<double> v)
        {
            Require(x.Length == y.Length && x.Length == v.Length, "Dimension mismatch");
            double innerXy = LorentzProduct(x, y);
            double innerYv = LorentzProduct(y, v);
            double denom = 1.0 - innerXy;
            double factor = Math.Abs(denom) < 1e-7 ? 0.0 : innerYv / denom;
            var result = new double[v.Length];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = v[i] + (factor * (x[i] + y[i]));
            }

            return result;
        }

        /// <summary>Hyperboloid to Poincare ball: <c>x_i / (1 + x_0)</c>.</summary>
        public static double[] LorentzToPoincare(ReadOnlySpan<double> x)
        {
            if (x.Length == 0)
            {
                return Array.Empty<double>();
            }

            double denom = 1.0 + x[0];
            if (Math.Abs(denom) < 1e-7)
            {
                return new double[x.Length - 1];
            }

            var p = new double[x.Length - 1];
            for (int i = 0; i < p.Length; i++)
            {
                p[i] = x[i + 1] / denom;
            }

            return p;
        }

        /// <summary>Poincare ball to hyperboloid: <c>((1+|p|^2), 2p) / (1-|p|^2)</c>.</summary>
        public static double[] PoincareToLorentz(ReadOnlySpan<double> p)
        {
            double sq = NormSq(p);
            double denom = 1.0 - sq;
            if (Math.Abs(denom) < 1e-7)
            {
                denom = 1e-7;
            }

            var x = new double[p.Length + 1];
            x[0] = (1.0 + sq) / denom;
            for (int i = 0; i < p.Length; i++)
            {
                x[i + 1] = 2.0 * p[i] / denom;
            }

            return x;
        }

        /// <summary>Lifts a spatial vector onto the hyperboloid by solving for the time coordinate <c>t = sqrt(1 + |x|^2)</c>.</summary>
        public static double[] LiftToHyperboloid(ReadOnlySpan<double> spatial)
        {
            var x = new double[spatial.Length + 1];
            x[0] = Math.Sqrt(1.0 + NormSq(spatial));
            spatial.CopyTo(x.AsSpan(1));
            return x;
        }

        // ------------------------------------------------------------------ cognitive math

        /// <summary>
        /// Dispersion of <paramref name="candidate"/> relative to its neighbours, in [0, 1): values near 1 mean high chaos
        /// (a likely hallucination); 1 if there are no neighbours.
        /// </summary>
        public static double LocalEntropy(ReadOnlySpan<double> candidate, IReadOnlyList<double[]> neighbors, double c = 1.0)
        {
            if (neighbors.Count == 0)
            {
                return 1.0;
            }

            double total = 0;
            foreach (var n in neighbors)
            {
                total += Math.Sqrt(NormSq(LogMap(candidate, n, c)));
            }

            return 1.0 - Math.Exp(-(total / neighbors.Count));
        }

        /// <summary>
        /// Average rate of change of the Lyapunov energy of a trajectory around its Frechet-mean attractor: negative when the
        /// trajectory converges, positive when it diverges.
        /// </summary>
        public static double LyapunovConvergence(IReadOnlyList<double[]> trajectory, double c = 1.0)
        {
            Require(trajectory.Count >= 3, "Need at least 3 points to evaluate convergence trend");
            var attractor = FrechetMean(trajectory, c, 32, 1e-6);
            double sum = 0;
            for (int i = 0; i < trajectory.Count - 1; i++)
            {
                double v0 = Math.Sqrt(NormSq(LogMap(attractor, trajectory[i], c)));
                double v1 = Math.Sqrt(NormSq(LogMap(attractor, trajectory[i + 1], c)));
                sum += v1 - v0;
            }

            return sum / (trajectory.Count - 1);
        }

        /// <summary>Extrapolates a trajectory <paramref name="steps"/> steps ahead by transporting its last velocity (Koopman linearization).</summary>
        public static double[] KoopmanExtrapolate(ReadOnlySpan<double> past, ReadOnlySpan<double> current, double steps, double c = 1.0)
        {
            var velocityAtPast = LogMap(past, current, c);
            var velocityAtCurrent = ParallelTransport(past, current, velocityAtPast, c);
            for (int i = 0; i < velocityAtCurrent.Length; i++)
            {
                velocityAtCurrent[i] *= steps;
            }

            return ExpMap(current, velocityAtCurrent, c);
        }

        /// <summary>Pulls <paramref name="thought"/> towards <paramref name="globalContext"/> along the geodesic by <paramref name="resonanceFactor"/> in [0, 1].</summary>
        public static double[] ContextResonance(ReadOnlySpan<double> thought, ReadOnlySpan<double> globalContext, double resonanceFactor, double c = 1.0)
        {
            var pull = LogMap(thought, globalContext, c);
            double factor = Math.Clamp(resonanceFactor, 0.0, 1.0);
            for (int i = 0; i < pull.Length; i++)
            {
                pull[i] *= factor;
            }

            return ExpMap(thought, pull, c);
        }

        // ------------------------------------------------------------------ momentum traversal

        /// <summary>
        /// Extrapolates a trajectory under the geometry of <paramref name="metric"/>: a Mobius/Euler step in the Poincare ball,
        /// the same step through the ball for the hyperboloid, a split step for <see cref="DistanceMetric.Hybrid"/>, and linear
        /// extrapolation for the Euclidean-like metrics.
        /// </summary>
        public static double[] ExtrapolateMomentum(DistanceMetric metric, ReadOnlySpan<double> past, ReadOnlySpan<double> current, double steps)
        {
            Require(past.Length == current.Length, "Dimension mismatch");
            switch (metric)
            {
                case DistanceMetric.Poincare:
                    return ExtrapolatePoincare(past, current, steps);
                case DistanceMetric.Lorentz:
                    {
                        var next = ExtrapolatePoincare(LorentzToPoincare(past), LorentzToPoincare(current), steps);
                        double sq = NormSq(next);
                        double den = Math.Max(1.0 - sq, 1e-15);
                        var result = new double[past.Length];
                        result[0] = (1.0 + sq) / den;
                        double factor = 2.0 / den;
                        for (int i = 0; i < next.Length; i++)
                        {
                            result[i + 1] = next[i] * factor;
                        }

                        return result;
                    }

                case DistanceMetric.Hybrid:
                    {
                        int l = Metrics.HybridLorentzDimensions;
                        Require(past.Length >= l, "Vector too short for Hybrid layout");
                        var head = ExtrapolateMomentum(DistanceMetric.Lorentz, past.Slice(0, l), current.Slice(0, l), steps);
                        var result = new double[past.Length];
                        head.CopyTo(result, 0);
                        for (int i = l; i < result.Length; i++)
                        {
                            result[i] = current[i] + (steps * (current[i] - past[i]));
                        }

                        return result;
                    }

                case DistanceMetric.Wasserstein:
                case DistanceMetric.Jaccard:
                case DistanceMetric.Hamming:
                    throw new NotSupportedException($"Momentum extrapolation not supported for metric: {Metrics.GetName(metric)}");
                default:
                    {
                        var result = new double[past.Length];
                        for (int i = 0; i < result.Length; i++)
                        {
                            result[i] = current[i] + (steps * (current[i] - past[i]));
                        }

                        return result;
                    }
            }
        }

        private static double[] ExtrapolatePoincare(ReadOnlySpan<double> past, ReadOnlySpan<double> current, double steps)
        {
            const double c = 1.0;
            int n = past.Length;
            double x2 = NormSq(past), y2 = NormSq(current), xy = Dot(past, current);
            double numLeft = 1.0 + (2.0 * c * xy) + (c * y2);
            double numRight = 1.0 - (c * x2);
            double den = 1.0 + (2.0 * c * xy) + (c * c * x2 * y2);
            if (Math.Abs(den) < 1e-15)
            {
                throw new InvalidOperationException("Mobius addition denominator too small");
            }

            var delta = new double[n];
            double deltaNormSq = 0;
            for (int i = 0; i < n; i++)
            {
                delta[i] = ((numLeft * -past[i]) + (numRight * current[i])) / den;
                deltaNormSq += delta[i] * delta[i];
            }

            double deltaNorm = Math.Sqrt(deltaNormSq);
            if (deltaNorm < 1e-15)
            {
                return current.ToArray();
            }

            double lambdaX = 2.0 / Math.Max(1.0 - (c * x2), 1e-15);
            double arg = Math.Min(Math.Sqrt(c) * deltaNorm, 1.0 - 1e-15);
            double factor = 2.0 / (lambdaX * Math.Sqrt(c)) * Math.Atanh(arg);

            var step = new double[n];
            double vNormSq = 0;
            for (int i = 0; i < n; i++)
            {
                step[i] = factor * delta[i] / deltaNorm * steps;
                vNormSq += step[i] * step[i];
            }

            double vNorm = Math.Sqrt(vNormSq);
            if (vNorm < 1e-15)
            {
                return current.ToArray();
            }

            double cx2 = y2;
            double lambdaC = 2.0 / Math.Max(1.0 - (c * cx2), 1e-15);
            double scale = Math.Tanh(Math.Sqrt(c) * lambdaC * vNorm / 2.0) / (Math.Sqrt(c) * vNorm);

            double asX2 = 0, cxAs = 0;
            var applied = new double[n];
            for (int i = 0; i < n; i++)
            {
                applied[i] = scale * step[i];
                asX2 += applied[i] * applied[i];
                cxAs += current[i] * applied[i];
            }

            double numLeft2 = 1.0 + (2.0 * c * cxAs) + (c * asX2);
            double numRight2 = 1.0 - (c * cx2);
            double den2 = 1.0 + (2.0 * c * cxAs) + (c * c * cx2 * asX2);
            if (Math.Abs(den2) < 1e-15)
            {
                throw new InvalidOperationException("Mobius addition denominator too small");
            }

            var result = new double[n];
            for (int i = 0; i < n; i++)
            {
                result[i] = ((numLeft2 * current[i]) + (numRight2 * applied[i])) / den2;
            }

            return result;
        }
    }
}
