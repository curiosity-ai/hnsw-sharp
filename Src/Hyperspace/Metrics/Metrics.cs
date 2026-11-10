// <copyright file="Metrics.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;
    using System.Numerics;
    using System.Runtime.CompilerServices;
    using System.Runtime.InteropServices;

    /// <summary>The built-in distance metrics (the "12-metric" engine of HyperspaceDB).</summary>
    public enum DistanceMetric
    {
        /// <summary>Squared Euclidean distance.</summary>
        L2 = 0,

        /// <summary>Cosine distance, expressed as the squared L2 distance of unit vectors (<c>2 - 2cos</c>); see <see cref="HyperspaceOptions.NormalizeVectors"/>.</summary>
        Cosine = 1,

        /// <summary><c>1 - a.b</c>, monotonic with the negative inner product.</summary>
        Dot = 2,

        /// <summary>Manhattan / taxicab distance.</summary>
        L1 = 3,

        /// <summary>Chebyshev distance.</summary>
        Linf = 4,

        /// <summary>Geodesic angle on the unit hypersphere, <c>acos(cos(a,b)) / pi</c>, which obeys the triangle inequality.</summary>
        Angular = 5,

        /// <summary>Continuous Jaccard / Tanimoto distance.</summary>
        Jaccard = 6,

        /// <summary>Normalized Hamming distance over coordinate signs.</summary>
        Hamming = 7,

        /// <summary>Hyperbolic geodesic distance in the Poincare ball.</summary>
        Poincare = 8,

        /// <summary>Hyperbolic geodesic distance on the Lorentz hyperboloid (first coordinate is the time axis).</summary>
        Lorentz = 9,

        /// <summary>33 Lorentz dimensions followed by Euclidean dimensions; the sum of the Lorentz and the squared L2 distance.</summary>
        Hybrid = 10,

        /// <summary>One-dimensional earth mover's distance between the (normalized, non-negative) vectors seen as histograms.</summary>
        Wasserstein = 11,
    }

    /// <summary>The distance functions behind <see cref="DistanceMetric"/>, over <see cref="float"/> and <see cref="double"/> spans.</summary>
    public static class Metrics
    {
        /// <summary>The number of Lorentz dimensions at the head of a <see cref="DistanceMetric.Hybrid"/> vector.</summary>
        public const int HybridLorentzDimensions = 33;

        /// <summary>Default tolerance of <see cref="Validate{T}"/> for the Lorentz hyperboloid membership check.</summary>
        public const double DefaultLorentzTolerance = 1e-6;

        /// <summary>Gets or sets the tolerance used when validating Lorentz vectors (<c>|-t^2 + |x|^2 + 1|</c>). Must be positive.</summary>
        public static double LorentzValidationTolerance { get; set; } = DefaultLorentzTolerance;

        private static readonly (string Name, DistanceMetric Metric)[] Names =
        {
            ("l2", DistanceMetric.L2), ("euclidean", DistanceMetric.L2),
            ("cosine", DistanceMetric.Cosine),
            ("dot", DistanceMetric.Dot), ("ip", DistanceMetric.Dot), ("inner_product", DistanceMetric.Dot),
            ("l1", DistanceMetric.L1), ("manhattan", DistanceMetric.L1), ("taxicab", DistanceMetric.L1),
            ("linf", DistanceMetric.Linf), ("chebyshev", DistanceMetric.Linf), ("chessboard", DistanceMetric.Linf),
            ("angular", DistanceMetric.Angular),
            ("jaccard", DistanceMetric.Jaccard), ("tanimoto", DistanceMetric.Jaccard),
            ("hamming", DistanceMetric.Hamming),
            ("poincare", DistanceMetric.Poincare),
            ("lorentz", DistanceMetric.Lorentz),
            ("hybrid", DistanceMetric.Hybrid),
            ("wasserstein", DistanceMetric.Wasserstein), ("emd", DistanceMetric.Wasserstein), ("sinkhorn", DistanceMetric.Wasserstein),
        };

        /// <summary>The canonical names of the 12 built-in metrics.</summary>
        public static IReadOnlyList<string> SupportedMetrics { get; } = new[]
        {
            "l2", "cosine", "dot", "l1", "linf", "angular", "jaccard", "hamming", "poincare", "lorentz", "hybrid", "wasserstein",
        };

        /// <summary>Gets the canonical name of a metric.</summary>
        public static string GetName(DistanceMetric metric) => SupportedMetrics[(int)metric];

        /// <summary>Parses a metric name (case-insensitive, with the aliases accepted by HyperspaceDB).</summary>
        public static bool TryParse(string? name, out DistanceMetric metric)
        {
            metric = DistanceMetric.L2;
            if (name is null)
            {
                return false;
            }

            var n = name.Trim().ToLowerInvariant();
            foreach (var (alias, m) in Names)
            {
                if (alias == n)
                {
                    metric = m;
                    return true;
                }
            }

            return false;
        }

        /// <summary>Parses a metric name, throwing <see cref="ArgumentException"/> if it is not supported.</summary>
        public static DistanceMetric Parse(string name)
        {
            if (!TryParse(name, out var metric))
            {
                throw new ArgumentException($"Unsupported metric '{name}'", nameof(name));
            }

            return metric;
        }

        /// <summary>Evaluates the distance between two vectors under the named metric.</summary>
        public static double Distance(string metric, ReadOnlySpan<double> a, ReadOnlySpan<double> b) => Distance(Parse(metric), a, b);

        /// <summary>Evaluates the distance between two vectors under the given metric.</summary>
        public static double Distance(DistanceMetric metric, ReadOnlySpan<float> a, ReadOnlySpan<float> b) => DistanceCore(metric, a, b);

        /// <summary>Evaluates the distance between two vectors under the given metric.</summary>
        public static double Distance(DistanceMetric metric, ReadOnlySpan<double> a, ReadOnlySpan<double> b) => DistanceCore(metric, a, b);

        private static double DistanceCore<T>(DistanceMetric metric, ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : unmanaged, IFloatingPointIeee754<T>
        {
            switch (metric)
            {
                case DistanceMetric.L2:
                case DistanceMetric.Cosine:
                    return L2Squared(a, b);
                case DistanceMetric.Dot:
                    return 1.0 - Dot(a, b);
                case DistanceMetric.L1:
                    return Manhattan(a, b);
                case DistanceMetric.Linf:
                    return Chebyshev(a, b);
                case DistanceMetric.Angular:
                    return Angular(a, b);
                case DistanceMetric.Jaccard:
                    return Jaccard(a, b);
                case DistanceMetric.Hamming:
                    return Hamming(a, b);
                case DistanceMetric.Poincare:
                    return Poincare(a, b);
                case DistanceMetric.Lorentz:
                    return Lorentz(a, b);
                case DistanceMetric.Hybrid:
                    return Hybrid(a, b);
                case DistanceMetric.Wasserstein:
                    return Wasserstein(a, b);
                default:
                    throw new ArgumentOutOfRangeException(nameof(metric));
            }
        }

        // ------------------------------------------------------------------ Euclidean family (SIMD)

        /// <summary>Squared Euclidean distance.</summary>
        public static double L2Squared<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : unmanaged, IFloatingPointIeee754<T>
        {
            int n = Math.Min(a.Length, b.Length);
            int i = 0;
            T sum = T.Zero;

            if (Vector.IsHardwareAccelerated && n >= Vector<T>.Count)
            {
                var acc = Vector<T>.Zero;
                ref T ra = ref MemoryMarshal.GetReference(a);
                ref T rb = ref MemoryMarshal.GetReference(b);
                int last = n - Vector<T>.Count;
                for (; i <= last; i += Vector<T>.Count)
                {
                    var d = Vector.LoadUnsafe(ref ra, (nuint)i) - Vector.LoadUnsafe(ref rb, (nuint)i);
                    acc += d * d;
                }

                sum = Vector.Sum(acc);
            }

            for (; i < n; i++)
            {
                var d = a[i] - b[i];
                sum += d * d;
            }

            return double.CreateChecked(sum);
        }

        /// <summary>Inner product.</summary>
        public static double Dot<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : unmanaged, IFloatingPointIeee754<T>
        {
            int n = Math.Min(a.Length, b.Length);
            int i = 0;
            T sum = T.Zero;

            if (Vector.IsHardwareAccelerated && n >= Vector<T>.Count)
            {
                var acc = Vector<T>.Zero;
                ref T ra = ref MemoryMarshal.GetReference(a);
                ref T rb = ref MemoryMarshal.GetReference(b);
                int last = n - Vector<T>.Count;
                for (; i <= last; i += Vector<T>.Count)
                {
                    acc += Vector.LoadUnsafe(ref ra, (nuint)i) * Vector.LoadUnsafe(ref rb, (nuint)i);
                }

                sum = Vector.Sum(acc);
            }

            for (; i < n; i++)
            {
                sum += a[i] * b[i];
            }

            return double.CreateChecked(sum);
        }

        /// <summary>Manhattan distance.</summary>
        public static double Manhattan<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : unmanaged, IFloatingPointIeee754<T>
        {
            int n = Math.Min(a.Length, b.Length);
            int i = 0;
            T sum = T.Zero;

            if (Vector.IsHardwareAccelerated && n >= Vector<T>.Count)
            {
                var acc = Vector<T>.Zero;
                ref T ra = ref MemoryMarshal.GetReference(a);
                ref T rb = ref MemoryMarshal.GetReference(b);
                int last = n - Vector<T>.Count;
                for (; i <= last; i += Vector<T>.Count)
                {
                    acc += Vector.Abs(Vector.LoadUnsafe(ref ra, (nuint)i) - Vector.LoadUnsafe(ref rb, (nuint)i));
                }

                sum = Vector.Sum(acc);
            }

            for (; i < n; i++)
            {
                sum += T.Abs(a[i] - b[i]);
            }

            return double.CreateChecked(sum);
        }

        /// <summary>Chebyshev distance.</summary>
        public static double Chebyshev<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : unmanaged, IFloatingPointIeee754<T>
        {
            int n = Math.Min(a.Length, b.Length);
            double max = 0;
            for (int i = 0; i < n; i++)
            {
                var d = double.CreateChecked(T.Abs(a[i] - b[i]));
                if (d > max)
                {
                    max = d;
                }
            }

            return max;
        }

        // ------------------------------------------------------------------ angular family

        /// <summary><c>acos(cos(a, b)) / pi</c>, in [0, 1]; 1 for zero vectors.</summary>
        public static double Angular<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : unmanaged, IFloatingPointIeee754<T>
        {
            int n = Math.Min(a.Length, b.Length);
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < n; i++)
            {
                double x = double.CreateChecked(a[i]);
                double y = double.CreateChecked(b[i]);
                dot += x * y;
                na += x * x;
                nb += y * y;
            }

            if (na <= 1e-24 || nb <= 1e-24)
            {
                return 1.0;
            }

            double cos = Math.Clamp(dot / Math.Sqrt(na * nb), -1.0, 1.0);
            return Math.Acos(cos) / Math.PI;
        }

        /// <summary><c>1 - a.b / (|a|^2 + |b|^2 - a.b)</c>, in [0, 2].</summary>
        public static double Jaccard<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : unmanaged, IFloatingPointIeee754<T>
        {
            int n = Math.Min(a.Length, b.Length);
            double dot = 0, na = 0, nb = 0;
            for (int i = 0; i < n; i++)
            {
                double x = double.CreateChecked(a[i]);
                double y = double.CreateChecked(b[i]);
                dot += x * y;
                na += x * x;
                nb += y * y;
            }

            double denom = na + nb - dot;
            if (denom <= 1e-24)
            {
                return 0.0;
            }

            return Math.Clamp(1.0 - (dot / denom), 0.0, 2.0);
        }

        /// <summary>The fraction of coordinates whose sign / activation differs.</summary>
        public static double Hamming<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : unmanaged, IFloatingPointIeee754<T>
        {
            int n = Math.Min(a.Length, b.Length);
            if (n == 0)
            {
                return 0.0;
            }

            int diff = 0;
            for (int i = 0; i < n; i++)
            {
                bool pa = a[i] > T.Zero, pb = b[i] > T.Zero;
                if (pa != pb || (a[i] == T.Zero) != (b[i] == T.Zero))
                {
                    diff++;
                }
            }

            return (double)diff / n;
        }

        /// <summary>
        /// One-dimensional Wasserstein-1 distance: both vectors are clipped to non-negative, normalized to sum to 1
        /// and compared through their cumulative distributions. Returns <see cref="double.MaxValue"/> if either has no mass.
        /// </summary>
        public static double Wasserstein<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : unmanaged, IFloatingPointIeee754<T>
        {
            double sumA = 0, sumB = 0;
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i] > T.Zero)
                {
                    sumA += double.CreateChecked(a[i]);
                }
            }

            for (int i = 0; i < b.Length; i++)
            {
                if (b[i] > T.Zero)
                {
                    sumB += double.CreateChecked(b[i]);
                }
            }

            if (sumA <= 1e-9 || sumB <= 1e-9)
            {
                return double.MaxValue;
            }

            double cdfA = 0, cdfB = 0, dist = 0;
            int n = Math.Min(a.Length, b.Length);
            for (int i = 0; i < n; i++)
            {
                double va = a[i] > T.Zero ? double.CreateChecked(a[i]) / sumA : 0.0;
                double vb = b[i] > T.Zero ? double.CreateChecked(b[i]) / sumB : 0.0;
                cdfA += va;
                cdfB += vb;
                dist += Math.Abs(cdfA - cdfB);
            }

            return dist;
        }

        // ------------------------------------------------------------------ hyperbolic

        /// <summary>
        /// Poincare ball geodesic distance <c>acosh(1 + 2|a-b|^2 / ((1-|a|^2)(1-|b|^2)))</c>.
        /// </summary>
        public static double Poincare<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : unmanaged, IFloatingPointIeee754<T>
        {
            int n = Math.Min(a.Length, b.Length);
            double nu = 0, nv = 0, nd = 0;
            for (int i = 0; i < n; i++)
            {
                double u = double.CreateChecked(a[i]);
                double v = double.CreateChecked(b[i]);
                nu += u * u;
                nv += v * v;
                double d = u - v;
                nd += d * d;
            }

            double denom = (1.0 - nu) * (1.0 - nv);
            double arg = 1.0 + (2.0 * nd / Math.Max(denom, 1e-9));
            return Math.Acosh(arg);
        }

        /// <summary>
        /// Poincare distance from a precomputed conformal factor, <c>alpha = 1 / (1 - |x|^2)</c> for each side and the squared
        /// Euclidean distance between the two points.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double PoincareFromAlpha(double squaredDifference, double alphaA, double alphaB)
            => Math.Acosh(1.0 + (2.0 * squaredDifference * alphaA * alphaB));

        /// <summary>Lorentz hyperboloid geodesic distance <c>acosh(max(1, -&lt;a, b&gt;_L))</c>.</summary>
        public static double Lorentz<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : unmanaged, IFloatingPointIeee754<T>
        {
            int n = Math.Min(a.Length, b.Length);
            if (n < 2)
            {
                return 0.0;
            }

            double spatial = 0;
            for (int i = 1; i < n; i++)
            {
                spatial += double.CreateChecked(a[i]) * double.CreateChecked(b[i]);
            }

            double inner = (-double.CreateChecked(a[0]) * double.CreateChecked(b[0])) + spatial;
            return Math.Acosh(Math.Max(-inner, 1.0 + 1e-12));
        }

        /// <summary>The sum of the Lorentz distance over the first 33 dimensions and the squared L2 over the rest.</summary>
        public static double Hybrid<T>(ReadOnlySpan<T> a, ReadOnlySpan<T> b) where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (a.Length < HybridLorentzDimensions || b.Length < HybridLorentzDimensions)
            {
                return 0.0;
            }

            return Lorentz(a.Slice(0, HybridLorentzDimensions), b.Slice(0, HybridLorentzDimensions))
                 + L2Squared(a.Slice(HybridLorentzDimensions), b.Slice(HybridLorentzDimensions));
        }

        // ------------------------------------------------------------------ validation / helpers

        /// <summary>Checks a vector is acceptable for a metric: finite, inside the ball for Poincare, on the upper sheet of the hyperboloid for Lorentz.</summary>
        /// <returns><c>null</c> if the vector is valid, otherwise the reason it is not.</returns>
        public static string? Validate<T>(DistanceMetric metric, ReadOnlySpan<T> vector) where T : unmanaged, IFloatingPointIeee754<T>
        {
            for (int i = 0; i < vector.Length; i++)
            {
                if (!T.IsFinite(vector[i]))
                {
                    return $"Vector contains non-finite values (NaN/Inf) for metric: {GetName(metric)}";
                }
            }

            switch (metric)
            {
                case DistanceMetric.Poincare:
                    {
                        double sq = 0;
                        foreach (var x in vector)
                        {
                            sq += double.CreateChecked(x) * double.CreateChecked(x);
                        }

                        return sq >= 1.0 - 1e-9 ? "Vector must be strictly inside the Poincare ball" : null;
                    }

                case DistanceMetric.Lorentz:
                    return ValidateLorentz(vector, LorentzValidationTolerance);
                case DistanceMetric.Hybrid:
                    if (vector.Length < HybridLorentzDimensions)
                    {
                        return $"Hybrid vectors need at least {HybridLorentzDimensions} dimensions";
                    }

                    return ValidateLorentz(vector.Slice(0, HybridLorentzDimensions), LorentzValidationTolerance);
                default:
                    return null;
            }
        }

        private static string? ValidateLorentz<T>(ReadOnlySpan<T> vector, double tolerance) where T : unmanaged, IFloatingPointIeee754<T>
        {
            if (vector.Length < 2)
            {
                return "Lorentz vector requires at least 2 dimensions";
            }

            if (double.CreateChecked(vector[0]) <= 0.0)
            {
                return "Lorentz vector must be on the upper sheet (t > 0)";
            }

            double spatial = 0;
            for (int i = 1; i < vector.Length; i++)
            {
                spatial += double.CreateChecked(vector[i]) * double.CreateChecked(vector[i]);
            }

            double t = double.CreateChecked(vector[0]);
            double mink = (-t * t) + spatial;

            // A float32 coordinate carries a relative error of about 6e-8, so -t^2 + |x|^2 can be off by that times t^2 even for a point
            // that was exactly on the hyperboloid in double precision: widen the tolerance in proportion, for float input only.
            if (typeof(T) == typeof(float))
            {
                tolerance += 8 * 1.19e-7 * (1.0 + (t * t));
            }

            return Math.Abs(mink + 1.0) > tolerance ? $"Lorentz vector is not on the unit hyperboloid: -t^2+|x|^2={mink}, expected -1" : null;
        }

        /// <summary>The conformal factor <c>1 / (1 - |x|^2)</c> of a point in the Poincare ball.</summary>
        public static double ConformalAlpha<T>(ReadOnlySpan<T> vector) where T : unmanaged, IFloatingPointIeee754<T>
        {
            double sq = 0;
            foreach (var x in vector)
            {
                sq += double.CreateChecked(x) * double.CreateChecked(x);
            }

            return 1.0 / (1.0 - sq);
        }

        /// <summary>Scales a vector to unit L2 norm in place. Zero vectors are left untouched.</summary>
        public static void Normalize(Span<float> vector)
        {
            double sq = 0;
            foreach (var x in vector)
            {
                sq += (double)x * x;
            }

            if (sq <= 1e-30)
            {
                return;
            }

            float inv = (float)(1.0 / Math.Sqrt(sq));
            for (int i = 0; i < vector.Length; i++)
            {
                vector[i] *= inv;
            }
        }
    }
}
