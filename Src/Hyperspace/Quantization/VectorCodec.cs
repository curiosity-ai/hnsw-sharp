// <copyright file="VectorCodec.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;

    /// <summary>How vectors are stored by a <see cref="HyperspaceIndex"/>.</summary>
    public enum QuantizationMode
    {
        /// <summary>Full precision, 64-bit floats (the default, and the only mode that keeps hyperbolic vectors exact near the boundary).</summary>
        None = 0,

        /// <summary>32-bit floats.</summary>
        Float32 = 1,

        /// <summary>8 bits per dimension plus a per-vector scale (4x smaller than <see cref="Float32"/>), optionally with anisotropic refinement.</summary>
        ScalarInt8 = 2,

        /// <summary>One sign bit per dimension of the Walsh-Hadamard-rotated vector (32x smaller than <see cref="Float32"/>), searched asymmetrically.</summary>
        Binary = 3,

        /// <summary>4 bits per dimension in blocks of 16 with a per-block scale, with an optional full-precision head.</summary>
        ScalarInt4 = 4,

        /// <summary>TurboQuant: rotation, norm extraction and 4-bit Lloyd-Max (Gaussian-optimal) centroids, with an optional full-precision head.</summary>
        Turbo = 5,

        /// <summary>1.58-bit BitNet ternary <c>{-1, 0, +1}</c> (2 bits per dimension, 16x smaller than <see cref="Float32"/>), with an optional full-precision head.</summary>
        Ternary = 6,

        /// <summary>Product quantization: one byte per sub-vector, searched with asymmetric lookup tables (squared L2).</summary>
        ProductQuantization = 7,

        /// <summary>Product quantization after a Walsh-Hadamard rotation (squared L2).</summary>
        OPQ = 8,

        /// <summary>A full-precision (float32) head, e.g. the 33 Lorentz dimensions of a hybrid vector, followed by an 8-bit tail.</summary>
        AsymmetricHybrid = 9,
    }

    /// <summary>A query prepared once per search so each distance evaluation is cheap (rotation, lookup tables, conformal factor).</summary>
    public sealed class PreparedQuery
    {
        internal PreparedQuery(double[] coords, float[] coords32)
        {
            Coords = coords;
            Coords32 = coords32;
        }

        /// <summary>Gets the query coordinates as doubles.</summary>
        public double[] Coords { get; }

        /// <summary>Gets the query coordinates as floats.</summary>
        public float[] Coords32 { get; }

        /// <summary>Gets the Euclidean norm of the query.</summary>
        public double Norm { get; internal set; }

        /// <summary>Gets the squared norm of the rotated tail (the part of the query a rotating codec quantizes).</summary>
        internal double TailNormSq { get; set; }

        /// <summary>Gets the Poincare conformal factor <c>1 / (1 - |q|^2)</c> (only meaningful for the Poincare metric).</summary>
        public double Alpha { get; internal set; } = 1.0;

        /// <summary>Gets the query in Klein coordinates, when fast routing is enabled.</summary>
        internal double[]? Klein { get; set; }

        /// <summary>Gets codec-specific state (rotated tail, lookup table...).</summary>
        internal float[]? Rotated { get; set; }

        /// <summary>Gets the per-query lookup table of the bit-packed codecs (see <see cref="QuantSimd"/>).</summary>
        internal float[]? Table { get; set; }

        /// <summary>Gets the product-quantization lookup table.</summary>
        internal ProductQuantizer.LookupTable? Lookup { get; set; }
    }

    /// <summary>Encodes vectors to a compact fixed-size byte representation and computes distances against it.</summary>
    internal abstract class VectorCodec
    {
        [ThreadStatic]
        private static double[]? scratch;

        protected VectorCodec(int dimension, DistanceMetric metric)
        {
            Dimension = dimension;
            Metric = metric;
        }

        public int Dimension { get; }

        public DistanceMetric Metric { get; }

        /// <summary>Gets the length in bytes of every encoded vector.</summary>
        public abstract int EncodedLength { get; }

        /// <summary>Gets a value indicating whether searches over a prefix of the dimensions (matryoshka truncation) are supported.</summary>
        public virtual bool SupportsTruncation => false;

        public abstract byte[] Encode(ReadOnlySpan<float> vector);

        /// <summary>
        /// Gets the metric that <see cref="PairDistance"/> evaluates, chosen so its scale matches <see cref="Distance"/> (squared L2 for
        /// product quantization, an angular distance for binary codes), which graph construction compares against.
        /// </summary>
        public virtual DistanceMetric PairMetric => Metric;

        /// <summary>Distance between two stored vectors, from their reconstructions (see <see cref="DecodeForPair"/>).</summary>
        public virtual double PairDistance(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
        {
            var x = PairScratchA;
            var y = PairScratchB;
            if (x is null || x.Length < Dimension)
            {
                x = PairScratchA = new float[Dimension];
                y = PairScratchB = new float[Dimension];
            }

            DecodeForPair(a, x.AsSpan(0, Dimension));
            DecodeForPair(b, y!.AsSpan(0, Dimension));
            return Metrics.Distance(PairMetric, (ReadOnlySpan<float>)x.AsSpan(0, Dimension), y.AsSpan(0, Dimension));
        }

        [ThreadStatic]
        private static float[]? PairScratchA;

        [ThreadStatic]
        private static float[]? PairScratchB;

        /// <summary>
        /// Reconstructs a vector for <see cref="PairDistance"/>. Codecs that rotate vectors before quantizing override this to stay in the rotated
        /// domain when the pair metric is rotation invariant: an orthonormal rotation preserves L2 distances and inner products exactly, so the
        /// inverse transform (and its cost, for every candidate of every insertion) can be skipped.
        /// </summary>
        protected virtual void DecodeForPair(ReadOnlySpan<byte> stored, Span<float> destination) => Decode(stored, destination);

        /// <summary>Whether <see cref="PairMetric"/> is unchanged by an orthonormal rotation of the tail (after the first <paramref name="head"/> coordinates).</summary>
        protected bool PairMetricIsRotationInvariant(int head)
            => PairMetric is DistanceMetric.L2 or DistanceMetric.Cosine or DistanceMetric.Dot or DistanceMetric.Angular
               || (PairMetric == DistanceMetric.Hybrid && head == Metrics.HybridLorentzDimensions);

        /// <summary>Reconstructs the (approximate) original vector.</summary>
        public abstract void Decode(ReadOnlySpan<byte> stored, Span<float> destination);

        public virtual PreparedQuery Prepare(ReadOnlySpan<float> query)
        {
            var d = new double[query.Length];
            for (int i = 0; i < d.Length; i++)
            {
                d[i] = query[i];
            }

            var q = new PreparedQuery(d, query.ToArray());
            double sq = 0;
            foreach (var x in d)
            {
                sq += x * x;
            }

            q.Norm = Math.Sqrt(sq);
            if (Metric == DistanceMetric.Poincare)
            {
                q.Alpha = Metrics.ConformalAlpha<double>(d);
            }

            return q;
        }

        /// <summary>Distance from a stored vector to a prepared query; if <paramref name="truncation"/> is positive only that many leading dimensions are compared.</summary>
        public abstract double Distance(ReadOnlySpan<byte> stored, PreparedQuery query, int truncation = 0);

        /// <summary>Distance used while routing through the upper layers; defaults to <see cref="Distance"/>.</summary>
        public virtual double UpperDistance(ReadOnlySpan<byte> stored, PreparedQuery query) => Distance(stored, query);

        protected static double[] Scratch(int length)
        {
            var s = scratch;
            if (s is null || s.Length < length)
            {
                s = scratch = new double[Math.Max(length, 64)];
            }

            return s;
        }

        /// <summary>Applies the metric to a reconstructed vector, using the stored conformal factor for Poincare when available.</summary>
        protected double DistanceFromDecoded(ReadOnlySpan<double> decoded, PreparedQuery query, double storedAlpha, int truncation)
        {
            var q = query.Coords.AsSpan();
            int n = Math.Min(decoded.Length, q.Length);
            if (truncation > 0)
            {
                n = Math.Min(n, truncation);
            }

            var a = decoded.Slice(0, n);
            var b = q.Slice(0, n);
            if (Metric == DistanceMetric.Poincare && storedAlpha > 0 && truncation <= 0)
            {
                return Metrics.PoincareFromAlpha(Metrics.L2Squared(a, b), storedAlpha, query.Alpha);
            }

            return Metrics.Distance(Metric, a, b);
        }

        public static VectorCodec Create(HyperspaceOptions options)
        {
            int dim = options.Dimension;
            var metric = options.Metric;
            int head = options.ResolvedHeadDimensions;
            switch (options.Quantization)
            {
                case QuantizationMode.None:
                    return new FloatCodec(dim, metric, doublePrecision: true, options.FastRouting);
                case QuantizationMode.Float32:
                    return new FloatCodec(dim, metric, doublePrecision: false, options.FastRouting);
                case QuantizationMode.ScalarInt8:
                    return new Int8Codec(dim, metric, options.AnisotropicRefinement);
                case QuantizationMode.Binary:
                    return new BinaryCodec(dim, metric);
                case QuantizationMode.ScalarInt4:
                    return new Int4Codec(dim, metric, head);
                case QuantizationMode.Turbo:
                    return new TurboCodec(dim, metric, head);
                case QuantizationMode.Ternary:
                    return new TernaryCodec(dim, metric, head, options.TernaryThreshold);
                case QuantizationMode.AsymmetricHybrid:
                    return new AsymmetricHybridCodec(dim, metric, head);
                case QuantizationMode.ProductQuantization:
                case QuantizationMode.OPQ:
                    {
                        bool opq = options.Quantization == QuantizationMode.OPQ;
                        var pq = options.ProductQuantizer ?? new ProductQuantizer(dim, ProductQuantizer.DefaultSubvectorCount(dim), opq);
                        if (pq.Dimension != dim || pq.IsOpq != opq)
                        {
                            throw new ArgumentException("The product quantizer does not match the index dimension / mode");
                        }

                        return new ProductQuantizationCodec(dim, metric, pq);
                    }

                default:
                    throw new ArgumentOutOfRangeException(nameof(options));
            }
        }
    }
}
