// <copyright file="Int8Codec.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Buffers.Binary;
    using System.Runtime.InteropServices;

    /// <summary>
    /// Scalar 8-bit quantization. Layout: <c>sbyte[dim] | float scale | float alpha</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// For the Poincare metric the scale is fixed to 1 (the ball is inside [-1, 1]^d) and the original conformal factor
    /// <c>alpha = 1 / (1 - |x|^2)</c> is stored, so the quantization noise cannot push a point to the boundary. For every other metric
    /// the scale is the largest absolute coordinate of the vector, which also covers the Lorentz time axis.
    /// </para>
    /// <para>
    /// With <c>anisotropic</c> refinement and a fixed scale, up to 128 dimensions, a coordinate-descent pass nudges each quantized
    /// coordinate by +/-1 to minimize <c>e_par^2 + 10 e_perp^2</c>: the error component along the vector hurts inner-product and
    /// hyperbolic distances far more than the orthogonal one.
    /// </para>
    /// </remarks>
    internal sealed class Int8Codec : VectorCodec
    {
        private readonly bool _fixedScale;
        private readonly bool _anisotropic;

        public Int8Codec(int dimension, DistanceMetric metric, bool anisotropic)
            : base(dimension, metric)
        {
            _fixedScale = metric == DistanceMetric.Poincare;
            _anisotropic = anisotropic;
        }

        public override int EncodedLength => Dimension + 8;

        public override bool SupportsTruncation => true;

        public override byte[] Encode(ReadOnlySpan<float> vector)
        {
            int n = Dimension;
            var bytes = new byte[EncodedLength];
            double scale = 1.0;
            if (!_fixedScale)
            {
                double max = 0;
                foreach (var x in vector)
                {
                    max = Math.Max(max, Math.Abs(x));
                }

                scale = Math.Max(max, 1e-12);
            }

            double inv = 127.0 / scale;
            var q = new sbyte[n];
            for (int i = 0; i < n; i++)
            {
                q[i] = (sbyte)Math.Clamp(Math.Round(vector[i] * inv, MidpointRounding.AwayFromZero), -127.0, 127.0);
            }

            if (_fixedScale && _anisotropic && n <= 128)
            {
                RefineAnisotropic(vector, q);
            }

            for (int i = 0; i < n; i++)
            {
                bytes[i] = (byte)q[i];
            }

            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(n, 4), (float)scale);
            double alpha = _fixedScale ? Metrics.ConformalAlpha(vector) : 1.0;
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(n + 4, 4), (float)alpha);
            return bytes;
        }

        private static void RefineAnisotropic(ReadOnlySpan<float> v, sbyte[] q)
        {
            int n = q.Length;
            double norm = 0;
            for (int i = 0; i < n; i++)
            {
                norm += (double)v[i] * v[i];
            }

            norm = Math.Sqrt(norm);
            if (norm <= 1e-9)
            {
                return;
            }

            const double tWeight = 10.0;
            double invNorm = 1.0 / norm;
            var f = new double[n];
            double dotErrX = 0, sqErr = 0;
            for (int i = 0; i < n; i++)
            {
                f[i] = q[i] / 127.0;
                double err = f[i] - v[i];
                dotErrX += err * v[i];
                sqErr += err * err;
            }

            for (int i = 0; i < n; i++)
            {
                double origErr = f[i] - v[i];
                double baseDot = dotErrX - (origErr * v[i]);
                double baseSq = sqErr - (origErr * origErr);

                sbyte bestQ = q[i];
                double bestF = f[i], bestLoss = double.MaxValue, bestDot = dotErrX, bestSq = sqErr;
                for (int delta = -1; delta <= 1; delta++)
                {
                    sbyte cq = (sbyte)Math.Clamp(q[i] + delta, -127, 127);
                    double cf = cq / 127.0;
                    double cerr = cf - v[i];
                    double dot = baseDot + (cerr * v[i]);
                    double sq = baseSq + (cerr * cerr);
                    double par = dot * invNorm;
                    double parSq = par * par;
                    double ortho = Math.Max(sq - parSq, 0.0);
                    double loss = parSq + (tWeight * ortho);
                    if (loss < bestLoss)
                    {
                        bestLoss = loss;
                        bestQ = cq;
                        bestF = cf;
                        bestDot = dot;
                        bestSq = sq;
                    }
                }

                q[i] = bestQ;
                f[i] = bestF;
                dotErrX = bestDot;
                sqErr = bestSq;
            }
        }

        public override void Decode(ReadOnlySpan<byte> stored, Span<float> destination)
        {
            float step = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(Dimension, 4)) / 127f;
            int n = Math.Min(destination.Length, Dimension);
            QuantSimd.DecodeInt8(MemoryMarshal.Cast<byte, sbyte>(stored.Slice(0, n)), step, destination);
        }

        public override double Distance(ReadOnlySpan<byte> stored, PreparedQuery query, int truncation = 0)
        {
            int n = truncation > 0 ? Math.Min(truncation, Dimension) : Dimension;
            double step = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(Dimension, 4)) / 127.0;
            var q = query.Coords;

            var codes = MemoryMarshal.Cast<byte, sbyte>(stored.Slice(0, n));
            var q32 = query.Coords32.AsSpan(0, n);
            float step32 = (float)step;

            switch (Metric)
            {
                case DistanceMetric.L2:
                case DistanceMetric.Cosine:
                    return QuantSimd.L2Int8(codes, q32, step32);

                case DistanceMetric.Dot:
                    return 1.0 - (QuantSimd.DotInt8(codes, q32) * step);

                case DistanceMetric.Poincare when truncation <= 0:
                    {
                        double alpha = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(Dimension + 4, 4));
                        return Metrics.PoincareFromAlpha(QuantSimd.L2Int8(codes, q32, step32), alpha, query.Alpha);
                    }

                case DistanceMetric.Lorentz when truncation <= 0 && n >= 2:
                    {
                        double spatial = 0;
                        for (int i = 1; i < n; i++)
                        {
                            spatial += (sbyte)stored[i] * step * q[i];
                        }

                        double inner = (-(sbyte)stored[0] * step * q[0]) + spatial;
                        return Math.Acosh(Math.Max(-inner, 1.0 + 1e-12));
                    }

                default:
                    {
                        var buf = Scratch(Dimension);
                        for (int i = 0; i < Dimension; i++)
                        {
                            buf[i] = (sbyte)stored[i] * step;
                        }

                        return DistanceFromDecoded(buf.AsSpan(0, Dimension), query, 0, truncation);
                    }
            }
        }
    }
}
