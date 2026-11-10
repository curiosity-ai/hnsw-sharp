// <copyright file="TernaryCodec.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Buffers.Binary;
    using System.Numerics;
    using System.Runtime.InteropServices;

    /// <summary>
    /// 1.58-bit BitNet ternary quantization. Layout: <c>float[head] | byte[ceil(tail/8)] nonZero | byte[ceil(tail/8)] sign | float scale | float alpha</c>.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The tail is rotated with <see cref="TurboRotation"/>, then every coordinate whose magnitude exceeds <c>tau = factor * mean|x|</c> becomes
    /// <c>+/-1</c> and the rest <c>0</c>. A single scale, the least-squares optimal <c>sum|x| over the non-zeros / count</c>, restores the magnitude.
    /// Two bit-planes (non-zero mask and sign) make dot products with a full-precision query a masked add.
    /// </para>
    /// </remarks>
    internal sealed class TernaryCodec : VectorCodec
    {
        private readonly int _head;
        private readonly float _tauFactor;

        public TernaryCodec(int dimension, DistanceMetric metric, int headDimensions, float tauFactor)
            : base(dimension, metric)
        {
            _head = Math.Min(headDimensions, dimension);
            _tauFactor = tauFactor;
        }

        private int TailDim => Dimension - _head;

        private int PlaneBytes => (TailDim + 7) / 8;

        public override int EncodedLength => (_head * 4) + (2 * PlaneBytes) + 8;

        public override byte[] Encode(ReadOnlySpan<float> vector)
        {
            var bytes = new byte[EncodedLength];
            for (int i = 0; i < _head; i++)
            {
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), vector[i]);
            }

            int nzOffset = _head * 4;
            int signOffset = nzOffset + PlaneBytes;
            int scaleOffset = signOffset + PlaneBytes;
            float scale = 1.0f;

            if (TailDim > 0)
            {
                var tail = ScratchFloats(TailDim);
                vector.Slice(_head).CopyTo(tail);
                TurboRotation.Rotate(tail);
                float meanAbs = 0;
                foreach (var x in tail)
                {
                    meanAbs += Math.Abs(x);
                }

                meanAbs /= TailDim;
                float tau = meanAbs * _tauFactor;
                float dot = 0, norm = 0;
                for (int i = 0; i < tail.Length; i++)
                {
                    float x = tail[i];
                    if (x > tau)
                    {
                        bytes[nzOffset + (i >> 3)] |= (byte)(1 << (i & 7));
                        bytes[signOffset + (i >> 3)] |= (byte)(1 << (i & 7));
                        dot += x;
                        norm += 1;
                    }
                    else if (x < -tau)
                    {
                        bytes[nzOffset + (i >> 3)] |= (byte)(1 << (i & 7));
                        dot -= x;
                        norm += 1;
                    }
                }

                scale = norm > 0 ? dot / norm : 1.0f;
            }

            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(scaleOffset, 4), scale);
            double alpha = Metric == DistanceMetric.Poincare ? Metrics.ConformalAlpha(vector) : 1.0;
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(scaleOffset + 4, 4), (float)alpha);
            return bytes;
        }

        /// <summary>Reconstructs the head and the ROTATED ternary tail.</summary>
        private void ReconstructRotated(ReadOnlySpan<byte> stored, Span<double> output)
        {
            for (int i = 0; i < _head; i++)
            {
                output[i] = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(i * 4, 4));
            }

            int nzOffset = _head * 4;
            int signOffset = nzOffset + PlaneBytes;
            double scale = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(signOffset + PlaneBytes, 4));
            for (int i = 0; i < TailDim; i++)
            {
                int mask = 1 << (i & 7);
                if ((stored[nzOffset + (i >> 3)] & mask) != 0)
                {
                    output[_head + i] = (stored[signOffset + (i >> 3)] & mask) != 0 ? scale : -scale;
                }
                else
                {
                    output[_head + i] = 0;
                }
            }
        }

        private void ReconstructUnrotated(ReadOnlySpan<byte> stored, Span<double> output)
        {
            ReconstructRotated(stored, output);
            if (TailDim > 0)
            {
                var tail = ScratchFloats(TailDim);
                for (int i = 0; i < tail.Length; i++)
                {
                    tail[i] = (float)output[_head + i];
                }

                TurboRotation.Unrotate(tail);
                for (int i = 0; i < tail.Length; i++)
                {
                    output[_head + i] = tail[i];
                }
            }
        }

        /// <summary>
        /// Squared L2 (or 1 - dot) between two ternary vectors from the bit planes alone: with scales sa, sb and counts of non-zeros na, nb,
        /// <c>|a - b|^2 = sa^2 na + sb^2 nb - 2 sa sb D</c>, where D (agreeing minus disagreeing non-zero signs) is two popcounts per word.
        /// The float head, if any, is compared directly.
        /// </summary>
        public override double PairDistance(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
        {
            if (Metric is not (DistanceMetric.L2 or DistanceMetric.Cosine or DistanceMetric.Dot))
            {
                return base.PairDistance(a, b);
            }

            double headL2 = 0, headDot = 0, headNormA = 0, headNormB = 0;
            for (int i = 0; i < _head; i++)
            {
                double x = BinaryPrimitives.ReadSingleLittleEndian(a.Slice(i * 4, 4));
                double y = BinaryPrimitives.ReadSingleLittleEndian(b.Slice(i * 4, 4));
                headL2 += (x - y) * (x - y);
                headDot += x * y;
                headNormA += x * x;
                headNormB += y * y;
            }

            int nzOffset = _head * 4;
            int signOffset = nzOffset + PlaneBytes;
            double sa = BinaryPrimitives.ReadSingleLittleEndian(a.Slice(signOffset + PlaneBytes, 4));
            double sb = BinaryPrimitives.ReadSingleLittleEndian(b.Slice(signOffset + PlaneBytes, 4));

            int wide = PlaneBytes & ~7;
            var nzA = MemoryMarshal.Cast<byte, ulong>(a.Slice(nzOffset, wide));
            var nzB = MemoryMarshal.Cast<byte, ulong>(b.Slice(nzOffset, wide));
            var sgA = MemoryMarshal.Cast<byte, ulong>(a.Slice(signOffset, wide));
            var sgB = MemoryMarshal.Cast<byte, ulong>(b.Slice(signOffset, wide));
            int countA = 0, countB = 0, agree = 0, disagree = 0;
            for (int i = 0; i < nzA.Length; i++)
            {
                countA += BitOperations.PopCount(nzA[i]);
                countB += BitOperations.PopCount(nzB[i]);
                ulong both = nzA[i] & nzB[i];
                ulong differ = both & (sgA[i] ^ sgB[i]);
                disagree += BitOperations.PopCount(differ);
                agree += BitOperations.PopCount(both & ~differ);
            }

            for (int i = wide; i < PlaneBytes; i++)
            {
                byte na = a[nzOffset + i], nb = b[nzOffset + i];
                countA += BitOperations.PopCount(na);
                countB += BitOperations.PopCount(nb);
                byte both = (byte)(na & nb);
                byte differ = (byte)(both & (a[signOffset + i] ^ b[signOffset + i]));
                disagree += BitOperations.PopCount(differ);
                agree += BitOperations.PopCount((uint)(both & ~differ & 0xFF));
            }

            double tailDot = sa * sb * (agree - disagree);
            if (Metric == DistanceMetric.Dot)
            {
                return 1.0 - (headDot + tailDot);
            }

            if (Metric == DistanceMetric.Cosine)
            {
                // Same form as the query distance, 2 (1 - cos). A decoded ternary vector is shorter than unit length, so comparing it by the squared L2
                // the generic path would use puts the pair distances on a smaller scale than the query distances that the neighbour-selection
                // heuristic weighs them against, and the graph it builds is poor.
                double normSq = (headNormA + (sa * sa * countA)) * (headNormB + (sb * sb * countB));
                if (normSq < 1e-18)
                {
                    return 2.0;
                }

                return Math.Max(2.0 * (1.0 - Math.Clamp((headDot + tailDot) / Math.Sqrt(normSq), -1.0, 1.0)), 0.0);
            }

            return headL2 + Math.Max((sa * sa * countA) + (sb * sb * countB) - (2.0 * tailDot), 0.0);
        }

        protected override void DecodeForPair(ReadOnlySpan<byte> stored, Span<float> destination)
        {
            if (!PairMetricIsRotationInvariant(_head))
            {
                Decode(stored, destination);
                return;
            }

            var buf = Scratch(Dimension);
            ReconstructRotated(stored, buf.AsSpan(0, Dimension));
            for (int i = 0; i < destination.Length; i++)
            {
                destination[i] = (float)buf[i];
            }
        }

        public override void Decode(ReadOnlySpan<byte> stored, Span<float> destination)
        {
            var buf = Scratch(Dimension);
            ReconstructUnrotated(stored, buf.AsSpan(0, Dimension));
            for (int i = 0; i < destination.Length; i++)
            {
                destination[i] = (float)buf[i];
            }
        }

        public override PreparedQuery Prepare(ReadOnlySpan<float> query, PreparedQuery? reuse = null)
        {
            var q = base.Prepare(query, reuse);
            if (TailDim > 0)
            {
                var tail = q.Rotated is { } r && r.Length == TailDim ? r : new float[TailDim];
                query.Slice(_head).CopyTo(tail);
                TurboRotation.Rotate(tail);
                q.Rotated = tail;
                double sq = 0;
                foreach (var x in tail)
                {
                    sq += (double)x * x;
                }

                q.TailNormSq = sq;
                if (Metric is DistanceMetric.L2 or DistanceMetric.Cosine or DistanceMetric.Dot)
                {
                    q.Table = QuantSimd.BuildTernaryTable(tail, PlaneBytes * 2, q.Table);
                }
            }
            else
            {
                q.Rotated = Array.Empty<float>();
            }

            return q;
        }

        public override double Distance(ReadOnlySpan<byte> stored, PreparedQuery query, int truncation = 0)
        {
            switch (Metric)
            {
                case DistanceMetric.L2:
                case DistanceMetric.Cosine:
                case DistanceMetric.Dot:
                    return FastDistance(stored, query);
                default:
                    {
                        var buf = Scratch(Dimension);
                        ReconstructUnrotated(stored, buf.AsSpan(0, Dimension));
                        double alpha = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(stored.Length - 4, 4));
                        return DistanceFromDecoded(buf.AsSpan(0, Dimension), query, alpha, 0);
                    }
            }
        }

        private double FastDistance(ReadOnlySpan<byte> stored, PreparedQuery query)
        {
            int nzOffset = _head * 4;
            int signOffset = nzOffset + PlaneBytes;
            double scale = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(signOffset + PlaneBytes, 4));
            var q = query.Coords;
            var rotated = query.Rotated!;

            double headDot = 0, headNormSq = 0, headDistSq = 0;
            for (int i = 0; i < _head; i++)
            {
                double h = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(i * 4, 4));
                double diff = h - q[i];
                headDistSq += diff * diff;
                headDot += h * q[i];
                headNormSq += h * h;
            }

            // Two lookups per byte of the planes (four dimensions each) replace the per-dimension loop.
            double tailDot = 0;
            int nzCount = 0;
            if (PlaneBytes > 0)
            {
                var table = query.Table!;
                float acc = 0;
                for (int b = 0; b < PlaneBytes; b++)
                {
                    byte nz = stored[nzOffset + b];
                    if (nz == 0)
                    {
                        continue;
                    }

                    byte sign = stored[signOffset + b];
                    nzCount += BitOperations.PopCount(nz);
                    acc += table[((2 * b) << 8) + ((nz & 15) << 4) + (sign & 15)] + table[(((2 * b) + 1) << 8) + ((nz >> 4) << 4) + (sign >> 4)];
                }

                tailDot = acc;
            }

            double vTailNormSq = nzCount * scale * scale;
            switch (Metric)
            {
                case DistanceMetric.L2:
                    return headDistSq + Math.Max(vTailNormSq + query.TailNormSq - (2.0 * tailDot * scale), 0.0);
                case DistanceMetric.Dot:
                    return 1.0 - (headDot + (tailDot * scale));
                default:
                    {
                        double dot = headDot + (tailDot * scale);
                        double denom = Math.Sqrt((headNormSq + vTailNormSq) * (query.Norm * query.Norm));
                        if (denom < 1e-9)
                        {
                            return 2.0;
                        }

                        return Math.Max(2.0 * (1.0 - Math.Clamp(dot / denom, -1.0, 1.0)), 0.0);
                    }
            }
        }
    }
}
