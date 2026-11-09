// <copyright file="TernaryCodec.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Buffers.Binary;
    using System.Numerics;

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
                var tail = vector.Slice(_head).ToArray();
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
                var tail = new float[TailDim];
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

        public override PreparedQuery Prepare(ReadOnlySpan<float> query)
        {
            var q = base.Prepare(query);
            if (TailDim > 0)
            {
                var tail = query.Slice(_head).ToArray();
                TurboRotation.Rotate(tail);
                q.Rotated = tail;
                double sq = 0;
                foreach (var x in tail)
                {
                    sq += (double)x * x;
                }

                q.TailNormSq = sq;
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

            double tailDot = 0;
            int nzCount = 0;
            for (int b = 0; b < PlaneBytes; b++)
            {
                byte nz = stored[nzOffset + b];
                if (nz == 0)
                {
                    continue;
                }

                nzCount += BitOperations.PopCount(nz);
                byte sign = stored[signOffset + b];
                int baseIdx = b * 8;
                int limit = Math.Min(8, TailDim - baseIdx);
                for (int bit = 0; bit < limit; bit++)
                {
                    int mask = 1 << bit;
                    if ((nz & mask) != 0)
                    {
                        tailDot += (sign & mask) != 0 ? rotated[baseIdx + bit] : -rotated[baseIdx + bit];
                    }
                }
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
