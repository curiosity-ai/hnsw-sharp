// <copyright file="BinaryCodec.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Buffers.Binary;

    /// <summary>
    /// One sign bit per dimension of the rotated vector. Layout: <c>byte[ceil(dim/8)] | float alpha</c>.
    /// </summary>
    /// <remarks>
    /// The vector is rotated with <see cref="TurboRotation"/> first so every bit carries a similar amount of information; the query is
    /// rotated the same way and compared asymmetrically (full-precision query against sign bits), giving a cosine-style distance
    /// <c>2 (1 - cos(q, sign(v)))</c>. All metrics use that angular approximation; use a re-ranking pass for exact distances.
    /// </remarks>
    internal sealed class BinaryCodec : VectorCodec
    {
        public BinaryCodec(int dimension, DistanceMetric metric)
            : base(dimension, metric)
        {
        }

        public override DistanceMetric PairMetric => DistanceMetric.Cosine;

        private int BitBytes => (Dimension + 7) / 8;

        public override int EncodedLength => BitBytes + 4;

        public override byte[] Encode(ReadOnlySpan<float> vector)
        {
            var rotated = TurboRotation.Rotated(vector);
            var bytes = new byte[EncodedLength];
            for (int i = 0; i < Dimension; i++)
            {
                if (rotated[i] > 0f)
                {
                    bytes[i >> 3] |= (byte)(1 << (i & 7));
                }
            }

            double alpha = Metric == DistanceMetric.Poincare ? Metrics.ConformalAlpha(vector) : 1.0;
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(BitBytes, 4), (float)alpha);
            return bytes;
        }

        protected override void DecodeForPair(ReadOnlySpan<byte> stored, Span<float> destination)
        {
            float s = 1.0f / MathF.Sqrt(Dimension);
            for (int i = 0; i < Dimension && i < destination.Length; i++)
            {
                destination[i] = ((stored[i >> 3] >> (i & 7)) & 1) == 1 ? s : -s;
            }
        }

        public override void Decode(ReadOnlySpan<byte> stored, Span<float> destination)
        {
            // The sign pattern, un-rotated, scaled to unit norm: the direction the bits describe.
            var v = new float[Dimension];
            float s = 1.0f / MathF.Sqrt(Dimension);
            for (int i = 0; i < Dimension; i++)
            {
                v[i] = ((stored[i >> 3] >> (i & 7)) & 1) == 1 ? s : -s;
            }

            TurboRotation.Unrotate(v);
            v.AsSpan(0, destination.Length).CopyTo(destination);
        }

        public override PreparedQuery Prepare(ReadOnlySpan<float> query)
        {
            var q = base.Prepare(query);
            q.Rotated = TurboRotation.Rotated(query);
            q.Table = QuantSimd.BuildSignTable(q.Rotated, BitBytes);
            return q;
        }

        public override double Distance(ReadOnlySpan<byte> stored, PreparedQuery query, int truncation = 0)
        {
            // sum over dimensions of sign(code) * rotatedQuery, one table lookup per byte of bits
            double signedDot = QuantSimd.SumTable(stored.Slice(0, BitBytes), query.Table!);
            double denom = Math.Sqrt(Dimension) * query.Norm;
            if (denom <= 1e-12)
            {
                return 2.0;
            }

            return Math.Max(2.0 * (1.0 - Math.Clamp(signedDot / denom, -1.0, 1.0)), 0.0);
        }
    }
}
