// <copyright file="TurboCodec.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Buffers.Binary;

    /// <summary>
    /// TurboQuant: rotate, normalize, then snap each coordinate to the nearest of 16 Lloyd-Max centroids of the standard normal
    /// distribution. Layout: <c>float norm | uint head | float[head] | byte[ceil(tail/2)] nibbles</c>.
    /// </summary>
    /// <remarks>
    /// After the Walsh-Hadamard rotation the coordinates of a vector are close to Gaussian with variance <c>|x|^2 / d</c>, which is what the
    /// fixed centroids are optimal for; only the tail norm needs storing. A bias-correction factor cancels the variance shrinkage of
    /// the nearest-centroid reconstruction.
    /// </remarks>
    internal sealed class TurboCodec : VectorCodec
    {
        /// <summary>The 4-bit Lloyd-Max centroids for N(0, 1).</summary>
        internal static readonly double[] Centroids =
        {
            -2.401, -1.844, -1.437, -1.099, -0.800, -0.524, -0.262, -0.066, 0.066, 0.262, 0.524, 0.800, 1.099, 1.437, 1.844, 2.401,
        };

        /// <summary>Cancels the MMSE shrinkage of the reconstruction (E[&lt;x^, q&gt;] / E[&lt;x, q&gt;] is about 0.98006).</summary>
        internal const double BiasCorrection = 1.0203407;

        private static readonly float[] CentroidsF = Array.ConvertAll(Centroids, c => (float)c);

        /// <summary>For a packed byte, the sum of the squares of its two centroids.</summary>
        private static readonly float[] PairSquares = BuildPairSquares();

        private readonly int _head;

        /// <summary>CentroidProducts[i * 16 + j] is the product of centroids i and j.</summary>
        private static readonly float[] CentroidProducts = BuildCentroidProducts();

        private static float[] BuildCentroidProducts()
        {
            var t = new float[256];
            for (int i = 0; i < 16; i++)
            {
                for (int j = 0; j < 16; j++)
                {
                    t[(i * 16) + j] = (float)(Centroids[i] * Centroids[j]);
                }
            }

            return t;
        }

        private static float[] BuildPairSquares()
        {
            var t = new float[256];
            for (int b = 0; b < 256; b++)
            {
                t[b] = (float)((Centroids[b >> 4] * Centroids[b >> 4]) + (Centroids[b & 15] * Centroids[b & 15]));
            }

            return t;
        }

        public TurboCodec(int dimension, DistanceMetric metric, int headDimensions)
            : base(dimension, metric)
        {
            _head = Math.Min(headDimensions, dimension);
        }

        private int TailDim => Dimension - _head;

        public override int EncodedLength => 8 + (_head * 4) + ((TailDim + 1) / 2);

        private static int Quantize(double v)
        {
            if (v >= 0)
            {
                if (v >= 0.9495)
                {
                    if (v >= 1.6405)
                    {
                        return v >= 2.1225 ? 15 : 14;
                    }

                    return v >= 1.268 ? 13 : 12;
                }

                if (v >= 0.393)
                {
                    return v >= 0.662 ? 11 : 10;
                }

                return v >= 0.164 ? 9 : 8;
            }

            if (v < -0.9495)
            {
                if (v < -1.6405)
                {
                    return v < -2.1225 ? 0 : 1;
                }

                return v < -1.268 ? 2 : 3;
            }

            if (v < -0.393)
            {
                return v < -0.662 ? 4 : 5;
            }

            return v < -0.164 ? 6 : 7;
        }

        public override byte[] Encode(ReadOnlySpan<float> vector)
        {
            var bytes = new byte[EncodedLength];
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), (uint)_head);
            for (int i = 0; i < _head; i++)
            {
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(8 + (i * 4), 4), vector[i]);
            }

            int tailDim = TailDim;
            if (tailDim == 0)
            {
                return bytes;
            }

            var tail = ScratchFloats(TailDim);
            vector.Slice(_head).CopyTo(tail);
            TurboRotation.Rotate(tail);
            double normSq = 0;
            foreach (var x in tail)
            {
                normSq += (double)x * x;
            }

            float norm = (float)Math.Sqrt(normSq);
            if (norm < 1e-9f)
            {
                return bytes;
            }

            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(0, 4), norm);
            double mult = Math.Sqrt(tailDim) / norm;
            int packedOffset = 8 + (_head * 4);
            for (int p = 0; p < tailDim / 2; p++)
            {
                int b1 = Quantize(tail[p * 2] * mult);
                int b2 = Quantize(tail[(p * 2) + 1] * mult);
                bytes[packedOffset + p] = (byte)((b1 << 4) | b2);
            }

            if ((tailDim & 1) == 1)
            {
                bytes[packedOffset + (tailDim / 2)] = (byte)(Quantize(tail[tailDim - 1] * mult) << 4);
            }

            return bytes;
        }

        private void ReconstructRotated(ReadOnlySpan<byte> stored, Span<double> output)
        {
            for (int i = 0; i < _head; i++)
            {
                output[i] = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(8 + (i * 4), 4));
            }

            int tailDim = TailDim;
            double norm = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(0, 4));
            if (tailDim == 0)
            {
                return;
            }

            if (norm < 1e-9)
            {
                output.Slice(_head).Clear();
                return;
            }

            double scale = (norm / Math.Sqrt(tailDim)) * BiasCorrection;
            int packedOffset = 8 + (_head * 4);
            for (int i = 0; i < tailDim; i++)
            {
                byte packed = stored[packedOffset + (i / 2)];
                int bin = (i & 1) == 1 ? packed & 0x0F : packed >> 4;
                output[_head + i] = Centroids[bin] * scale;
            }
        }

        /// <summary>
        /// Squared L2 (or 1 - dot) between two TurboQuant vectors from the nibbles, without decoding: with tail scales sa, sb,
        /// <c>|a - b|^2 = sa^2 sum(ca^2) + sb^2 sum(cb^2) - 2 sa sb sum(ca cb)</c>, each sum a table lookup per byte (the squares per byte, the products per nibble pair).
        /// </summary>
        public override double PairDistance(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
        {
            if (Metric is not (DistanceMetric.L2 or DistanceMetric.Cosine or DistanceMetric.Dot))
            {
                return base.PairDistance(a, b);
            }

            double headL2 = 0, headDot = 0;
            for (int i = 0; i < _head; i++)
            {
                double x = BinaryPrimitives.ReadSingleLittleEndian(a.Slice(8 + (i * 4), 4));
                double y = BinaryPrimitives.ReadSingleLittleEndian(b.Slice(8 + (i * 4), 4));
                headL2 += (x - y) * (x - y);
                headDot += x * y;
            }

            int tailDim = TailDim;
            double normA = BinaryPrimitives.ReadSingleLittleEndian(a.Slice(0, 4));
            double normB = BinaryPrimitives.ReadSingleLittleEndian(b.Slice(0, 4));
            double sa = tailDim == 0 || normA < 1e-9 ? 0.0 : (normA / Math.Sqrt(tailDim)) * BiasCorrection;
            double sb = tailDim == 0 || normB < 1e-9 ? 0.0 : (normB / Math.Sqrt(tailDim)) * BiasCorrection;

            int packedOffset = 8 + (_head * 4);
            int pairs = tailDim / 2;
            float squaresA = 0, squaresB = 0, products = 0;
            for (int p = 0; p < pairs; p++)
            {
                byte x = a[packedOffset + p], y = b[packedOffset + p];
                squaresA += PairSquares[x];
                squaresB += PairSquares[y];
                products += CentroidProducts[((x >> 4) << 4) + (y >> 4)] + CentroidProducts[((x & 15) << 4) + (y & 15)];
            }

            double sqA = squaresA, sqB = squaresB, prod = products;
            if ((tailDim & 1) == 1)
            {
                // The unused low nibble of the last byte is not a coordinate.
                double ca = Centroids[a[packedOffset + pairs] >> 4], cb = Centroids[b[packedOffset + pairs] >> 4];
                sqA += ca * ca;
                sqB += cb * cb;
                prod += ca * cb;
            }

            double tailDot = sa * sb * prod;
            if (Metric == DistanceMetric.Dot)
            {
                return 1.0 - (headDot + tailDot);
            }

            return headL2 + Math.Max((sa * sa * sqA) + (sb * sb * sqB) - (2.0 * tailDot), 0.0);
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
            ReconstructRotated(stored, buf.AsSpan(0, Dimension));
            var tail = ScratchFloats(TailDim);
            for (int i = 0; i < tail.Length; i++)
            {
                tail[i] = (float)buf[_head + i];
            }

            TurboRotation.Unrotate(tail);
            for (int i = 0; i < destination.Length; i++)
            {
                destination[i] = i < _head ? (float)buf[i] : tail[i - _head];
            }
        }

        public override PreparedQuery Prepare(ReadOnlySpan<float> query, PreparedQuery? reuse = null)
        {
            var q = base.Prepare(query, reuse);
            var tail = TailDim == 0 ? Array.Empty<float>() : q.Rotated is { } r && r.Length == TailDim ? r : new float[TailDim];
            query.Slice(_head).CopyTo(tail);
            TurboRotation.Rotate(tail);
            q.Rotated = tail;
            double sq = 0;
            foreach (var x in tail)
            {
                sq += (double)x * x;
            }

            q.TailNormSq = sq;
            if (Metric is DistanceMetric.L2 or DistanceMetric.Cosine or DistanceMetric.Dot && tail.Length >= 2)
            {
                q.Table = QuantSimd.BuildNibblePairTable(tail, CentroidsF, tail.Length / 2, q.Table);
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
                    break;
                default:
                    {
                        // Back to the original basis for the geometries that are not rotation invariant.
                        var buf = Scratch(Dimension);
                        ReconstructRotated(stored, buf.AsSpan(0, Dimension));
                        var tail = ScratchFloats(TailDim);
                        for (int i = 0; i < tail.Length; i++)
                        {
                            tail[i] = (float)buf[_head + i];
                        }

                        TurboRotation.Unrotate(tail);
                        for (int i = 0; i < tail.Length; i++)
                        {
                            buf[_head + i] = tail[i];
                        }

                        return DistanceFromDecoded(buf.AsSpan(0, Dimension), query, 0, 0);
                    }
            }

            var q = query.Coords;
            var rotated = query.Rotated!;
            double headDistSq = 0, headDot = 0;
            for (int i = 0; i < _head; i++)
            {
                double h = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(8 + (i * 4), 4));
                double diff = h - q[i];
                headDistSq += diff * diff;
                headDot += h * q[i];
            }

            int tailDim = TailDim;
            double norm = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(0, 4));
            if (tailDim == 0 || norm < 1e-9)
            {
                double tailQ = 0;
                foreach (var x in rotated)
                {
                    tailQ += (double)x * x;
                }

                return Metric == DistanceMetric.L2 ? headDistSq + tailQ
                     : Metric == DistanceMetric.Dot ? 1.0 - headDot
                     : Math.Max(2.0 * (1.0 - headDot), 0.0);
            }

            double scale = (norm / Math.Sqrt(tailDim)) * BiasCorrection;
            int packedOffset = 8 + (_head * 4);

            // The tail is a sum over packed bytes (two dimensions each): one table lookup per byte gives its dot product with the query,
            // and a fixed table gives the squared norm of the codes, so L2 follows from |v - q|^2 = |v|^2 - 2 v.q + |q|^2.
            int pairs = tailDim / 2;
            var packed = stored.Slice(packedOffset, (tailDim + 1) / 2);
            double tailDot = pairs > 0 ? QuantSimd.SumTable(packed.Slice(0, pairs), query.Table!) : 0.0;
            double codeSq = 0;
            if (Metric == DistanceMetric.L2)
            {
                float sqAcc = 0;
                for (int p = 0; p < pairs; p++)
                {
                    sqAcc += PairSquares[packed[p]];
                }

                codeSq = sqAcc;
            }

            if ((tailDim & 1) == 1)
            {
                // The unused low nibble of the last byte is not a coordinate.
                double c = Centroids[packed[pairs] >> 4];
                tailDot += c * rotated[tailDim - 1];
                codeSq += c * c;
            }

            if (Metric == DistanceMetric.L2)
            {
                return headDistSq + Math.Max((scale * scale * codeSq) - (2.0 * scale * tailDot) + query.TailNormSq, 0.0);
            }

            double dot = headDot + (tailDot * scale);
            return Metric == DistanceMetric.Dot ? 1.0 - dot : Math.Max(2.0 * (1.0 - dot), 0.0);
        }
    }

    /// <summary>
    /// A float32 head (e.g. the Lorentz block of a hybrid vector) followed by an 8-bit tail.
    /// Layout: <c>float[head] | float tailScale | sbyte[tail]</c>.
    /// </summary>
    internal sealed class AsymmetricHybridCodec : VectorCodec
    {
        private readonly int _head;

        public AsymmetricHybridCodec(int dimension, DistanceMetric metric, int headDimensions)
            : base(dimension, metric)
        {
            _head = Math.Min(headDimensions, dimension);
        }

        private int TailDim => Dimension - _head;

        public override int EncodedLength => (_head * 4) + 4 + TailDim;

        public override bool SupportsTruncation => true;

        public override byte[] Encode(ReadOnlySpan<float> vector)
        {
            var bytes = new byte[EncodedLength];
            for (int i = 0; i < _head; i++)
            {
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4, 4), vector[i]);
            }

            double max = 0;
            for (int i = _head; i < Dimension; i++)
            {
                max = Math.Max(max, Math.Abs(vector[i]));
            }

            double scale = Math.Max(max, 1e-12);
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(_head * 4, 4), (float)scale);
            double inv = 127.0 / scale;
            for (int i = 0; i < TailDim; i++)
            {
                bytes[(_head * 4) + 4 + i] = (byte)(sbyte)Math.Clamp(Math.Round(vector[_head + i] * inv, MidpointRounding.AwayFromZero), -127.0, 127.0);
            }

            return bytes;
        }

        private void Reconstruct(ReadOnlySpan<byte> stored, Span<double> output)
        {
            for (int i = 0; i < _head; i++)
            {
                output[i] = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(i * 4, 4));
            }

            double step = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(_head * 4, 4)) / 127.0;
            for (int i = 0; i < TailDim; i++)
            {
                output[_head + i] = (sbyte)stored[(_head * 4) + 4 + i] * step;
            }
        }

        public override void Decode(ReadOnlySpan<byte> stored, Span<float> destination)
        {
            var buf = Scratch(Dimension);
            Reconstruct(stored, buf.AsSpan(0, Dimension));
            for (int i = 0; i < destination.Length; i++)
            {
                destination[i] = (float)buf[i];
            }
        }

        public override double Distance(ReadOnlySpan<byte> stored, PreparedQuery query, int truncation = 0)
        {
            var buf = Scratch(Dimension);
            Reconstruct(stored, buf.AsSpan(0, Dimension));
            return DistanceFromDecoded(buf.AsSpan(0, Dimension), query, 0, truncation);
        }
    }
}
