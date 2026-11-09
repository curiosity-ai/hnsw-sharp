// <copyright file="Int4Codec.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Buffers.Binary;

    /// <summary>
    /// 4-bit scalar quantization in blocks of 16. Layout (little-endian):
    /// <c>float[head] | uint blockCount | float[blockCount] scales | byte[blockCount * 8] nibbles</c>.
    /// </summary>
    /// <remarks>
    /// The first <c>head</c> coordinates (the Lorentz part of a hybrid vector) stay in float32. Each block of 16 tail coordinates
    /// gets <c>scale = max|x| / 7</c> and nibbles <c>round(x / scale) + 8</c>, two per byte, high nibble first.
    /// </remarks>
    internal sealed class Int4Codec : VectorCodec
    {
        private const int BlockSize = 16;
        private readonly int _head;

        public Int4Codec(int dimension, DistanceMetric metric, int headDimensions)
            : base(dimension, metric)
        {
            _head = Math.Min(headDimensions, dimension);
        }

        private int TailDim => Dimension - _head;

        private int Blocks => (TailDim + BlockSize - 1) / BlockSize;

        public override int EncodedLength => (_head * 4) + 4 + (Blocks * 4) + (Blocks * (BlockSize / 2));

        public override byte[] Encode(ReadOnlySpan<float> vector)
        {
            var bytes = new byte[EncodedLength];
            int pos = 0;
            for (int i = 0; i < _head; i++, pos += 4)
            {
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(pos, 4), vector[i]);
            }

            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(pos, 4), (uint)Blocks);
            pos += 4;
            int scalesOffset = pos;
            int packedOffset = scalesOffset + (Blocks * 4);

            Span<int> q = stackalloc int[BlockSize];
            for (int b = 0; b < Blocks; b++)
            {
                int start = b * BlockSize;
                int end = Math.Min(start + BlockSize, TailDim);
                double max = 0;
                for (int i = start; i < end; i++)
                {
                    max = Math.Max(max, Math.Abs(vector[_head + i]));
                }

                double scale = max > 1e-9 ? max / 7.0 : 1.0;
                BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(scalesOffset + (b * 4), 4), (float)scale);

                double inv = 1.0 / scale;
                for (int i = 0; i < BlockSize; i++)
                {
                    int idx = start + i;
                    q[i] = idx < TailDim ? (int)Math.Clamp(Math.Round(vector[_head + idx] * inv, MidpointRounding.AwayFromZero), -7.0, 7.0) : 0;
                }

                for (int pair = 0; pair < BlockSize / 2; pair++)
                {
                    int q1 = (q[pair * 2] + 8) & 0x0F;
                    int q2 = (q[(pair * 2) + 1] + 8) & 0x0F;
                    bytes[packedOffset + (b * (BlockSize / 2)) + pair] = (byte)((q1 << 4) | q2);
                }
            }

            return bytes;
        }

        private void Reconstruct(ReadOnlySpan<byte> stored, Span<double> output)
        {
            for (int i = 0; i < _head; i++)
            {
                output[i] = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(i * 4, 4));
            }

            int pos = _head * 4;
            int blocks = (int)BinaryPrimitives.ReadUInt32LittleEndian(stored.Slice(pos, 4));
            int scalesOffset = pos + 4;
            int packedOffset = scalesOffset + (blocks * 4);
            for (int b = 0; b < blocks; b++)
            {
                double scale = BinaryPrimitives.ReadSingleLittleEndian(stored.Slice(scalesOffset + (b * 4), 4));
                int start = b * BlockSize;
                int end = Math.Min(start + BlockSize, TailDim);
                for (int i = start; i < end; i++)
                {
                    int rel = i - start;
                    byte packed = stored[packedOffset + (b * (BlockSize / 2)) + (rel / 2)];
                    int u = (rel & 1) == 1 ? packed & 0x0F : packed >> 4;
                    output[_head + i] = (u - 8) * scale;
                }
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

        public override bool SupportsTruncation => true;

        public override double Distance(ReadOnlySpan<byte> stored, PreparedQuery query, int truncation = 0)
        {
            var buf = Scratch(Dimension);
            Reconstruct(stored, buf.AsSpan(0, Dimension));
            return DistanceFromDecoded(buf.AsSpan(0, Dimension), query, 0, truncation);
        }
    }
}
