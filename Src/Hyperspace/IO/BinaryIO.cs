// <copyright file="BinaryIO.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Buffers;
    using System.Buffers.Binary;
    using System.IO;
    using System.Runtime.InteropServices;
    using System.Text;

    /// <summary>Little-endian primitives over an <see cref="IBufferWriter{T}"/>, the writer the flat formats are produced through.</summary>
    internal static class BufferWriterExtensions
    {
        public static void WriteInt32(this IBufferWriter<byte> writer, int value)
        {
            var span = writer.GetSpan(4);
            BinaryPrimitives.WriteInt32LittleEndian(span, value);
            writer.Advance(4);
        }

        public static void WriteInt64(this IBufferWriter<byte> writer, long value)
        {
            var span = writer.GetSpan(8);
            BinaryPrimitives.WriteInt64LittleEndian(span, value);
            writer.Advance(8);
        }

        public static void WriteSingle(this IBufferWriter<byte> writer, float value)
        {
            var span = writer.GetSpan(4);
            BinaryPrimitives.WriteSingleLittleEndian(span, value);
            writer.Advance(4);
        }

        public static void WriteByte(this IBufferWriter<byte> writer, byte value)
        {
            var span = writer.GetSpan(1);
            span[0] = value;
            writer.Advance(1);
        }

        public static void WriteBytes(this IBufferWriter<byte> writer, ReadOnlySpan<byte> value)
        {
            while (!value.IsEmpty)
            {
                var span = writer.GetSpan(1);
                int n = Math.Min(span.Length, value.Length);
                value.Slice(0, n).CopyTo(span);
                writer.Advance(n);
                value = value.Slice(n);
            }
        }

        /// <summary>Writes a run of 32-bit integers (a span copy on little-endian machines).</summary>
        public static void WriteInt32Span(this IBufferWriter<byte> writer, ReadOnlySpan<int> values)
        {
            if (BitConverter.IsLittleEndian)
            {
                writer.WriteBytes(MemoryMarshal.AsBytes(values));
                return;
            }

            foreach (var v in values)
            {
                writer.WriteInt32(v);
            }
        }

        /// <summary>Writes a run of 32-bit floats (a span copy on little-endian machines).</summary>
        public static void WriteSingleSpan(this IBufferWriter<byte> writer, ReadOnlySpan<float> values)
        {
            if (BitConverter.IsLittleEndian)
            {
                writer.WriteBytes(MemoryMarshal.AsBytes(values));
                return;
            }

            foreach (var v in values)
            {
                writer.WriteSingle(v);
            }
        }

        /// <summary>Writes a length-prefixed (int32) UTF-8 string.</summary>
        public static void WriteString(this IBufferWriter<byte> writer, string value)
        {
            int count = Encoding.UTF8.GetByteCount(value);
            writer.WriteInt32(count);
            var span = writer.GetSpan(count);
            Encoding.UTF8.GetBytes(value, span);
            writer.Advance(count);
        }
    }

    /// <summary>An <see cref="IBufferWriter{T}"/> that drains into a <see cref="Stream"/> through a pooled buffer of fixed size.</summary>
    internal sealed class StreamBufferWriter : IBufferWriter<byte>, IDisposable
    {
        private const int BufferSize = 64 * 1024;
        private readonly Stream _stream;
        private byte[] _buffer;
        private int _written;

        public StreamBufferWriter(Stream stream)
        {
            _stream = stream;
            _buffer = ArrayPool<byte>.Shared.Rent(BufferSize);
        }

        public void Advance(int count) => _written += count;

        public Memory<byte> GetMemory(int sizeHint = 0)
        {
            Ensure(sizeHint);
            return _buffer.AsMemory(_written);
        }

        public Span<byte> GetSpan(int sizeHint = 0)
        {
            Ensure(sizeHint);
            return _buffer.AsSpan(_written);
        }

        public void Flush()
        {
            if (_written > 0)
            {
                _stream.Write(_buffer, 0, _written);
                _written = 0;
            }
        }

        public void Dispose()
        {
            Flush();
            var b = _buffer;
            _buffer = Array.Empty<byte>();
            if (b.Length > 0)
            {
                ArrayPool<byte>.Shared.Return(b);
            }
        }

        private void Ensure(int sizeHint)
        {
            if (sizeHint < 1)
            {
                sizeHint = 1;
            }

            if (_buffer.Length - _written >= sizeHint)
            {
                return;
            }

            Flush();
            if (_buffer.Length < sizeHint)
            {
                ArrayPool<byte>.Shared.Return(_buffer);
                _buffer = ArrayPool<byte>.Shared.Rent(sizeHint);
            }
        }
    }

    /// <summary>Reads little-endian primitives from a stream.</summary>
    internal sealed class StreamReader2
    {
        private readonly Stream _stream;
        private readonly byte[] _scratch = new byte[8];

        public StreamReader2(Stream stream)
        {
            _stream = stream;
        }

        public void ReadExactly(Span<byte> destination) => _stream.ReadExactly(destination);

        public byte ReadByte()
        {
            int b = _stream.ReadByte();
            if (b < 0)
            {
                throw new EndOfStreamException();
            }

            return (byte)b;
        }

        public int ReadInt32()
        {
            _stream.ReadExactly(_scratch, 0, 4);
            return BinaryPrimitives.ReadInt32LittleEndian(_scratch);
        }

        public long ReadInt64()
        {
            _stream.ReadExactly(_scratch, 0, 8);
            return BinaryPrimitives.ReadInt64LittleEndian(_scratch);
        }

        public float ReadSingle()
        {
            _stream.ReadExactly(_scratch, 0, 4);
            return BinaryPrimitives.ReadSingleLittleEndian(_scratch);
        }

        public void ReadInt32Span(Span<int> destination)
        {
            _stream.ReadExactly(MemoryMarshal.AsBytes(destination));
            if (!BitConverter.IsLittleEndian)
            {
                for (int i = 0; i < destination.Length; i++)
                {
                    destination[i] = BinaryPrimitives.ReverseEndianness(destination[i]);
                }
            }
        }

        public void ReadSingleSpan(Span<float> destination)
        {
            _stream.ReadExactly(MemoryMarshal.AsBytes(destination));
            if (!BitConverter.IsLittleEndian)
            {
                for (int i = 0; i < destination.Length; i++)
                {
                    destination[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReverseEndianness(BitConverter.SingleToInt32Bits(destination[i])));
                }
            }
        }

        public string ReadString(int maxLength = 64 * 1024 * 1024)
        {
            int count = ReadInt32();
            if (count < 0 || count > maxLength)
            {
                throw new InvalidDataException($"Invalid string length {count}");
            }

            if (count == 0)
            {
                return string.Empty;
            }

            var rented = ArrayPool<byte>.Shared.Rent(count);
            try
            {
                _stream.ReadExactly(rented, 0, count);
                return Encoding.UTF8.GetString(rented, 0, count);
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
        }
    }
}
