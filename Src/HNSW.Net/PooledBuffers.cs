// <copyright file="PooledBuffers.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Buffers;
    using System.Buffers.Binary;
    using System.IO;
    using System.Runtime.CompilerServices;
    using System.Runtime.InteropServices;
    using MessagePack;

    /// <summary>
    /// An <see cref="IBufferWriter{T}"/> that buffers into a single array rented from
    /// <see cref="ArrayPool{T}.Shared"/> and flushes it to a <see cref="Stream"/> whenever it fills up.
    /// </summary>
    /// <remarks>
    /// Peak memory is one rented chunk regardless of how large the graph is, which is what lets a graph be
    /// written straight to its destination stream instead of being staged in a growing
    /// <see cref="MemoryStream"/> (or a temporary file) first.
    /// </remarks>
    internal sealed class PooledStreamBufferWriter : IBufferWriter<byte>, IDisposable
    {
        private const int CHUNK_SIZE = 128 * 1024;

        private readonly Stream _stream;
        private byte[] _buffer;
        private int _written;

        internal PooledStreamBufferWriter(Stream stream)
        {
            _stream = stream;
            _buffer = ArrayPool<byte>.Shared.Rent(CHUNK_SIZE);
        }

        public void Advance(int count)
        {
            _written += count;
        }

        public Memory<byte> GetMemory(int sizeHint = 0) => EnsureRoom(sizeHint).AsMemory(_written);

        public Span<byte> GetSpan(int sizeHint = 0) => EnsureRoom(sizeHint).AsSpan(_written);

        /// <summary>
        /// Pushes everything buffered so far to the underlying stream. Called on dispose, so a caller only
        /// needs this when it has to observe the stream's position before the writer goes away.
        /// </summary>
        internal void Flush()
        {
            if (_written > 0)
            {
                _stream.Write(_buffer, 0, _written);
                _written = 0;
            }
        }

        private byte[] EnsureRoom(int sizeHint)
        {
            if (sizeHint <= 0) sizeHint = 1;

            if (_buffer.Length - _written >= sizeHint) return _buffer;

            Flush();

            if (_buffer.Length < sizeHint)
            {
                // A single write larger than the chunk (only possible through GetSpan with a big hint):
                // swap in a rental that fits rather than failing.
                ArrayPool<byte>.Shared.Return(_buffer);
                _buffer = ArrayPool<byte>.Shared.Rent(sizeHint);
            }

            return _buffer;
        }

        public void Dispose()
        {
            if (_buffer is null) return;

            Flush();
            ArrayPool<byte>.Shared.Return(_buffer);
            _buffer = null;
        }
    }

    /// <summary>
    /// Reads the fixed-width primitives of the flat graph format out of a <see cref="Stream"/> through a
    /// single array rented from <see cref="ArrayPool{T}.Shared"/>.
    /// </summary>
    internal sealed class PooledStreamBufferReader : IDisposable
    {
        private const int CHUNK_SIZE = 128 * 1024;

        private readonly Stream _stream;
        private byte[] _buffer;
        private int _start;
        private int _end;

        internal PooledStreamBufferReader(Stream stream)
        {
            _stream = stream;
            _buffer = ArrayPool<byte>.Shared.Rent(CHUNK_SIZE);
        }

        internal int ReadInt32()
        {
            var span = Take(sizeof(int));
            return BinaryPrimitives.ReadInt32LittleEndian(span);
        }

        internal long ReadInt64()
        {
            var span = Take(sizeof(long));
            return BinaryPrimitives.ReadInt64LittleEndian(span);
        }

        /// <summary>
        /// Reads a length-prefixed MessagePack value.
        /// </summary>
        /// <remarks>
        /// The length is what makes the whole format strictly forward-reading:
        /// <c>MessagePackSerializer.Deserialize(Stream)</c> buffers ahead and rewinds afterwards, which needs
        /// a seekable stream and would leave this reader's own position wrong.
        /// </remarks>
        internal T ReadMessagePackBlock<T>(int maximumLength)
        {
            int length = ReadInt32();

            if (length < 0 || length > maximumLength)
            {
                throw new InvalidDataException($"Invalid HNSW graph: a {length} byte header block is not plausible");
            }

            var buffer = ArrayPool<byte>.Shared.Rent(length);

            try
            {
                ReadBytes(buffer.AsSpan(0, length));
                return MessagePackSerializer.Deserialize<T>(buffer.AsMemory(0, length));
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(buffer);
            }
        }

        internal void ReadBytes(Span<byte> destination)
        {
            while (!destination.IsEmpty)
            {
                if (Buffered() == 0) Refill(1);

                int count = Math.Min(destination.Length, Buffered());
                _buffer.AsSpan(_start, count).CopyTo(destination);

                _start      += count;
                destination =  destination.Slice(count);
            }
        }

        /// <summary>
        /// Fills <paramref name="destination"/> from the stream. The destination is written in place - for
        /// deserialization it points straight into the flattened connection storage, so no intermediate copy
        /// of a node's connections is ever made.
        /// </summary>
        internal void ReadInt32Span(Span<int> destination)
        {
            while (!destination.IsEmpty)
            {
                var available = Buffered();

                if (available < sizeof(int))
                {
                    Refill(sizeof(int));
                    available = Buffered();
                }

                int count = Math.Min(destination.Length, available / sizeof(int));
                var source = _buffer.AsSpan(_start, count * sizeof(int));

                PooledBuffers.ReadInt32LittleEndian(source, destination.Slice(0, count));

                _start      += count * sizeof(int);
                destination =  destination.Slice(count);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int Buffered() => _end - _start;

        private ReadOnlySpan<byte> Take(int count)
        {
            if (Buffered() < count) Refill(count);

            var span = _buffer.AsSpan(_start, count);
            _start += count;
            return span;
        }

        private void Refill(int required)
        {
            var buffered = Buffered();

            if (buffered > 0 && _start > 0)
            {
                _buffer.AsSpan(_start, buffered).CopyTo(_buffer);
            }

            _start = 0;
            _end   = buffered;

            while (_end < required)
            {
                int read = _stream.Read(_buffer, _end, _buffer.Length - _end);

                if (read <= 0)
                {
                    throw new InvalidDataException("Unexpected end of stream while reading the HNSW graph: the data is truncated or corrupted");
                }

                _end += read;
            }
        }

        public void Dispose()
        {
            if (_buffer is null) return;

            ArrayPool<byte>.Shared.Return(_buffer);
            _buffer = null;
        }
    }

    /// <summary>
    /// Little-endian bulk conversion between <see cref="int"/> spans and their serialized bytes.
    /// </summary>
    internal static class PooledBuffers
    {
        /// <summary>
        /// The largest span handed to <see cref="IBufferWriter{T}.GetSpan"/> in one go. Bounded so a writer
        /// backed by a growing contiguous buffer is not asked for a graph-sized allocation at once.
        /// </summary>
        private const int MAX_WRITE_CHUNK_BYTES = 64 * 1024;

        internal static void WriteInt32(this IBufferWriter<byte> writer, int value)
        {
            BinaryPrimitives.WriteInt32LittleEndian(writer.GetSpan(sizeof(int)), value);
            writer.Advance(sizeof(int));
        }

        internal static void WriteInt64(this IBufferWriter<byte> writer, long value)
        {
            BinaryPrimitives.WriteInt64LittleEndian(writer.GetSpan(sizeof(long)), value);
            writer.Advance(sizeof(long));
        }

        internal static void WriteInt32Span(this IBufferWriter<byte> writer, ReadOnlySpan<int> values)
        {
            while (!values.IsEmpty)
            {
                int count       = Math.Min(values.Length, MAX_WRITE_CHUNK_BYTES / sizeof(int));
                var destination = writer.GetSpan(count * sizeof(int));

                // GetSpan may hand back more than asked for; only the requested prefix is advanced.
                WriteInt32LittleEndian(values.Slice(0, count), destination);

                writer.Advance(count * sizeof(int));
                values = values.Slice(count);
            }
        }

        internal static void WriteInt32LittleEndian(ReadOnlySpan<int> source, Span<byte> destination)
        {
            if (BitConverter.IsLittleEndian)
            {
                MemoryMarshal.AsBytes(source).CopyTo(destination);
                return;
            }

            for (int i = 0; i < source.Length; i++)
            {
                BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(i * sizeof(int)), source[i]);
            }
        }

        internal static void ReadInt32LittleEndian(ReadOnlySpan<byte> source, Span<int> destination)
        {
            if (BitConverter.IsLittleEndian)
            {
                source.CopyTo(MemoryMarshal.AsBytes(destination));
                return;
            }

            for (int i = 0; i < destination.Length; i++)
            {
                destination[i] = BinaryPrimitives.ReadInt32LittleEndian(source.Slice(i * sizeof(int)));
            }
        }
    }
}
