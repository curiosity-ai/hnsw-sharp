// <copyright file="GraphSerializationTests.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net.Tests
{
    using System;
    using System.Buffers;
    using System.Collections.Generic;
    using System.Globalization;
    using System.IO;
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    /// <summary>
    /// Tests for the flat ("HNSW2") graph format and for reading graphs written in the legacy
    /// MessagePack ("HNSW") one.
    /// </summary>
    [TestClass]
    public class GraphSerializationTests
    {
        private IReadOnlyList<float[]> vectors;

        [TestInitialize]
        public void TestInitialize()
        {
            var data = File.ReadAllLines(@"vectors.txt");
            vectors = data.Select(r => Array.ConvertAll(r.Split('\t'), x => float.Parse(x, CultureInfo.CurrentCulture))).ToList();
        }

        private SmallWorld<float[], float> BuildGraph(IReadOnlyList<float[]> items = null)
        {
            var parameters = new SmallWorldParameters()
            {
                M           = 15,
                LevelLambda = 1 / Math.Log(15),
            };

            var graph = new SmallWorld<float[], float>(CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, parameters);
            graph.AddItems(items ?? vectors);
            return graph;
        }

        private static SmallWorld<float[], float> Reload(IReadOnlyList<float[]> items, Stream stream)
        {
            stream.Position = 0;
            var (graph, remaining) = SmallWorld<float[], float>.DeserializeGraph(items, CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, stream);
            Assert.AreEqual(0, remaining.Length);
            return graph;
        }

        /// <summary>
        /// A graph written by the flat format and read back must have the exact same edges, and answer
        /// searches identically.
        /// </summary>
        [TestMethod]
        public void FlatFormatRoundTripsEdgesAndSearches()
        {
            var graph    = BuildGraph();
            var expected = graph.Print();

            var stream = new MemoryStream();
            graph.SerializeGraph(stream);

            var copy = Reload(vectors, stream);

            Assert.AreEqual(expected, copy.Print());

            for (int i = 0; i < vectors.Count; i += 7)
            {
                var original = graph.KNNSearch(vectors[i], 10).Select(r => r.Id).ToArray();
                var reloaded = copy.KNNSearch(vectors[i], 10).Select(r => r.Id).ToArray();
                CollectionAssert.AreEqual(original, reloaded, $"search results differ for vector {i}");
            }
        }

        /// <summary>
        /// Serialization must not depend on whether the graph happens to be flattened: a freshly built graph
        /// holds its connections as lists, one that was loaded (or optimized) holds them in the flat cache,
        /// and both have to produce the same bytes.
        /// </summary>
        [TestMethod]
        public void FlatFormatIsIndependentOfWhetherTheGraphIsOptimized()
        {
            var graph = BuildGraph();

            var beforeOptimize = new MemoryStream();
            graph.SerializeGraph(beforeOptimize);

            graph.OptimizeIfNeeded(force: true);

            var afterOptimize = new MemoryStream();
            graph.SerializeGraph(afterOptimize);

            CollectionAssert.AreEqual(beforeOptimize.ToArray(), afterOptimize.ToArray());
        }

        /// <summary>
        /// Serializing must leave the graph alone - the MessagePack path used to hydrate every node's
        /// connections as a side effect, and then re-flatten the whole graph to undo it.
        /// </summary>
        [TestMethod]
        public void SerializingDoesNotChangeTheGraph()
        {
            var graph = BuildGraph();
            graph.OptimizeIfNeeded(force: true);

            var expected = graph.Print();

            for (int i = 0; i < 3; i++)
            {
                var stream = new MemoryStream();
                graph.SerializeGraph(stream);
                Assert.AreEqual(expected, graph.Print());
            }
        }

        /// <summary>
        /// The <see cref="IBufferWriter{T}"/> overload is the same format as the stream one, byte for byte.
        /// </summary>
        [TestMethod]
        public void BufferWriterOverloadWritesTheSameBytes()
        {
            var graph = BuildGraph();

            var stream = new MemoryStream();
            graph.SerializeGraph(stream);

            var buffer = new ArrayBufferWriter<byte>();
            graph.SerializeGraph(buffer);

            CollectionAssert.AreEqual(stream.ToArray(), buffer.WrittenSpan.ToArray());

            var copy = Reload(vectors, new MemoryStream(buffer.WrittenSpan.ToArray()));
            Assert.AreEqual(graph.Print(), copy.Print());
        }

        /// <summary>
        /// Graphs persisted before the flat format exists must keep loading.
        /// </summary>
        [TestMethod]
        public void LegacyMessagePackFormatIsStillReadable()
        {
            var graph    = BuildGraph();
            var expected = graph.Print();

            var stream = new MemoryStream();
            graph.SerializeGraphLegacy(stream);

            var copy = Reload(vectors, stream);

            Assert.AreEqual(expected, copy.Print());
        }

        /// <summary>
        /// Both formats describe the same graph, so a legacy-written graph and a flat-written one load into
        /// the same edges.
        /// </summary>
        [TestMethod]
        public void BothFormatsLoadIntoTheSameGraph()
        {
            var graph = BuildGraph();

            var legacy = new MemoryStream();
            graph.SerializeGraphLegacy(legacy);

            var flat = new MemoryStream();
            graph.SerializeGraph(flat);

            Assert.AreEqual(Reload(vectors, legacy).Print(), Reload(vectors, flat).Print());
        }

        /// <summary>
        /// Items the persisted graph does not cover are handed back so the caller can re-add them - this is
        /// how a crash between "vectors written" and "graph written" is recovered from.
        /// </summary>
        [TestMethod]
        public void ItemsBeyondTheGraphAreReturnedAsRemaining()
        {
            int indexed = vectors.Count / 2;

            var graph = BuildGraph(vectors.Take(indexed).ToArray());

            var stream = new MemoryStream();
            graph.SerializeGraph(stream);
            stream.Position = 0;

            var (copy, remaining) = SmallWorld<float[], float>.DeserializeGraph(vectors, CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, stream);

            Assert.AreEqual(vectors.Count - indexed, remaining.Length);
            Assert.AreEqual(indexed, copy.Items.Count);
            CollectionAssert.AreEqual(vectors[indexed], remaining[0]);
        }

        /// <summary>
        /// A single item is the degenerate graph: one node, and it is its own entry point.
        /// </summary>
        [TestMethod]
        public void SingleItemGraphRoundTrips()
        {
            var graph = BuildGraph(new[] { vectors[0] });

            var stream = new MemoryStream();
            graph.SerializeGraph(stream);

            var copy = Reload(new[] { vectors[0] }, stream);

            Assert.AreEqual(graph.Print(), copy.Print());
            Assert.AreEqual(1, copy.KNNSearch(vectors[0], 5).Count);
        }

        /// <summary>
        /// Truncated data is reported as such instead of producing a graph with missing edges.
        /// </summary>
        [TestMethod]
        public void TruncatedStreamIsRejected()
        {
            var graph = BuildGraph();

            var stream = new MemoryStream();
            graph.SerializeGraph(stream);

            var truncated = new MemoryStream(stream.ToArray().AsSpan(0, (int)(stream.Length * 0.6)).ToArray());

            Assert.ThrowsExactly<InvalidDataException>(() =>
                SmallWorld<float[], float>.DeserializeGraph(vectors, CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, truncated));
        }

        /// <summary>
        /// A stream that is not a graph at all keeps failing with the header error, and leaves the stream
        /// where it found it so a caller can try another reader.
        /// </summary>
        [TestMethod]
        public void UnknownHeaderIsRejectedAndTheStreamIsRewound()
        {
            var stream = new MemoryStream();
            stream.Write(new byte[] { 0xA4, (byte)'N', (byte)'O', (byte)'P', (byte)'E', 1, 2, 3, 4 });
            stream.Position = 0;

            Assert.ThrowsExactly<InvalidDataException>(() =>
                SmallWorld<float[], float>.DeserializeGraph(vectors, CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, stream));

            Assert.AreEqual(0, stream.Position);
        }

        /// <summary>
        /// Loading reads the stream strictly forwards, so a graph can be read straight off a
        /// non-seekable source.
        /// </summary>
        [TestMethod]
        public void LoadsFromANonSeekableStream()
        {
            var graph = BuildGraph();

            var stream = new MemoryStream();
            graph.SerializeGraph(stream);

            var copy = SmallWorld<float[], float>.DeserializeGraph(vectors, CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, new ForwardOnlyStream(stream.ToArray()));

            Assert.AreEqual(graph.Print(), copy.Graph.Print());
        }

        /// <summary>A read-only, non-seekable stream over a byte array.</summary>
        private sealed class ForwardOnlyStream : Stream
        {
            private readonly byte[] _data;
            private int _position;

            internal ForwardOnlyStream(byte[] data) => _data = data;

            public override bool CanRead  => true;
            public override bool CanSeek  => false;
            public override bool CanWrite => false;

            public override long Length   => throw new NotSupportedException();
            public override long Position { get => throw new NotSupportedException(); set => throw new NotSupportedException(); }

            public override int Read(byte[] buffer, int offset, int count)
            {
                // Deliberately stingy, so every reader has to cope with a partial read.
                int read = Math.Min(Math.Min(count, 7), _data.Length - _position);
                _data.AsSpan(_position, read).CopyTo(buffer.AsSpan(offset));
                _position += read;
                return read;
            }

            public override void Flush() { }
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
            public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        }

        /// <summary>
        /// A neighbour id that does not name a node is caught while loading, so a corrupted file becomes a
        /// rebuild instead of an index-out-of-range thrown from inside a search later.
        /// </summary>
        [TestMethod]
        public void CorruptedNeighbourIdIsRejected()
        {
            var graph = BuildGraph();

            var stream = new MemoryStream();
            graph.SerializeGraph(stream);

            var bytes = stream.ToArray();

            // The last int of the payload is a neighbour id; point it past the end of the graph.
            BitConverter.TryWriteBytes(bytes.AsSpan(bytes.Length - sizeof(int)), int.MaxValue);

            Assert.ThrowsExactly<InvalidDataException>(() =>
                SmallWorld<float[], float>.DeserializeGraph(vectors, CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, new MemoryStream(bytes)));
        }

        /// <summary>
        /// A graph that was loaded and then grown must serialize both its flattened nodes and the newly
        /// added, still-hydrated ones.
        /// </summary>
        [TestMethod]
        public void GraphGrownAfterLoadRoundTrips()
        {
            int indexed = vectors.Count / 2;

            var first = new MemoryStream();
            BuildGraph(vectors.Take(indexed).ToArray()).SerializeGraph(first);
            first.Position = 0;

            var (graph, remaining) = SmallWorld<float[], float>.DeserializeGraph(vectors, CosineDistance.NonOptimized, DefaultRandomGenerator.Instance, first);
            graph.AddItems(remaining);

            var expected = graph.Print();

            var second = new MemoryStream();
            graph.SerializeGraph(second);

            Assert.AreEqual(expected, Reload(vectors, second).Print());
        }
    }
}
