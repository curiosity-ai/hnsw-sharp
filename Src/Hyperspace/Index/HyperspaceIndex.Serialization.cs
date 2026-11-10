// <copyright file="HyperspaceIndex.Serialization.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Buffers;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    using System.IO;
    using System.Runtime.InteropServices;
    using System.Text;

    /// <summary>
    /// Two persistence formats, both flat little-endian and read without any intermediate object graph:
    /// <list type="bullet">
    /// <item>
    /// <description>
    /// the <b>graph</b> format (<see cref="HyperspaceIndex.SerializeGraph(Stream)"/>), the counterpart of the <c>HNSW2</c> format of the HNSW package: the
    /// parameters, then each node's links as one run of <see cref="int"/>s (layer start offsets, total, neighbour ids), without the vectors,
    /// which the caller keeps and hands back on load;
    /// </description>
    /// </item>
    /// <item>
    /// <description>
    /// the <b>snapshot</b> format (<see cref="HyperspaceIndex.Save(Stream)"/>): the graph section followed by the encoded vectors, the metadata and the property graph.
    /// </description>
    /// </item>
    /// </list>
    /// </summary>
    /// <remarks>
    /// Graph section layout:
    /// <code>
    /// byte[8]  magic "HSPG0001"
    /// int32    parametersLength, then the parameters block
    /// int32    nodeCount
    /// int32    entryPointId          // -1 when empty
    /// int32    maxLevel              // -1 when empty
    /// int64    totalRecordInts       // sum of recordLength
    /// per node, in id order:
    ///   int32    id
    ///   int32    maxLayers
    ///   int32    recordLength        // == maxLayers + 1 + neighbourCount
    ///   int32[]  record              // layer start offsets, total, then the neighbour ids
    /// int32    deletedCount, then deletedCount x int32 ids
    /// </code>
    /// A snapshot is <c>"HSPS0001"</c>, the graph section, <c>int32 vectorLength</c>, <c>nodeCount x byte[vectorLength]</c>, then (only if the parameters
    /// say the original vectors are kept) <c>nodeCount x float32[dimension]</c>, per node
    /// <c>int32 metadataCount</c> (-1 for none) with that many (string, string) pairs, and finally <c>int32 graphLength</c> followed by a
    /// <c>HYPR_CSR</c> image of the property graph (0 when empty).
    /// </remarks>
    public sealed partial class HyperspaceIndex
    {
        private static readonly byte[] GraphMagic = Encoding.ASCII.GetBytes("HSPG0001");
        private static readonly byte[] SnapshotMagic = Encoding.ASCII.GetBytes("HSPS0001");
        private const int ParametersVersion = 1;
        private const int MaximumParametersSize = 256 * 1024 * 1024;

        // ------------------------------------------------------------------ writing

        /// <summary>Writes the graph (parameters and links, no vectors) in the flat format.</summary>
        public void SerializeGraph(Stream stream)
        {
            using var writer = new StreamBufferWriter(stream);
            SerializeGraph(writer);
        }

        /// <summary>Writes the graph (parameters and links, no vectors) in the flat format to a buffer writer.</summary>
        public void SerializeGraph(IBufferWriter<byte> writer)
        {
            EnterRead();
            try
            {
                WriteGraphSection(writer);
            }
            finally
            {
                ExitRead();
            }
        }

        /// <summary>Writes a complete snapshot: graph, vectors, metadata and property graph.</summary>
        public void Save(Stream stream)
        {
            using var writer = new StreamBufferWriter(stream);
            Save(writer);
        }

        /// <summary>Writes a complete snapshot to a file.</summary>
        public void Save(string path)
        {
            string tmp = path + ".tmp." + Environment.ProcessId;
            using (var fs = new FileStream(tmp, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16))
            {
                Save(fs);
                fs.Flush(true);
            }

            File.Move(tmp, path, overwrite: true);
        }

        /// <summary>Writes a complete snapshot to a buffer writer.</summary>
        public void Save(IBufferWriter<byte> writer)
        {
            EnterRead();
            try
            {
                writer.WriteBytes(SnapshotMagic);
                WriteGraphSection(writer);

                writer.WriteInt32(_codec.EncodedLength);
                foreach (var v in _vectors)
                {
                    writer.WriteBytes(v);
                }

                if (_originals is not null)
                {
                    foreach (var original in _originals)
                    {
                        writer.WriteSingleSpan(original);
                    }
                }

                foreach (var m in _metadata)
                {
                    if (m is null)
                    {
                        writer.WriteInt32(-1);
                        continue;
                    }

                    writer.WriteInt32(m.Count);
                    foreach (var (key, value) in m)
                    {
                        writer.WriteString(key);
                        writer.WriteString(value);
                    }
                }

                using var graph = new MemoryStream();
                var csr = Graph.Dynamic.ToCsr();
                if (csr.EdgeCount > 0)
                {
                    csr.Write(graph);
                }

                writer.WriteInt32((int)graph.Length);
                writer.WriteBytes(graph.GetBuffer().AsSpan(0, (int)graph.Length));
            }
            finally
            {
                ExitRead();
            }
        }

        private void WriteGraphSection(IBufferWriter<byte> writer)
        {
            writer.WriteBytes(GraphMagic);

            var parameters = new ArrayBufferWriter<byte>();
            WriteParameters(parameters);
            writer.WriteInt32(parameters.WrittenCount);
            writer.WriteBytes(parameters.WrittenSpan);

            long totalRecordInts = 0;
            for (int id = 0; id < _nodes.Count; id++)
            {
                totalRecordInts += RecordLength(id);
            }

            writer.WriteInt32(_nodes.Count);
            writer.WriteInt32(_entryPoint);
            writer.WriteInt32(_maxLevel);
            writer.WriteInt64(totalRecordInts);

            var offsets = new int[_options.MaxLayers + 1];
            for (int id = 0; id < _nodes.Count; id++)
            {
                var node = _nodes[id];
                int layers = node.Links.Length;
                int total = 0;
                for (int l = 0; l < layers; l++)
                {
                    offsets[l] = total;
                    total += l == 0 ? Layer0Of(id).Length : node.Counts[l];
                }

                offsets[layers] = total;
                writer.WriteInt32(id);
                writer.WriteInt32(layers);
                writer.WriteInt32(layers + 1 + total);
                writer.WriteInt32Span(offsets.AsSpan(0, layers + 1));
                for (int l = 0; l < layers; l++)
                {
                    writer.WriteInt32Span(l == 0 ? Layer0Of(id) : node.Get(l));
                }
            }

            writer.WriteInt32(_deletedCount);
            foreach (int id in _deleted)
            {
                writer.WriteInt32(id);
            }
        }

        private long RecordLength(int id)
        {
            var node = _nodes[id];
            long length = node.Links.Length + 1;
            for (int l = 0; l < node.Links.Length; l++)
            {
                length += l == 0 ? Layer0Of(id).Length : node.Counts[l];
            }

            return length;
        }

        private void WriteParameters(IBufferWriter<byte> w)
        {
            var o = _options;
            w.WriteInt32(ParametersVersion);
            w.WriteInt32(o.Dimension);
            w.WriteInt32((int)o.Metric);
            w.WriteInt32((int)o.Quantization);
            w.WriteInt32(o.M);
            w.WriteInt32(o.EfConstruction);
            w.WriteInt32(o.EfSearch);
            w.WriteInt32(o.MaxLayers);
            w.WriteInt32(o.ResolvedHeadDimensions);
            w.WriteSingle((float)o.ResolvedLevelProbability);
            w.WriteSingle(o.TernaryThreshold);
            int flags = (o.FastRouting ? 1 : 0) | (o.DensityPruning ? 2 : 0) | (o.ResolvedNormalize ? 4 : 0) | (o.AnisotropicRefinement ? 8 : 0) | (o.ValidateVectors ? 16 : 0) | (o.KeepOriginalVectors ? 32 : 0);
            w.WriteInt32(flags);
            w.WriteInt32(o.ResolvedRerankOversample);
            w.WriteString(o.FusionMethod);
            w.WriteInt32((int)o.Bm25.Method);
            w.WriteSingle(o.Bm25.K1);
            w.WriteSingle(o.Bm25.B);
            w.WriteSingle(o.Bm25.Delta);
            w.WriteInt32(o.Bm25.NGrams);
            w.WriteString(o.Bm25.Language);
            w.WriteInt32(o.Graph.Enabled ? 1 : 0);
            w.WriteInt32((int)o.Graph.StorageLayout);
            w.WriteInt32(o.Graph.MaxHopsLimit);
            w.WriteInt32(o.Graph.MaxFanoutPerNode);

            if (Quantizer is { } pq)
            {
                using var ms = new MemoryStream();
                pq.Save(ms);
                w.WriteInt32((int)ms.Length);
                w.WriteBytes(ms.GetBuffer().AsSpan(0, (int)ms.Length));
            }
            else
            {
                w.WriteInt32(0);
            }
        }

        // ------------------------------------------------------------------ reading

        /// <summary>Reads a snapshot written by <see cref="Save(Stream)"/>.</summary>
        /// <param name="stream">The stream to read from.</param>
        /// <param name="tokenizer">A custom tokenizer, since delegates are not persisted.</param>
        public static HyperspaceIndex Load(Stream stream, Tokenizer? tokenizer = null)
        {
            var reader = new StreamReader2(stream);
            Span<byte> magic = stackalloc byte[8];
            reader.ReadExactly(magic);
            if (!magic.SequenceEqual(SnapshotMagic))
            {
                throw new InvalidDataException("Not a Hyperspace snapshot (bad magic)");
            }

            var section = ReadGraphSection(reader, tokenizer);
            var index = new HyperspaceIndex(section.Options);

            int vectorLength = reader.ReadInt32();
            if (vectorLength != index._codec.EncodedLength)
            {
                throw new InvalidDataException($"The snapshot stores vectors of {vectorLength} bytes but the index expects {index._codec.EncodedLength}");
            }

            var vectors = new List<byte[]>(section.Nodes.Length);
            for (int i = 0; i < section.Nodes.Length; i++)
            {
                var bytes = new byte[vectorLength];
                reader.ReadExactly(bytes);
                vectors.Add(bytes);
            }

            List<float[]>? originals = null;
            if (index._options.KeepOriginalVectors)
            {
                originals = new List<float[]>(section.Nodes.Length);
                for (int i = 0; i < section.Nodes.Length; i++)
                {
                    var original = new float[index._options.Dimension];
                    reader.ReadSingleSpan(original);
                    originals.Add(original);
                }
            }

            var metadata = new List<Dictionary<string, string>?>(section.Nodes.Length);
            for (int i = 0; i < section.Nodes.Length; i++)
            {
                int count = reader.ReadInt32();
                if (count < 0)
                {
                    metadata.Add(null);
                    continue;
                }

                var m = new Dictionary<string, string>(count, StringComparer.Ordinal);
                for (int j = 0; j < count; j++)
                {
                    string key = reader.ReadString();
                    m[key] = reader.ReadString();
                }

                metadata.Add(m);
            }

            index.Populate(section, vectors, metadata, originals);

            int graphLength = reader.ReadInt32();
            if (graphLength < 0)
            {
                throw new InvalidDataException("Invalid property graph length");
            }

            if (graphLength > 0)
            {
                var image = new byte[graphLength];
                reader.ReadExactly(image);
                var csr = CsrGraph.Read(new MemoryStream(image, writable: false));
                for (uint src = 0; src < (uint)csr.NumNodes; src++)
                {
                    foreach (var e in csr.Neighbors(src))
                    {
                        index.Graph.Dynamic.InsertEdge(src, e.Target, e.Relation, e.Weight);
                    }
                }

                index.Graph.Compact();
            }

            return index;
        }

        /// <summary>Reads a snapshot from memory.</summary>
        public static HyperspaceIndex Load(ReadOnlyMemory<byte> data, Tokenizer? tokenizer = null)
        {
            using var ms = MemoryMarshal.TryGetArray(data, out var seg) && seg.Array is not null
                ? new MemoryStream(seg.Array, seg.Offset, seg.Count, writable: false)
                : new MemoryStream(data.ToArray(), writable: false);
            return Load(ms, tokenizer);
        }

        /// <summary>Reads a snapshot from a file.</summary>
        public static HyperspaceIndex Load(string path, Tokenizer? tokenizer = null)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16);
            return Load(fs, tokenizer);
        }

        /// <summary>
        /// Rebuilds an index from a graph written by <see cref="SerializeGraph(Stream)"/> and the items it was built from, which must be given
        /// in the same order. The vectors are re-encoded deterministically, so searches behave exactly as before the save.
        /// </summary>
        /// <returns>
        /// The index, and the items beyond the ones the graph covers (for example vectors persisted just before a crash): add them with
        /// <see cref="AddRange(IReadOnlyList{HyperspaceItem}, Action{int}?)"/>.
        /// </returns>
        public static (HyperspaceIndex Index, IReadOnlyList<HyperspaceItem> ItemsNotInGraph) DeserializeGraph(IReadOnlyList<HyperspaceItem> items, Stream stream, Tokenizer? tokenizer = null)
        {
            var reader = new StreamReader2(stream);
            var section = ReadGraphSection(reader, tokenizer);
            var index = new HyperspaceIndex(section.Options);

            if (items.Count < section.Nodes.Length)
            {
                throw new InvalidDataException($"The graph has {section.Nodes.Length} nodes but only {items.Count} items were supplied");
            }

            var vectors = new List<byte[]>(section.Nodes.Length);
            var metadata = new List<Dictionary<string, string>?>(section.Nodes.Length);
            var originals = index._options.KeepOriginalVectors ? new List<float[]>(section.Nodes.Length) : null;
            for (int i = 0; i < section.Nodes.Length; i++)
            {
                var vector = items[i].Vector;
                if (vector.Length != index._options.Dimension)
                {
                    throw new InvalidDataException($"Item {i} has {vector.Length} dimensions but the graph was built for {index._options.Dimension}");
                }

                vectors.Add(index.EncodeForLoad(vector, out var normalized));
                originals?.Add(normalized);
                metadata.Add(ToPlainDictionary(items[i].Metadata));
            }

            index.Populate(section, vectors, metadata, originals);

            var rest = new HyperspaceItem[items.Count - section.Nodes.Length];
            for (int i = 0; i < rest.Length; i++)
            {
                rest[i] = items[section.Nodes.Length + i];
            }

            return (index, rest);
        }

        /// <summary>Rebuilds an index from a graph and plain vectors, see <see cref="DeserializeGraph(IReadOnlyList{HyperspaceItem}, Stream, Tokenizer?)"/>.</summary>
        public static (HyperspaceIndex Index, IReadOnlyList<HyperspaceItem> ItemsNotInGraph) DeserializeGraph(IReadOnlyList<float[]> vectors, Stream stream, Tokenizer? tokenizer = null)
            => DeserializeGraph(HyperspaceItem.FromVectors(vectors), stream, tokenizer);

        /// <summary>Rebuilds an index from a graph held in memory.</summary>
        public static (HyperspaceIndex Index, IReadOnlyList<HyperspaceItem> ItemsNotInGraph) DeserializeGraph(IReadOnlyList<HyperspaceItem> items, ReadOnlyMemory<byte> graph, Tokenizer? tokenizer = null)
        {
            using var ms = MemoryMarshal.TryGetArray(graph, out var seg) && seg.Array is not null
                ? new MemoryStream(seg.Array, seg.Offset, seg.Count, writable: false)
                : new MemoryStream(graph.ToArray(), writable: false);
            return DeserializeGraph(items, ms, tokenizer);
        }

        private static Dictionary<string, string>? ToPlainDictionary(IReadOnlyDictionary<string, string>? source)
        {
            if (source is null || source.Count == 0)
            {
                return null;
            }

            var copy = new Dictionary<string, string>(source.Count, StringComparer.Ordinal);
            foreach (var kv in source)
            {
                copy[kv.Key] = kv.Value;
            }

            return copy;
        }

        private byte[] EncodeForLoad(float[] vector, out float[] normalized)
        {
            normalized = (float[])vector.Clone();
            if (_options.ResolvedNormalize)
            {
                Metrics.Normalize(normalized);
            }

            return _codec.Encode(normalized);
        }

        private sealed class GraphSection
        {
            public GraphSection(HyperspaceOptions options, Node[] nodes, int entryPoint, int maxLevel, int[] deleted)
            {
                Options = options;
                Nodes = nodes;
                EntryPoint = entryPoint;
                MaxLevel = maxLevel;
                Deleted = deleted;
            }

            public HyperspaceOptions Options { get; }

            public Node[] Nodes { get; }

            public int EntryPoint { get; }

            public int MaxLevel { get; }

            public int[] Deleted { get; }
        }

        private static GraphSection ReadGraphSection(StreamReader2 reader, Tokenizer? tokenizer)
        {
            Span<byte> magic = stackalloc byte[8];
            reader.ReadExactly(magic);
            if (!magic.SequenceEqual(GraphMagic))
            {
                throw new InvalidDataException("Not a Hyperspace graph (bad magic)");
            }

            int parametersLength = reader.ReadInt32();
            if (parametersLength < 0 || parametersLength > MaximumParametersSize)
            {
                throw new InvalidDataException($"Invalid parameters block size {parametersLength}");
            }

            var parameterBytes = new byte[parametersLength];
            reader.ReadExactly(parameterBytes);
            var options = ReadParameters(new StreamReader2(new MemoryStream(parameterBytes, writable: false)));
            options.Tokenizer = tokenizer;

            int nodeCount = reader.ReadInt32();
            int entryPoint = reader.ReadInt32();
            int maxLevel = reader.ReadInt32();
            long totalRecordInts = reader.ReadInt64();
            if (nodeCount < 0 || totalRecordInts < 0 || entryPoint >= nodeCount || maxLevel < -1 || maxLevel >= options.MaxLayers || (nodeCount == 0) != (entryPoint < 0))
            {
                throw new InvalidDataException($"Invalid graph header (nodes: {nodeCount}, entry point: {entryPoint}, max level: {maxLevel}, size: {totalRecordInts})");
            }

            var nodes = new Node[nodeCount];
            long remaining = totalRecordInts;
            for (int i = 0; i < nodeCount; i++)
            {
                int id = reader.ReadInt32();
                int maxLayers = reader.ReadInt32();
                int recordLength = reader.ReadInt32();

                if (id != i)
                {
                    throw new InvalidDataException($"Invalid node record at index {i}: it carries id {id}");
                }

                // Bounded by the total announced in the header, so corrupt data cannot demand an arbitrarily large allocation.
                if (maxLayers < 1 || maxLayers > options.MaxLayers || recordLength < maxLayers + 1 || recordLength > remaining)
                {
                    throw new InvalidDataException($"Invalid node record at index {i} (layers: {maxLayers}, size: {recordLength})");
                }

                remaining -= recordLength;
                var record = new int[recordLength];
                reader.ReadInt32Span(record);
                ValidateRecord(record, i, maxLayers, nodeCount);

                var links = new int[maxLayers][];
                int neighbours = recordLength - maxLayers - 1;
                for (int l = 0; l < maxLayers; l++)
                {
                    int start = record[l];
                    int end = l + 1 < maxLayers ? record[l + 1] : neighbours;
                    links[l] = record.AsSpan(maxLayers + 1 + start, end - start).ToArray();
                }

                nodes[i] = new Node(links);
            }

            if (remaining != 0)
            {
                throw new InvalidDataException("The graph header's size does not match its node records");
            }

            int deletedCount = reader.ReadInt32();
            if (deletedCount < 0 || deletedCount > nodeCount)
            {
                throw new InvalidDataException($"Invalid deleted count {deletedCount}");
            }

            var deleted = new int[deletedCount];
            reader.ReadInt32Span(deleted);
            foreach (int d in deleted)
            {
                if ((uint)d >= (uint)nodeCount)
                {
                    throw new InvalidDataException($"Deleted id {d} is not one of the {nodeCount} nodes");
                }
            }

            return new GraphSection(options, nodes, entryPoint, maxLevel, deleted);
        }

        /// <summary>
        /// Checks a node record describes a graph the searcher can walk: monotonic layer offsets covering exactly the record,
        /// and neighbour ids that name a node. A corrupt file becomes an <see cref="InvalidDataException"/> the caller can rebuild from,
        /// instead of an index-out-of-range thrown from inside a search hours later.
        /// </summary>
        private static void ValidateRecord(ReadOnlySpan<int> record, int index, int maxLayers, int nodeCount)
        {
            int neighbours = record.Length - maxLayers - 1;
            if (record[maxLayers] != neighbours)
            {
                throw new InvalidDataException($"Invalid node record at index {index}: the layer offsets do not describe {neighbours} neighbours");
            }

            if (record[0] != 0)
            {
                throw new InvalidDataException($"Invalid node record at index {index}: layer 0 does not start at 0");
            }

            for (int layer = 0; layer < maxLayers; layer++)
            {
                if (record[layer] < 0 || record[layer] > record[layer + 1])
                {
                    throw new InvalidDataException($"Invalid node record at index {index}: layer {layer} starts at {record[layer]} but layer {layer + 1} starts at {record[layer + 1]}");
                }
            }

            var connections = record.Slice(maxLayers + 1);
            for (int i = 0; i < connections.Length; i++)
            {
                if ((uint)connections[i] >= (uint)nodeCount)
                {
                    throw new InvalidDataException($"Invalid node record at index {index}: neighbour {connections[i]} is not one of the {nodeCount} nodes in the graph");
                }
            }
        }

        private static HyperspaceOptions ReadParameters(StreamReader2 r)
        {
            int version = r.ReadInt32();
            if (version != ParametersVersion)
            {
                throw new InvalidDataException($"Unsupported parameters version {version}");
            }

            var o = new HyperspaceOptions
            {
                Dimension = r.ReadInt32(),
                Metric = (DistanceMetric)r.ReadInt32(),
                Quantization = (QuantizationMode)r.ReadInt32(),
                M = r.ReadInt32(),
                EfConstruction = r.ReadInt32(),
                EfSearch = r.ReadInt32(),
                MaxLayers = r.ReadInt32(),
                HeadDimensions = r.ReadInt32(),
                LevelProbability = r.ReadSingle(),
                TernaryThreshold = r.ReadSingle(),
            };

            int flags = r.ReadInt32();
            o.FastRouting = (flags & 1) != 0;
            o.DensityPruning = (flags & 2) != 0;
            o.NormalizeVectors = (flags & 4) != 0;
            o.AnisotropicRefinement = (flags & 8) != 0;
            o.ValidateVectors = (flags & 16) != 0;
            o.KeepOriginalVectors = (flags & 32) != 0;
            o.RerankOversample = r.ReadInt32();
            o.FusionMethod = r.ReadString();
            o.Bm25 = new Bm25Parameters
            {
                Method = (Bm25Method)r.ReadInt32(),
                K1 = r.ReadSingle(),
                B = r.ReadSingle(),
                Delta = r.ReadSingle(),
                NGrams = r.ReadInt32(),
                Language = r.ReadString(),
            };
            o.Graph = new GraphConfig
            {
                Enabled = r.ReadInt32() != 0,
                StorageLayout = (GraphStorageLayout)r.ReadInt32(),
                MaxHopsLimit = r.ReadInt32(),
                MaxFanoutPerNode = r.ReadInt32(),
            };

            int pqLength = r.ReadInt32();
            if (pqLength < 0 || pqLength > MaximumParametersSize)
            {
                throw new InvalidDataException($"Invalid codebook size {pqLength}");
            }

            if (pqLength > 0)
            {
                var pqBytes = new byte[pqLength];
                r.ReadExactly(pqBytes);
                o.ProductQuantizer = ProductQuantizer.Load(new MemoryStream(pqBytes, writable: false));
            }

            if (o.Dimension <= 0 || !Enum.IsDefined(o.Metric) || !Enum.IsDefined(o.Quantization))
            {
                throw new InvalidDataException("Invalid index parameters");
            }

            o.Validate();
            return o;
        }

        /// <summary>Installs the contents read from disk into a freshly created index.</summary>
        private void Populate(GraphSection section, List<byte[]> vectors, List<Dictionary<string, string>?> metadata, List<float[]>? originals)
        {
            foreach (var d in section.Deleted)
            {
                if (_deleted.Add(d))
                {
                    _deletedCount++;
                }
            }

            for (int i = 0; i < section.Nodes.Length; i++)
            {
                _nodes.Add(section.Nodes[i]);
                _vectors.Add(vectors[i]);
                if (originals is not null)
                {
                    _originals!.Add(originals[i]);
                }

                var m = metadata[i] is { } loaded ? new ReadOnlyDictionary<string, string>(loaded) : null;
                _metadata.Add(m);
                if (m is not null && !_deleted.Contains(i))
                {
                    _metaIndex.Add(i, m);
                }
            }

            _entryPoint = section.EntryPoint;
            _maxLevel = section.MaxLevel;
            if (_options.FreezeOnLoad && _nodes.Count > 0)
            {
                FreezeFlatCore();
            }
        }
    }
}
