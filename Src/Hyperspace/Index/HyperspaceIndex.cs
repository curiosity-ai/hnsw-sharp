// <copyright file="HyperspaceIndex.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading;
    using System.Threading.Tasks;

    /// <summary>
    /// An embedding database: a hierarchical navigable small-world graph over (optionally quantized) vectors with a choice of 12 metrics,
    /// metadata and geometric filters, BM25 hybrid search, fuzzy logic queries, a property graph, and flat binary persistence.
    /// </summary>
    /// <remarks>
    /// <para>
    /// Items are identified by consecutive <see cref="int"/> ids assigned by <see cref="Add(ReadOnlySpan{float}, IReadOnlyDictionary{string, string})"/>. Ids are stable: deleting an item only
    /// hides it (use <see cref="Compact"/> to physically remove deleted items and obtain the id mapping).
    /// </para>
    /// <para>
    /// By default the index guards itself with a reader-writer lock, so searches run in parallel and writes are exclusive
    /// (see <see cref="HyperspaceOptions.ThreadSafe"/>).
    /// </para>
    /// </remarks>
    public sealed partial class HyperspaceIndex
    {
        private readonly HyperspaceOptions _options;
        private readonly VectorCodec _codec;
        private readonly ReaderWriterLockSlim? _rw;
        private readonly List<byte[]> _vectors = new List<byte[]>();
        private readonly List<float[]>? _originals;
        private readonly List<Node> _nodes = new List<Node>();
        private readonly List<Dictionary<string, string>?> _metadata = new List<Dictionary<string, string>?>();
        private readonly IdSet _deleted = new IdSet();
        private readonly MetadataIndex _metaIndex = new MetadataIndex();
        private readonly Random _random;
        private readonly object _randomLock = new object();
        private int _deletedCount;
        private int _entryPoint = -1;
        private int _maxLevel = -1;
        private LexicalIndex? _lexical;
        private FlatLayer0? _flat;

        /// <summary>Initializes a new instance of the <see cref="HyperspaceIndex"/> class.</summary>
        public HyperspaceIndex(HyperspaceOptions options)
        {
            if (options is null)
            {
                throw new ArgumentNullException(nameof(options));
            }

            _options = options.Clone();
            _options.Validate();
            _codec = VectorCodec.Create(_options);
            _rw = _options.ThreadSafe ? new ReaderWriterLockSlim() : null;
            _originals = _options.KeepOriginalVectors ? new List<float[]>() : null;
            _random = _options.Seed is int seed ? new Random(seed) : new Random();
            Graph = new GraphEngine(_options.Graph);
        }

        /// <summary>Gets the number of live (not deleted) items.</summary>
        public int Count
        {
            get
            {
                EnterRead();
                try
                {
                    return _vectors.Count - _deletedCount;
                }
                finally
                {
                    ExitRead();
                }
            }
        }

        /// <summary>Gets the number of ids ever assigned, deleted ones included.</summary>
        public int TotalCount
        {
            get
            {
                EnterRead();
                try
                {
                    return _vectors.Count;
                }
                finally
                {
                    ExitRead();
                }
            }
        }

        /// <summary>Gets the number of deleted items still occupying an id.</summary>
        public int DeletedCount
        {
            get
            {
                EnterRead();
                try
                {
                    return _deletedCount;
                }
                finally
                {
                    ExitRead();
                }
            }
        }

        /// <summary>Gets the dimension of the vectors.</summary>
        public int Dimension => _options.Dimension;

        /// <summary>Gets the distance metric.</summary>
        public DistanceMetric Metric => _options.Metric;

        /// <summary>Gets the storage mode.</summary>
        public QuantizationMode Quantization => _options.Quantization;

        /// <summary>Gets a copy of the options the index was created with.</summary>
        public HyperspaceOptions Options => _options.Clone();

        /// <summary>Gets the property graph (edges between item ids).</summary>
        public GraphEngine Graph { get; private set; }

        /// <summary>Gets the product quantizer when the index stores product-quantized vectors.</summary>
        public ProductQuantizer? Quantizer => (_codec as ProductQuantizationCodec)?.Quantizer;

        /// <summary>Gets the number of bytes used to store one vector.</summary>
        public int BytesPerVector => _codec.EncodedLength;

        private void EnterRead() => _rw?.EnterReadLock();

        private void ExitRead() => _rw?.ExitReadLock();

        private void EnterWrite() => _rw?.EnterWriteLock();

        private void ExitWrite() => _rw?.ExitWriteLock();

        private sealed class Node
        {
            public Node(int level)
            {
                Links = new int[level + 1][];
                for (int i = 0; i < Links.Length; i++)
                {
                    Links[i] = Array.Empty<int>();
                }
            }

            public Node(int[][] links)
            {
                Links = links;
            }

            public int[][] Links { get; }

            public int Level => Links.Length - 1;
        }

        private sealed class FlatLayer0
        {
            public FlatLayer0(int[] offsets, int[] neighbors)
            {
                Offsets = offsets;
                Neighbors = neighbors;
            }

            public int[] Offsets { get; }

            public int[] Neighbors { get; }

            public ReadOnlySpan<int> Of(int node) => Neighbors.AsSpan(Offsets[node], Offsets[node + 1] - Offsets[node]);
        }

        // ------------------------------------------------------------------ adding

        /// <summary>Adds an item and returns its id.</summary>
        /// <param name="vector">The embedding; must have <see cref="Dimension"/> components.</param>
        /// <param name="metadata">Optional metadata. String values are searchable with <see cref="FilterExpr"/>; all values feed the BM25 index of hybrid search.</param>
        public int Add(ReadOnlySpan<float> vector, IReadOnlyDictionary<string, string>? metadata = null)
        {
            var prepared = PrepareVector(vector, out var encoded);
            return Insert(encoded, prepared, metadata);
        }

        /// <summary>Adds an item.</summary>
        public int Add(HyperspaceItem item) => Add(item.Vector, item.Metadata);

        /// <summary>
        /// Adds many items, encoding and preparing them in parallel and linking them in order. Returns the ids in input order.
        /// </summary>
        /// <param name="items">The items to add.</param>
        /// <param name="progress">Called with the number of items added so far, after each internal batch.</param>
        public int[] AddRange(IReadOnlyList<HyperspaceItem> items, Action<int>? progress = null)
        {
            var ids = new int[items.Count];
            const int batch = 512;
            for (int start = 0; start < items.Count; start += batch)
            {
                int count = Math.Min(batch, items.Count - start);
                var encoded = new byte[count][];
                var prepared = new PreparedQuery[count];
                int from = start;
                Parallel.For(0, count, i =>
                {
                    prepared[i] = PrepareVector(items[from + i].Vector, out encoded[i]);
                });

                for (int i = 0; i < count; i++)
                {
                    ids[start + i] = Insert(encoded[i], prepared[i], items[start + i].Metadata);
                }

                progress?.Invoke(start + count);
            }

            return ids;
        }

        /// <summary>Adds many plain vectors.</summary>
        public int[] AddRange(IReadOnlyList<float[]> vectors, Action<int>? progress = null) => AddRange(HyperspaceItem.FromVectors(vectors), progress);

        private PreparedQuery PrepareVector(ReadOnlySpan<float> vector, out byte[] encoded)
        {
            if (vector.Length != _options.Dimension)
            {
                throw new ArgumentException($"Expected a vector of {_options.Dimension} dimensions but got {vector.Length}", nameof(vector));
            }

            Span<float> work = vector.Length <= 1024 ? stackalloc float[vector.Length] : new float[vector.Length];
            vector.CopyTo(work);
            if (_options.ResolvedNormalize)
            {
                Metrics.Normalize(work);
            }

            if (_options.ValidateVectors)
            {
                var error = Metrics.Validate<float>(_options.Metric, work);
                if (error is not null)
                {
                    throw new ArgumentException(error, nameof(vector));
                }
            }

            encoded = _codec.Encode(work);
            return _codec.Prepare(work);
        }

        private int Insert(byte[] encoded, PreparedQuery prepared, IReadOnlyDictionary<string, string>? metadata)
        {
            var copy = CopyMetadata(metadata);
            EnterWrite();
            try
            {
                int id = _vectors.Count;
                _vectors.Add(encoded);
                _originals?.Add(prepared.Coords32);
                _nodes.Add(new Node(RandomLevel()));
                _metadata.Add(copy);
                if (copy is not null)
                {
                    _metaIndex.Add(id, copy);
                    _lexical?.Upsert(id, null, copy);
                }

                ConnectNode(id, prepared);
                _flat = null;
                return id;
            }
            finally
            {
                ExitWrite();
            }
        }

        private static Dictionary<string, string>? CopyMetadata(IReadOnlyDictionary<string, string>? metadata)
            => metadata is null || metadata.Count == 0 ? null : new Dictionary<string, string>(metadata.ToDictionary(kv => kv.Key, kv => kv.Value), StringComparer.Ordinal);

        private int RandomLevel()
        {
            lock (_randomLock)
            {
                int level = 0;
                while (_random.NextDouble() < _options.LevelProbability && level < _options.MaxLayers - 1)
                {
                    level++;
                }

                return level;
            }
        }

        // ------------------------------------------------------------------ updating / deleting

        /// <summary>Marks an item as deleted. It disappears from searches, filters and the lexical index; its id is not reused.</summary>
        /// <returns><c>true</c> if the item existed and was live.</returns>
        public bool Delete(int id)
        {
            EnterWrite();
            try
            {
                if ((uint)id >= (uint)_vectors.Count || _deleted.Contains(id))
                {
                    return false;
                }

                _deleted.Add(id);
                _deletedCount++;
                var meta = _metadata[id];
                if (meta is not null)
                {
                    _metaIndex.Remove(id, meta);
                }

                _lexical?.Remove(id, meta);
                return true;
            }
            finally
            {
                ExitWrite();
            }
        }

        /// <summary>Replaces the vector (and, if given, the metadata) of an item and re-links it in the graph.</summary>
        public void Update(int id, ReadOnlySpan<float> vector, IReadOnlyDictionary<string, string>? metadata = null)
        {
            var prepared = PrepareVector(vector, out var encoded);
            var copy = CopyMetadata(metadata);
            EnterWrite();
            try
            {
                CheckLive(id);
                _vectors[id] = encoded;
                if (_originals is not null)
                {
                    _originals[id] = prepared.Coords32;
                }

                if (metadata is not null)
                {
                    ReplaceMetadata(id, copy);
                }

                // The old links stay in place while re-linking: they keep the node reachable as a bridge during the search and are
                // replaced layer by layer as the new neighbours are chosen.
                ConnectNode(id, prepared);
                _flat = null;
            }
            finally
            {
                ExitWrite();
            }
        }

        /// <summary>Replaces the metadata of an item without touching its vector or its links.</summary>
        public void UpdateMetadata(int id, IReadOnlyDictionary<string, string>? metadata)
        {
            var copy = CopyMetadata(metadata);
            EnterWrite();
            try
            {
                CheckLive(id);
                ReplaceMetadata(id, copy);
            }
            finally
            {
                ExitWrite();
            }
        }

        private void ReplaceMetadata(int id, Dictionary<string, string>? copy)
        {
            var old = _metadata[id];
            if (old is not null)
            {
                _metaIndex.Remove(id, old);
            }

            _metadata[id] = copy;
            if (copy is not null)
            {
                _metaIndex.Add(id, copy);
            }

            _lexical?.Upsert(id, old, copy ?? new Dictionary<string, string>());
        }

        private void CheckLive(int id)
        {
            if ((uint)id >= (uint)_vectors.Count || _deleted.Contains(id))
            {
                throw new KeyNotFoundException($"No live item with id {id}");
            }
        }

        // ------------------------------------------------------------------ reading

        /// <summary>Whether a live item with this id exists.</summary>
        public bool Contains(int id)
        {
            EnterRead();
            try
            {
                return (uint)id < (uint)_vectors.Count && !_deleted.Contains(id);
            }
            finally
            {
                ExitRead();
            }
        }

        /// <summary>The stored vector: the original if <see cref="HyperspaceOptions.KeepOriginalVectors"/> is on (after normalization, if the metric normalizes), else reconstructed from its (possibly quantized) encoding.</summary>
        public float[] GetVector(int id)
        {
            EnterRead();
            try
            {
                CheckLive(id);
                if (_originals is not null)
                {
                    return (float[])_originals[id].Clone();
                }

                var v = new float[_options.Dimension];
                _codec.Decode(_vectors[id], v);
                return v;
            }
            finally
            {
                ExitRead();
            }
        }

        /// <summary>The metadata of an item, or <c>null</c> if it has none.</summary>
        public IReadOnlyDictionary<string, string>? GetMetadata(int id)
        {
            EnterRead();
            try
            {
                CheckLive(id);
                var m = _metadata[id];
                return m is null ? null : new Dictionary<string, string>(m);
            }
            finally
            {
                ExitRead();
            }
        }

        /// <summary>The ids of all live items, ascending.</summary>
        public int[] GetIds()
        {
            EnterRead();
            try
            {
                var ids = new List<int>(_vectors.Count - _deletedCount);
                for (int i = 0; i < _vectors.Count; i++)
                {
                    if (!_deleted.Contains(i))
                    {
                        ids.Add(i);
                    }
                }

                return ids.ToArray();
            }
            finally
            {
                ExitRead();
            }
        }

        /// <summary>The ids linked to a node on a layer of the HNSW graph (for visualization and diagnostics).</summary>
        public int[] GetNeighbors(int id, int layer = 0)
        {
            EnterRead();
            try
            {
                if ((uint)id >= (uint)_nodes.Count || layer < 0 || layer >= _nodes[id].Links.Length)
                {
                    return Array.Empty<int>();
                }

                return (int[])_nodes[id].Links[layer].Clone();
            }
            finally
            {
                ExitRead();
            }
        }

        /// <summary>Gets the number of layers of the graph.</summary>
        public int LayerCount
        {
            get
            {
                EnterRead();
                try
                {
                    return _maxLevel + 1;
                }
                finally
                {
                    ExitRead();
                }
            }
        }

        // ------------------------------------------------------------------ maintenance

        /// <summary>
        /// Rebuilds the index without the deleted items (and the stale edges they left in the graph). Ids are renumbered
        /// in order; <paramref name="idMap"/> maps every old id to its new one, or -1 for deleted items.
        /// The encoded vectors are copied as they are, so quantized data is not re-quantized.
        /// </summary>
        public HyperspaceIndex Compact(out int[] idMap)
        {
            var copy = new HyperspaceIndex(_options);
            EnterRead();
            try
            {
                idMap = new int[_vectors.Count];
                var floats = new float[_options.Dimension];
                for (int i = 0; i < _vectors.Count; i++)
                {
                    if (_deleted.Contains(i))
                    {
                        idMap[i] = -1;
                        continue;
                    }

                    _codec.Decode(_vectors[i], floats);
                    var prepared = copy._codec.Prepare(_originals is not null ? _originals[i] : floats);
                    idMap[i] = copy.Insert(_vectors[i], prepared, _metadata[i]);
                }

                // Carry over the property graph for surviving nodes.
                for (int i = 0; i < idMap.Length; i++)
                {
                    if (idMap[i] < 0)
                    {
                        continue;
                    }

                    foreach (var e in Graph.EdgesFor((uint)i))
                    {
                        int to = e.Target < (uint)idMap.Length ? idMap[e.Target] : -1;
                        if (to >= 0)
                        {
                            copy.Graph.InsertEdge((uint)idMap[i], (uint)to, e.Relation, e.Weight);
                        }
                    }
                }
            }
            finally
            {
                ExitRead();
            }

            copy.Graph.Compact();
            return copy;
        }

        /// <summary>Trains the product-quantization codebooks. Only possible while the index is empty, since stored codes depend on the codebooks.</summary>
        public void TrainProductQuantizer(IReadOnlyList<float[]> samples, int iterations = 10)
        {
            if (Quantizer is not { } pq)
            {
                throw new InvalidOperationException("The index does not use product quantization");
            }

            EnterWrite();
            try
            {
                if (_vectors.Count > 0)
                {
                    throw new InvalidOperationException("Product quantizer codebooks can only be trained while the index is empty");
                }

                var work = new List<float[]>(samples.Count);
                foreach (var s in samples)
                {
                    var c = (float[])s.Clone();
                    if (_options.ResolvedNormalize)
                    {
                        Metrics.Normalize(c);
                    }

                    work.Add(c);
                }

                pq.Train(work, iterations);
            }
            finally
            {
                ExitWrite();
            }
        }

        /// <summary>
        /// Freezes layer 0 of the graph into two contiguous arrays (offsets and neighbours), removing pointer chasing from the search hot path.
        /// Any later write unfreezes it again.
        /// </summary>
        public void FreezeFlatLayer0()
        {
            EnterWrite();
            try
            {
                FreezeFlatCore();
            }
            finally
            {
                ExitWrite();
            }
        }

        /// <summary>Whether layer 0 is currently frozen.</summary>
        public bool IsLayer0Frozen => _flat is not null;

        private void FreezeFlatCore()
        {
            var offsets = new int[_nodes.Count + 1];
            int total = 0;
            for (int i = 0; i < _nodes.Count; i++)
            {
                offsets[i] = total;
                total += _nodes[i].Links[0].Length;
            }

            offsets[_nodes.Count] = total;
            var neighbors = new int[total];
            for (int i = 0; i < _nodes.Count; i++)
            {
                _nodes[i].Links[0].CopyTo(neighbors, offsets[i]);
            }

            _flat = new FlatLayer0(offsets, neighbors);
        }

        // ------------------------------------------------------------------ property graph

        /// <summary>Adds a directed edge to the property graph.</summary>
        public void Link(int source, int target, ushort relation, float weight = 1f)
        {
            if (source < 0 || target < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(source));
            }

            Graph.InsertEdge((uint)source, (uint)target, relation, weight);
        }

        /// <summary>Removes a directed edge from the property graph.</summary>
        public bool Unlink(int source, int target, ushort relation) => Graph.DeleteEdge((uint)source, (uint)target, relation);

        /// <summary>Creates edges from the metadata of the live items according to declarative rules; returns the number of edges added.</summary>
        public int InduceGraphFromMetadata(IReadOnlyList<MetadataInductionRule> rules)
        {
            List<(uint, IReadOnlyDictionary<string, string>)> nodes;
            EnterRead();
            try
            {
                nodes = new List<(uint, IReadOnlyDictionary<string, string>)>();
                for (int i = 0; i < _metadata.Count; i++)
                {
                    if (_metadata[i] is { } m && !_deleted.Contains(i))
                    {
                        nodes.Add(((uint)i, m));
                    }
                }
            }
            finally
            {
                ExitRead();
            }

            return Graph.InduceFromMetadata(nodes, rules);
        }
    }
}
