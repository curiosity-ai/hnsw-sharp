// <copyright file="Node.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using MessagePack;
    using System;
    using System.Buffers;
    using System.Collections.Generic;
    using System.Runtime.InteropServices;

    /// <summary>
    /// The implementation of the node in hnsw graph.
    /// </summary>
    [MessagePackObject]
    public struct Node
    {
        [Key(0)]
        public List<List<int>> Connections 
        { 
            get 
            { 
                return _connections ?? HydrateConnections(); 
            } 
        }

        private List<List<int>> HydrateConnections()
        {
            // Build the full structure into a local first and publish it only once it is complete. A concurrent
            // reader (non thread-safe mode relies on the version/retry mechanism) must never observe _connections
            // as a non-null but partially-populated list, or indexing it by layer would throw out of range.
            var hydrated = new List<List<int>>(_maxLayers);
            for (int l = 0; l < _maxLayers; l++)
            {
                var nl = new List<int>();
                foreach (var v in _cache.GetLayer(_bucketIndex, _position, l, _maxLayers))
                {
                    nl.Add(v);
                }
                hydrated.Add(nl);
            }
            _connections = hydrated;
            return hydrated;
        }

        [Key(1)] public int Id { get; private set; }

        private int _bucketIndex;
        private int _position;
        private int _maxLayers;
        private List<List<int>> _connections;
        private CachedNodeData _cache;

        [SerializationConstructor]
        public Node(List<List<int>> connections, int id)
        {
            _connections = connections;
            _maxLayers = connections?.Count ?? 0;
            Id = id;
        }

        public static void FlattenToCache(ref Node node, CachedNodeData cache)
        {
            if (node._connections is object)
            {
                var data = cache.Add(node._connections);
                node._connections = null;
                node._bucketIndex = data.bucketIndex;
                node._position    = data.position;
                node._maxLayers   = data.maxLayers;
            }
            else
            {
                var data = cache.Add(node._cache.GetAll(node._bucketIndex, node._position, node._maxLayers), node._maxLayers);
                node._bucketIndex = data.bucketIndex;
                node._position    = data.position;
                node._maxLayers   = data.maxLayers;
            }

            node._cache = cache;
        }

        /// <summary>
        /// Gets the max layer where the node is presented.
        /// </summary>
        [IgnoreMember]
        public int MaxLayer
        {
            get
            {
                return _maxLayers - 1;
            }
        }

        [IgnoreMember] public bool IsCached => _connections is null;

        /// <summary>
        /// Gets connections ids of the node at the given layer
        /// </summary>
        /// <param name="layer">The layer to get connections at.</param>
        /// <returns>The connections of the node at the given layer.</returns>
        public ReadOnlySpan<int> this[int layer]
        {
            get
            {
                return EnumerateLayer(layer);
            }
        }

        public ReadOnlySpan<int> EnumerateLayer(int layer)
        {
            if (_connections is null)
            {
                return _cache.GetLayer(_bucketIndex, _position, layer, _maxLayers);
            }
            else
            {
                // The span aliases the live connection list (no copy); it is only valid until the
                // connections of this node are modified.
                return System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_connections[layer]);
            }
        }

        /// <summary>
        /// Copies the neighbour ids of the given layer into <paramref name="destination"/> (cleared on entry).
        /// Unlike <see cref="EnumerateLayer"/>, this does not alias the live connection list, so the snapshot
        /// stays valid (and in-bounds) even if a concurrent writer mutates the node while we read it: a racing
        /// modification surfaces as a catchable exception (absorbed by the version-retry in the searcher)
        /// rather than handing back an out-of-bounds span over a reallocated backing array.
        /// </summary>
        internal void CopyLayerTo(int layer, List<int> destination)
        {
            destination.Clear();
            if (_connections is null)
            {
                // Flattened cache is immutable; copy the span element-wise.
                var span = _cache.GetLayer(_bucketIndex, _position, layer, _maxLayers);
                for (int i = 0; i < span.Length; ++i)
                {
                    destination.Add(span[i]);
                }
            }
            else
            {
                // List.AddRange uses ICollection.CopyTo (bounds-checked against the source array), so a
                // concurrent Add/Clear/AddRange throws instead of yielding a torn, out-of-bounds read.
                destination.AddRange(_connections[layer]);
            }
        }

        public void SetLayer(int layer, List<int> layerContent)
        {
            if (_connections is null)
            {
                HydrateConnections();
            }

            _connections[layer] = layerContent;
        }

        internal List<int> GetLayerForModifying(int layer)
        {
            if (_connections is null)
            {
                HydrateConnections();
            }

            return _connections[layer];
        }

        /// <summary>
        /// Number of ints this node occupies in the flat format: the per-layer start offsets, the total, and
        /// the neighbour ids themselves.
        /// </summary>
        internal int FlatRecordLength
        {
            get
            {
                if (_maxLayers == 0) return 0;

                if (_connections is null)
                {
                    return _cache.GetAll(_bucketIndex, _position, _maxLayers).Length;
                }

                int total = 0;
                for (int layer = 0; layer < _maxLayers; layer++)
                {
                    total += _connections[layer].Count;
                }

                return total + _maxLayers + 1;
            }
        }

        /// <summary>
        /// Writes this node in the flat format, without materializing its connections as lists.
        /// </summary>
        /// <remarks>
        /// A node that has already been flattened (the common case after <c>OptimizeIfNeeded</c>) is copied
        /// straight out of <see cref="CachedNodeData"/> as one span, which is the whole point of the format:
        /// reading <see cref="Connections"/> instead would hydrate a <c>List&lt;List&lt;int&gt;&gt;</c> per
        /// node and leave the graph un-flattened behind it.
        /// </remarks>
        internal void WriteTo(IBufferWriter<byte> writer)
        {
            writer.WriteInt32(Id);
            writer.WriteInt32(_maxLayers);

            if (_maxLayers == 0)
            {
                writer.WriteInt32(0);
                return;
            }

            if (_connections is null)
            {
                var flat = _cache.GetAll(_bucketIndex, _position, _maxLayers);
                writer.WriteInt32(flat.Length);
                writer.WriteInt32Span(flat);
                return;
            }

            int total = 0;
            for (int layer = 0; layer < _maxLayers; layer++)
            {
                total += _connections[layer].Count;
            }

            writer.WriteInt32(total + _maxLayers + 1);

            int offset = 0;
            for (int layer = 0; layer < _maxLayers; layer++)
            {
                writer.WriteInt32(offset);
                offset += _connections[layer].Count;
            }

            writer.WriteInt32(offset);

            for (int layer = 0; layer < _maxLayers; layer++)
            {
                writer.WriteInt32Span(CollectionsMarshal.AsSpan(_connections[layer]));
            }
        }

        /// <summary>
        /// Rebuilds a node whose flat record has already been read into <paramref name="cache"/>.
        /// </summary>
        internal static Node FromCache(int id, CachedNodeData cache, int bucketIndex, int position, int maxLayers)
        {
            return new Node
            {
                Id           = id,
                _connections = null,
                _cache       = cache,
                _bucketIndex = bucketIndex,
                _position    = position,
                _maxLayers   = maxLayers,
            };
        }
    }
}
