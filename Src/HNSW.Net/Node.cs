// <copyright file="Node.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using MessagePack;
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// The implementation of the node in hnsw graph.
    /// </summary>
    [MessagePackObject]
    public struct Node
    {
        [Key(0)]
        public List<List<int>> Connections { get; private set; }

        [Key(1)] public int Id { get; private set; }

        private int _bucketIndex;
        private int _position;
        private int _maxLayers;
        private CachedNodeData _cache;

        [SerializationConstructor]
        public Node(List<List<int>> connections, int id)
        {
            Connections = connections;
            _maxLayers = connections.Count;
            Id = id;
        }

        public static void FlattenToCache(ref Node node, CachedNodeData cache)
        {
            if (node.Connections is object)
            {
                var data = cache.Add(node.Connections);
                node.Connections  = null;
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

        [IgnoreMember] public bool IsCached => Connections is null;

        /// <summary>
        /// Gets connections ids of the node at the given layer
        /// </summary>
        /// <param name="layer">The layer to get connections at.</param>
        /// <returns>The connections of the node at the given layer.</returns>
        public ReadOnlySpan<int> this[int layer]
        {
            get
            {
                return Connections is object ? Connections[layer].ToArray().AsSpan() : _cache.GetLayer(_bucketIndex, _position, layer, _maxLayers);
            }
        }

        public ReadOnlySpan<int> EnumerateLayer(int layer)
        {
            if (Connections is null)
            {
                return _cache.GetLayer(_bucketIndex, _position, layer, _maxLayers);
            }
            else
            {
                var l = Connections[layer];
                return l.ToArray().AsSpan();
            }
        }

        public void SetLayer(int layer, List<int> layerContent)
        {
            if (Connections is null)
            {
                Connections = new List<List<int>>();
                for (int l = 0; l < _maxLayers; l++)
                {
                    var nl = new List<int>();
                    foreach (var v in _cache.GetLayer(_bucketIndex, _position, l, _maxLayers))
                    {
                        nl.Add(v);
                    }
                    Connections.Add(nl);
                }
            }

            Connections[layer] = layerContent;
        }

        internal List<int> GetLayerForModifying(int layer)
        {
            if (Connections is null)
            {
                Connections = new List<List<int>>();
                for (int l = 0; l < _maxLayers; l++)
                {
                    var nl = new List<int>();
                    foreach (var v in _cache.GetLayer(_bucketIndex, _position, l, _maxLayers))
                    {
                        nl.Add(v);
                    }
                    Connections.Add(nl);
                }
            }

            return Connections[layer];
        }
    }
}
