// <copyright file="Node.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using MessagePack;
    using System.Collections.Generic;

    /// <summary>
    /// The implementation of the node in hnsw graph.
    /// </summary>
    [MessagePackObject]
    public struct Node
    {
        [Key(0)]
        public List<List<int>> Connections;

        [Key(1)] public int Id;

        /// <summary>
        /// Gets the max layer where the node is presented.
        /// </summary>
        [IgnoreMember]
        public int MaxLayer
        {
            get
            {
                return Connections.Count - 1;
            }
        }

        /// <summary>
        /// Gets connections ids of the node at the given layer
        /// </summary>
        /// <param name="layer">The layer to get connections at.</param>
        /// <returns>The connections of the node at the given layer.</returns>
        public List<int> this[int layer]
        {
            get
            {
                return Connections[layer];
            }
            set
            {
                Connections[layer] = value;
            }
        }
    }
}
