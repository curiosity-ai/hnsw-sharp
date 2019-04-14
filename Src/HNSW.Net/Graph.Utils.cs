// <copyright file="Graph.Utils.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <content>
    /// The part with the auxiliary graph tools.
    /// </content>
    internal partial class Graph<TItem, TDistance>
    {
        /// <summary>
        /// Runs breadth first search.
        /// </summary>
        /// <param name="core">The graph core.</param>
        /// <param name="entryPoint">The entry point.</param>
        /// <param name="layer">The layer of the graph where to run BFS.</param>
        /// <param name="visitAction">The action to perform on each node.</param>
        internal static void BFS(Core core, Node entryPoint, int layer, Action<Node> visitAction)
        {
            var visitedIds = new HashSet<int>();
            var expansionQueue = new Queue<int>(new[] { entryPoint.Id });

            while (expansionQueue.Any())
            {
                var currentNode = core.Nodes[expansionQueue.Dequeue()];
                if (!visitedIds.Contains(currentNode.Id))
                {
                    visitAction(currentNode);
                    visitedIds.Add(currentNode.Id);
                    foreach (var neighbourId in currentNode[layer])
                    {
                        expansionQueue.Enqueue(neighbourId);
                    }
                }
            }
        }

        /// <summary>
        /// Bitset for tracking visited nodes.
        /// </summary>
        internal class VisitedBitSet
        {
            private int[] buffer;

            /// <summary>
            /// Initializes a new instance of the <see cref="VisitedBitSet"/> class.
            /// </summary>
            /// <param name="nodesCount">The number of nodes to track in the set.</param>
            internal VisitedBitSet(int nodesCount)
            {
                this.buffer = new int[(nodesCount >> 5) + 1];
            }

            /// <summary>
            /// Checks whether the node is already in the set.
            /// </summary>
            /// <param name="nodeId">The identifier of the node.</param>
            /// <returns>True if the node is in the set.</returns>
            internal bool Contains(int nodeId)
            {
                int carrier = this.buffer[nodeId >> 5];
                return ((1 << (nodeId & 31)) & carrier) != 0;
            }

            /// <summary>
            /// Adds the node id to the set.
            /// </summary>
            /// <param name="nodeId">The node id to add.</param>
            internal void Add(int nodeId)
            {
                int mask = 1 << (nodeId & 31);
                this.buffer[nodeId >> 5] |= mask;
            }

            /// <summary>
            /// Clears the set.
            /// </summary>
            internal void Clear()
            {
                Array.Clear(this.buffer, 0, this.buffer.Length);
            }
        }
    }
}