// <copyright file="Graph.Utils.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

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

            while (expansionQueue.Count > 0)
            {
                var currentNode = core.Nodes[expansionQueue.Dequeue()];
                if (!visitedIds.Contains(currentNode.Id))
                {
                    visitAction(currentNode);
                    visitedIds.Add(currentNode.Id);
                    foreach (var neighbourId in currentNode.EnumerateLayer(layer))
                    {
                        expansionQueue.Enqueue(neighbourId);
                    }
                }
            }
        }
    }
}