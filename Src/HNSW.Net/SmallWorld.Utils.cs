// <copyright file="SmallWorld.Utils.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics.CodeAnalysis;
    using System.Linq;

    /// <content>
    /// The part with the auxiliary tools for hnsw algorithm.
    /// </content>
    public partial class SmallWorld<TItem, TDistance>
    {
        /// <summary>
        /// Distance is Lower Than.
        /// </summary>
        /// <param name="x">Left argument.</param>
        /// <param name="y">Right argument.</param>
        /// <returns>True if x &lt; y.</returns>
        [SuppressMessage("Design", "CA1000:Do not declare static members on generic types", Justification = "By Design")]
        public static bool DLt(TDistance x, TDistance y)
        {
            return x.CompareTo(y) < 0;
        }

        /// <summary>
        /// Distance is Greater Than.
        /// </summary>
        /// <param name="x">Left argument.</param>
        /// <param name="y">Right argument.</param>
        /// <returns>True if x &gt; y.</returns>
        [SuppressMessage("Design", "CA1000:Do not declare static members on generic types", Justification = "By Design")]
        public static bool DGt(TDistance x, TDistance y)
        {
            return x.CompareTo(y) > 0;
        }

        /// <summary>
        /// Distances are Equal.
        /// </summary>
        /// <param name="x">Left argument.</param>
        /// <param name="y">Right argument.</param>
        /// <returns>True if x == y.</returns>
        [SuppressMessage("Design", "CA1000:Do not declare static members on generic types", Justification = "By Design")]
        public static bool DEq(TDistance x, TDistance y)
        {
            return x.CompareTo(y) == 0;
        }

        /// <summary>
        /// Runs breadth first search.
        /// </summary>
        /// <param name="entryPoint">The entry point.</param>
        /// <param name="level">The level of the graph where to run BFS.</param>
        /// <param name="visitAction">The action to perform on each node.</param>
        internal static void BFS(Node entryPoint, int level, Action<Node> visitAction)
        {
            var visitedIds = new HashSet<int>();
            var expansionQueue = new Queue<Node>(new[] { entryPoint });

            while (expansionQueue.Any())
            {
                var currentNode = expansionQueue.Dequeue();
                if (!visitedIds.Contains(currentNode.Id))
                {
                    visitAction(currentNode);
                    visitedIds.Add(currentNode.Id);
                    foreach (var neighbour in currentNode.GetConnections(level))
                    {
                        expansionQueue.Enqueue(neighbour);
                    }
                }
            }
        }
    }
}