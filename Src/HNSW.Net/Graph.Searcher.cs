// <copyright file="Graph.Searcher.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System.Collections.Generic;

    /// <content>
    /// The implementation of knn search.
    /// </content>
    internal partial class Graph<TItem, TDistance>
    {
        /// <summary>
        /// The graph searcher.
        /// </summary>
        internal struct Searcher
        {
            private readonly Core core;
            private readonly IList<int> expansionBuffer;
            private readonly VisitedBitSet visitedSet;

            /// <summary>
            /// Initializes a new instance of the <see cref="Searcher"/> struct.
            /// </summary>
            /// <param name="core">The core of the graph.</param>
            internal Searcher(Core core)
            {
                this.core = core;
                this.expansionBuffer = new List<int>();
                this.visitedSet = new VisitedBitSet(core.Nodes.Count);
            }

            /// <summary>
            /// The implementaiton of SEARCH-LAYER(q, ep, ef, lc) algorithm.
            /// Article: Section 4. Algorithm 2.
            /// </summary>
            /// <param name="entryPointId">The identifier of the entry point for the search.</param>
            /// <param name="targetCosts">The traveling costs for the search target.</param>
            /// <param name="resultList">The list of identifiers of the nearest neighbours at the level.</param>
            /// <param name="layer">The layer to perform search at.</param>
            /// <param name="k">The number of the nearest neighbours to get from the layer.</param>
            /// <returns>The number of expanded nodes during the run.</returns>
            internal int RunKnnAtLayer(int entryPointId, TravelingCosts<int, TDistance> targetCosts, IList<int> resultList, int layer, int k)
            {
                /*
                 * v ← ep // set of visited elements
                 * C ← ep // set of candidates
                 * W ← ep // dynamic list of found nearest neighbors
                 * while │C│ > 0
                 *   c ← extract nearest element from C to q
                 *   f ← get furthest element from W to q
                 *   if distance(c, q) > distance(f, q)
                 *     break // all elements in W are evaluated
                 *   for each e ∈ neighbourhood(c) at layer lc // update C and W
                 *     if e ∉ v
                 *       v ← v ⋃ e
                 *       f ← get furthest element from W to q
                 *       if distance(e, q) < distance(f, q) or │W│ < ef
                 *         C ← C ⋃ e
                 *         W ← W ⋃ e
                 *         if │W│ > ef
                 *           remove furthest element from W to q
                 * return W
                 */

                // prepare tools
                IComparer<int> fartherIsOnTop = targetCosts;
                IComparer<int> closerIsOnTop = fartherIsOnTop.Reverse();

                // prepare collections
                // TODO: Optimize by providing buffers
                var resultHeap = new BinaryHeap<int>(resultList, fartherIsOnTop);
                var expansionHeap = new BinaryHeap<int>(this.expansionBuffer, closerIsOnTop);

                resultHeap.Push(entryPointId);
                expansionHeap.Push(entryPointId);
                this.visitedSet.Add(entryPointId);

                // run bfs
                int visitedNodesCount = 1;
                while (expansionHeap.Buffer.Count > 0)
                {
                    // get next candidate to check and expand
                    var toExpandId = expansionHeap.Pop();
                    var farthestResultId = resultHeap.Buffer[0];
                    if (DistanceUtils.Gt(targetCosts.From(toExpandId), targetCosts.From(farthestResultId)))
                    {
                        // the closest candidate is farther than farthest result
                        break;
                    }

                    // expand candidate
                    var neighboursIds = this.core.Nodes[toExpandId][layer];
                    for (int i = 0; i < neighboursIds.Count; ++i)
                    {
                        int neighbourId = neighboursIds[i];
                        if (!this.visitedSet.Contains(neighbourId))
                        {
                            // enqueue perspective neighbours to expansion list
                            farthestResultId = resultHeap.Buffer[0];
                            if (resultHeap.Buffer.Count < k
                            || DistanceUtils.Lt(targetCosts.From(neighbourId), targetCosts.From(farthestResultId)))
                            {
                                expansionHeap.Push(neighbourId);
                                resultHeap.Push(neighbourId);
                                if (resultHeap.Buffer.Count > k)
                                {
                                    resultHeap.Pop();
                                }
                            }

                            // update visited list
                            ++visitedNodesCount;
                            this.visitedSet.Add(neighbourId);
                        }
                    }
                }

                this.expansionBuffer.Clear();
                this.visitedSet.Clear();

                return visitedNodesCount;
            }
        }
    }
}
