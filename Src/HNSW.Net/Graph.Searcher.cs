﻿// <copyright file="Graph.Searcher.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Threading;

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
            private readonly Core Core;
            private readonly List<int> ExpansionBuffer;
            private readonly VisitedBitSet VisitedSet;

            /// <summary>
            /// Initializes a new instance of the <see cref="Searcher"/> struct.
            /// </summary>
            /// <param name="core">The core of the graph.</param>
            internal Searcher(Core core)
            {
                Core = core;
                ExpansionBuffer = new List<int>();
                VisitedSet = new VisitedBitSet(core.Nodes.Count);
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
            /// <param name="version">The version of the graph, will retry the search if the version changed</param>
            /// <returns>The number of expanded nodes during the run.</returns>
            internal int RunKnnAtLayer(int entryPointId, TravelingCosts<int, TDistance> targetCosts, List<int> resultList, int layer, int k, ref long version, long versionAtStart, Func<int, bool> keepResult, CancellationToken cancellationToken = default)
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
                var resultHeap    = new BinaryHeap(resultList, fartherIsOnTop);
                var expansionHeap = new BinaryHeap(ExpansionBuffer, closerIsOnTop);

                if (keepResult(entryPointId))
                {
                    resultHeap.Push(entryPointId);
                }

                expansionHeap.Push(entryPointId);
                VisitedSet.Add(entryPointId);

                try
                {
                    // run bfs
                    int visitedNodesCount = 1;
                    while (expansionHeap.Buffer.Count > 0)
                    {
                        if (cancellationToken.IsCancellationRequested)
                        {
                            return visitedNodesCount;
                        }

                        GraphChangedException.ThrowIfChanged(ref version, versionAtStart);

                        // get next candidate to check and expand
                        var toExpandId = expansionHeap.Pop();
                        var farthestResultId = resultHeap.Buffer.Count > 0 ? resultHeap.Buffer[0] : -1;
                        if (farthestResultId > 0 && DistanceUtils.GreaterThan(targetCosts.From(toExpandId), targetCosts.From(farthestResultId)))
                        {
                            // the closest candidate is farther than farthest result
                            break;
                        }

                        // expand candidate
                        var neighboursIds = Core.Nodes[toExpandId].EnumerateLayer(layer);
                        
                        foreach(var neighbourId in neighboursIds) 
                        {
                            if (cancellationToken.IsCancellationRequested)
                            {
                                return visitedNodesCount;
                            }

                            if (!VisitedSet.Contains(neighbourId))
                            {
                                // enqueue perspective neighbours to expansion list
                                farthestResultId = resultHeap.Buffer.Count > 0 ? resultHeap.Buffer[0] : -1;
                                if (resultHeap.Buffer.Count < k || (farthestResultId >= 0 && DistanceUtils.LowerThan(targetCosts.From(neighbourId), targetCosts.From(farthestResultId))))
                                {
                                    expansionHeap.Push(neighbourId);
                                    
                                    if (keepResult(neighbourId))
                                    {
                                        resultHeap.Push(neighbourId);
                                    }

                                    if (resultHeap.Buffer.Count > k)
                                    {
                                        resultHeap.Pop();
                                    }
                                }

                                // update visited list
                                ++visitedNodesCount;
                                VisitedSet.Add(neighbourId);
                            }
                        }
                    }

                    ExpansionBuffer.Clear();
                    VisitedSet.Clear();

                    
                    return visitedNodesCount;
                }
                catch (Exception ex) 
                {
                    //Throws if the collection changed, otherwise propagates the original exception
                    GraphChangedException.ThrowIfChanged(ref version, versionAtStart);
                    throw;
                }
            }
        }
    }
}
