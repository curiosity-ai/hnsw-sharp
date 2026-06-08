// <copyright file="Graph.Searcher.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
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
            /// <param name="enableEarlyTermination">Whether to stop the traversal early once the result set stops improving (see <see cref="SmallWorldParameters.EnableEarlyTermination"/>).</param>
            /// <returns>The number of expanded nodes during the run.</returns>
            internal int RunKnnAtLayer(int entryPointId, TravelingCosts<int, TDistance> targetCosts, List<int> resultList, int layer, int k, ref long version, long versionAtStart, Func<int, bool> keepResult, CancellationToken cancellationToken = default, bool enableEarlyTermination = false)
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

                // Early termination ("patience") state: once the result set stops improving for a sustained
                // number of consecutive hops we stop exploring. Only meaningful while collecting more than a single
                // result (k > 1); the greedy ef=1 descents already terminate as soon as no closer node is found.
                double saturationThreshold = Core.Parameters.EarlyTerminationSaturationThreshold;
                int patience = enableEarlyTermination && k > 1 ? ResolvePatience(Core.Parameters.EarlyTerminationPatience, k) : 0;
                bool earlyTermination = patience > 0;
                int consecutiveSaturatedHops = 0;

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

                        // number of top-k results changed by expanding the current candidate (used for early termination)
                        int resultChangesThisHop = 0;

                        // get next candidate to check and expand
                        var toExpandId = expansionHeap.Pop();
                        var farthestResultId = resultHeap.Buffer.Count > 0 ? resultHeap.Buffer[0] : -1;
                        if (farthestResultId >= 0 && DistanceUtils.GreaterThan(targetCosts.From(toExpandId), targetCosts.From(farthestResultId)))
                        {
                            // the closest candidate is farther than farthest result
                            break;
                        }

                        // expand candidate
                        var rawNeighboursIds = Core.Nodes[toExpandId].EnumerateLayer(layer);

                        IEnumerable<int> neighboursIds = rawNeighboursIds.ToArray();

                        // Apply ACORN filtering (https://arxiv.org/html/2403.04871v1)
                        if (Core.Parameters.OptimizeForFiltering)
                        {
                            int targetM = layer == 0 ? 2 * Core.Parameters.M : Core.Parameters.M;
                            var rawArr = rawNeighboursIds.ToArray();

                            if (layer > 0)
                            {
                                var filtered = new List<int>();
                                foreach (var n in rawArr)
                                {
                                    if (keepResult(n))
                                    {
                                        filtered.Add(n);
                                        if (filtered.Count >= targetM)
                                            break;
                                    }
                                }
                                neighboursIds = filtered;
                            }
                            else
                            {
                                var filtered = new List<int>();

                                if (Core.Parameters.Gamma == 1) // ACORN-1
                                {
                                    foreach (var n in rawArr)
                                    {
                                        if (keepResult(n))
                                        {
                                            filtered.Add(n);
                                            if (filtered.Count >= targetM)
                                                break;
                                        }
                                        else
                                        {
                                            var twoHop = Core.Nodes[n].EnumerateLayer(layer);
                                            foreach (var nn in twoHop)
                                            {
                                                if (keepResult(nn) && !filtered.Contains(nn))
                                                {
                                                    filtered.Add(nn);
                                                    if (filtered.Count >= targetM)
                                                    {
                                                        break;
                                                    }
                                                }
                                            }
                                            if (filtered.Count >= targetM)
                                            {
                                                break;
                                            }
                                        }
                                    }
                                }
                                else // ACORN-gamma
                                {
                                    int mb = Math.Min(Core.Parameters.Mb, rawArr.Length);

                                    for (int i = 0; i < mb; i++)
                                    {
                                        if (keepResult(rawArr[i]))
                                        {
                                            filtered.Add(rawArr[i]);
                                        }
                                    }

                                    if (filtered.Count < targetM && rawArr.Length > mb)
                                    {
                                        for (int i = mb; i < rawArr.Length; i++)
                                        {
                                            int n = rawArr[i];
                                            var twoHop = Core.Nodes[n].EnumerateLayer(layer);
                                            foreach (var nn in twoHop)
                                            {
                                                if (keepResult(nn) && !filtered.Contains(nn))
                                                {
                                                    filtered.Add(nn);
                                                    if (filtered.Count >= targetM)
                                                    {
                                                        break;
                                                    }
                                                }
                                            }

                                            if (filtered.Count >= targetM)
                                            {
                                                break;
                                            }
                                        }
                                    }
                                }
                                neighboursIds = filtered;
                            }
                        }

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
                                        ++resultChangesThisHop;
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

                        // Patience based early termination: once the result set is full, measure how saturated it is
                        // after this hop (the fraction of the top-k that stayed unchanged). When the result set stays
                        // saturated for `patience` consecutive hops it is unlikely further exploration improves the
                        // result, so we stop. A single improving hop resets the patience window.
                        if (earlyTermination && resultHeap.Buffer.Count >= k)
                        {
                            double saturation = (double)(resultHeap.Buffer.Count - resultChangesThisHop) / resultHeap.Buffer.Count;
                            if (saturation >= saturationThreshold)
                            {
                                if (++consecutiveSaturatedHops >= patience)
                                {
                                    break;
                                }
                            }
                            else
                            {
                                consecutiveSaturatedHops = 0;
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

            /// <summary>
            /// Resolves the number of consecutive non-improving hops allowed before the search terminates early.
            /// A configured value of 0 or less selects an adaptive patience that scales inversely with the
            /// exploration factor (ef), ranging from ~9 at low ef down to 6 at very high ef, matching the behaviour
            /// described in https://manticoresearch.com/blog/knn-early-termination/.
            /// </summary>
            /// <param name="configuredPatience">The user configured patience, or 0/negative for adaptive.</param>
            /// <param name="ef">The exploration factor (number of candidates kept) for this search.</param>
            private static int ResolvePatience(int configuredPatience, int ef)
            {
                if (configuredPatience > 0)
                {
                    return configuredPatience;
                }

                // Adaptive: more exploration (higher ef) means more evidence per decision, so less patience is needed.
                int adaptive = (int)Math.Round(9.0 - Math.Log(Math.Max(ef, 1) / 16.0, 2.0));
                return Math.Min(9, Math.Max(6, adaptive));
            }
        }
    }
}
