// <copyright file="Graph.Searcher.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Runtime.CompilerServices;
    using System.Runtime.InteropServices;
    using System.Threading;

    /// <content>
    /// The implementation of knn search.
    /// </content>
    internal partial class Graph<TItem, TDistance>
    {
        /// <summary>
        /// The graph searcher. Holds the reusable per-search scratch state (expansion frontier and
        /// visited marks) so that running a query allocates (almost) nothing. Instances are pooled
        /// by <see cref="Core"/> and are not thread-safe; a searcher must be used by one search at a time.
        /// </summary>
        internal sealed class Searcher
        {
            private readonly Core Core;
            private readonly List<Candidate<TDistance>> ExpansionBuffer;

            // Reusable snapshot of the neighbour ids of the node currently being expanded. Used to read live
            // (still mutable) connection lists without aliasing them, so a concurrent writer cannot hand us an
            // out-of-bounds span. Pooled with the searcher, so taking the snapshot stays allocation-free.
            private readonly List<int> NeighbourBuffer = new List<int>();

            // Epoch based visited set: a node is visited in the current search iff VisitedMarks[id] == VisitedEpoch.
            // Resetting between searches is O(1) (bump the epoch) instead of clearing a bit set proportional
            // to the size of the graph.
            private int[] VisitedMarks;
            private int VisitedEpoch;

            // Alternative visited set: a packed bit set (1 bit/node). 32x denser than VisitedMarks, so it
            // stays hot in cache for large graphs, at the cost of an O(nodes/64) clear per search.
            private readonly bool UseBitSet;
            private ulong[] VisitedBits;

            /// <summary>
            /// Initializes a new instance of the <see cref="Searcher"/> class.
            /// </summary>
            /// <param name="core">The core of the graph.</param>
            internal Searcher(Core core)
            {
                Core = core;
                ExpansionBuffer = new List<Candidate<TDistance>>();
                UseBitSet = core.UseBitSetVisited;
                if (UseBitSet)
                {
                    VisitedMarks = Array.Empty<int>();
                    VisitedBits = new ulong[(Math.Max(1024, core.Nodes.Count) + 63) >> 6];
                }
                else
                {
                    VisitedMarks = new int[Math.Max(1024, core.Nodes.Count)];
                    VisitedBits = Array.Empty<ulong>();
                }
                VisitedEpoch = 0;
            }

            private void Reset()
            {
                ExpansionBuffer.Clear();

                int nodesCount = Core.Nodes.Count;

                if (UseBitSet)
                {
                    int words = (nodesCount + 63) >> 6;
                    if (VisitedBits.Length < words)
                    {
                        VisitedBits = new ulong[Math.Max(words, VisitedBits.Length * 2)];
                    }
                    else
                    {
                        Array.Clear(VisitedBits, 0, words);
                    }
                    return;
                }

                if (VisitedMarks.Length < nodesCount)
                {
                    VisitedMarks = new int[Math.Max(nodesCount, VisitedMarks.Length * 2)];
                    VisitedEpoch = 0;
                }

                if (VisitedEpoch == int.MaxValue)
                {
                    Array.Clear(VisitedMarks, 0, VisitedMarks.Length);
                    VisitedEpoch = 0;
                }

                ++VisitedEpoch;
            }

            // Marks a node visited unconditionally (used for the entry point).
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private void MarkVisited(int id)
            {
                if (UseBitSet)
                {
                    VisitedBits[id >> 6] |= 1UL << (id & 63);
                }
                else
                {
                    VisitedMarks[id] = VisitedEpoch;
                }
            }

            // Returns true if the node had not been visited yet this search, marking it visited.
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private bool TryVisit(int id)
            {
                if (UseBitSet)
                {
                    int w = id >> 6;
                    ulong bit = 1UL << (id & 63);
                    if ((VisitedBits[w] & bit) != 0) return false;
                    VisitedBits[w] |= bit;
                    return true;
                }
                else
                {
                    if (VisitedMarks[id] == VisitedEpoch) return false;
                    VisitedMarks[id] = VisitedEpoch;
                    return true;
                }
            }

            /// <summary>
            /// The implementaiton of SEARCH-LAYER(q, ep, ef, lc) algorithm.
            /// Article: Section 4. Algorithm 2.
            /// The distance from the query to every node is computed exactly once (when the node is first
            /// visited) and is carried together with the node id through the candidate heaps.
            /// </summary>
            /// <param name="entryPointId">The identifier of the entry point for the search.</param>
            /// <param name="targetCosts">The traveling costs for the search target.</param>
            /// <param name="resultList">The list of candidates (id + distance) of the nearest neighbours at the level. Cleared on entry; left in max-heap order (farthest on top).</param>
            /// <param name="layer">The layer to perform search at.</param>
            /// <param name="k">The number of the nearest neighbours to get from the layer.</param>
            /// <param name="version">The version of the graph, will retry the search if the version changed</param>
            /// <param name="keepResult">Optional filter: nodes are added to the result set only when it returns true. Pass null to keep everything.</param>
            /// <param name="enableEarlyTermination">Whether to stop the traversal early once the result set stops improving (see <see cref="SmallWorldParameters.EnableEarlyTermination"/>).</param>
            /// <returns>The number of expanded nodes during the run.</returns>
            internal int RunKnnAtLayer(int entryPointId, TravelingCosts<int, TDistance> targetCosts, List<Candidate<TDistance>> resultList, int layer, int k, ref long version, long versionAtStart, Func<int, bool> keepResult, CancellationToken cancellationToken = default, bool enableEarlyTermination = false)
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

                Reset();
                resultList.Clear();

                var resultHeap = new CandidateMaxHeap<TDistance>(resultList);
                var expansionHeap = new CandidateMinHeap<TDistance>(ExpansionBuffer);

                var entryPoint = new Candidate<TDistance>(targetCosts.From(entryPointId), entryPointId);

                if (keepResult is null || keepResult(entryPointId))
                {
                    resultHeap.Push(entryPoint);
                }

                expansionHeap.Push(entryPoint);
                MarkVisited(entryPointId);

                // Early termination ("patience") state: once the result set stops improving for a sustained
                // number of consecutive hops we stop exploring. Only meaningful while collecting more than a single
                // result (k > 1); the greedy ef=1 descents already terminate as soon as no closer node is found.
                double saturationThreshold = Core.Parameters.EarlyTerminationSaturationThreshold;
                int patience = enableEarlyTermination && k > 1 ? ResolvePatience(Core.Parameters.EarlyTerminationPatience, k) : 0;
                bool earlyTermination = patience > 0;
                int consecutiveSaturatedHops = 0;

                bool optimizeForFiltering = Core.Parameters.OptimizeForFiltering;

                try
                {
                    // run bfs
                    int visitedNodesCount = 1;
                    while (expansionHeap.Count > 0)
                    {
                        if (cancellationToken.IsCancellationRequested)
                        {
                            return visitedNodesCount;
                        }

                        GraphChangedException.ThrowIfChanged(ref version, versionAtStart);

                        // number of top-k results changed by expanding the current candidate (used for early termination)
                        int resultChangesThisHop = 0;

                        // get next candidate to check and expand
                        var toExpand = expansionHeap.Pop();
                        if (resultHeap.Count > 0 && DistanceUtils.GreaterThan(toExpand.Distance, resultHeap.Top.Distance))
                        {
                            // the closest candidate is farther than farthest result
                            break;
                        }

                        if (optimizeForFiltering)
                        {
                            // Apply ACORN filtering (https://arxiv.org/html/2403.04871v1)
                            var neighboursIds = SelectAcornNeighbours(toExpand.Id, layer, keepResult);
                            foreach (var neighbourId in neighboursIds)
                            {
                                if (cancellationToken.IsCancellationRequested)
                                {
                                    return visitedNodesCount;
                                }

                                if (TryVisit(neighbourId))
                                {
                                    ++visitedNodesCount;
                                    resultChangesThisHop += ProcessNeighbour(neighbourId, targetCosts, ref resultHeap, ref expansionHeap, k, keepResult);
                                }
                            }
                        }
                        else
                        {
                            var node = Core.Nodes[toExpand.Id];

                            // Flattened nodes expose immutable storage and can be iterated in place; live nodes
                            // may be mutated by a concurrent AddItems, so snapshot their connections first to
                            // avoid aliasing a list whose backing array can be reallocated mid-read.
                            ReadOnlySpan<int> neighbours;
                            if (node.IsCached)
                            {
                                neighbours = node.EnumerateLayer(layer);
                            }
                            else
                            {
                                node.CopyLayerTo(layer, NeighbourBuffer);
                                neighbours = CollectionsMarshal.AsSpan(NeighbourBuffer);
                            }

                            for (int i = 0; i < neighbours.Length; ++i)
                            {
                                int neighbourId = neighbours[i];
                                if (TryVisit(neighbourId))
                                {
                                    ++visitedNodesCount;
                                    resultChangesThisHop += ProcessNeighbour(neighbourId, targetCosts, ref resultHeap, ref expansionHeap, k, keepResult);
                                }
                            }
                        }

                        // Patience based early termination: once the result set is full, measure how saturated it is
                        // after this hop (the fraction of the top-k that stayed unchanged). When the result set stays
                        // saturated for `patience` consecutive hops it is unlikely further exploration improves the
                        // result, so we stop. A single improving hop resets the patience window.
                        if (earlyTermination && resultHeap.Count >= k)
                        {
                            double saturation = (double)(resultHeap.Count - resultChangesThisHop) / resultHeap.Count;
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

                    return visitedNodesCount;
                }
                catch (Exception)
                {
                    //Throws if the collection changed, otherwise propagates the original exception
                    GraphChangedException.ThrowIfChanged(ref version, versionAtStart);
                    throw;
                }
            }

            /// <summary>
            /// Evaluates a single not-yet-visited neighbour: computes its distance to the query once and
            /// inserts it into the expansion frontier / result set when it can improve the result.
            /// Returns 1 when the result set changed, 0 otherwise.
            /// </summary>
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private int ProcessNeighbour(int neighbourId, TravelingCosts<int, TDistance> targetCosts, ref CandidateMaxHeap<TDistance> resultHeap, ref CandidateMinHeap<TDistance> expansionHeap, int k, Func<int, bool> keepResult)
            {
                var neighbourDistance = targetCosts.From(neighbourId);
                if (resultHeap.Count < k || DistanceUtils.LowerThan(neighbourDistance, resultHeap.Top.Distance))
                {
                    var neighbour = new Candidate<TDistance>(neighbourDistance, neighbourId);
                    expansionHeap.Push(neighbour);

                    if (keepResult is null || keepResult(neighbourId))
                    {
                        resultHeap.Push(neighbour);
                        if (resultHeap.Count > k)
                        {
                            resultHeap.Pop();
                        }

                        return 1;
                    }
                }

                return 0;
            }

            /// <summary>
            /// Selects the neighbours to traverse from an expanded node when ACORN filtering is active
            /// (https://arxiv.org/html/2403.04871v1).
            /// </summary>
            private List<int> SelectAcornNeighbours(int toExpandId, int layer, Func<int, bool> filter)
            {
                Func<int, bool> keepResult = filter ?? (static _ => true);
                var rawNeighbours = Core.Nodes[toExpandId].EnumerateLayer(layer);
                int targetM = layer == 0 ? 2 * Core.Parameters.M : Core.Parameters.M;
                var filtered = new List<int>();

                if (layer > 0)
                {
                    foreach (var n in rawNeighbours)
                    {
                        if (keepResult(n))
                        {
                            filtered.Add(n);
                            if (filtered.Count >= targetM)
                                break;
                        }
                    }
                }
                else if (Core.Parameters.Gamma == 1) // ACORN-1
                {
                    foreach (var n in rawNeighbours)
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
                    int mb = Math.Min(Core.Parameters.Mb, rawNeighbours.Length);

                    for (int i = 0; i < mb; i++)
                    {
                        if (keepResult(rawNeighbours[i]))
                        {
                            filtered.Add(rawNeighbours[i]);
                        }
                    }

                    if (filtered.Count < targetM && rawNeighbours.Length > mb)
                    {
                        for (int i = mb; i < rawNeighbours.Length; i++)
                        {
                            int n = rawNeighbours[i];
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

                return filtered;
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
