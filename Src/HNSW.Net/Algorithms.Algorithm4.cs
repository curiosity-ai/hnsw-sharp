// <copyright file="Node.Algorithm4.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;

    internal partial class Algorithms
    {
        /// <summary>
        /// The implementation of the SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates, keepPrunedConnections) algorithm.
        /// Article: Section 4. Algorithm 4.
        /// </summary>
        /// <typeparam name="TItem">The typeof the items in the small world.</typeparam>
        /// <typeparam name="TDistance">The type of the distance in the small world.</typeparam>
        internal sealed class Algorithm4<TItem, TDistance> : Algorithm<TItem, TDistance> where TDistance : struct, IComparable<TDistance>
        {
            private readonly List<Candidate<TDistance>> _discarded = new List<Candidate<TDistance>>();

            public Algorithm4(Graph<TItem, TDistance>.Core graphCore) : base(graphCore)
            {
            }

            /// <inheritdoc/>
            internal override void SelectBestForConnecting(List<Candidate<TDistance>> candidates, int targetId, int layer, List<int> output)
            {
                /*
                 * q ← this
                 * R ← ∅    // result
                 * W ← C    // working queue for the candidates
                 * if expandCandidates  // expand candidates
                 *   for each e ∈ C
                 *     for each eadj ∈ neighbourhood(e) at layer lc
                 *       if eadj ∉ W
                 *         W ← W ⋃ eadj
                 *
                 * Wd ← ∅ // queue for the discarded candidates
                 * while │W│ gt 0 and │R│ lt M
                 *   e ← extract nearest element from W to q
                 *   if e is closer to q compared to any element from R
                 *     R ← R ⋃ e
                 *   else
                 *     Wd ← Wd ⋃ e
                 *
                 * if keepPrunedConnections // add some of the discarded connections from Wd
                 *   while │Wd│ gt 0 and │R│ lt M
                 *   R ← R ⋃ extract nearest element from Wd to q
                 *
                 * return R
                 */

                output.Clear();

                var layerM = GetM(layer);
                var keepPrunedConnections = GraphCore.Parameters.KeepPrunedConnections;

                // expand candidates option is enabled
                if (GraphCore.Parameters.ExpandBestSelection)
                {
                    var visited = new HashSet<int>();
                    foreach (var candidate in candidates)
                    {
                        visited.Add(candidate.Id);
                    }

                    int originalCount = candidates.Count;
                    for (int i = 0; i < originalCount; ++i)
                    {
                        var candidateNeighboursIds = GraphCore.Nodes[candidates[i].Id].EnumerateLayer(layer);
                        foreach (var candidateNeighbourId in candidateNeighboursIds)
                        {
                            if (visited.Add(candidateNeighbourId))
                            {
                                candidates.Add(new Candidate<TDistance>(NodeDistance(candidateNeighbourId, targetId), candidateNeighbourId));
                            }
                        }
                    }
                }

                candidates.Sort(CandidateComparer<TDistance>.Instance);

                // ACORN-gamma compression heuristic for layer 0 (https://arxiv.org/html/2403.04871v1)
                if (GraphCore.Parameters.OptimizeForFiltering && layer == 0 && GraphCore.Parameters.Gamma > 1)
                {
                    AcornCompress(candidates, layer, layerM, output);
                    return;
                }

                // Main stage of moving candidates to the result: a candidate (visited in the order of increasing
                // distance to the target) is selected only when it is closer to the target than to any of the
                // already selected neighbours. This keeps the neighbourhood spread out in different directions
                // which is essential for the navigability of the graph.
                _discarded.Clear();
                foreach (var candidate in candidates)
                {
                    if (output.Count >= layerM)
                    {
                        break;
                    }

                    bool good = true;
                    foreach (var selectedId in output)
                    {
                        if (DistanceUtils.LowerThan(NodeDistance(candidate.Id, selectedId), candidate.Distance))
                        {
                            good = false;
                            break;
                        }
                    }

                    if (good)
                    {
                        output.Add(candidate.Id);
                    }
                    else if (keepPrunedConnections)
                    {
                        _discarded.Add(candidate);
                    }
                }

                // keep pruned option is enabled
                if (keepPrunedConnections)
                {
                    for (int i = 0; i < _discarded.Count && output.Count < layerM; ++i)
                    {
                        output.Add(_discarded[i].Id);
                    }
                }
            }
        }
    }
}
