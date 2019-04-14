// <copyright file="Node.Algorithm4.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <content>
    /// The part with <see cref="Algorithm4{TItem, TDistance}"/> implementation.
    /// </content>
    internal partial struct Node
    {
        /// <summary>
        /// The implementation of the SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates, keepPrunedConnections) algorithm.
        /// Article: Section 4. Algorithm 4.
        /// </summary>
        /// <typeparam name="TItem">The typeof the items in the small world.</typeparam>
        /// <typeparam name="TDistance">The type of the distance in the small world.</typeparam>
        internal sealed class Algorithm4<TItem, TDistance> : Algorithm<TItem, TDistance>
            where TDistance : struct, IComparable<TDistance>
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="Algorithm4{TItem, TDistance}"/> class.
            /// </summary>
            /// <param name="graphCore">The core of the graph.</param>
            public Algorithm4(Graph<TItem, TDistance>.Core graphCore)
                : base(graphCore)
            {
            }

            /// <inheritdoc/>
            internal override IList<int> SelectBestForConnecting(IList<int> candidatesIds, TravelingCosts<int, TDistance> travelingCosts, int layer)
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

                IComparer<int> fartherIsOnTop = travelingCosts;
                IComparer<int> closerIsOnTop = fartherIsOnTop.Reverse();

                var layerM = this.GetM(layer);

                var resultHeap = new BinaryHeap<int>(new List<int>(layerM + 1), fartherIsOnTop);
                var candidatesHeap = new BinaryHeap<int>(candidatesIds, closerIsOnTop);

                // expand candidates option is enabled
                if (this.graphCore.Parameters.ExpandBestSelection)
                {
                    var visited = new HashSet<int>(candidatesHeap.Buffer);
                    foreach (var candidateId in candidatesHeap.Buffer)
                    {
                        foreach (var candidateNeighbourId in this.graphCore.Nodes[candidateId][layer])
                        {
                            if (!visited.Contains(candidateNeighbourId))
                            {
                                candidatesHeap.Push(candidateNeighbourId);
                                visited.Add(candidateNeighbourId);
                            }
                        }
                    }
                }

                // main stage of moving candidates to result
                var discardedHeap = new BinaryHeap<int>(new List<int>(candidatesHeap.Buffer.Count), closerIsOnTop);
                while (candidatesHeap.Buffer.Any() && resultHeap.Buffer.Count < layerM)
                {
                    var candidateId = candidatesHeap.Pop();
                    var farestResultId = resultHeap.Buffer.FirstOrDefault();

                    if (!resultHeap.Buffer.Any()
                    || DistanceUtils.Lt(travelingCosts.From(candidateId), travelingCosts.From(farestResultId)))
                    {
                        resultHeap.Push(candidateId);
                    }
                    else if (this.graphCore.Parameters.KeepPrunedConnections)
                    {
                        discardedHeap.Push(candidateId);
                    }
                }

                // keep pruned option is enabled
                if (this.graphCore.Parameters.KeepPrunedConnections)
                {
                    while (discardedHeap.Buffer.Any() && resultHeap.Buffer.Count < layerM)
                    {
                        resultHeap.Push(discardedHeap.Pop());
                    }
                }

                return resultHeap.Buffer;
            }
        }
    }
}
