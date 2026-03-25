// <copyright file="Node.Algorithm3.cs" company="Microsoft">
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
        /// The implementation of the SELECT-NEIGHBORS-SIMPLE(q, C, M) algorithm.
        /// Article: Section 4. Algorithm 3.
        /// </summary>
        /// <typeparam name="TItem">The typeof the items in the small world.</typeparam>
        /// <typeparam name="TDistance">The type of the distance in the small world.</typeparam>
        internal class Algorithm3<TItem, TDistance> : Algorithm<TItem, TDistance> where TDistance : struct, IComparable<TDistance>
        {
            public Algorithm3(Graph<TItem, TDistance>.Core graphCore) : base(graphCore)
            {
            }

            /// <inheritdoc/>
            internal override List<int> SelectBestForConnecting(List<int> candidatesIds, TravelingCosts<int, TDistance> travelingCosts, int layer)
            {
                /*
                 * q ← this
                 * return M nearest elements from C to q
                 */

                var bestN = GetM(layer);
                var candidatesHeap = new BinaryHeap(candidatesIds, travelingCosts);

                // ACORN-gamma compression heuristic for layer 0 (https://arxiv.org/html/2403.04871v1)
                if (GraphCore.Parameters.OptimizeForFiltering && layer == 0 && GraphCore.Parameters.Gamma > 1)
                {
                    var sortedCandidates = new List<int>(candidatesHeap.Buffer);
                    sortedCandidates.Sort((a, b) => travelingCosts.From(a).CompareTo(travelingCosts.From(b)));

                    int mb = GraphCore.Parameters.Mb;
                    var result = new List<int>(bestN);

                    for (int i = 0; i < Math.Min(mb, sortedCandidates.Count); i++)
                    {
                        result.Add(sortedCandidates[i]);
                    }

                    var h = new HashSet<int>();
                    for (int i = mb; i < sortedCandidates.Count; i++)
                    {
                        if (result.Count + h.Count >= bestN)
                        {
                            break;
                        }

                        int c = sortedCandidates[i];
                        if (h.Contains(c))
                        {
                            continue;
                        }

                        result.Add(c);

                        var neighbors = GraphCore.Nodes[c].EnumerateLayer(layer);
                        foreach (var neighbor in neighbors)
                        {
                            h.Add(neighbor);
                        }
                    }

                    return result;
                }
                else
                {
                    // !NO COPY! in-place selection
                    while (candidatesHeap.Buffer.Count > bestN)
                    {
                        candidatesHeap.Pop();
                    }

                    return candidatesHeap.Buffer;
                }
            }
        }
    }
}