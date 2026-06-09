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
            internal override void SelectBestForConnecting(List<Candidate<TDistance>> candidates, int targetId, int layer, List<int> output)
            {
                /*
                 * q ← this
                 * return M nearest elements from C to q
                 */

                output.Clear();

                var bestN = GetM(layer);
                candidates.Sort(CandidateComparer<TDistance>.Instance);

                // ACORN-gamma compression heuristic for layer 0 (https://arxiv.org/html/2403.04871v1)
                if (GraphCore.Parameters.OptimizeForFiltering && layer == 0 && GraphCore.Parameters.Gamma > 1)
                {
                    AcornCompress(candidates, layer, bestN, output);
                    return;
                }

                int count = Math.Min(bestN, candidates.Count);
                for (int i = 0; i < count; ++i)
                {
                    output.Add(candidates[i].Id);
                }
            }
        }
    }
}
