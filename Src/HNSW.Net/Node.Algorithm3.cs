// <copyright file="Node.Algorithm3.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;

    /// <content>
    /// The part with <see cref="Algorithm3{TItem, TDistance}"/> implementation.
    /// </content>
    internal partial struct Node
    {
        /// <summary>
        /// The implementation of the SELECT-NEIGHBORS-SIMPLE(q, C, M) algorithm.
        /// Article: Section 4. Algorithm 3.
        /// </summary>
        /// <typeparam name="TItem">The typeof the items in the small world.</typeparam>
        /// <typeparam name="TDistance">The type of the distance in the small world.</typeparam>
        internal class Algorithm3<TItem, TDistance> : Algorithm<TItem, TDistance>
            where TDistance : struct, IComparable<TDistance>
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="Algorithm3{TItem, TDistance}"/> class.
            /// </summary>
            /// <param name="graphCore">The core of the graph.</param>
            public Algorithm3(Graph<TItem, TDistance>.Core graphCore)
                : base(graphCore)
            {
            }

            /// <inheritdoc/>
            internal override IList<int> SelectBestForConnecting(IList<int> candidatesIds, TravelingCosts<int, TDistance> travelingCosts, int layer)
            {
                /*
                 * q ← this
                 * return M nearest elements from C to q
                 */

                // !NO COPY! in-place selection
                var bestN = this.GetM(layer);
                var candidatesHeap = new BinaryHeap<int>(candidatesIds, travelingCosts);
                while (candidatesHeap.Buffer.Count > bestN)
                {
                    candidatesHeap.Pop();
                }

                return candidatesHeap.Buffer;
            }
        }
    }
}