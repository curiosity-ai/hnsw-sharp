// <copyright file="Node.cs" company="Microsoft">
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
        /// The abstract class representing algorithm to control node capacity.
        /// </summary>
        /// <typeparam name="TItem">The typeof the items in the small world.</typeparam>
        /// <typeparam name="TDistance">The type of the distance in the small world.</typeparam>
        internal abstract class Algorithm<TItem, TDistance> where TDistance : struct, IComparable<TDistance>
        {
            protected readonly Graph<TItem, TDistance>.Core GraphCore;

            protected readonly Func<int, int, TDistance> NodeDistance;

            // Scratch buffers for Connect. Graph construction is single-writer, so reusing them is safe.
            private readonly List<Candidate<TDistance>> _connectCandidates = new List<Candidate<TDistance>>();
            private readonly List<int> _connectSelected = new List<int>();

            public Algorithm(Graph<TItem, TDistance>.Core graphCore)
            {
                GraphCore = graphCore;
                NodeDistance = graphCore.GetDistance;
            }

            /// <summary>
            /// Creates a new instance of the <see cref="Node"/> struct. Controls the exact type of connection lists.
            /// </summary>
            /// <param name="nodeId">The identifier of the node.</param>
            /// <param name="maxLayer">The max layer where the node is presented.</param>
            /// <returns>The new instance.</returns>
            internal virtual Node NewNode(int nodeId, int maxLayer)
            {
                var connections = new List<List<int>>(maxLayer + 1);
                for (int layer = 0; layer <= maxLayer; ++layer)
                {
                    // M + 1 neighbours to not realloc in AddConnection when the level is full
                    int layerM = GetM(layer) + 1;
                    connections.Add(new List<int>(layerM));
                }

                return new Node(connections, nodeId);
            }

            /// <summary>
            /// The algorithm which selects best neighbours from the candidates for the given node.
            /// </summary>
            /// <param name="candidates">The candidates to the neighbourhood together with their distances to the target node. The list may be reordered in place.</param>
            /// <param name="targetId">The identifier of the node the neighbourhood is being built for.</param>
            /// <param name="layer">The layer of the neighbourhood.</param>
            /// <param name="output">The list to fill with the identifiers of the selected neighbours. Cleared on entry.</param>
            internal abstract void SelectBestForConnecting(List<Candidate<TDistance>> candidates, int targetId, int layer, List<int> output);

            /// <summary>
            /// Get maximum allowed connections for the given level.
            /// </summary>
            /// <remarks>
            /// Article: Section 4.1:
            /// "Selection of the Mmax0 (the maximum number of connections that an element can have in the zero layer) also
            /// has a strong influence on the search performance, especially in case of high quality(high recall) search.
            /// Simulations show that setting Mmax0 to M(this corresponds to kNN graphs on each layer if the neighbors
            /// selection heuristic is not used) leads to a very strong performance penalty at high recall.
            /// Simulations also suggest that 2∙M is a good choice for Mmax0;
            /// setting the parameter higher leads to performance degradation and excessive memory usage."
            /// </remarks>
            /// <param name="layer">The level of the layer.</param>
            /// <returns>The maximum number of connections.</returns>
            internal int GetM(int layer)
            {
                int m = layer == 0 ? 2 * GraphCore.Parameters.M : GraphCore.Parameters.M;
                if (GraphCore.Parameters.OptimizeForFiltering) // ACORN graph expansion (https://arxiv.org/html/2403.04871v1)
                {
                    m *= GraphCore.Parameters.Gamma;
                }
                return m;
            }

            /// <summary>
            /// Tries to connect the node with the new neighbour.
            /// </summary>
            /// <param name="node">The node to add neighbour to.</param>
            /// <param name="neighbour">The new neighbour.</param>
            /// <param name="layer">The layer to add neighbour to.</param>
            internal void Connect(ref Node node, ref Node neighbour, int layer)
            {
                var nodeLayer = node.GetLayerForModifying(layer);
                nodeLayer.Add(neighbour.Id);
                if (nodeLayer.Count > GetM(layer))
                {
                    _connectCandidates.Clear();
                    foreach (var candidateId in nodeLayer)
                    {
                        _connectCandidates.Add(new Candidate<TDistance>(NodeDistance(node.Id, candidateId), candidateId));
                    }

                    SelectBestForConnecting(_connectCandidates, node.Id, layer, _connectSelected);

                    nodeLayer.Clear();
                    nodeLayer.AddRange(_connectSelected);
                    node.SetLayer(layer, nodeLayer);
                }
            }

            /// <summary>
            /// ACORN-gamma compression heuristic for layer 0 (https://arxiv.org/html/2403.04871v1).
            /// Expects the candidates to be sorted by ascending distance to the target.
            /// </summary>
            protected void AcornCompress(List<Candidate<TDistance>> sortedCandidates, int layer, int bestN, List<int> output)
            {
                int mb = GraphCore.Parameters.Mb;

                for (int i = 0; i < Math.Min(mb, sortedCandidates.Count); i++)
                {
                    output.Add(sortedCandidates[i].Id);
                }

                var h = new HashSet<int>();
                for (int i = mb; i < sortedCandidates.Count; i++)
                {
                    if (output.Count + h.Count >= bestN)
                    {
                        break;
                    }

                    int c = sortedCandidates[i].Id;
                    if (h.Contains(c))
                    {
                        continue;
                    }

                    output.Add(c);

                    var neighbors = GraphCore.Nodes[c].EnumerateLayer(layer);
                    foreach (var neighbor in neighbors)
                    {
                        h.Add(neighbor);
                    }
                }
            }
        }
    }
}
