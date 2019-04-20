// <copyright file="Node.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics.CodeAnalysis;

    /// <summary>
    /// The implementation of the node in hnsw graph.
    /// </summary>
    [SuppressMessage("Usage", "CA2235:Mark all non-serializable fields", Justification = "Analyzer bug: https://github.com/dotnet/roslyn-analyzers/issues/2156")]
    [Serializable]
    internal partial struct Node
    {
        private int id;
        private IList<IList<int>> connections;

        /// <summary>
        /// Gets the identifier of the node.
        /// </summary>
        public int Id => this.id;

        /// <summary>
        /// Gets the max layer where the node is presented.
        /// </summary>
        public int MaxLayer
        {
            get
            {
                return this.connections.Count - 1;
            }
        }

        /// <summary>
        /// Gets connections ids of the node at the given layer
        /// </summary>
        /// <param name="layer">The layer to get connections at.</param>
        /// <returns>The connections of the node at the given layer.</returns>
        public IReadOnlyList<int> this[int layer]
        {
            get
            {
                // connections[i] must implement IReadOnlyList
                return this.connections[layer] as IReadOnlyList<int>;
            }
        }

        /// <summary>
        /// The abstract class representing algorithm to control node capacity.
        /// </summary>
        /// <typeparam name="TItem">The typeof the items in the small world.</typeparam>
        /// <typeparam name="TDistance">The type of the distance in the small world.</typeparam>
        [SuppressMessage("StyleCop.CSharp.MaintainabilityRules", "SA1401:Fields must be private", Justification = "By design")]
        internal abstract class Algorithm<TItem, TDistance>
            where TDistance : struct, IComparable<TDistance>
        {
            /// <summary>
            /// Gives access to the core of the graph.
            /// </summary>
            protected readonly Graph<TItem, TDistance>.Core graphCore;

            /// <summary>
            /// Cache of the distance function between the nodes.
            /// </summary>
            protected readonly Func<int, int, TDistance> nodeDistance;

            /// <summary>
            /// Initializes a new instance of the <see cref="Algorithm{TItem, TDistance}"/> class
            /// </summary>
            /// <param name="graphCore">The core of the graph.</param>
            public Algorithm(Graph<TItem, TDistance>.Core graphCore)
            {
                this.graphCore = graphCore;
                this.nodeDistance = graphCore.GetDistance;
            }

            /// <summary>
            /// Creates a new instance of the <see cref="Node"/> struct.
            /// Controls the exact type of connection lists.
            /// </summary>
            /// <param name="nodeId">The identifier of the node.</param>
            /// <param name="maxLayer">The max layer where the node is presented.</param>
            /// <returns>The new instance.</returns>
            internal virtual Node NewNode(int nodeId, int maxLayer)
            {
                var connections = new List<IList<int>>(maxLayer + 1);
                for (int layer = 0; layer <= maxLayer; ++layer)
                {
                    // M + 1 neighbours to not realloc in AddConnection when the level is full
                    int layerM = this.GetM(layer) + 1;
                    connections.Add(new List<int>(layerM));
                }

                return new Node
                {
                    id = nodeId,
                    connections = connections
                };
            }

            /// <summary>
            /// The algorithm which selects best neighbours from the candidates for the given node.
            /// </summary>
            /// <param name="candidatesIds">The identifiers of candidates to neighbourhood.</param>
            /// <param name="travelingCosts">Traveling costs to compare candidates.</param>
            /// <param name="layer">The layer of the neighbourhood.</param>
            /// <returns>Best nodes selected from the candidates.</returns>
            internal abstract IList<int> SelectBestForConnecting(IList<int> candidatesIds, TravelingCosts<int, TDistance> travelingCosts, int layer);

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
                return layer == 0 ? 2 * this.graphCore.Parameters.M : this.graphCore.Parameters.M;
            }

            /// <summary>
            /// Tries to connect the node with the new neighbour.
            /// </summary>
            /// <param name="node">The node to add neighbour to.</param>
            /// <param name="neighbour">The new neighbour.</param>
            /// <param name="layer">The layer to add neighbour to.</param>
            internal void Connect(Node node, Node neighbour, int layer)
            {
                node.connections[layer].Add(neighbour.id);
                if (node.connections[layer].Count > this.GetM(layer))
                {
                    var travelingCosts = new TravelingCosts<int, TDistance>(this.nodeDistance, node.id);
                    node.connections[layer] = this.SelectBestForConnecting(node.connections[layer], travelingCosts, layer);
                }
            }
        }
    }
}
