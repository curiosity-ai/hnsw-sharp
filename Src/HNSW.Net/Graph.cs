// <copyright file="Graph.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Runtime.Serialization.Formatters.Binary;
    using System.Text;

    using static HNSW.Net.EventSources;

    /// <summary>
    /// The implementation of a hierarchical small world graph.
    /// </summary>
    /// <typeparam name="TItem">The type of items to connect into small world.</typeparam>
    /// <typeparam name="TDistance">The type of distance between items (expects any numeric type: float, double, decimal, int, ...).</typeparam>
    internal partial class Graph<TItem, TDistance>
        where TDistance : struct, IComparable<TDistance>
    {
        /// <summary>
        /// The distance.
        /// </summary>
        private readonly Func<TItem, TItem, TDistance> distance;

        /// <summary>
        /// The core.
        /// </summary>
        private Core core;

        /// <summary>
        /// The entry point.
        /// </summary>
        private Node entryPoint;

        /// <summary>
        /// Initializes a new instance of the <see cref="Graph{TItem, TDistance}"/> class.
        /// </summary>
        /// <param name="distance">The distance function.</param>
        /// <param name="parameters">The parameters of the world.</param>
        internal Graph(Func<TItem, TItem, TDistance> distance, SmallWorld<TItem, TDistance>.Parameters parameters)
        {
            this.distance = distance;
            this.Parameters = parameters;
        }

        /// <summary>
        /// Gets the parameters.
        /// </summary>
        internal SmallWorld<TItem, TDistance>.Parameters Parameters { get; }

        /// <summary>
        /// Creates graph from the given items.
        /// Contains implementation of INSERT(hnsw, q, M, Mmax, efConstruction, mL) algorithm.
        /// Article: Section 4. Algorithm 1.
        /// </summary>
        /// <param name="items">The items to insert.</param>
        /// <param name="generator">The random number generator to distribute nodes across layers.</param>
        internal void Build(IReadOnlyList<TItem> items, Random generator)
        {
            if (!items?.Any() ?? false)
            {
                return;
            }

            var core = new Core(this.distance, this.Parameters, items);
            core.AllocateNodes(generator);

            var entryPoint = core.Nodes[0];
            var searcher = new Searcher(core);
            Func<int, int, TDistance> nodeDistance = core.GetDistance;
            var neighboursIdsBuffer = new List<int>(core.Algorithm.GetM(0) + 1);

            for (int nodeId = 1; nodeId < core.Nodes.Count; ++nodeId)
            {
                using (new ScopeLatencyTracker(GraphBuildEventSource.Instance?.GraphInsertNodeLatencyReporter))
                {
                    /*
                     * W ← ∅ // list for the currently found nearest elements
                     * ep ← get enter point for hnsw
                     * L ← level of ep // top layer for hnsw
                     * l ← ⌊-ln(unif(0..1))∙mL⌋ // new element’s level
                     * for lc ← L … l+1
                     *   W ← SEARCH-LAYER(q, ep, ef=1, lc)
                     *   ep ← get the nearest element from W to q
                     * for lc ← min(L, l) … 0
                     *   W ← SEARCH-LAYER(q, ep, efConstruction, lc)
                     *   neighbors ← SELECT-NEIGHBORS(q, W, M, lc) // alg. 3 or alg. 4
                     *     for each e ∈ neighbors // shrink connections if needed
                     *       eConn ← neighbourhood(e) at layer lc
                     *       if │eConn│ > Mmax // shrink connections of e if lc = 0 then Mmax = Mmax0
                     *         eNewConn ← SELECT-NEIGHBORS(e, eConn, Mmax, lc) // alg. 3 or alg. 4
                     *         set neighbourhood(e) at layer lc to eNewConn
                     *   ep ← W
                     * if l > L
                     *   set enter point for hnsw to q
                     */

                    // zoom in and find the best peer on the same level as newNode
                    var bestPeer = entryPoint;
                    var currentNode = core.Nodes[nodeId];
                    var currentNodeTravelingCosts = new TravelingCosts<int, TDistance>(nodeDistance, nodeId);
                    for (int layer = bestPeer.MaxLayer; layer > currentNode.MaxLayer; --layer)
                    {
                        searcher.RunKnnAtLayer(bestPeer.Id, currentNodeTravelingCosts, neighboursIdsBuffer, layer, 1);
                        bestPeer = core.Nodes[neighboursIdsBuffer[0]];
                        neighboursIdsBuffer.Clear();
                    }

                    // connecting new node to the small world
                    for (int layer = Math.Min(currentNode.MaxLayer, entryPoint.MaxLayer); layer >= 0; --layer)
                    {
                        searcher.RunKnnAtLayer(bestPeer.Id, currentNodeTravelingCosts, neighboursIdsBuffer, layer, this.Parameters.ConstructionPruning);
                        var bestNeighboursIds = core.Algorithm.SelectBestForConnecting(neighboursIdsBuffer, currentNodeTravelingCosts, layer);

                        for (int i = 0; i < bestNeighboursIds.Count; ++i)
                        {
                            int newNeighbourId = bestNeighboursIds[i];
                            core.Algorithm.Connect(currentNode, core.Nodes[newNeighbourId], layer);
                            core.Algorithm.Connect(core.Nodes[newNeighbourId], currentNode, layer);

                            // if distance from newNode to newNeighbour is better than to bestPeer => update bestPeer
                            if (DistanceUtils.Lt(currentNodeTravelingCosts.From(newNeighbourId), currentNodeTravelingCosts.From(bestPeer.Id)))
                            {
                                bestPeer = core.Nodes[newNeighbourId];
                            }
                        }

                        neighboursIdsBuffer.Clear();
                    }

                    // zoom out to the highest level
                    if (currentNode.MaxLayer > entryPoint.MaxLayer)
                    {
                        entryPoint = currentNode;
                    }

                    // report distance cache hit rate
                    GraphBuildEventSource.Instance?.CoreGetDistanceCacheHitRateReporter?.Invoke(core.DistanceCacheHitRate);
                }
            }

            // construction is done
            this.core = core;
            this.entryPoint = entryPoint;
        }

        /// <summary>
        /// Get k nearest items for a given one.
        /// Contains implementation of K-NN-SEARCH(hnsw, q, K, ef) algorithm.
        /// Article: Section 4. Algorithm 5.
        /// </summary>
        /// <param name="destination">The given node to get the nearest neighbourhood for.</param>
        /// <param name="k">The size of the neighbourhood.</param>
        /// <returns>The list of the nearest neighbours.</returns>
        internal IList<SmallWorld<TItem, TDistance>.KNNSearchResult> KNearest(TItem destination, int k)
        {
            using (new ScopeLatencyTracker(GraphSearchEventSource.Instance?.GraphKNearestLatencyReporter))
            {
                // TODO: hack we know that destination id is -1.
                TDistance RuntimeDistance(int x, int y)
                {
                    int nodeId = x >= 0 ? x : y;
                    return this.distance(destination, this.core.Items[nodeId]);
                }

                var bestPeer = this.entryPoint;
                var searcher = new Searcher(this.core);
                var destiantionTravelingCosts = new TravelingCosts<int, TDistance>(RuntimeDistance, -1);
                var resultIds = new List<int>(k + 1);

                int visitedNodesCount = 0;
                for (int layer = this.entryPoint.MaxLayer; layer > 0; --layer)
                {
                    visitedNodesCount += searcher.RunKnnAtLayer(bestPeer.Id, destiantionTravelingCosts, resultIds, layer, 1);
                    bestPeer = this.core.Nodes[resultIds[0]];
                    resultIds.Clear();
                }

                visitedNodesCount += searcher.RunKnnAtLayer(bestPeer.Id, destiantionTravelingCosts, resultIds, 0, k);
                GraphSearchEventSource.Instance?.GraphKNearestVisitedNodesReporter?.Invoke(visitedNodesCount);

                return resultIds.Select(id => new SmallWorld<TItem, TDistance>.KNNSearchResult
                {
                    Id = id,
                    Item = this.core.Items[id],
                    Distance = RuntimeDistance(id, -1)
                }).ToList();
            }
        }

        /// <summary>
        /// Serializes core of the graph.
        /// </summary>
        /// <returns>Bytes representing edges.</returns>
        internal byte[] Serialize()
        {
            using (var stream = new MemoryStream())
            {
                var formatter = new BinaryFormatter();
                formatter.Serialize(stream, this.core.Serialize());
                formatter.Serialize(stream, this.entryPoint);
                return stream.ToArray();
            }
        }

        /// <summary>
        /// Deserilaizes graph edges and assigns nodes to the items.
        /// </summary>
        /// <param name="items">The underlying items.</param>
        /// <param name="bytes">The serialized edges.</param>
        internal void Deserialize(IReadOnlyList<TItem> items, byte[] bytes)
        {
            using (var stream = new MemoryStream(bytes))
            {
                var formatter = new BinaryFormatter();

                var coreBytes = (byte[])formatter.Deserialize(stream);
                var core = new Core(this.distance, this.Parameters, items);
                core.Deserialize(coreBytes);

                this.entryPoint = (Node)formatter.Deserialize(stream);
                this.core = core;
            }
        }

        /// <summary>
        /// Prints edges of the graph.
        /// </summary>
        /// <returns>String representation of the graph's edges.</returns>
        internal string Print()
        {
            var buffer = new StringBuilder();
            for (int layer = this.entryPoint.MaxLayer; layer >= 0; --layer)
            {
                buffer.AppendLine($"[LEVEL {layer}]");
                BFS(this.core, this.entryPoint, layer, (node) =>
                {
                    var neighbours = string.Join(", ", node[layer]);
                    buffer.AppendLine($"({node.Id}) -> {{{neighbours}}}");
                });

                buffer.AppendLine();
            }

            return buffer.ToString();
        }
    }
}
