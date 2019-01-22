// <copyright file="SmallWorld.Graph.cs" company="Microsoft">
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

    /// <content>
    /// The part with the implemnation of a hierarchical small world graph.
    /// </content>
    public partial class SmallWorld<TItem, TDistance>
    {
        /// <summary>
        /// The layered graph implementation.
        /// </summary>
        internal class Graph
        {
            private Node entryPoint;

            /// <summary>
            /// Initializes a new instance of the <see cref="Graph"/> class.
            /// </summary>
            /// <param name="distance">The distance funtion to use in the small world.</param>
            /// <param name="parameters">The parameters of the algorithm.</param>
            public Graph(Func<TItem, TItem, TDistance> distance, Parameters parameters)
            {
                this.Parameters = parameters;
                switch (this.Parameters.NeighbourHeuristic)
                {
                    case NeighbourSelectionHeuristic.SelectHeuristic:
                        this.NewNode = (id, item, level) => new NodeAlg4(id, item, level, distance, this.Parameters);
                        break;

                    case NeighbourSelectionHeuristic.SelectSimple:
                    default:
                        this.NewNode = (id, item, level) => new NodeAlg3(id, item, level, distance, this.Parameters);
                        break;
                }
            }

            /// <summary>
            /// Gets parameters of the algorithm.
            /// </summary>
            public Parameters Parameters { get; private set; }

            /// <summary>
            /// Gets the node factory associated with the graph.
            /// The node construction arguments are:
            /// 1st: int -> the id of the new node;
            /// 2nd: TItem -> the item to attach to the node;
            /// 3rd: int -> the level of the node.
            /// </summary>
            public Func<int, TItem, int, Node> NewNode { get; private set; }

            /// <summary>
            /// Creates graph from the given items.
            /// Contains implementation of INSERT(hnsw, q, M, Mmax, efConstruction, mL) algorithm.
            /// Article: Section 4. Algorithm 1.
            /// </summary>
            /// <param name="items">The items to insert.</param>
            /// <param name="generator">The random number generator to use in <see cref="RandomLevel"/></param>
            public void Create(IList<TItem> items, Random generator)
            {
                if (!items?.Any() ?? false)
                {
                    return;
                }

                int id = 0;
                var entryPoint = this.NewNode(id, items[id], RandomLevel(generator, this.Parameters.LevelLambda));

                for (id = 1; id < items.Count; ++id)
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
                    var newNode = this.NewNode(id, items[id], RandomLevel(generator, this.Parameters.LevelLambda));
                    for (int level = bestPeer.MaxLevel; level > newNode.MaxLevel; --level)
                    {
                        bestPeer = KNearestAtLevel(bestPeer, newNode, 1, level).Single();
                    }

                    // connecting new node to the small world
                    for (int level = Math.Min(newNode.MaxLevel, entryPoint.MaxLevel); level >= 0; --level)
                    {
                        var potentialNeighbours = KNearestAtLevel(bestPeer, newNode, this.Parameters.ConstructionPruning, level);
                        var bestNeighbours = newNode.SelectBestForConnecting(potentialNeighbours);

                        foreach (var newNeighbour in bestNeighbours)
                        {
                            newNode.AddConnection(newNeighbour, level);
                            newNeighbour.AddConnection(newNode, level);

                            // if distance from newNode to newNeighbour is better than to bestPeer => update bestPeer
                            if (DLt(newNode.TravelingCosts.From(newNeighbour), newNode.TravelingCosts.From(bestPeer)))
                            {
                                bestPeer = newNeighbour;
                            }
                        }
                    }

                    // zoom out to the highest level
                    if (newNode.MaxLevel > entryPoint.MaxLevel)
                    {
                        entryPoint = newNode;
                    }
                }

                // construction is done
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
            public IList<Node> KNearest(Node destination, int k)
            {
                var bestPeer = this.entryPoint;
                for (int level = this.entryPoint.MaxLevel; level > 0; --level)
                {
                    bestPeer = KNearestAtLevel(bestPeer, destination, 1, level).Single();
                }

                return KNearestAtLevel(bestPeer, destination, k, 0);
            }

            /// <summary>
            /// Serializes edges of the graph.
            /// </summary>
            /// <returns>Bytes representing edges.</returns>
            public byte[] Serialize()
            {
                using (var stream = new MemoryStream())
                {
                    var formatter = new BinaryFormatter();
                    formatter.Serialize(stream, this.entryPoint.Id);
                    formatter.Serialize(stream, this.entryPoint.MaxLevel);

                    for (int level = this.entryPoint.MaxLevel; level >= 0; --level)
                    {
                        var edges = new Dictionary<int, List<int>>();
                        BFS(this.entryPoint, level, (node) =>
                        {
                            edges[node.Id] = node.GetConnections(level).Select(x => x.Id).ToList();
                        });

                        formatter.Serialize(stream, edges);
                    }

                    return stream.ToArray();
                }
            }

            /// <summary>
            /// Deserilaizes graph edges and assigns nodes to the items.
            /// </summary>
            /// <param name="items">The underlying items.</param>
            /// <param name="bytes">The serialized edges.</param>
            public void Deserialize(IList<TItem> items, byte[] bytes)
            {
                var nodeList = Enumerable.Repeat<Node>(null, items.Count).ToList();
                Func<int, int, Node> getOrAdd = (id, level) => nodeList[id] = nodeList[id] ?? this.NewNode(id, items[id], level);

                using (var stream = new MemoryStream(bytes))
                {
                    var formatter = new BinaryFormatter();
                    int entryId = (int)formatter.Deserialize(stream);
                    int maxLevel = (int)formatter.Deserialize(stream);

                    nodeList[entryId] = this.NewNode(entryId, items[entryId], maxLevel);
                    for (int level = maxLevel; level >= 0; --level)
                    {
                        var edges = (Dictionary<int, List<int>>)formatter.Deserialize(stream);
                        foreach (var pair in edges)
                        {
                            var currentNode = getOrAdd(pair.Key, level);
                            foreach (var adjacentId in pair.Value)
                            {
                                var neighbour = getOrAdd(adjacentId, level);
                                currentNode.AddConnection(neighbour, level);
                            }
                        }
                    }

                    this.entryPoint = nodeList[entryId];
                }
            }

            /// <summary>
            /// Prints edges of the graph.
            /// </summary>
            /// <returns>String representation of the graph's edges.</returns>
            internal string Print()
            {
                var buffer = new StringBuilder();
                for (int level = this.entryPoint.MaxLevel; level >= 0; --level)
                {
                    buffer.AppendLine($"[LEVEL {level}]");
                    BFS(this.entryPoint, level, (node) =>
                    {
                        var neighbours = string.Join(", ", node.GetConnections(level).Select(x => x.Id));
                        buffer.AppendLine($"({node.Id}) -> {{{neighbours}}}");
                    });

                    buffer.AppendLine();
                }

                return buffer.ToString();
            }

            /// <summary>
            /// The implementaiton of SEARCH-LAYER(q, ep, ef, lc) algorithm.
            /// Article: Section 4. Algorithm 2.
            /// </summary>
            /// <param name="entryPoint">The entry point for the search.</param>
            /// <param name="destination">The search target.</param>
            /// <param name="k">The number of the nearest neighbours to get from the layer.</param>
            /// <param name="level">Level of the layer.</param>
            /// <returns>The list of the nearest neighbours at the level.</returns>
            private static IList<Node> KNearestAtLevel(Node entryPoint, Node destination, int k, int level)
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
                IComparer<Node> closerIsLess = destination.TravelingCosts;
                IComparer<Node> fartherIsLess = closerIsLess.Reverse();

                // prepare heaps
                var resultHeap = new BinaryHeap<Node>(new List<Node>(k + 1) { entryPoint }, closerIsLess);
                var expansionHeap = new BinaryHeap<Node>(new List<Node>() { entryPoint }, fartherIsLess);

                // run bfs
                var visited = new HashSet<int>() { entryPoint.Id };
                while (expansionHeap.Buffer.Any())
                {
                    // get next candidate to check and expand
                    var toExpand = expansionHeap.Pop();
                    var farthestResult = resultHeap.Buffer.First();
                    if (DGt(destination.TravelingCosts.From(toExpand), destination.TravelingCosts.From(farthestResult)))
                    {
                        // the closest candidate is farther than farthest result
                        break;
                    }

                    // expand candidate
                    foreach (var neighbour in toExpand.GetConnections(level))
                    {
                        if (!visited.Contains(neighbour.Id))
                        {
                            // enque perspective neighbours to expansion list
                            farthestResult = resultHeap.Buffer.First();
                            if (resultHeap.Buffer.Count < k
                            || DLt(destination.TravelingCosts.From(neighbour), destination.TravelingCosts.From(farthestResult)))
                            {
                                expansionHeap.Push(neighbour);
                                resultHeap.Push(neighbour);
                                if (resultHeap.Buffer.Count > k)
                                {
                                    resultHeap.Pop();
                                }
                            }

                            // update visited list
                            visited.Add(neighbour.Id);
                        }
                    }
                }

                return resultHeap.Buffer;
            }

            /// <summary>
            /// Gets the level for the layer.
            /// </summary>
            /// <param name="generator">The random numbers generator.</param>
            /// <param name="lambda">Poisson lambda.</param>
            /// <returns>The level value.</returns>
            private static int RandomLevel(Random generator, double lambda)
            {
                var r = -Math.Log(generator.NextDouble()) * lambda;
                return (int)r;
            }
        }
    }
}