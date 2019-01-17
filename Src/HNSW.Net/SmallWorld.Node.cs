// <copyright file="SmallWorld.Node.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <content>
    /// The part with the implementaion of a node in the hnsw graph.
    /// </content>
    public partial class SmallWorld<TItem, TDistance>
    {
        /// <summary>
        /// The abstract node implementation.
        /// The <see cref="SelectBestForConnecting(IList{Node})"/> must be implemented by the subclass.
        /// </summary>
        private abstract class Node
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="Node"/> class.
            /// </summary>
            /// <param name="id">The identifier of the node.</param>
            /// <param name="item">The item which is represented by the node.</param>
            /// <param name="maxLevel">The maximum level until which the node exists.</param>
            /// <param name="parameters">The parameters of the algorithm.</param>
            public Node(int id, TItem item, int maxLevel, Parameters parameters)
            {
                this.Id = id;
                this.Item = item;
                this.MaxLevel = maxLevel;
                this.Parameters = parameters;

                this.Connections = new List<IList<Node>>(this.MaxLevel + 1);
                this.Connections.Add(new List<Node>(2 * this.Parameters.M));
                for (int level = 1; level <= this.MaxLevel; ++level)
                {
                    this.Connections.Add(new List<Node>(this.Parameters.M));
                }

                Func<Node, Node, TDistance> nodesDistance = (x, y) => this.Parameters.Distance(x.Item, y.Item);
                this.TravelingCosts = new TravelingCosts<Node, TDistance>(nodesDistance, this);
            }

            /// <summary>
            /// Gets the identifier of the node.
            /// </summary>
            public int Id { get; private set; }

            /// <summary>
            /// Gets the maximum level of the node.
            /// </summary>
            public int MaxLevel { get; private set; }

            /// <summary>
            /// Gets the item associated with the node.
            /// </summary>
            public TItem Item { get; private set; }

            /// <summary>
            /// Gets traveling costs from any other node to this one.
            /// </summary>
            public TravelingCosts<Node, TDistance> TravelingCosts { get; private set; }

            /// <summary>
            /// Gets parameters of the algorithm.
            /// </summary>
            protected Parameters Parameters { get; private set; }

            /// <summary>
            /// Gets all connections of the node on all layers.
            /// </summary>
            protected IList<IList<Node>> Connections { get; private set; }

            /// <summary>
            /// Get connections of the node on the given layer.
            /// </summary>
            /// <param name="level">The level of the layer.</param>
            /// <returns>List of connected nodes.</returns>
            public IReadOnlyList<Node> GetConnections(int level)
            {
                if (level < this.Connections.Count)
                {
                    // this cast is needed
                    // https://visualstudio.uservoice.com/forums/121579-visual-studio-ide/suggestions/2845892-make-ilist-t-inherited-from-ireadonlylist-t
                    return (List<Node>)this.Connections[level];
                }

                return Enumerable.Empty<Node>().ToList();
            }

            /// <summary>
            /// Add connections to the node on the specific layer.
            /// </summary>
            /// <param name="newNeighbour">The node to connect with.</param>
            /// <param name="level">The level of the layer.</param>
            public void AddConnection(Node newNeighbour, int level)
            {
                var levelNeighbours = this.Connections[level];
                levelNeighbours.Add(newNeighbour);
                if (levelNeighbours.Count > this.GetM(level))
                {
                    this.Connections[level] = this.SelectBestForConnecting(levelNeighbours);
                }
            }

            /// <summary>
            /// The algorithm which selects best neighbours from the candidates for this node.
            /// </summary>
            /// <param name="candidates">The candidates for connecting.</param>
            /// <returns>Best nodes selected from the candidates.</returns>
            public abstract IList<Node> SelectBestForConnecting(IList<Node> candidates);

            /// <summary>
            /// Get maximum allowed connections for the given layer.
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
            /// <param name="level">The level of the layer.</param>
            /// <returns>The maximum number of connections.</returns>
            protected int GetM(int level)
            {
                return level == 0 ? 2 * this.Parameters.M : this.Parameters.M;
            }
        }

        /// <summary>
        /// The implementation of the SELECT-NEIGHBORS-SIMPLE(q, C, M) algorithm.
        /// Article: Section 4. Algorithm 3.
        /// </summary>
        private class NodeAlg3 : Node
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="NodeAlg3"/> class.
            /// </summary>
            /// <param name="id">The identifier of the node.</param>
            /// <param name="item">The item which is represented by the node.</param>
            /// <param name="maxLevel">The maximum level until which the node exists.</param>
            /// <param name="parameters">The parameters of the algorithm.</param>
            public NodeAlg3(int id, TItem item, int maxLevel, Parameters parameters)
                : base(id, item, maxLevel, parameters)
            {
            }

            /// <inheritdoc />
            public override IList<Node> SelectBestForConnecting(IList<Node> candidates)
            {
                /*
                 * q ← this
                 * return M nearest elements from C to q
                 */

                IComparer<Node> fartherIsLess = this.TravelingCosts.Reverse();
                var candidatesHeap = new BinaryHeap<Node>(candidates, fartherIsLess);

                var result = new List<Node>(this.GetM(this.MaxLevel) + 1);
                while (candidatesHeap.Buffer.Any() && result.Count < this.GetM(this.MaxLevel))
                {
                    result.Add(candidatesHeap.Pop());
                }

                return result;
            }
        }

        /// <summary>
        /// The implementation of the SELECT-NEIGHBORS-HEURISTIC(q, C, M, lc, extendCandidates, keepPrunedConnections) algorithm.
        /// Article: Section 4. Algorithm 4.
        /// </summary>
        private class NodeAlg4 : Node
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="NodeAlg4"/> class.
            /// </summary>
            /// <param name="id">The identifier of the node.</param>
            /// <param name="item">The item which is represented by the node.</param>
            /// <param name="maxLevel">The maximum level until which the node exists.</param>
            /// <param name="parameters">The parameters of the algorithm.</param>
            public NodeAlg4(int id, TItem item, int maxLevel, Parameters parameters)
                : base(id, item, maxLevel, parameters)
            {
            }

            /// <inheritdoc />
            public override IList<Node> SelectBestForConnecting(IList<Node> candidates)
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

                IComparer<Node> closerIsLess = this.TravelingCosts;
                IComparer<Node> fartherIsLess = closerIsLess.Reverse();

                var resultHeap = new BinaryHeap<Node>(new List<Node>(this.GetM(this.MaxLevel) + 1), closerIsLess);
                var candidatesHeap = new BinaryHeap<Node>(candidates, fartherIsLess);

                // expand candidates option is enabled
                if (this.Parameters.ExpandBestSelection)
                {
                    var candidatesIds = new HashSet<int>(candidates.Select(c => c.Id));
                    foreach (var neighbour in this.GetConnections(this.MaxLevel))
                    {
                        if (!candidatesIds.Contains(neighbour.Id))
                        {
                            candidatesHeap.Push(neighbour);
                            candidatesIds.Add(neighbour.Id);
                        }
                    }
                }

                // main stage of moving candidates to result
                var discardedHeap = new BinaryHeap<Node>(new List<Node>(candidatesHeap.Buffer.Count), fartherIsLess);
                while (candidatesHeap.Buffer.Any() && resultHeap.Buffer.Count < this.GetM(this.MaxLevel))
                {
                    var candidate = candidatesHeap.Pop();
                    var farestResult = resultHeap.Buffer.FirstOrDefault();

                    if (farestResult == null
                    || DLt(this.TravelingCosts.From(candidate), this.TravelingCosts.From(farestResult)))
                    {
                        resultHeap.Push(candidate);
                    }
                    else if (this.Parameters.KeepPrunedConnections)
                    {
                        discardedHeap.Push(candidate);
                    }
                }

                // keep pruned option is enabled
                if (this.Parameters.KeepPrunedConnections)
                {
                    while (discardedHeap.Buffer.Any() && resultHeap.Buffer.Count < this.GetM(this.MaxLevel))
                    {
                        resultHeap.Push(discardedHeap.Pop());
                    }
                }

                return resultHeap.Buffer;
            }
        }
    }
}
