// <copyright file="SmallWorld.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics.CodeAnalysis;
    using System.Linq;

    /// <summary>
    /// The Hierarchical Navigable Small World Graphs.
    /// https://arxiv.org/abs/1603.09320
    /// </summary>
    /// <typeparam name="TItem">The type of items to connect into small world.</typeparam>
    /// <typeparam name="TDistance">The type of distance between items (expect any numeric type: float, double, decimal, int, ...).</typeparam>
    public partial class SmallWorld<TItem, TDistance>
        where TDistance : IComparable<TDistance>
    {
        /// <summary>
        /// The parameters for the hnsw algorithm.
        /// </summary>
        private readonly Parameters parameters;

        /// <summary>
        /// The hierarchical small world graph instance.
        /// </summary>
        private Graph graph;

        /// <summary>
        /// Initializes a new instance of the <see cref="SmallWorld{TItem, TDistance}"/> class.
        /// </summary>
        /// <param name="parameters">Parameters of the algorithm.</param>
        public SmallWorld(Parameters parameters)
        {
            this.parameters = parameters;
        }

        /// <summary>
        /// Type of heuristic to select best neighbours for a node.
        /// </summary>
        public enum NeighbourSelectionHeuristic
        {
            /// <summary>
            /// Marker for the Algorithm 3 (SELECT-NEIGHBORS-SIMPLE) from the article.
            /// Implemented in <see cref="SmallWorld{TItem, TDistance}.NodeAlg3"/>
            /// </summary>
            SelectSimple,

            /// <summary>
            /// Marker for the Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC) from the article.
            /// Implemented in <see cref="SmallWorld{TItem, TDistance}.NodeAlg4"/>
            /// </summary>
            SelectHeuristic
        }

        /// <summary>
        /// Builds hnsw graph from the items.
        /// </summary>
        /// <param name="items">The items to connect into the graph.</param>
        public void BuildGraph(IEnumerable<TItem> items)
        {
            var graph = new Graph(this.parameters);
            graph.Create(items);
            this.graph = graph;
        }

        /// <summary>
        /// Run knn search for a given item.
        /// </summary>
        /// <param name="item">The item to search nearest neighbours.</param>
        /// <param name="k">The number of nearest neighbours.</param>
        /// <returns>The list of found nearest neighbours.</returns>
        public IList<KNNSearchResult> KNNSearch(TItem item, int k)
        {
            var destination = this.graph.NewNode(-1, item, 0);
            var neighbourhood = this.graph.KNearest(destination, k);
            return neighbourhood.Select(
                n => new KNNSearchResult
                {
                    Id = n.Id,
                    Item = n.Item,
                    Distance = destination.TravelingCosts.From(n)
                }).ToList();
        }

        /// <summary>
        /// Serializes the internal graph.
        /// </summary>
        /// <returns>Bytes representing the graph.</returns>
        public byte[] SerializeGraph()
        {
            // TODO: implement serialization.
            throw new NotImplementedException("TODO");
        }

        /// <summary>
        /// Deserializes the graph from byte array.
        /// </summary>
        /// <param name="items">The items to assign to the graph's verticies.</param>
        /// <param name="graph">The serialized grpah.</param>
        public void DeserializeGraph(IEnumerable<TItem> items, byte[] graph)
        {
            // TODO: implement deserialization.
            throw new NotImplementedException("TODO");
        }

        /// <summary>
        /// Parameters of the algorithm.
        /// </summary>
        [SuppressMessage("Design", "CA1034:Nested types should not be visible", Justification = "By Design")]
        public class Parameters
        {
            /// <summary>
            /// Initializes a new instance of the <see cref="Parameters"/> class.
            /// </summary>
            /// <param name="distance">The distance computation function.</param>
            public Parameters(Func<TItem, TItem, TDistance> distance)
            {
                this.Distance = distance;
                this.M = 10;
                this.LevelLambda = 1 / Math.Log(this.M);
                this.Generator = new Random(42);
                this.NeighbourHeuristic = NeighbourSelectionHeuristic.SelectSimple;
                this.ConstructionPruning = 200;
                this.ExpandBestSelection = false;
                this.KeepPrunedConnections = true;
            }

            /// <summary>
            /// Gets function representing distance in the items space.
            /// </summary>
            public Func<TItem, TItem, TDistance> Distance { get; private set; }

            /// <summary>
            /// Gets or sets the parameter which defines the maximum number of neighbors in the zero and above-zero layers.
            /// The maximum number of neighbors for the zero layer is 2 * M.
            /// The maximum number of neighbors for higher layers is M.
            /// </summary>
            public int M { get; set; }

            /// <summary>
            /// Gets or sets the max level decay parameter.
            /// https://en.wikipedia.org/wiki/Exponential_distribution
            /// See 'mL' parameter in the HNSW article.
            /// </summary>
            public double LevelLambda { get; set; }

            /// <summary>
            /// Gets or sets the seed for random numbers generator.
            /// </summary>
            public Random Generator { get; set; }

            /// <summary>
            /// Gets or sets parameter which specifies the type of heuristic to use for best neighbours selection.
            /// </summary>
            public NeighbourSelectionHeuristic NeighbourHeuristic { get; set; }

            /// <summary>
            /// Gets or sets the number of candidates to consider as neighbousr for a given node at the graph construction phase.
            /// See 'efConstruction' parameter in the article.
            /// </summary>
            public int ConstructionPruning { get; set; }

            /// <summary>
            /// Gets or sets a value indicating whether to expand candidates if <see cref="NeighbourSelectionHeuristic.SelectHeuristic"/> is used.
            /// See 'extendCandidates' parameter in the article.
            /// </summary>
            public bool ExpandBestSelection { get; set; }

            /// <summary>
            /// Gets or sets a value indicating whether to keep pruned candidates if <see cref="NeighbourSelectionHeuristic.SelectHeuristic"/> is used.
            /// See 'keepPrunedConnections' parameter in the article.
            /// </summary>
            public bool KeepPrunedConnections { get; set; }
        }

        /// <summary>
        /// Representation of knn search result.
        /// </summary>
        [SuppressMessage("Design", "CA1034:Nested types should not be visible", Justification = "By Design")]
        public class KNNSearchResult
        {
            /// <summary>
            /// Gets or sets the id of the item = rank of the item in source collection
            /// </summary>
            public int Id { get; set; }

            /// <summary>
            /// Gets or sets the item itself.
            /// </summary>
            public TItem Item { get; set; }

            /// <summary>
            /// Gets or sets the distance between the item and the knn search query.
            /// </summary>
            public TDistance Distance { get; set; }
        }
    }
}
