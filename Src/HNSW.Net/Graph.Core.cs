// <copyright file="Graph.Core.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Runtime.Serialization.Formatters.Binary;

    using static HNSW.Net.EventSources;

    /// <content>
    /// The implementation of graph core structure.
    /// </content>
    internal partial class Graph<TItem, TDistance>
    {
        /// <summary>
        /// The graph core.
        /// </summary>
        internal class Core
        {
            /// <summary>
            /// The original distance function.
            /// </summary>
            private readonly Func<TItem, TItem, TDistance> distance;

            /// <summary>
            /// The distance cache.
            /// </summary>
            private readonly DistanceCache<TDistance> distanceCache;

            /// <summary>
            /// The number of times cache was hit.
            /// </summary>
            private long distanceCacheHitCount;

            /// <summary>
            /// The number of times distance calculation was requested.
            /// </summary>
            private long distanceCalculationsCount;

            /// <summary>
            /// Initializes a new instance of the <see cref="Core"/> class.
            /// </summary>
            /// <param name="distance">The distance function in the items space.</param>
            /// <param name="parameters">The parameters of the world.</param>
            /// <param name="items">The original items.</param>
            internal Core(Func<TItem, TItem, TDistance> distance, SmallWorld<TItem, TDistance>.Parameters parameters, IReadOnlyList<TItem> items)
            {
                this.distance = distance;
                Parameters = parameters;
                Items = items;

                switch (Parameters.NeighbourHeuristic)
                {
                    case SmallWorld<TItem, TDistance>.NeighbourSelectionHeuristic.SelectSimple:
                        Algorithm = new Node.Algorithm3<TItem, TDistance>(this);
                        break;
                    case SmallWorld<TItem, TDistance>.NeighbourSelectionHeuristic.SelectHeuristic:
                        Algorithm = new Node.Algorithm4<TItem, TDistance>(this);
                        break;
                }

                if (Parameters.EnableDistanceCacheForConstruction)
                {
                    distanceCache = new DistanceCache<TDistance>(Items.Count);
                }

                distanceCacheHitCount = 0;
                distanceCalculationsCount = 0;
            }

            /// <summary>
            /// Gets the graph nodes corresponding to <see cref="Items"/>
            /// </summary>
            internal IReadOnlyList<Node> Nodes { get; private set; }

            /// <summary>
            /// Gets the items associated with the <see cref="Nodes"/>
            /// </summary>
            internal IReadOnlyList<TItem> Items { get; private set; }

            /// <summary>
            /// Gets the algorithm for allocating and managing nodes capacity.
            /// </summary>
            internal Node.Algorithm<TItem, TDistance> Algorithm { get; private set; }

            /// <summary>
            /// Gets parameters of the small world.
            /// </summary>
            internal SmallWorld<TItem, TDistance>.Parameters Parameters { get; private set; }

            /// <summary>
            /// Gets distance cache hit rate.
            /// </summary>
            internal float DistanceCacheHitRate => (float)distanceCacheHitCount / distanceCalculationsCount;

            /// <summary>
            /// Initializes node array for building graph.
            /// </summary>
            /// <param name="generator">The random number generator to assign layers.</param>
            internal void AllocateNodes(Random generator)
            {
                var nodes = new List<Node>(Items.Count);
                for (int id = 0; id < Items.Count; ++id)
                {
                    nodes.Add(Algorithm.NewNode(id, RandomLayer(generator, Parameters.LevelLambda)));
                }

                Nodes = nodes;
            }

            /// <summary>
            /// Serializes nodes of the core.
            /// </summary>
            /// <returns>Bytes representing nodes.</returns>
            internal byte[] Serialize()
            {
                using (var stream = new MemoryStream())
                {
                    var formatter = new BinaryFormatter();
                    formatter.Serialize(stream, Nodes);
                    return stream.ToArray();
                }
            }

            /// <summary>
            /// Deserializes the graph core from byte array.
            /// </summary>
            /// <param name="bytes">The byte array representing graph core.</param>
            internal void Deserialize(byte[] bytes)
            {
                using (var stream = new MemoryStream(bytes))
                {
                    var formatter = new BinaryFormatter();
                    Nodes = (List<Node>)formatter.Deserialize(stream);
                }
            }

            /// <summary>
            /// Gets the distance between 2 items.
            /// </summary>
            /// <param name="fromId">The identifier of the "from" item.</param>
            /// <param name="toId">The identifier of the "to" item.</param>
            /// <returns>The distance between items.</returns>
            internal TDistance GetDistance(int fromId, int toId)
            {
                distanceCalculationsCount++;

                TDistance result;
                if (distanceCache != null && distanceCache.TryGetValue(fromId, toId, out result))
                {
                    distanceCacheHitCount++;
                    return result;
                }

                result = distance(Items[fromId], Items[toId]);
                distanceCache?.SetValue(fromId, toId, result);

                return result;
            }

            /// <summary>
            /// Gets the random layer.
            /// </summary>
            /// <param name="generator">The random numbers generator.</param>
            /// <param name="lambda">Poisson lambda.</param>
            /// <returns>The layer value.</returns>
            private static int RandomLayer(Random generator, double lambda)
            {
                var r = -Math.Log(generator.NextDouble()) * lambda;
                return (int)r;
            }
        }
    }
}
