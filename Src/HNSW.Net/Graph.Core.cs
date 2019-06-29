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

    internal partial class Graph<TItem, TDistance>
    {
        internal class Core
        {
            private readonly Func<TItem, TItem, TDistance> Distance;

            private readonly DistanceCache<TDistance> distanceCache;

            private long DistanceCacheHitCount;

            private long DistanceCalculationsCount;

            internal List<Node> Nodes { get; private set; }

            internal List<TItem> Items { get; private set; }

            internal Node.Algorithm<TItem, TDistance> Algorithm { get; private set; }

            internal SmallWorld<TItem, TDistance>.Parameters Parameters { get; private set; }

            internal float DistanceCacheHitRate => (float)DistanceCacheHitCount / DistanceCalculationsCount;

            internal Core(Func<TItem, TItem, TDistance> distance, SmallWorld<TItem, TDistance>.Parameters parameters)
            {
                Distance = distance;
                Parameters = parameters;
                Nodes = new List<Node>();
                Items = new List<TItem>();

                switch (Parameters.NeighbourHeuristic)
                {
                    case SmallWorld<TItem, TDistance>.NeighbourSelectionHeuristic.SelectSimple:
                    {
                        Algorithm = new Node.Algorithm3<TItem, TDistance>(this);
                        break;
                    }
                    case SmallWorld<TItem, TDistance>.NeighbourSelectionHeuristic.SelectHeuristic:
                    {
                        Algorithm = new Node.Algorithm4<TItem, TDistance>(this);
                        break;
                    }
                }

                if (Parameters.EnableDistanceCacheForConstruction)
                {
                    distanceCache = new DistanceCache<TDistance>(Items.Count);
                }

                DistanceCacheHitCount = 0;
                DistanceCalculationsCount = 0;
            }


            internal void AddItems(IReadOnlyList<TItem> items, IProvideRandomValues generator)
            {
                Items.AddRange(items);
                int id0 = Nodes.Count;
                Nodes.Capacity += items.Count;
                for (int id = 0; id < items.Count; ++id)
                {
                    Nodes.Add(Algorithm.NewNode(id0 + id, RandomLayer(generator, Parameters.LevelLambda)));
                }
            }

            internal byte[] Serialize()
            {
                using (var stream = new MemoryStream())
                {
                    var formatter = new BinaryFormatter();
                    formatter.Serialize(stream, Nodes);
                    return stream.ToArray();
                }
            }

            internal void Deserialize(IReadOnlyList<TItem> items, byte[] bytes)
            {
                using (var stream = new MemoryStream(bytes))
                {
                    var formatter = new BinaryFormatter();
                    Nodes = (List<Node>)formatter.Deserialize(stream);
                }
                Items.AddRange(items);
            }

            internal TDistance GetDistance(int fromId, int toId)
            {
                DistanceCalculationsCount++;

                TDistance result;
                if (distanceCache != null && distanceCache.TryGetValue(fromId, toId, out result))
                {
                    DistanceCacheHitCount++;
                    return result;
                }

                result = Distance(Items[fromId], Items[toId]);
                distanceCache?.SetValue(fromId, toId, result);

                return result;
            }

            private static int RandomLayer(IProvideRandomValues generator, double lambda)
            {
                var r = -Math.Log(generator.NextFloat()) * lambda;
                return (int)r;
            }
        }
    }
}
