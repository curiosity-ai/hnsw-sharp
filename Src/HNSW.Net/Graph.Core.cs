// <copyright file="Graph.Core.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Runtime.CompilerServices;
    using MessagePack;

    using static HNSW.Net.EventSources;

    internal partial class Graph<TItem, TDistance>
    {
        internal class Core
        {
            private readonly Func<TItem, TItem, TDistance> Distance;

            private readonly DistanceCache<TDistance> DistanceCache;

            private long DistanceCalculationsCount;

            internal List<Node> Nodes { get; private set; }

            internal List<TItem> Items { get; private set; }

            internal Algorithms.Algorithm<TItem, TDistance> Algorithm { get; private set; }

            internal SmallWorld<TItem, TDistance>.Parameters Parameters { get; private set; }

            internal float DistanceCacheHitRate => (float)(DistanceCache?.HitCount ?? 0) / DistanceCalculationsCount;

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
                        Algorithm = new Algorithms.Algorithm3<TItem, TDistance>(this);
                        break;
                    }
                    case SmallWorld<TItem, TDistance>.NeighbourSelectionHeuristic.SelectHeuristic:
                    {
                        Algorithm = new Algorithms.Algorithm4<TItem, TDistance>(this);
                        break;
                    }
                }

                if (Parameters.EnableDistanceCacheForConstruction)
                {
                    DistanceCache = new DistanceCache<TDistance>();
                    DistanceCache.Resize(parameters.InitialDistanceCacheSize);
                }

                DistanceCalculationsCount = 0;
            }


            internal void AddItems(IReadOnlyList<TItem> items, IProvideRandomValues generator)
            {
                Items.AddRange(items);
                DistanceCache?.Resize(items.Count);
                int id0 = Nodes.Count;
                Nodes.Capacity += items.Count;
                for (int id = 0; id < items.Count; ++id)
                {
                    Nodes.Add(Algorithm.NewNode(id0 + id, RandomLayer(generator, Parameters.LevelLambda)));
                }
            }

            internal void Serialize(Stream stream)
            {
                MessagePackSerializer.Serialize(stream, Nodes);
            }

            internal void Deserialize(IReadOnlyList<TItem> items, Stream stream)
            {
                // readStrict: true -> removed, as not available anymore on MessagePack 2.0 - also probably not necessary anymore
                //                     see https://github.com/neuecc/MessagePack-CSharp/pull/663
                Nodes = MessagePackSerializer.Deserialize<List<Node>>(stream);
                Items.AddRange(items);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal TDistance GetDistance(int fromId, int toId)
            {
                DistanceCalculationsCount++;
                if (DistanceCache is object)
                {
                    return DistanceCache.GetValue(fromId, toId, GetDistanceSkipCache);
                }
                else
                {
                    return Distance(Items[fromId], Items[toId]);
                }
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private TDistance GetDistanceSkipCache(int fromId, int toId)
            {
                return Distance(Items[fromId], Items[toId]);
            }

            private static int RandomLayer(IProvideRandomValues generator, double lambda)
            {
                var r = -Math.Log(generator.NextFloat()) * lambda;
                return (int)r;
            }
        }
    }
}
