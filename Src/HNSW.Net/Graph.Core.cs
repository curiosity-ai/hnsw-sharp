// <copyright file="Graph.Core.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Runtime.CompilerServices;
    using System.Runtime.InteropServices;
    using System.Threading;
    using MessagePack;

    using static HNSW.Net.EventSources;

    internal partial class Graph<TItem, TDistance>
    {
        internal partial class Core
        {
            private readonly Func<TItem, TItem, TDistance> Distance;

            // cached delegate to avoid allocating one per GetDistance call
            private readonly Func<int, int, TDistance> GetDistanceSkipCacheFunc;

            private DistanceCache<TDistance> DistanceCache;

            private long DistanceCalculationsCount;

            // Pool of searchers so that concurrent queries reuse the (graph sized) scratch buffers
            // instead of allocating them on every call.
            private readonly System.Collections.Concurrent.ConcurrentBag<Searcher> SearcherPool = new System.Collections.Concurrent.ConcurrentBag<Searcher>();

            internal List<Node> Nodes { get; private set; }

            internal List<TItem> Items { get; private set; }

            internal Algorithms.Algorithm<TItem, TDistance> Algorithm { get; private set; }

            internal SmallWorldParameters Parameters { get; private set; }

            internal float DistanceCacheHitRate => (float)(DistanceCache?.HitCount ?? 0) / DistanceCalculationsCount;

            internal Core(Func<TItem, TItem, TDistance> distance, SmallWorldParameters parameters)
            {
                Distance = distance;
                Parameters = parameters;
                GetDistanceSkipCacheFunc = GetDistanceSkipCache;

                var initialSize = Math.Max(1024, parameters.InitialItemsSize);

                Nodes = new List<Node>(initialSize);
                Items = new List<TItem>(initialSize);

                switch (Parameters.NeighbourHeuristic)
                {
                    case NeighbourSelectionHeuristic.SelectSimple:
                    {
                        Algorithm = new Algorithms.Algorithm3<TItem, TDistance>(this);
                        break;
                    }
                    case NeighbourSelectionHeuristic.SelectHeuristic:
                    {
                        Algorithm = new Algorithms.Algorithm4<TItem, TDistance>(this);
                        break;
                    }
                }

                if (Parameters.EnableDistanceCacheForConstruction)
                {
                    DistanceCache = new DistanceCache<TDistance>();

                    // InitialDistanceCacheSize is the number of cache entries (not points): passing it
                    // through Resize would square it and eagerly allocate gigabytes for the default settings.
                    DistanceCache.ResizeToEntries(parameters.InitialDistanceCacheSize, false);
                }

                DistanceCalculationsCount = 0;
            }

            internal IReadOnlyList<int> AddItems(IReadOnlyList<TItem> items, IProvideRandomValues generator)
            {
                int newCount = items.Count;

                var newIDs = new List<int>();
                Items.AddRange(items);

                // size the cache for the total number of points in the graph, so the hit rate
                // does not degrade as the graph grows incrementally
                DistanceCache?.Resize(Items.Count, false);

                int id0 = Nodes.Count;

                for (int id = 0; id < newCount; ++id)
                {
                    Nodes.Add(Algorithm.NewNode(id0 + id, RandomLayer(generator, Parameters.LevelLambda)));
                    newIDs.Add(id0 + id);
                }
                return newIDs;
            }

            internal Searcher RentSearcher()
            {
                return SearcherPool.TryTake(out var searcher) ? searcher : new Searcher(this);
            }

            internal void ReturnSearcher(Searcher searcher)
            {
                SearcherPool.Add(searcher);
            }

            internal void ResizeDistanceCache(int newSize)
            {
                if (newSize >= 0)
                {
                    DistanceCache?.Resize(newSize, true);
                }
                else
                {
                    DistanceCache = null;
                }
            }

            /// <summary>
            /// Writes the nodes in the legacy MessagePack format. Kept so a graph written by this version can
            /// still be read by one that predates the flat format; <see cref="Graph{TItem,TDistance}.SerializeFlat"/>
            /// is what <see cref="SmallWorld{TItem,TDistance}.SerializeGraph(Stream)"/> writes.
            /// </summary>
            internal void SerializeMessagePack(Stream stream)
            {
                MessagePackSerializer.Serialize(stream, Nodes);
            }

            internal bool NeedsOptimization()
            {
                if (Nodes.Count == 0) return false;

                int notCached = 0;
                foreach (var n in Nodes)
                {
                    if(!n.IsCached)
                    {
                        notCached++;
                    }
                }

                return notCached > 1000 && notCached > (0.1 * Nodes.Count);
            }
            internal void Optimize(CachedNodeData cachedNodeData)
            {
                var nodesSpan = CollectionsMarshal.AsSpan(Nodes);

                for (int i = 0; i < nodesSpan.Length; i++)
                {
                    Node.FlattenToCache(ref nodesSpan[i], cachedNodeData);
                }
            }

            internal TItem[] DeserializeMessagePack(IReadOnlyList<TItem> items, Stream stream, CachedNodeData cachedNodeData)
            {
                // readStrict: true -> removed, as not available anymore on MessagePack 2.0 - also probably not necessary anymore
                //                     see https://github.com/neuecc/MessagePack-CSharp/pull/663
                Nodes = MessagePackSerializer.Deserialize<List<Node>>(stream);
                
                var nodesSpan = CollectionsMarshal.AsSpan(Nodes);

                for (int i = 0; i < nodesSpan.Length; i++)
                {
                    Node.FlattenToCache(ref nodesSpan[i], cachedNodeData);
                }

                return AssignItems(items, Nodes.Count);
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal TDistance GetDistance(int fromId, int toId)
            {
                DistanceCalculationsCount++;
                if (DistanceCache is object)
                {
                    return DistanceCache.GetOrCacheValue(fromId, toId, GetDistanceSkipCacheFunc);
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
                var u = generator.NextFloat();

                // NextFloat() can return exactly 0 (probability ~2^-31), and Log(0) = -inf would
                // saturate the cast to int.MaxValue, making NewNode try to allocate billions of
                // layers. Clamp to the smallest value the generator can otherwise produce (2^-31),
                // which corresponds to the deepest layer reachable by a non-zero draw.
                if (u <= 0f) u = 4.656613e-10f;

                var r = -Math.Log(u) * lambda;
                return (int)r;
            }
        }
    }
}
