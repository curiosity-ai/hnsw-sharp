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
    using System.Numerics;
    using System.Runtime.CompilerServices;
    using System.Runtime.InteropServices;
    using System.Threading;
    using MessagePack;

    using static HNSW.Net.EventSources;

    internal partial class Graph<TItem, TDistance>
    {
        internal class Core
        {
            private readonly Func<TItem, TItem, TDistance> Distance;

            // cached delegate to avoid allocating one per GetDistance call
            private readonly Func<int, int, TDistance> GetDistanceSkipCacheFunc;

            private DistanceCache<TDistance> DistanceCache;

            private long DistanceCalculationsCount;

            // Built-in fast path for unit-normalised float[] vectors: a single contiguous
            // backing buffer plus an inlined SIMD inner-product, avoiding the per-comparison
            // delegate call and the jagged-array dereference. See SmallWorldParameters.UseBuiltInUnitInnerProduct.
            internal readonly bool FastFloatInnerProduct;
            internal readonly bool NoDistanceCache;
            internal readonly bool UseBitSetVisited;
            private int _dim = -1;
            private float[] _flat = Array.Empty<float>();
            private int _flatRows;

            // Pool of searchers so that concurrent queries reuse the (graph sized) scratch buffers
            // instead of allocating them on every call.
            private readonly System.Collections.Concurrent.ConcurrentBag<Searcher> SearcherPool = new System.Collections.Concurrent.ConcurrentBag<Searcher>();

            internal List<Node> Nodes { get; private set; }

            internal List<TItem> Items { get; private set; }

            internal Algorithms.Algorithm<TItem, TDistance> Algorithm { get; private set; }

            internal SmallWorldParameters Parameters { get; private set; }

            internal float DistanceCacheHitRate => DistanceCalculationsCount == 0
                ? 0f
                : (float)(DistanceCache?.HitCount ?? 0) / DistanceCalculationsCount;

            internal Core(Func<TItem, TItem, TDistance> distance, SmallWorldParameters parameters)
            {
                Distance = distance;
                Parameters = parameters;
                GetDistanceSkipCacheFunc = GetDistanceSkipCache;
                FastFloatInnerProduct = parameters.UseBuiltInUnitInnerProduct
                    && typeof(TItem) == typeof(float[])
                    && typeof(TDistance) == typeof(float);
                NoDistanceCache = parameters.RemoveDistanceCache;
                UseBitSetVisited = parameters.UseBitSetVisited;

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

                if (Parameters.EnableDistanceCacheForConstruction && !NoDistanceCache)
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

                if (FastFloatInnerProduct && newCount > 0)
                {
                    PackFlat(items);
                }

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

            internal void Serialize(Stream stream)
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

            internal TItem[] Deserialize(IReadOnlyList<TItem> items, Stream stream, CachedNodeData cachedNodeData)
            {
                // readStrict: true -> removed, as not available anymore on MessagePack 2.0 - also probably not necessary anymore
                //                     see https://github.com/neuecc/MessagePack-CSharp/pull/663
                Nodes = MessagePackSerializer.Deserialize<List<Node>>(stream);
                
                var nodesSpan = CollectionsMarshal.AsSpan(Nodes);

                for (int i = 0; i < nodesSpan.Length; i++)
                {
                    Node.FlattenToCache(ref nodesSpan[i], cachedNodeData);
                }

                var remainingItems = items.Skip(Nodes.Count).ToArray();
                Items.AddRange(items.Take(Nodes.Count));
                return remainingItems;
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal TDistance GetDistance(int fromId, int toId)
            {
                if (FastFloatInnerProduct)
                {
                    // No delegate, no jagged-array deref, no distance-cache bookkeeping on the hot path.
                    float d = InnerProductDistanceByRow(fromId, toId);
                    return Unsafe.As<float, TDistance>(ref d);
                }

                if (NoDistanceCache)
                {
                    // Distance cache removed completely: no counter, no cache-lookup branch,
                    // matching the leaner distance path of hnswlib / Lucene HNSW.
                    return Distance(Items[fromId], Items[toId]);
                }

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

            // ---- built-in unit inner-product fast path --------------------------------

            private void PackFlat(IReadOnlyList<TItem> items)
            {
                if (_dim < 0)
                {
                    _dim = ((float[])(object)items[0]).Length;
                }

                int total = Items.Count;
                long needed = (long)total * _dim;
                if (_flat.Length < needed)
                {
                    // grow with headroom to amortise incremental AddItems calls
                    long cap = Math.Max(needed, _flat.Length == 0 ? needed : _flat.Length * 2L);
                    Array.Resize(ref _flat, (int)cap);
                }

                var dst = _flat.AsSpan();
                for (int i = 0; i < items.Count; i++)
                {
                    var v = (float[])(object)items[i];
                    v.AsSpan(0, _dim).CopyTo(dst.Slice(_flatRows * _dim, _dim));
                    _flatRows++;
                }
            }

            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private float InnerProductDistanceByRow(int a, int b)
            {
                var flat = _flat;
                int dim = _dim;
                return 1f - Dot(flat.AsSpan(a * dim, dim), flat.AsSpan(b * dim, dim));
            }

            // Distance from an external unit vector (e.g. a query) to a stored row.
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            internal float InnerProductDistanceToRow(ReadOnlySpan<float> query, int row)
            {
                int dim = _dim;
                return 1f - Dot(query, _flat.AsSpan(row * dim, dim));
            }

            // Four-accumulator SIMD dot product over contiguous spans (dim assumed equal).
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            private static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
            {
                int n = a.Length;
                int step = Vector<float>.Count;
                int unroll = step * 4;

                var acc0 = Vector<float>.Zero;
                var acc1 = Vector<float>.Zero;
                var acc2 = Vector<float>.Zero;
                var acc3 = Vector<float>.Zero;

                int i = 0;
                for (; i <= n - unroll; i += unroll)
                {
                    acc0 += new Vector<float>(a.Slice(i, step)) * new Vector<float>(b.Slice(i, step));
                    acc1 += new Vector<float>(a.Slice(i + step, step)) * new Vector<float>(b.Slice(i + step, step));
                    acc2 += new Vector<float>(a.Slice(i + step * 2, step)) * new Vector<float>(b.Slice(i + step * 2, step));
                    acc3 += new Vector<float>(a.Slice(i + step * 3, step)) * new Vector<float>(b.Slice(i + step * 3, step));
                }

                float sum = Vector.Sum((acc0 + acc1) + (acc2 + acc3));

                for (; i < n; i++)
                {
                    sum += a[i] * b[i];
                }

                return sum;
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
