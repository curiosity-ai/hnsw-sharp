// <copyright file="HyperspaceIndex.Build.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;
    using System.Runtime.InteropServices;

    public sealed partial class HyperspaceIndex
    {
        /// <summary>Per-thread reusable state of graph traversals: an epoch-stamped visited set and the two heaps.</summary>
        private sealed class SearchScratch
        {
            private int[] _marks = new int[1024];
            private int _epoch;

            public PriorityQueue<int, double> Candidates { get; } = new PriorityQueue<int, double>();

            public PriorityQueue<int, double> Results { get; } = new PriorityQueue<int, double>();

            private int[] _batch = new int[64];

            /// <summary>A reusable buffer for the unvisited neighbours of the node being expanded.</summary>
            public int[] Batch(int length)
            {
                if (_batch.Length < length)
                {
                    _batch = new int[Math.Max(length, _batch.Length * 2)];
                }

                return _batch;
            }

            /// <summary>Output of a layer search during construction.</summary>
            public List<(int Id, double Distance)> Found { get; } = new List<(int, double)>();

            /// <summary>Output of the neighbour-selection heuristic.</summary>
            public List<int> Chosen { get; } = new List<int>();

            /// <summary>Pruning runs while <see cref="Chosen"/> is being walked, so it gets its own lists.</summary>
            public List<(int Id, double Distance)> PruneCandidates { get; } = new List<(int, double)>();

            public List<int> PruneChosen { get; } = new List<int>();

            /// <summary>Hits of a query, nearest first.</summary>
            public List<(int Id, double Distance)> Hits { get; } = new List<(int, double)>();

            /// <summary>The hits again, for the exact re-ranking pass.</summary>
            public List<(int Id, double Distance)> Reranked { get; } = new List<(int, double)>();

            /// <summary>A prepared query whose buffers are reused by every search of this thread on <see cref="QueryOwner"/>.</summary>
            public PreparedQuery? Query { get; set; }

            public object? QueryOwner { get; set; }

            /// <summary>Starts a new traversal: a fresh epoch makes every mark stale without clearing the array.</summary>
            public void Begin(int nodeCount)
            {
                if (_marks.Length < nodeCount)
                {
                    _marks = new int[Math.Max(nodeCount, _marks.Length * 2)];
                    _epoch = 0;
                }

                if (++_epoch == int.MaxValue)
                {
                    Array.Clear(_marks);
                    _epoch = 1;
                }

                Candidates.Clear();
                Results.Clear();
            }

            /// <summary>Marks a node visited; returns <c>false</c> if it already was.</summary>
            public bool Visit(int id)
            {
                if (_marks[id] == _epoch)
                {
                    return false;
                }

                _marks[id] = _epoch;
                return true;
            }
        }

        [ThreadStatic]
        private static SearchScratch? scratch;

        private static SearchScratch Scratch => scratch ??= new SearchScratch();

#if DIAG
        [ThreadStatic]
        internal static long DiagEvaluations;
#endif

        /// <summary>
        /// Asks the CPU to start loading the first cache lines of a stored vector. Neighbours are visited in an order that follows the graph, not
        /// memory, so every distance would otherwise wait for its vector to arrive from RAM; issuing the requests for all of a node's neighbours before
        /// computing any distance lets those loads overlap. Only a hint: it never faults and is ignored where unsupported.
        /// </summary>
        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private unsafe void PrefetchVector(int id)
        {
            if (System.Runtime.Intrinsics.X86.Sse.IsSupported)
            {
                var vector = _vectors[id];
                byte* p = (byte*)System.Runtime.CompilerServices.Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(vector));
                System.Runtime.Intrinsics.X86.Sse.Prefetch0(p);
                if (vector.Length > 64)
                {
                    System.Runtime.Intrinsics.X86.Sse.Prefetch0(p + 64);
                }

                if (PrefetchLines > 2 && vector.Length > 128)
                {
                    System.Runtime.Intrinsics.X86.Sse.Prefetch0(p + 128);
                    System.Runtime.Intrinsics.X86.Sse.Prefetch0(p + 192);
                }
            }
        }

        /// <summary>Cache lines of each neighbour's vector to prefetch (0 turns prefetching off); a field so benchmarks can compare settings in one process.</summary>
        internal static int PrefetchLines = 4;

        private double Dist(int id, PreparedQuery q, int truncation = 0)
        {
#if DIAG
            DiagEvaluations++;
#endif
            return _codec.Distance(_vectors[id], q, truncation);
        }

        private double UpperDist(int id, PreparedQuery q) => _codec.UpperDistance(_vectors[id], q);

        /// <summary>The distance graph construction uses between a node and the vector being inserted: exact when building from the originals.</summary>
        private double BuildDist(int id, PreparedQuery q, bool upper)
            => _exactBuild ? Metrics.Distance(_options.Metric, (ReadOnlySpan<float>)_originals![id], q.Coords32) : upper ? UpperDist(id, q) : Dist(id, q);

        private double Pair(int a, int b)
            => _exactBuild
                ? Metrics.Distance(_options.Metric, (ReadOnlySpan<float>)_originals![a], _originals[b])
                : _codec.PairDistance(_vectors[a], _vectors[b]);

        // ------------------------------------------------------------------ insertion

        private static readonly Comparison<(int Id, double Distance)> ByDistance = static (a, b) => a.Distance.CompareTo(b.Distance);

        /// <summary>Wires a node (whose vector is already stored) into the graph. Must hold the write lock, with layer 0 thawed.</summary>
        private void ConnectNode(int id, PreparedQuery q)
        {
            var node = _nodes[id];
            int level = node.Level;

            if (_entryPoint < 0 || _entryPoint == id && _nodes.Count == 1)
            {
                _entryPoint = id;
                _maxLevel = level;
                return;
            }

            int current = _entryPoint;
            double currentDist = BuildDist(current, q, upper: true);

            // Phase 1: zoom in greedily through the layers above the node's own.
            for (int l = _maxLevel; l > level; l--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    foreach (int nb in LinksOf(current, l))
                    {
                        if (nb == id)
                        {
                            continue;
                        }

                        double d = BuildDist(nb, q, upper: true);
                        if (d < currentDist)
                        {
                            currentDist = d;
                            current = nb;
                            changed = true;
                        }
                    }
                }
            }

            // Phase 2: from the node's top layer down, find candidates, pick diverse neighbours and link both ways.
            var scratch = Scratch;
            int mBase = _options.M;
            for (int l = Math.Min(level, _maxLevel); l >= 0; l--)
            {
                int mMax = l == 0 ? mBase * 2 : mBase;
                var candidates = SearchLayerForBuild(current, q, l, _options.EfConstruction, id);
                SelectNeighbors(candidates, mMax, scratch.Chosen);
                node.Set(l, CollectionsMarshal.AsSpan(scratch.Chosen));

                var selected = node.Get(l);
                foreach (int nb in selected)
                {
                    AddLink(nb, id, l);
                    var nbNode = _nodes[nb];
                    if (l < nbNode.Links.Length && nbNode.Counts[l] > Math.Max(mMax * 5 / 4, mMax + 2))
                    {
                        PruneConnections(nb, l, mMax);
                    }
                }

                if (selected.Length > 0)
                {
                    current = selected[0];
                }
            }

            if (level > _maxLevel)
            {
                _maxLevel = level;
                _entryPoint = id;
            }
        }

        /// <summary>The links of a node on a layer. Layer 0 comes from the flat arrays while they exist.</summary>
        private ReadOnlySpan<int> LinksOf(int node, int layer)
        {
            if (layer == 0)
            {
                return Layer0Of(node);
            }

            var n = _nodes[node];
            return layer < n.Links.Length ? n.Get(layer) : ReadOnlySpan<int>.Empty;
        }

        /// <summary>
        /// Best-first search on one layer returning up to <paramref name="ef"/> candidates, nearest first, in the thread's reusable list (valid until
        /// the next call). Deleted nodes are traversed and returned.
        /// </summary>
        private List<(int Id, double Distance)> SearchLayerForBuild(int entry, PreparedQuery q, int layer, int ef, int exclude)
        {
            var s = Scratch;
            s.Begin(_nodes.Count);
            var candidates = s.Candidates;
            var results = s.Results;
            bool upper = layer > 0;

            double d0 = BuildDist(entry, q, upper);
            s.Visit(entry);
            candidates.Enqueue(entry, d0);
            if (entry != exclude)
            {
                results.Enqueue(entry, -d0);
            }

            while (candidates.TryDequeue(out int c, out double cd))
            {
                if (results.Count >= ef && results.TryPeek(out _, out double negWorst) && cd > -negWorst)
                {
                    break;
                }

                var links = LinksOf(c, layer);
                var batch = s.Batch(links.Length);
                int m = 0;
                foreach (int nb in links)
                {
                    if (nb == exclude || !s.Visit(nb))
                    {
                        continue;
                    }

                    batch[m++] = nb;
                    if (PrefetchLines > 0)
                    {
                        PrefetchVector(nb);
                    }
                }

                for (int i = 0; i < m; i++)
                {
                    int nb = batch[i];
                    double d = BuildDist(nb, q, upper);
                    if (results.Count < ef || (results.TryPeek(out _, out double nw) && d < -nw))
                    {
                        candidates.Enqueue(nb, d);
                        results.Enqueue(nb, -d);
                        if (results.Count > ef)
                        {
                            results.Dequeue();
                        }
                    }
                }
            }

            var found = s.Found;
            found.Clear();
            while (results.TryDequeue(out int id, out double nd))
            {
                found.Add((id, -nd));
            }

            found.Reverse();
            return found;
        }

        /// <summary>
        /// The HNSW neighbour-selection heuristic: walking the candidates nearest first, keep one only if it is closer to the query
        /// than to every neighbour already kept, so links spread over different directions instead of crowding one cluster.
        /// The kept ids replace the contents of <paramref name="chosen"/>.
        /// </summary>
        private void SelectNeighbors(List<(int Id, double Distance)> candidates, int m, List<int> chosen)
        {
            int actualM = m;
            if (_options.DensityPruning && candidates.Count > 0)
            {
                int samples = Math.Min(candidates.Count, m);
                double sum = 0;
                for (int i = 0; i < samples; i++)
                {
                    sum += candidates[i].Distance;
                }

                if (sum / samples > 1.95)
                {
                    actualM = Math.Max(m / 2, 1);
                }
            }

            chosen.Clear();
            foreach (var (id, dist) in candidates)
            {
                if (chosen.Count >= actualM)
                {
                    break;
                }

                bool good = true;
                foreach (int s in chosen)
                {
                    if (Pair(id, s) < dist)
                    {
                        good = false;
                        break;
                    }
                }

                if (good)
                {
                    chosen.Add(id);
                }
            }
        }

        private void AddLink(int source, int target, int layer)
        {
            var node = _nodes[source];
            if (layer >= node.Links.Length || node.Contains(layer, target))
            {
                return;
            }

            node.Add(layer, target);
        }

        private void PruneConnections(int id, int layer, int maxLinks)
        {
            var s = Scratch;
            var node = _nodes[id];
            var candidates = s.PruneCandidates;
            candidates.Clear();
            foreach (int n in node.Get(layer))
            {
                candidates.Add((n, Pair(id, n)));
            }

            candidates.Sort(ByDistance);
            SelectNeighbors(candidates, maxLinks, s.PruneChosen);
            node.Set(layer, CollectionsMarshal.AsSpan(s.PruneChosen));
        }

        // ------------------------------------------------------------------ DiskANN / Vamana pruning

        /// <summary>
        /// Rewrites the links of layer 0 with the Vamana "RobustPrune" rule of DiskANN, turning the HNSW base layer into a sparser spatial
        /// navigable graph that touches fewer pages when it lives on disk: a neighbour is dropped when an already-kept one reaches
        /// it with <c>alpha * d(kept, candidate) &lt;= d(node, candidate)</c>.
        /// </summary>
        /// <param name="alpha">The pruning slack, at least 1; larger keeps more links.</param>
        public void OptimizeAsSng(double alpha = 1.2)
        {
            if (alpha < 1.0)
            {
                throw new ArgumentOutOfRangeException(nameof(alpha), "alpha must be at least 1");
            }

            EnterWrite();
            try
            {
                ThawLayer0();
                int maxDegree = _options.M;
                var rewritten = new int[_nodes.Count][];
                for (int i = 0; i < _nodes.Count; i++)
                {
                    var links = _nodes[i].Get(0);
                    if (links.Length == 0)
                    {
                        rewritten[i] = Array.Empty<int>();
                        continue;
                    }

                    var sorted = new List<(int Id, double Distance)>(links.Length);
                    foreach (int c in links)
                    {
                        if (c != i)
                        {
                            sorted.Add((c, Pair(i, c)));
                        }
                    }

                    sorted.Sort(ByDistance);
                    var kept = new List<int>(maxDegree);
                    foreach (var (c, cd) in sorted)
                    {
                        if (kept.Count >= maxDegree)
                        {
                            break;
                        }

                        bool prune = false;
                        foreach (int e in kept)
                        {
                            if (alpha * Pair(e, c) <= cd)
                            {
                                prune = true;
                                break;
                            }
                        }

                        if (!prune)
                        {
                            kept.Add(c);
                        }
                    }

                    rewritten[i] = kept.ToArray();
                }

                for (int i = 0; i < _nodes.Count; i++)
                {
                    _nodes[i].Set(0, rewritten[i]);
                }
            }
            finally
            {
                ExitWrite();
            }
        }
    }
}
