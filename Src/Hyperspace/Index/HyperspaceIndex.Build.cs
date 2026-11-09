// <copyright file="HyperspaceIndex.Build.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;

    public sealed partial class HyperspaceIndex
    {
        /// <summary>Per-thread reusable state of graph traversals: an epoch-stamped visited set and the two heaps.</summary>
        private sealed class SearchScratch
        {
            private int[] _marks = new int[1024];
            private int _epoch;

            public PriorityQueue<int, double> Candidates { get; } = new PriorityQueue<int, double>();

            public PriorityQueue<int, double> Results { get; } = new PriorityQueue<int, double>();

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

        private double Dist(int id, PreparedQuery q, int truncation = 0) => _codec.Distance(_vectors[id], q, truncation);

        private double UpperDist(int id, PreparedQuery q) => _codec.UpperDistance(_vectors[id], q);

        private double Pair(int a, int b) => _codec.PairDistance(_vectors[a], _vectors[b]);

        // ------------------------------------------------------------------ insertion

        /// <summary>Wires a node (whose vector is already stored) into the graph. Must hold the write lock.</summary>
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
            double currentDist = UpperDist(current, q);

            // Phase 1: zoom in greedily through the layers above the node's own.
            for (int l = _maxLevel; l > level; l--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    var links = LinksOf(current, l);
                    foreach (var nb in links)
                    {
                        if (nb == id)
                        {
                            continue;
                        }

                        double d = UpperDist(nb, q);
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
            int mBase = _options.M;
            for (int l = Math.Min(level, _maxLevel); l >= 0; l--)
            {
                int mMax = l == 0 ? mBase * 2 : mBase;
                var candidates = SearchLayerForBuild(current, q, l, _options.EfConstruction, id);
                var selected = SelectNeighbors(candidates, mMax);
                node.Links[l] = selected;

                foreach (int nb in selected)
                {
                    AddLink(nb, id, l);
                    var nbNode = _nodes[nb];
                    if (l < nbNode.Links.Length && nbNode.Links[l].Length > Math.Max(mMax * 5 / 4, mMax + 2))
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

        private int[] LinksOf(int node, int layer)
        {
            var links = _nodes[node].Links;
            return layer < links.Length ? links[layer] : Array.Empty<int>();
        }

        /// <summary>Best-first search on one layer returning up to <paramref name="ef"/> candidates, nearest first. Deleted nodes are traversed and returned.</summary>
        private List<(int Id, double Distance)> SearchLayerForBuild(int entry, PreparedQuery q, int layer, int ef, int exclude)
        {
            var s = Scratch;
            s.Begin(_nodes.Count);
            var candidates = s.Candidates;
            var results = s.Results;
            bool upper = layer > 0;

            double d0 = upper ? UpperDist(entry, q) : Dist(entry, q);
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

                foreach (int nb in LinksOf(c, layer))
                {
                    if (nb == exclude || !s.Visit(nb))
                    {
                        continue;
                    }

                    double d = upper ? UpperDist(nb, q) : Dist(nb, q);
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

            var list = new List<(int, double)>(results.Count);
            while (results.TryDequeue(out int id, out double nd))
            {
                list.Add((id, -nd));
            }

            list.Reverse();
            return list;
        }

        /// <summary>
        /// The HNSW neighbour-selection heuristic: walking the candidates nearest first, keep one only if it is closer to the query
        /// than to every neighbour already kept, so links spread over different directions instead of crowding one cluster.
        /// </summary>
        private int[] SelectNeighbors(List<(int Id, double Distance)> candidates, int m)
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

            var chosen = new List<int>(actualM);
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

            return chosen.ToArray();
        }

        private void AddLink(int source, int target, int layer)
        {
            var node = _nodes[source];
            if (layer >= node.Links.Length)
            {
                return;
            }

            var links = node.Links[layer];
            if (Array.IndexOf(links, target) >= 0)
            {
                return;
            }

            var grown = new int[links.Length + 1];
            links.CopyTo(grown, 0);
            grown[^1] = target;
            node.Links[layer] = grown;
        }

        private void PruneConnections(int id, int layer, int maxLinks)
        {
            var links = _nodes[id].Links[layer];
            var candidates = new List<(int Id, double Distance)>(links.Length);
            foreach (int n in links)
            {
                candidates.Add((n, Pair(id, n)));
            }

            candidates.Sort((a, b) => a.Distance.CompareTo(b.Distance));
            _nodes[id].Links[layer] = SelectNeighbors(candidates, maxLinks);
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
                int maxDegree = _options.M;
                var rewritten = new int[_nodes.Count][];
                for (int i = 0; i < _nodes.Count; i++)
                {
                    var links = _nodes[i].Links[0];
                    if (links.Length == 0)
                    {
                        rewritten[i] = links;
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

                    sorted.Sort((a, b) => a.Distance.CompareTo(b.Distance));
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
                    _nodes[i].Links[0] = rewritten[i];
                }

                _flat = null;
            }
            finally
            {
                ExitWrite();
            }
        }
    }
}
