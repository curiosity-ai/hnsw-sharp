// <copyright file="MetaRouter.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading;

    /// <summary>A segment (shard, chunk, tenant) registered with the <see cref="MetaRouter"/> by the centroid of its vectors.</summary>
    /// <param name="Id">The unique segment id.</param>
    /// <param name="Centroid">The mean of the vectors of the segment.</param>
    /// <param name="VectorCount">The number of vectors in the segment.</param>
    public sealed record SegmentInfo(string Id, float[] Centroid, int VectorCount);

    /// <summary>
    /// IVF-style router that maps a query to the few segments worth searching, by comparing it with one centroid per segment.
    /// Beyond 32 segments a second level of k-means macro-clusters is kept, so the cost of routing falls from O(C) to O(sqrt C).
    /// </summary>
    public sealed class MetaRouter
    {
        private const int MacroClusterThreshold = 32;

        private readonly object _lock = new object();
        private readonly List<SegmentInfo> _segments = new List<SegmentInfo>();
        private readonly Dictionary<string, int> _index = new Dictionary<string, int>(StringComparer.Ordinal);
        private List<(float[] Centroid, List<int> Members)> _macro = new List<(float[], List<int>)>();
        private long _queries, _probes;

        /// <summary>Gets the number of segments.</summary>
        public int Count
        {
            get
            {
                lock (_lock)
                {
                    return _segments.Count;
                }
            }
        }

        /// <summary>Gets the number of vectors over all segments.</summary>
        public long TotalVectorCount
        {
            get
            {
                lock (_lock)
                {
                    return _segments.Sum(s => (long)s.VectorCount);
                }
            }
        }

        /// <summary>Gets the number of queries routed.</summary>
        public long QueryCount => Interlocked.Read(ref _queries);

        /// <summary>Gets the number of centroid distances computed.</summary>
        public long ProbeOperations => Interlocked.Read(ref _probes);

        /// <summary>Registers a segment, replacing any with the same id.</summary>
        public void Register(SegmentInfo segment)
        {
            lock (_lock)
            {
                if (_index.TryGetValue(segment.Id, out var i))
                {
                    _segments[i] = segment;
                }
                else
                {
                    _index[segment.Id] = _segments.Count;
                    _segments.Add(segment);
                }

                RebuildMacro();
            }
        }

        /// <summary>Removes a segment.</summary>
        public bool Unregister(string id)
        {
            lock (_lock)
            {
                if (!_index.TryGetValue(id, out var i))
                {
                    return false;
                }

                _segments.RemoveAt(i);
                _index.Clear();
                for (int j = 0; j < _segments.Count; j++)
                {
                    _index[_segments[j].Id] = j;
                }

                RebuildMacro();
                return true;
            }
        }

        /// <summary>Atomically replaces all segments.</summary>
        public void ReplaceAll(IEnumerable<SegmentInfo> segments)
        {
            lock (_lock)
            {
                _segments.Clear();
                _index.Clear();
                foreach (var s in segments)
                {
                    _index[s.Id] = _segments.Count;
                    _segments.Add(s);
                }

                RebuildMacro();
            }
        }

        /// <summary>A snapshot of the registered segments.</summary>
        public IReadOnlyList<SegmentInfo> Segments()
        {
            lock (_lock)
            {
                return _segments.ToArray();
            }
        }

        private void RebuildMacro()
        {
            if (_segments.Count < MacroClusterThreshold)
            {
                _macro = new List<(float[], List<int>)>();
                return;
            }

            int k = Math.Clamp((int)Math.Ceiling(Math.Sqrt(_segments.Count)), 2, 512);
            var result = KMeans.Fit(_segments.Select(s => s.Centroid).ToArray(), k, maxIterations: 15);
            var macro = result.Centroids.Select(c => (c, new List<int>())).ToList();
            for (int i = 0; i < result.Assignments.Length; i++)
            {
                macro[result.Assignments[i]].Item2.Add(i);
            }

            macro.RemoveAll(m => m.Item2.Count == 0);
            _macro = macro;
        }

        /// <summary>The <paramref name="probeCount"/> segments whose centroids are nearest to the query, nearest first.</summary>
        /// <remarks>If there are no more than <paramref name="probeCount"/> segments (or <paramref name="probeCount"/> is 0) all are returned.</remarks>
        public List<(SegmentInfo Segment, double DistanceSquared)> Route(ReadOnlySpan<float> query, int probeCount)
        {
            Interlocked.Increment(ref _queries);
            var q = query.ToArray();
            lock (_lock)
            {
                if (_segments.Count == 0)
                {
                    return new List<(SegmentInfo, double)>();
                }

                if (probeCount <= 0 || probeCount >= _segments.Count)
                {
                    return _segments.Select(s => (s, 0.0)).ToList();
                }

                IEnumerable<int> candidates;
                if (_macro.Count > 0)
                {
                    int macroK = Math.Clamp(Math.Max(probeCount, 1) * 2, 2, _macro.Count);
                    var chosen = _macro.Select((m, i) => (i, d: Metrics.L2Squared<float>(q, m.Centroid)))
                                       .OrderBy(x => x.d)
                                       .Take(macroK)
                                       .ToList();
                    Interlocked.Add(ref _probes, _macro.Count);
                    candidates = chosen.SelectMany(c => _macro[c.i].Members).ToList();
                }
                else
                {
                    candidates = Enumerable.Range(0, _segments.Count);
                }

                var scored = candidates.Select(i => (Segment: _segments[i], d: Metrics.L2Squared<float>(q, _segments[i].Centroid))).ToList();
                Interlocked.Add(ref _probes, scored.Count);
                return scored.OrderBy(x => x.d).Take(probeCount).Select(x => (x.Segment, x.d)).ToList();
            }
        }
    }
}
