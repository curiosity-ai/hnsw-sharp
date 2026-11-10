// <copyright file="Graph.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Buffers.Binary;
    using System.Collections.Generic;
    using System.IO;
    using System.Linq;
    using System.Runtime.InteropServices;
    using System.Threading;

    /// <summary>The standard relation labels of the knowledge graph.</summary>
    public static class EdgeRelation
    {
        /// <summary>A owns B.</summary>
        public const ushort Owns = 1;

        /// <summary>A depends on B.</summary>
        public const ushort DependsOn = 2;

        /// <summary>A supersedes B (hierarchical).</summary>
        public const ushort Supersedes = 3;

        /// <summary>A amends B.</summary>
        public const ushort Amends = 4;

        /// <summary>A contains B (hierarchical).</summary>
        public const ushort Contains = 5;

        /// <summary>A references B.</summary>
        public const ushort References = 6;

        /// <summary>The code of a relation label that is not one of the standard ones.</summary>
        public const ushort Custom = 100;

        /// <summary>Whether the relation expresses a hierarchy (parent to child).</summary>
        public static bool IsHierarchical(ushort relation) => relation == Supersedes || relation == Contains;

        /// <summary>Maps a label (<c>OWNS</c>, <c>depends_on</c>, <c>42</c>...) to its code.</summary>
        public static ushort FromLabel(string label)
        {
            if (ushort.TryParse(label, out var numeric))
            {
                return numeric;
            }

            switch (label.ToUpperInvariant())
            {
                case "OWNS": return Owns;
                case "DEPENDS_ON":
                case "DEPENDSON": return DependsOn;
                case "SUPERSEDES": return Supersedes;
                case "AMENDS": return Amends;
                case "CONTAINS": return Contains;
                case "REFERENCES": return References;
                default: return Custom;
            }
        }
    }

    /// <summary>A directed edge packed in exactly 12 bytes.</summary>
    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    public struct CompactEdge : IEquatable<CompactEdge>
    {
        /// <summary>The flag marking an edge as live.</summary>
        public const ushort FlagActive = 0x0001;

        /// <summary>The destination node.</summary>
        public uint Target;

        /// <summary>The relation code, see <see cref="EdgeRelation"/>.</summary>
        public ushort Relation;

        /// <summary>The status flags.</summary>
        public ushort Flags;

        /// <summary>The edge weight.</summary>
        public float Weight;

        /// <summary>Initializes a new instance of the <see cref="CompactEdge"/> struct as an active edge.</summary>
        public CompactEdge(uint target, ushort relation, float weight)
        {
            Target = target;
            Relation = relation;
            Flags = FlagActive;
            Weight = weight;
        }

        /// <summary>Gets a value indicating whether the edge is live.</summary>
        public readonly bool IsActive => (Flags & FlagActive) != 0;

        /// <inheritdoc />
        public readonly bool Equals(CompactEdge other) => Target == other.Target && Relation == other.Relation && Flags == other.Flags && Weight.Equals(other.Weight);

        /// <inheritdoc />
        public override readonly bool Equals(object? obj) => obj is CompactEdge e && Equals(e);

        /// <inheritdoc />
        public override readonly int GetHashCode() => HashCode.Combine(Target, Relation, Flags, Weight);
    }

    /// <summary>A declarative rule creating edges from item metadata.</summary>
    /// <param name="Field">The metadata field the rule reads.</param>
    /// <param name="RuleType"><c>path_hierarchy</c> (parent of <c>a/b/c</c> is <c>a/b</c>), <c>foreign_key</c> or <c>exact_match</c> (clique of items sharing a value).</param>
    /// <param name="Relation">The relation label of the new edges.</param>
    /// <param name="TargetField">For <c>foreign_key</c>: the field holding the key being referenced (default: the field value is the target id).</param>
    /// <param name="Weight">The edge weight (default 1).</param>
    public sealed record MetadataInductionRule(string Field, string RuleType, string Relation, string? TargetField = null, float? Weight = null);

    /// <summary>Static, read-optimized compressed-sparse-row graph.</summary>
    public sealed class CsrGraph
    {
        /// <summary>The magic header of a serialized graph (<c>HYPR_CSR</c>).</summary>
        public static readonly byte[] Magic = { (byte)'H', (byte)'Y', (byte)'P', (byte)'R', (byte)'_', (byte)'C', (byte)'S', (byte)'R' };

        /// <summary>Initializes a new instance of the <see cref="CsrGraph"/> class.</summary>
        public CsrGraph(uint[] rowOffsets, CompactEdge[] edges, int numNodes)
        {
            RowOffsets = rowOffsets;
            Edges = edges;
            NumNodes = numNodes;
        }

        /// <summary>Gets the row pointers (length <c>NumNodes + 1</c>).</summary>
        public uint[] RowOffsets { get; }

        /// <summary>Gets the contiguous array of live edges.</summary>
        public CompactEdge[] Edges { get; }

        /// <summary>Gets the number of nodes.</summary>
        public int NumNodes { get; }

        /// <summary>Gets the number of edges.</summary>
        public int EdgeCount => Edges.Length;

        /// <summary>Compacts an adjacency list, dropping inactive edges.</summary>
        public static CsrGraph FromAdjacencyList(IReadOnlyList<IReadOnlyList<CompactEdge>> adjacency)
        {
            var offsets = new uint[adjacency.Count + 1];
            var edges = new List<CompactEdge>();
            for (int i = 0; i < adjacency.Count; i++)
            {
                foreach (var e in adjacency[i])
                {
                    if (e.IsActive)
                    {
                        edges.Add(e);
                    }
                }

                offsets[i + 1] = (uint)edges.Count;
            }

            return new CsrGraph(offsets, edges.ToArray(), adjacency.Count);
        }

        /// <summary>The outgoing edges of a node.</summary>
        public ReadOnlySpan<CompactEdge> Neighbors(uint node)
        {
            if (node >= (uint)NumNodes || node + 1 >= (uint)RowOffsets.Length)
            {
                return ReadOnlySpan<CompactEdge>.Empty;
            }

            uint start = RowOffsets[node], end = RowOffsets[node + 1];
            return start <= end && end <= Edges.Length ? Edges.AsSpan((int)start, (int)(end - start)) : ReadOnlySpan<CompactEdge>.Empty;
        }

        /// <summary>The out-degree of a node.</summary>
        public int OutDegree(uint node) => Neighbors(node).Length;

        /// <summary>The memory held by the arrays, in bytes.</summary>
        public long MemoryUsageBytes => ((long)RowOffsets.Length * sizeof(uint)) + ((long)Edges.Length * 12);

        /// <summary>
        /// Breadth-first reachability from <paramref name="seeds"/> within <paramref name="maxHops"/> hops, visiting at most
        /// <paramref name="maxFanout"/> edges per node (a circuit breaker for hub nodes), optionally restricted to some relations.
        /// </summary>
        public IdSet BfsReachability(IReadOnlyList<uint> seeds, int maxHops, int maxFanout, IReadOnlyCollection<ushort>? relationFilter = null)
        {
            var visited = new IdSet();
            if (seeds.Count == 0)
            {
                return visited;
            }

            var frontier = new List<uint>(seeds);
            foreach (var s in seeds)
            {
                visited.Add((int)s);
            }

            var next = new List<uint>();
            for (int hop = 0; hop < maxHops && frontier.Count > 0; hop++)
            {
                next.Clear();
                foreach (var node in frontier)
                {
                    int count = 0;
                    foreach (var edge in Neighbors(node))
                    {
                        if (!edge.IsActive || (relationFilter is not null && !relationFilter.Contains(edge.Relation)))
                        {
                            continue;
                        }

                        if (visited.Add((int)edge.Target))
                        {
                            next.Add(edge.Target);
                        }

                        if (++count >= maxFanout)
                        {
                            break;
                        }
                    }
                }

                (frontier, next) = (next, frontier);
            }

            return visited;
        }

        /// <summary>
        /// Writes <c>"HYPR_CSR" | u64 nodes | u64 edges | u32[nodes + 1] offsets | 12-byte edges</c>, little-endian, the file format
        /// of HyperspaceDB's <c>graph.csr</c>.
        /// </summary>
        public void Write(Stream stream)
        {
            Span<byte> header = stackalloc byte[24];
            Magic.CopyTo(header);
            BinaryPrimitives.WriteUInt64LittleEndian(header.Slice(8), (ulong)NumNodes);
            BinaryPrimitives.WriteUInt64LittleEndian(header.Slice(16), (ulong)Edges.Length);
            stream.Write(header);

            var buffer = new byte[12 * 512];
            for (int i = 0; i < RowOffsets.Length;)
            {
                int n = Math.Min(RowOffsets.Length - i, 1024);
                for (int j = 0; j < n; j++)
                {
                    BinaryPrimitives.WriteUInt32LittleEndian(buffer.AsSpan(j * 4, 4), RowOffsets[i + j]);
                }

                stream.Write(buffer, 0, n * 4);
                i += n;
            }

            for (int i = 0; i < Edges.Length;)
            {
                int n = Math.Min(Edges.Length - i, 512);
                for (int j = 0; j < n; j++)
                {
                    var e = Edges[i + j];
                    var span = buffer.AsSpan(j * 12, 12);
                    BinaryPrimitives.WriteUInt32LittleEndian(span, e.Target);
                    BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(4), e.Relation);
                    BinaryPrimitives.WriteUInt16LittleEndian(span.Slice(6), e.Flags);
                    BinaryPrimitives.WriteSingleLittleEndian(span.Slice(8), e.Weight);
                }

                stream.Write(buffer, 0, n * 12);
                i += n;
            }
        }

        /// <summary>Reads a graph written by <see cref="Write"/>.</summary>
        public static CsrGraph Read(Stream stream)
        {
            Span<byte> header = stackalloc byte[24];
            stream.ReadExactly(header);
            if (!header.Slice(0, 8).SequenceEqual(Magic))
            {
                throw new InvalidDataException("Invalid CSR graph magic header");
            }

            ulong nodes = BinaryPrimitives.ReadUInt64LittleEndian(header.Slice(8));
            ulong edgeCount = BinaryPrimitives.ReadUInt64LittleEndian(header.Slice(16));
            if (nodes > int.MaxValue - 1 || edgeCount > int.MaxValue)
            {
                throw new InvalidDataException($"Invalid CSR graph header (nodes: {nodes}, edges: {edgeCount})");
            }

            var offsets = new uint[nodes + 1];
            var buffer = new byte[12 * 512];
            for (int i = 0; i < offsets.Length;)
            {
                int n = Math.Min(offsets.Length - i, 1024);
                stream.ReadExactly(buffer, 0, n * 4);
                for (int j = 0; j < n; j++)
                {
                    offsets[i + j] = BinaryPrimitives.ReadUInt32LittleEndian(buffer.AsSpan(j * 4, 4));
                }

                i += n;
            }

            var edges = new CompactEdge[edgeCount];
            for (int i = 0; i < edges.Length;)
            {
                int n = Math.Min(edges.Length - i, 512);
                stream.ReadExactly(buffer, 0, n * 12);
                for (int j = 0; j < n; j++)
                {
                    var span = buffer.AsSpan(j * 12, 12);
                    edges[i + j] = new CompactEdge
                    {
                        Target = BinaryPrimitives.ReadUInt32LittleEndian(span),
                        Relation = BinaryPrimitives.ReadUInt16LittleEndian(span.Slice(4)),
                        Flags = BinaryPrimitives.ReadUInt16LittleEndian(span.Slice(6)),
                        Weight = BinaryPrimitives.ReadSingleLittleEndian(span.Slice(8)),
                    };
                }

                i += n;
            }

            // The offsets must be monotonic and end exactly at the edge count, or a corrupt file would make Neighbors read out of range.
            if (offsets[0] != 0 || offsets[^1] != edgeCount)
            {
                throw new InvalidDataException("Invalid CSR graph: row offsets do not span the edge array");
            }

            for (int i = 1; i < offsets.Length; i++)
            {
                if (offsets[i] < offsets[i - 1])
                {
                    throw new InvalidDataException("Invalid CSR graph: row offsets are not monotonic");
                }
            }

            return new CsrGraph(offsets, edges, (int)nodes);
        }

        /// <summary>Saves the graph atomically (write a temp file, then rename).</summary>
        public void Save(string path)
        {
            var tmp = path + ".tmp." + Environment.ProcessId + "." + Interlocked.Increment(ref tmpCounter);
            using (var fs = new FileStream(tmp, FileMode.Create, FileAccess.Write))
            {
                Write(fs);
                fs.Flush(true);
            }

            File.Move(tmp, path, overwrite: true);
        }

        private static int tmpCounter;

        /// <summary>Loads a graph from a file.</summary>
        public static CsrGraph Load(string path)
        {
            using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16);
            return Read(fs);
        }
    }

    /// <summary>A thread-safe, writable adjacency-list graph with immediate edge invalidation.</summary>
    public sealed class DynamicGraph
    {
        /// <summary>The largest node id the graph will allocate for, as a safety limit.</summary>
        public const int MaxNodes = 20_000_000;

        private readonly ReaderWriterLockSlim _lock = new ReaderWriterLockSlim();
        private readonly List<List<CompactEdge>> _adjacency;
        private int _edgeCount;

        /// <summary>Initializes a new instance of the <see cref="DynamicGraph"/> class.</summary>
        public DynamicGraph(int initialNodes = 0)
        {
            _adjacency = new List<List<CompactEdge>>(initialNodes);
            for (int i = 0; i < initialNodes; i++)
            {
                _adjacency.Add(new List<CompactEdge>());
            }
        }

        /// <summary>Gets the number of live edges.</summary>
        public int TotalEdges => Volatile.Read(ref _edgeCount);

        /// <summary>Gets the number of nodes allocated.</summary>
        public int NodeCount
        {
            get
            {
                _lock.EnterReadLock();
                try
                {
                    return _adjacency.Count;
                }
                finally
                {
                    _lock.ExitReadLock();
                }
            }
        }

        /// <summary>Inserts (or revives and re-weights) a directed edge, growing the graph if needed.</summary>
        public void InsertEdge(uint source, uint target, ushort relation, float weight)
        {
            uint max = Math.Max(source, target);
            if (max >= MaxNodes)
            {
                throw new ArgumentOutOfRangeException(nameof(source), $"Node id {max} exceeds the safety limit of {MaxNodes}");
            }

            _lock.EnterWriteLock();
            try
            {
                while (_adjacency.Count <= (int)max)
                {
                    int grow = Math.Max(_adjacency.Count / 2, 1024);
                    int target2 = Math.Min(Math.Max((int)max + 1, _adjacency.Count + grow), MaxNodes);
                    while (_adjacency.Count < target2)
                    {
                        _adjacency.Add(new List<CompactEdge>());
                    }
                }

                var list = _adjacency[(int)source];
                for (int i = 0; i < list.Count; i++)
                {
                    var e = list[i];
                    if (e.Target == target && e.Relation == relation)
                    {
                        bool wasActive = e.IsActive;
                        e.Flags = CompactEdge.FlagActive;
                        e.Weight = weight;
                        list[i] = e;
                        if (!wasActive)
                        {
                            Interlocked.Increment(ref _edgeCount);
                        }

                        return;
                    }
                }

                list.Add(new CompactEdge(target, relation, weight));
                Interlocked.Increment(ref _edgeCount);
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>Deactivates an edge; returns <c>true</c> if a live edge was found.</summary>
        public bool DeleteEdge(uint source, uint target, ushort relation)
        {
            _lock.EnterWriteLock();
            try
            {
                if (source >= (uint)_adjacency.Count)
                {
                    return false;
                }

                var list = _adjacency[(int)source];
                for (int i = 0; i < list.Count; i++)
                {
                    var e = list[i];
                    if (e.Target == target && e.Relation == relation && e.IsActive)
                    {
                        e.Flags = (ushort)(e.Flags & ~CompactEdge.FlagActive);
                        list[i] = e;
                        Interlocked.Decrement(ref _edgeCount);
                        return true;
                    }
                }

                return false;
            }
            finally
            {
                _lock.ExitWriteLock();
            }
        }

        /// <summary>The live outgoing edges of a node.</summary>
        public List<CompactEdge> EdgesFor(uint node)
        {
            _lock.EnterReadLock();
            try
            {
                var result = new List<CompactEdge>();
                if (node < (uint)_adjacency.Count)
                {
                    foreach (var e in _adjacency[(int)node])
                    {
                        if (e.IsActive)
                        {
                            result.Add(e);
                        }
                    }
                }

                return result;
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }

        /// <summary>Breadth-first reachability, see <see cref="CsrGraph.BfsReachability"/>.</summary>
        public IdSet BfsReachability(IReadOnlyList<uint> seeds, int maxHops, int maxFanout, IReadOnlyCollection<ushort>? relationFilter = null)
        {
            var visited = new IdSet();
            if (seeds.Count == 0)
            {
                return visited;
            }

            foreach (var s in seeds)
            {
                visited.Add((int)s);
            }

            var frontier = new List<uint>(seeds);
            var next = new List<uint>();
            _lock.EnterReadLock();
            try
            {
                for (int hop = 0; hop < maxHops && frontier.Count > 0; hop++)
                {
                    next.Clear();
                    foreach (var node in frontier)
                    {
                        if (node >= (uint)_adjacency.Count)
                        {
                            continue;
                        }

                        int count = 0;
                        foreach (var edge in _adjacency[(int)node])
                        {
                            if (!edge.IsActive || (relationFilter is not null && !relationFilter.Contains(edge.Relation)))
                            {
                                continue;
                            }

                            if (visited.Add((int)edge.Target))
                            {
                                next.Add(edge.Target);
                            }

                            if (++count >= maxFanout)
                            {
                                break;
                            }
                        }
                    }

                    (frontier, next) = (next, frontier);
                }
            }
            finally
            {
                _lock.ExitReadLock();
            }

            return visited;
        }

        /// <summary>Compacts the live edges into an immutable CSR graph.</summary>
        public CsrGraph ToCsr()
        {
            _lock.EnterReadLock();
            try
            {
                var snapshot = new List<IReadOnlyList<CompactEdge>>(_adjacency.Count);
                foreach (var l in _adjacency)
                {
                    snapshot.Add(l.ToArray());
                }

                return CsrGraph.FromAdjacencyList(snapshot);
            }
            finally
            {
                _lock.ExitReadLock();
            }
        }
    }

    /// <summary>How the graph engine answers reachability queries.</summary>
    public enum GraphStorageLayout
    {
        /// <summary>Walk the dynamic adjacency lists (always current; the default).</summary>
        AdjacencyList = 0,

        /// <summary>Walk the last compacted CSR snapshot (faster; stale until <see cref="GraphEngine.Compact"/>).</summary>
        Csr = 1,
    }

    /// <summary>Configuration of the <see cref="GraphEngine"/>.</summary>
    public sealed class GraphConfig
    {
        /// <summary>Gets or sets a value indicating whether the graph is active. When off, inserts are ignored and reachability returns only the seeds.</summary>
        public bool Enabled { get; set; } = true;

        /// <summary>Gets or sets the storage layout used by queries.</summary>
        public GraphStorageLayout StorageLayout { get; set; } = GraphStorageLayout.AdjacencyList;

        /// <summary>Gets or sets the hard limit on the hops of a query.</summary>
        public int MaxHopsLimit { get; set; } = 3;

        /// <summary>Gets or sets the number of edges followed per node and hop.</summary>
        public int MaxFanoutPerNode { get; set; } = 64;
    }

    /// <summary>The native vector-graph (KAG) engine: dynamic edges, CSR compaction, bounded BFS and metadata-driven edge induction.</summary>
    public sealed class GraphEngine
    {
        private readonly object _snapshotLock = new object();
        private CsrGraph _csr;

        /// <summary>Initializes a new instance of the <see cref="GraphEngine"/> class.</summary>
        public GraphEngine(GraphConfig? config = null, int initialNodes = 0)
        {
            Config = config ?? new GraphConfig();
            Dynamic = new DynamicGraph(initialNodes);
            _csr = Dynamic.ToCsr();
        }

        /// <summary>Gets the configuration.</summary>
        public GraphConfig Config { get; }

        /// <summary>Gets the writable graph.</summary>
        public DynamicGraph Dynamic { get; }

        /// <summary>Gets the latest compacted snapshot.</summary>
        public CsrGraph Snapshot
        {
            get
            {
                lock (_snapshotLock)
                {
                    return _csr;
                }
            }
        }

        /// <summary>Adds a directed edge.</summary>
        public void InsertEdge(uint source, uint target, ushort relation, float weight = 1f)
        {
            if (Config.Enabled)
            {
                Dynamic.InsertEdge(source, target, relation, weight);
            }
        }

        /// <summary>Removes a directed edge; returns <c>true</c> if it existed.</summary>
        public bool DeleteEdge(uint source, uint target, ushort relation) => Config.Enabled && Dynamic.DeleteEdge(source, target, relation);

        /// <summary>The live outgoing edges of a node.</summary>
        public List<CompactEdge> EdgesFor(uint node) => Dynamic.EdgesFor(node);

        /// <summary>The nodes reachable from the seeds within <paramref name="hops"/> hops (capped by <see cref="GraphConfig.MaxHopsLimit"/>).</summary>
        public IdSet GetReachabilityMask(IReadOnlyList<uint> seeds, int hops, IReadOnlyCollection<ushort>? relationFilter = null)
        {
            if (!Config.Enabled)
            {
                var only = new IdSet();
                foreach (var s in seeds)
                {
                    only.Add((int)s);
                }

                return only;
            }

            int bounded = Math.Min(hops, Config.MaxHopsLimit);
            return Config.StorageLayout == GraphStorageLayout.Csr
                ? Snapshot.BfsReachability(seeds, bounded, Config.MaxFanoutPerNode, relationFilter)
                : Dynamic.BfsReachability(seeds, bounded, Config.MaxFanoutPerNode, relationFilter);
        }

        /// <summary>Compacts the dynamic edges into a new CSR snapshot.</summary>
        public void Compact()
        {
            var csr = Dynamic.ToCsr();
            lock (_snapshotLock)
            {
                _csr = csr;
            }
        }

        /// <summary>Saves the current edges to a CSR file.</summary>
        public void SaveToCsrFile(string path) => Dynamic.ToCsr().Save(path);

        /// <summary>Creates an engine from a CSR file.</summary>
        public static GraphEngine LoadFromCsrFile(string path, GraphConfig? config = null)
        {
            var csr = CsrGraph.Load(path);
            var engine = new GraphEngine(config, csr.NumNodes);
            for (uint src = 0; src < (uint)csr.NumNodes; src++)
            {
                foreach (var e in csr.Neighbors(src))
                {
                    engine.Dynamic.InsertEdge(src, e.Target, e.Relation, e.Weight);
                }
            }

            engine.Compact();
            return engine;
        }

        /// <summary>Creates edges between items from their metadata according to declarative rules; returns the number of edges added.</summary>
        public int InduceFromMetadata(IReadOnlyList<(uint Id, IReadOnlyDictionary<string, string> Metadata)> nodes, IReadOnlyList<MetadataInductionRule> rules)
        {
            if (!Config.Enabled || rules.Count == 0)
            {
                return 0;
            }

            int added = 0;
            foreach (var rule in rules)
            {
                ushort rel = EdgeRelation.FromLabel(rule.Relation);
                float weight = rule.Weight ?? 1f;
                switch (rule.RuleType)
                {
                    case "path_hierarchy":
                        {
                            var pathToId = new Dictionary<string, uint>(StringComparer.Ordinal);
                            foreach (var (id, meta) in nodes)
                            {
                                if (meta.TryGetValue(rule.Field, out var path))
                                {
                                    pathToId[path.TrimEnd('/')] = id;
                                }
                            }

                            foreach (var (id, meta) in nodes)
                            {
                                if (!meta.TryGetValue(rule.Field, out var path))
                                {
                                    continue;
                                }

                                var clean = path.TrimEnd('/');
                                int slash = clean.LastIndexOf('/');
                                if (slash >= 0 && pathToId.TryGetValue(clean.Substring(0, slash), out var parent) && parent != id)
                                {
                                    InsertEdge(parent, id, rel, weight);
                                    added++;
                                }
                            }

                            break;
                        }

                    case "foreign_key":
                        {
                            if (rule.TargetField is { } targetField)
                            {
                                var valueToId = new Dictionary<string, uint>(StringComparer.Ordinal);
                                foreach (var (id, meta) in nodes)
                                {
                                    if (meta.TryGetValue(targetField, out var v))
                                    {
                                        valueToId[v] = id;
                                    }
                                }

                                foreach (var (id, meta) in nodes)
                                {
                                    if (meta.TryGetValue(rule.Field, out var fk) && valueToId.TryGetValue(fk, out var target) && target != id)
                                    {
                                        InsertEdge(id, target, rel, weight);
                                        added++;
                                    }
                                }
                            }
                            else
                            {
                                foreach (var (id, meta) in nodes)
                                {
                                    if (meta.TryGetValue(rule.Field, out var fk) && uint.TryParse(fk, out var target) && target != id)
                                    {
                                        InsertEdge(id, target, rel, weight);
                                        added++;
                                    }
                                }
                            }

                            break;
                        }

                    case "exact_match":
                        {
                            var groups = new Dictionary<string, List<uint>>(StringComparer.Ordinal);
                            foreach (var (id, meta) in nodes)
                            {
                                if (meta.TryGetValue(rule.Field, out var v) && v.Length > 0)
                                {
                                    if (!groups.TryGetValue(v, out var members))
                                    {
                                        groups[v] = members = new List<uint>();
                                    }

                                    members.Add(id);
                                }
                            }

                            foreach (var members in groups.Values)
                            {
                                int n = Math.Min(members.Count, 16);
                                for (int i = 0; i < n; i++)
                                {
                                    for (int j = i + 1; j < n; j++)
                                    {
                                        InsertEdge(members[i], members[j], rel, weight);
                                        InsertEdge(members[j], members[i], rel, weight);
                                        added += 2;
                                    }
                                }
                            }

                            break;
                        }
                }
            }

            return added;
        }
    }

    /// <summary>Geometry of 33-dimensional Lorentz manifolds used to embed trees (light cones = descendants).</summary>
    public static class LorentzTreeProjector
    {
        /// <summary>The Minkowski product over the first 33 dimensions.</summary>
        public static double MinkowskiDot(ReadOnlySpan<double> u, ReadOnlySpan<double> v)
        {
            double spatial = 0;
            for (int i = 1; i < Metrics.HybridLorentzDimensions; i++)
            {
                spatial += u[i] * v[i];
            }

            return -(u[0] * v[0]) + spatial;
        }

        /// <summary>The geodesic distance between two 33-dimensional hyperboloid points.</summary>
        public static double LorentzDistance(ReadOnlySpan<double> u, ReadOnlySpan<double> v) => Math.Acosh(Math.Max(-MinkowskiDot(u, v), 1.0 + 1e-15));

        /// <summary>Whether <paramref name="child"/> lies in the future light cone of <paramref name="parent"/> within <paramref name="maxRadius"/>.</summary>
        public static bool IsInLightCone(ReadOnlySpan<double> parent, ReadOnlySpan<double> child, double maxRadius)
            => MinkowskiDot(parent, child) >= -Math.Cosh(maxRadius) && child[0] >= parent[0];

        /// <summary>Places an ideal tree node at the given depth and branch on the hyperboloid.</summary>
        public static double[] ProjectTreeNode(int depth, int branchIndex, double stepRadius)
        {
            double r = depth * stepRadius;
            var coords = new double[Metrics.HybridLorentzDimensions];
            coords[0] = Math.Cosh(r);
            double sinh = Math.Sinh(r);
            if (depth > 0 && sinh > 1e-12)
            {
                double angle = branchIndex * 0.45;
                coords[1] = sinh * Math.Cos(angle);
                coords[2] = sinh * Math.Sin(angle);
            }

            return coords;
        }

        /// <summary>The mean relative distortion between embedded and graph distances.</summary>
        public static double ComputeDistortion(IReadOnlyList<(double[] U, double[] V, double GraphDistance)> edges)
        {
            if (edges.Count == 0)
            {
                return 0.0;
            }

            double total = 0;
            foreach (var (u, v, expected) in edges)
            {
                total += Math.Abs(LorentzDistance(u, v) - expected) / Math.Max(expected, 1e-6);
            }

            return total / edges.Count;
        }
    }
}
