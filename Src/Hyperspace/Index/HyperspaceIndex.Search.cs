// <copyright file="HyperspaceIndex.Search.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading;
    using System.Threading.Tasks;

    public sealed partial class HyperspaceIndex
    {
        private readonly object _lexicalLock = new object();

        /// <summary>The settings of a query that passed none; shared, never mutated.</summary>
        private static readonly SearchOptions DefaultSearchOptions = new SearchOptions();

        /// <summary>Finds the <paramref name="k"/> items nearest to a query.</summary>
        /// <param name="query">The query embedding.</param>
        /// <param name="k">How many results to return.</param>
        /// <param name="options">Filters, hybrid text, candidate-list size...</param>
        /// <returns>The hits, nearest first. Fewer than <paramref name="k"/> if fewer items qualify.</returns>
        /// <remarks>Apart from the returned array this does not allocate for an unfiltered, non-hybrid query; see <see cref="Search(ReadOnlySpan{float}, Span{SearchResult}, SearchOptions)"/> to avoid that too.</remarks>
        public SearchResult[] Search(ReadOnlySpan<float> query, int k, SearchOptions? options = null)
        {
            options ??= DefaultSearchOptions;
            var prepared = PrepareQuery(query, k, pooled: true);
            int truncation = ResolveTruncation(options);

            EnterRead();
            try
            {
                var hits = ExecuteSearch(prepared, k, truncation, options, out bool hybrid);
                var results = new SearchResult[hits.Count];
                FillResults(hits, hybrid, options.IncludeMetadata, results);
                return results;
            }
            finally
            {
                ExitRead();
            }
        }

        /// <summary>
        /// Finds the <c>results.Length</c> items nearest to a query and writes them, nearest first, into <paramref name="results"/>.
        /// </summary>
        /// <returns>The number of results written, which is smaller than <c>results.Length</c> if fewer items qualify.</returns>
        /// <remarks>
        /// A query without filters, hybrid text or a graph restriction allocates nothing once the calling thread has warmed up (its scratch
        /// buffers and prepared query are reused). Filtered queries allocate the id sets they compute, and hybrid queries their rankings.
        /// </remarks>
        public int Search(ReadOnlySpan<float> query, Span<SearchResult> results, SearchOptions? options = null)
        {
            options ??= DefaultSearchOptions;
            var prepared = PrepareQuery(query, results.Length, pooled: true);
            int truncation = ResolveTruncation(options);

            EnterRead();
            try
            {
                var hits = ExecuteSearch(prepared, results.Length, truncation, options, out bool hybrid);
                int count = Math.Min(hits.Count, results.Length);
                FillResults(hits, hybrid, options.IncludeMetadata, results.Slice(0, count));
                return count;
            }
            finally
            {
                ExitRead();
            }
        }

        /// <summary>Runs a query under the read lock; the hits are in the calling thread's reusable list and valid until its next search.</summary>
        private List<(int Id, double Distance)> ExecuteSearch(PreparedQuery prepared, int k, int truncation, SearchOptions options, out bool hybrid)
        {
            options.CancellationToken.ThrowIfCancellationRequested();
            var allowed = BuildAllowed(options);
            int ef = Math.Max(options.Ef ?? _options.EfSearch, k);
            var hits = Scratch.Hits;

            hybrid = !string.IsNullOrWhiteSpace(options.HybridQuery);
            if (hybrid)
            {
                hits.Clear();
                foreach (var (id, distance, _) in SearchHybridCore(prepared, k, ef, allowed, truncation, options))
                {
                    hits.Add((id, distance));
                }

                return hits;
            }

            int oversample = EffectiveOversample(options);
            int firstPass = oversample > 1 ? checked(k * oversample) : k;
            SearchCore(prepared, firstPass, Math.Max(ef, firstPass), allowed, truncation, options.CancellationToken, hits);
            if (oversample > 1)
            {
                RerankExactly(hits, prepared, k);
            }

            return hits;
        }

        private void FillResults(List<(int Id, double Distance)> hits, bool hybrid, bool includeMetadata, Span<SearchResult> destination)
        {
            for (int i = 0; i < destination.Length; i++)
            {
                var (id, distance) = hits[i];
                destination[i] = new SearchResult(id, distance, includeMetadata ? _metadata[id] : null, hybrid ? 10.0 - distance : null);
            }
        }

        /// <summary>Searches for many queries in parallel.</summary>
        public SearchResult[][] SearchBatch(IReadOnlyList<float[]> queries, int k, SearchOptions? options = null)
        {
            var results = new SearchResult[queries.Count][];
            Parallel.For(0, queries.Count, i => results[i] = Search(queries[i], k, options));
            return results;
        }

        /// <summary>
        /// Searches ahead of a trajectory: extrapolates <paramref name="past"/> to <paramref name="current"/> by <paramref name="steps"/> more steps in
        /// the geometry of the metric (Mobius / Koopman in the hyperbolic models, linear otherwise) and returns the neighbours of the predicted point.
        /// </summary>
        public SearchResult[] SearchMomentum(ReadOnlySpan<float> past, ReadOnlySpan<float> current, double steps, int k, SearchOptions? options = null)
        {
            if (past.Length != _options.Dimension || current.Length != _options.Dimension)
            {
                throw new ArgumentException($"Expected vectors of {_options.Dimension} dimensions");
            }

            var p = new double[past.Length];
            var c = new double[current.Length];
            for (int i = 0; i < p.Length; i++)
            {
                p[i] = past[i];
                c[i] = current[i];
            }

            var next = Riemannian.ExtrapolateMomentum(_options.Metric, p, c, steps);
            var q = new float[next.Length];
            for (int i = 0; i < q.Length; i++)
            {
                q[i] = (float)next[i];
            }

            return Search(q, k, options);
        }

        /// <summary>
        /// Answers a fuzzy logic query (AND / OR / NOT over vectors): every item gets a membership in [0, 1] and the best <paramref name="k"/> are returned,
        /// with <see cref="SearchResult.Score"/> the membership and <see cref="SearchResult.Distance"/> its complement.
        /// </summary>
        /// <remarks>
        /// Candidates come from the neighbourhood of every vector in the query. A query containing <c>NOT</c> can be satisfied by items far from
        /// every vector, so it is evaluated over all (allowed) items.
        /// </remarks>
        public SearchResult[] SearchFuzzy(FuzzyQuery query, int k, SearchOptions? options = null)
        {
            if (k <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(k));
            }

            options ??= new SearchOptions();
            var leaves = new List<float[]>();
            query.ExtractVectors(leaves);
            if (leaves.Count == 0)
            {
                return Array.Empty<SearchResult>();
            }

            var prepared = leaves.Select(l => PrepareQuery(l, k, pooled: false)).ToArray();
            EnterRead();
            try
            {
                var allowed = BuildAllowed(options);
                int ef = Math.Max(options.Ef ?? _options.EfSearch, k);
                var candidates = new HashSet<int>();
                if (query.HasNegation)
                {
                    if (allowed is null)
                    {
                        for (int i = 0; i < _vectors.Count; i++)
                        {
                            if (!_deleted.Contains(i))
                            {
                                candidates.Add(i);
                            }
                        }
                    }
                    else
                    {
                        foreach (int i in allowed)
                        {
                            candidates.Add(i);
                        }
                    }
                }
                else
                {
                    int candidateK = Math.Max(k * 4, ef);
                    var found = new List<(int Id, double Distance)>();
                    foreach (var p in prepared)
                    {
                        SearchCore(p, candidateK, Math.Max(ef, candidateK), allowed, 0, options.CancellationToken, found);
                        foreach (var (id, _) in found)
                        {
                            candidates.Add(id);
                        }
                    }
                }

                var scored = new List<(int Id, double Distance, double? Score)>(candidates.Count);
                foreach (int id in candidates)
                {
                    int counter = 0;
                    var dist = new float[prepared.Length];
                    for (int i = 0; i < dist.Length; i++)
                    {
                        dist[i] = (float)Dist(id, prepared[i]);
                    }

                    float membership = query.Evaluate(ref counter, i => dist[i]);
                    scored.Add((id, 1.0 - membership, membership));
                }

                scored.Sort((a, b) => a.Distance.CompareTo(b.Distance));
                var top = scored.Take(k).ToList();
                var results = new SearchResult[top.Count];
                for (int i = 0; i < results.Length; i++)
                {
                    results[i] = new SearchResult(top[i].Id, top[i].Distance, options.IncludeMetadata ? _metadata[top[i].Id] : null, top[i].Score);
                }

                return results;
            }
            finally
            {
                ExitRead();
            }
        }

        private int EffectiveOversample(SearchOptions options) => _originals is null ? 1 : Math.Max(1, options.RerankOversample ?? _options.RerankOversample);

        /// <summary>
        /// The second pass of two-pass search: the candidates the (approximate) first pass collected are scored again, in place, with the exact metric
        /// against the original vectors, and the best <paramref name="k"/> are kept.
        /// </summary>
        private void RerankExactly(List<(int Id, double Distance)> candidates, PreparedQuery q, int k)
        {
            for (int i = 0; i < candidates.Count; i++)
            {
                int id = candidates[i].Id;
                candidates[i] = (id, Metrics.Distance(_options.Metric, (ReadOnlySpan<float>)_originals![id], q.Coords32));
            }

            candidates.Sort(ByDistance);
            if (candidates.Count > k)
            {
                candidates.RemoveRange(k, candidates.Count - k);
            }
        }

        [ThreadStatic]
        private static float[]? queryBuffer;

        /// <summary>
        /// Validates, normalizes and prepares a query. A pooled query reuses the buffers of the previous search of this thread on this index
        /// (it is only valid until that thread's next search); an unpooled one is independent, for callers that hold several at once.
        /// </summary>
        private PreparedQuery PrepareQuery(ReadOnlySpan<float> query, int k, bool pooled)
        {
            if (k <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(k), "k must be positive");
            }

            if (query.Length != _options.Dimension)
            {
                throw new ArgumentException($"Expected a query of {_options.Dimension} dimensions but got {query.Length}", nameof(query));
            }

            foreach (var x in query)
            {
                if (!float.IsFinite(x))
                {
                    throw new ArgumentException("The query contains non-finite values (NaN/Inf)", nameof(query));
                }
            }

            var buffer = queryBuffer;
            if (buffer is null || buffer.Length < query.Length)
            {
                buffer = queryBuffer = new float[Math.Max(query.Length, 256)];
            }

            var work = buffer.AsSpan(0, query.Length);
            query.CopyTo(work);
            if (_options.ResolvedNormalize)
            {
                Metrics.Normalize(work);
            }

            if (!pooled)
            {
                return _codec.Prepare(work);
            }

            var scratch = Scratch;
            var prepared = _codec.Prepare(work, ReferenceEquals(scratch.QueryOwner, _codec) ? scratch.Query : null);
            scratch.Query = prepared;
            scratch.QueryOwner = _codec;
            return prepared;
        }

        private int ResolveTruncation(SearchOptions options)
        {
            if (options.TruncateDimensions is not int t)
            {
                return 0;
            }

            if (t < 1 || t > _options.Dimension)
            {
                throw new ArgumentOutOfRangeException(nameof(options), "TruncateDimensions must be within [1, Dimension]");
            }

            if (!_codec.SupportsTruncation)
            {
                throw new NotSupportedException($"{_options.Quantization} storage does not support searching a prefix of the dimensions");
            }

            return t == _options.Dimension ? 0 : t;
        }

        // ------------------------------------------------------------------ core ANN search

        private void SearchCore(PreparedQuery q, int k, int ef, IdSet? allowed, int truncation, CancellationToken ct, List<(int Id, double Distance)> output)
        {
            output.Clear();
            if (_entryPoint < 0 || _vectors.Count - _deletedCount == 0)
            {
                return;
            }

            ef = Math.Max(ef, k);
            if (allowed is not null)
            {
                int count = allowed.Count;
                if (count == 0)
                {
                    return;
                }

                if (count <= FilteredBruteForceThreshold(_vectors.Count, ef))
                {
                    output.AddRange(BruteForce(q, k, allowed, truncation, ct));
                    return;
                }
            }

            int current = _entryPoint;
            double currentDist = truncation > 0 ? Dist(current, q, truncation) : UpperDist(current, q);
            for (int l = _maxLevel; l > 0; l--)
            {
                bool changed = true;
                while (changed)
                {
                    changed = false;
                    foreach (int nb in LinksOf(current, l))
                    {
                        double d = truncation > 0 ? Dist(nb, q, truncation) : UpperDist(nb, q);
                        if (d < currentDist)
                        {
                            currentDist = d;
                            current = nb;
                            changed = true;
                        }
                    }
                }
            }

            SearchLayer0(current, q, k, ef, allowed, truncation, ct, output);
        }

        [System.Runtime.CompilerServices.MethodImpl(System.Runtime.CompilerServices.MethodImplOptions.AggressiveInlining)]
        private bool IsAllowed(int id, IdSet? allowed) => allowed is not null ? allowed.Contains(id) : _deletedCount == 0 || !_deleted.Contains(id);

        private void SearchLayer0(int entry, PreparedQuery q, int k, int ef, IdSet? allowed, int truncation, CancellationToken ct, List<(int Id, double Distance)> output)
        {
            var s = Scratch;
            s.Begin(_nodes.Count);
            var candidates = s.Candidates;
            var results = s.Results;

            double d0 = Dist(entry, q, truncation);
            s.Visit(entry);
            candidates.Enqueue(entry, d0);
            if (IsAllowed(entry, allowed))
            {
                results.Enqueue(entry, -d0);
            }

            int iterations = 0;
            while (candidates.TryDequeue(out int c, out double cd))
            {
                if (results.Count >= ef && results.TryPeek(out _, out double negWorst) && cd > -negWorst)
                {
                    break;
                }

                if ((++iterations & 255) == 0)
                {
                    ct.ThrowIfCancellationRequested();
                }

                foreach (int nb in Layer0Of(c))
                {
                    if (!s.Visit(nb))
                    {
                        continue;
                    }

                    double d = Dist(nb, q, truncation);
                    if (results.Count >= ef && results.TryPeek(out _, out double nw) && d > -nw)
                    {
                        continue;
                    }

                    candidates.Enqueue(nb, d);
                    if (IsAllowed(nb, allowed))
                    {
                        results.Enqueue(nb, -d);
                        if (results.Count > ef)
                        {
                            results.Dequeue();
                        }
                    }
                }
            }

            while (results.TryDequeue(out int id, out double nd))
            {
                output.Add((id, -nd));
            }

            output.Reverse();
            if (output.Count > k)
            {
                output.RemoveRange(k, output.Count - k);
            }
        }

        /// <summary>
        /// The number of allowed items up to which a filtered search scans them all instead of walking the graph: a selective filter leaves the
        /// graph too sparse to navigate, whereas an exact scan of a small set is cheap and has perfect recall.
        /// </summary>
        private int FilteredBruteForceThreshold(int nodeCount, int ef)
        {
            if (nodeCount <= 2000)
            {
                return nodeCount;
            }

            double m = Math.Max(_options.M, 4);
            double efF = Math.Max(ef, 16);
            double baseValue = Math.Sqrt(nodeCount * efF * m) * 1.75;
            int max = Math.Max(nodeCount / 2, 2000);
            int min = Math.Min(1000, nodeCount);
            return (int)Math.Clamp(baseValue, min, max);
        }

        private List<(int Id, double Distance)> BruteForce(PreparedQuery q, int k, IdSet allowed, int truncation, CancellationToken ct)
        {
            var ids = allowed.ToArray();
            var merged = new PriorityQueue<int, double>();
            var mergeLock = new object();

            void Offer(PriorityQueue<int, double> heap, int id, double d)
            {
                if (heap.Count < k)
                {
                    heap.Enqueue(id, -d);
                }
                else if (heap.TryPeek(out _, out double nw) && d < -nw)
                {
                    heap.Dequeue();
                    heap.Enqueue(id, -d);
                }
            }

            if (ids.Length < 4096)
            {
                foreach (int id in ids)
                {
                    if ((uint)id < (uint)_vectors.Count && !_deleted.Contains(id))
                    {
                        Offer(merged, id, Dist(id, q, truncation));
                    }
                }
            }
            else
            {
                Parallel.ForEach(
                    Partitioner.Create(0, ids.Length, Math.Max(256, ids.Length / (Environment.ProcessorCount * 4))),
                    new ParallelOptions { CancellationToken = ct },
                    () => new PriorityQueue<int, double>(),
                    (range, _, local) =>
                    {
                        for (int i = range.Item1; i < range.Item2; i++)
                        {
                            int id = ids[i];
                            if ((uint)id < (uint)_vectors.Count && !_deleted.Contains(id))
                            {
                                Offer(local, id, Dist(id, q, truncation));
                            }
                        }

                        return local;
                    },
                    local =>
                    {
                        lock (mergeLock)
                        {
                            while (local.TryDequeue(out int id, out double nd))
                            {
                                Offer(merged, id, -nd);
                            }
                        }
                    });
            }

            var output = new List<(int, double)>(merged.Count);
            while (merged.TryDequeue(out int id, out double nd))
            {
                output.Add((id, -nd));
            }

            output.Reverse();
            return output;
        }

        // ------------------------------------------------------------------ hybrid search (vector + BM25)

        private LexicalIndex EnsureLexical()
        {
            var lex = _lexical;
            if (lex is not null)
            {
                return lex;
            }

            lock (_lexicalLock)
            {
                if (_lexical is null)
                {
                    var built = new LexicalIndex(CreateTokenizerFunction());
                    for (int i = 0; i < _metadata.Count; i++)
                    {
                        if (_metadata[i] is { } m && !_deleted.Contains(i))
                        {
                            built.Upsert(i, null, m);
                        }
                    }

                    _lexical = built;
                }

                return _lexical;
            }
        }

        private Func<string, List<string>> CreateTokenizerFunction()
        {
            var tokenizer = _options.Tokenizer ?? Tokenizer.ForLanguage(_options.Bm25.Language);
            int ngrams = _options.Bm25.NGrams;
            return text => tokenizer.Tokenize(text, ngrams);
        }

        private List<(int Id, double Distance, double? Score)> SearchHybridCore(PreparedQuery q, int k, int ef, IdSet? allowed, int truncation, SearchOptions o)
        {
            var lexical = EnsureLexical();
            if (allowed is not null && allowed.IsEmpty)
            {
                return new List<(int, double, double?)>();
            }

            int vectorK = Math.Max(k * 10, 100);
            var vectorHits = new List<(int Id, double Distance)>();
            SearchCore(q, vectorK, Math.Max(ef, vectorK), allowed, truncation, o.CancellationToken, vectorHits);
            if (EffectiveOversample(o) > 1)
            {
                RerankExactly(vectorHits, q, vectorHits.Count);
            }

            var tokens = CreateTokenizerFunction()(o.HybridQuery!);
            if (tokens.Count == 0)
            {
                return vectorHits.Take(k).Select(h => (h.Id, h.Distance, (double?)null)).ToList();
            }

            var bm25 = o.Bm25 ?? _options.Bm25;
            int liveDocs = _vectors.Count - _deletedCount;
            int keywordLimit = Math.Max(k * 3, 200);
            var keywordHits = lexical.Search(tokens, bm25, liveDocs, id => !_deleted.Contains(id) && (allowed is null || allowed.Contains(id)), keywordLimit);

            string fusion = (o.FusionMethod ?? _options.FusionMethod).ToLowerInvariant();
            var fused = new Dictionary<int, double>();
            if (fusion == "weighted")
            {
                double vectorWeight = Math.Clamp(o.HybridAlpha ?? 0.5f, 0.0, 1.0);
                double keywordWeight = 1.0 - vectorWeight;
                if (vectorHits.Count > 0)
                {
                    double vMin = vectorHits.Min(h => h.Distance), vMax = vectorHits.Max(h => h.Distance);
                    double vRange = Math.Max(vMax - vMin, 1e-9);
                    foreach (var (id, dist) in vectorHits)
                    {
                        fused[id] = fused.GetValueOrDefault(id) + ((1.0 - ((dist - vMin) / vRange)) * vectorWeight);
                    }
                }

                if (keywordHits.Count > 0)
                {
                    double kMin = keywordHits.Min(h => h.Score), kMax = keywordHits.Max(h => h.Score);
                    double kRange = Math.Max(kMax - kMin, 1e-9);
                    foreach (var (id, score) in keywordHits)
                    {
                        fused[id] = fused.GetValueOrDefault(id) + (((score - kMin) / kRange) * keywordWeight);
                    }
                }
            }
            else
            {
                // Reciprocal rank fusion: 1 / (c + rank), summed over the two rankings.
                double c = o.HybridAlpha ?? 60f;
                for (int rank = 0; rank < vectorHits.Count; rank++)
                {
                    fused[vectorHits[rank].Id] = fused.GetValueOrDefault(vectorHits[rank].Id) + (1.0 / (c + rank + 1));
                }

                for (int rank = 0; rank < keywordHits.Count; rank++)
                {
                    fused[keywordHits[rank].Id] = fused.GetValueOrDefault(keywordHits[rank].Id) + (1.0 / (c + rank + 1));
                }
            }

            return fused.OrderByDescending(kv => kv.Value)
                        .ThenBy(kv => kv.Key)
                        .Take(k)
                        .Select(kv => (kv.Key, 10.0 - kv.Value, (double?)kv.Value))
                        .ToList();
        }

        // ------------------------------------------------------------------ filters

        /// <summary>Compiles the filters of a query into the set of ids allowed in the results, or <c>null</c> if there are none. Must hold the read lock.</summary>
        private IdSet? BuildAllowed(SearchOptions o)
        {
            IdSet? set = null;

            void Apply(IdSet s)
            {
                if (set is null)
                {
                    set = s.Clone();
                }
                else
                {
                    set.IntersectWith(s);
                }
            }

            if (o.ExactMetadata is { Count: > 0 } exact)
            {
                foreach (var (key, value) in exact)
                {
                    Apply(Evaluate(FilterExpr.Match(key, value), set));
                }
            }

            if (o.Filter is not null)
            {
                set = Evaluate(o.Filter, set);
            }

            if (o.Graph is { } g)
            {
                Apply(Graph.GetReachabilityMask(g.Seeds, g.Hops, g.Relations));
            }

            set?.ExceptWith(_deleted);
            return set;
        }

        /// <summary>The live ids matching an expression, within <paramref name="scope"/> if given. Always returns a set the caller owns.</summary>
        private IdSet Evaluate(FilterExpr expr, IdSet? scope)
        {
            IdSet Restrict(IdSet s)
            {
                var r = s.Clone();
                if (scope is not null)
                {
                    r.IntersectWith(scope);
                }

                return r;
            }

            switch (expr)
            {
                case FilterExpr.MatchFilter m when !_metaIndex.HasTyped:
                    return _metaIndex.Match(m.Key, m.Value) is { } hit ? Restrict(hit) : new IdSet();
                case FilterExpr.PrefixFilter p when !_metaIndex.HasTyped:
                    return Restrict(_metaIndex.Prefix(p.Key, p.Value));
                case FilterExpr.RangeFilter r when _metaIndex.Range(r.Key, r.Gte, r.Lte) is { } range:
                    return Restrict(range);
                case FilterExpr.AndFilter and:
                    {
                        var current = scope;
                        IdSet? result = null;
                        foreach (var child in and.Filters)
                        {
                            result = Evaluate(child, current);
                            current = result;
                            if (result.IsEmpty)
                            {
                                break;
                            }
                        }

                        return result ?? (scope is null ? Universe() : scope.Clone());
                    }

                case FilterExpr.OrFilter or:
                    {
                        var union = new IdSet();
                        foreach (var child in or.Filters)
                        {
                            union.UnionWith(Evaluate(child, scope));
                        }

                        return union;
                    }

                case FilterExpr.NotFilter not:
                    {
                        var universe = scope is null ? Universe() : scope.Clone();
                        universe.ExceptWith(Evaluate(not.Inner, scope));
                        return universe;
                    }

                default:
                    return Scan(expr, scope);
            }
        }

        private IdSet Universe()
        {
            var all = IdSet.Range(_vectors.Count);
            all.ExceptWith(_deleted);
            return all;
        }

        /// <summary>Tests an expression item by item (metadata via <see cref="FilterExpr.CheckMetadata"/>, geometry on the decoded vector).</summary>
        private IdSet Scan(FilterExpr expr, IdSet? scope)
        {
            var result = new IdSet();
            var coords = new double[_options.Dimension];
            var floats = new float[_options.Dimension];
            var noMetadata = new Dictionary<string, string>();
            IEnumerable<int> ids = scope ?? (IEnumerable<int>)Enumerable.Range(0, _vectors.Count);
            foreach (int id in ids)
            {
                if ((uint)id >= (uint)_vectors.Count || _deleted.Contains(id))
                {
                    continue;
                }

                _codec.Decode(_vectors[id], floats);
                for (int i = 0; i < coords.Length; i++)
                {
                    coords[i] = floats[i];
                }

                if (expr.Check(coords, _metadata[id] ?? (IReadOnlyDictionary<string, string>)noMetadata))
                {
                    result.Add(id);
                }
            }

            return result;
        }
    }
}
