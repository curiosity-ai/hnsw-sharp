// <copyright file="LexicalIndex.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// An in-memory inverted index over the text values of the metadata of every item, scored with BM25.
    /// Metadata keys starting with <c>__hs_</c> are internal and are not indexed.
    /// </summary>
    internal sealed class LexicalIndex
    {
        private readonly Func<string, List<string>> _tokenize;
        private readonly Dictionary<string, Dictionary<int, ushort>> _postings = new Dictionary<string, Dictionary<int, ushort>>(StringComparer.Ordinal);
        private readonly List<int> _docLength = new List<int>();
        private long _totalLength;

        public LexicalIndex(Func<string, List<string>> tokenize)
        {
            _tokenize = tokenize;
        }

        public long TotalLength => _totalLength;

        public int TermCount => _postings.Count;

        private (Dictionary<string, ushort> Terms, int Length) Analyze(IReadOnlyDictionary<string, string> metadata)
        {
            var tf = new Dictionary<string, ushort>(StringComparer.Ordinal);
            int length = 0;
            foreach (var kv in metadata)
            {
                if (kv.Key.StartsWith("__hs_", StringComparison.Ordinal))
                {
                    continue;
                }

                foreach (var token in _tokenize(kv.Value))
                {
                    length++;
                    tf.TryGetValue(token, out var current);
                    tf[token] = current == ushort.MaxValue ? current : (ushort)(current + 1);
                }
            }

            return (tf, length);
        }

        public void Remove(int id, IReadOnlyDictionary<string, string>? oldMetadata)
        {
            if (id < _docLength.Count)
            {
                _totalLength -= _docLength[id];
                _docLength[id] = 0;
            }

            if (oldMetadata is null)
            {
                return;
            }

            foreach (var term in Analyze(oldMetadata).Terms.Keys)
            {
                if (_postings.TryGetValue(term, out var docs) && docs.Remove(id) && docs.Count == 0)
                {
                    _postings.Remove(term);
                }
            }
        }

        public void Upsert(int id, IReadOnlyDictionary<string, string>? oldMetadata, IReadOnlyDictionary<string, string> metadata)
        {
            Remove(id, oldMetadata);
            var (terms, length) = Analyze(metadata);
            while (_docLength.Count <= id)
            {
                _docLength.Add(0);
            }

            _docLength[id] = length;
            _totalLength += length;
            foreach (var kv in terms)
            {
                if (!_postings.TryGetValue(kv.Key, out var docs))
                {
                    _postings[kv.Key] = docs = new Dictionary<int, ushort>();
                }

                docs[id] = kv.Value;
            }
        }

        public int DocumentLength(int id) => id < _docLength.Count ? _docLength[id] : 0;

        public int DocumentFrequency(string term) => _postings.TryGetValue(term, out var docs) ? docs.Count : 0;

        /// <summary>Scores every document containing at least one query term and returns the best <paramref name="limit"/>.</summary>
        public List<(int Id, double Score)> Search(IEnumerable<string> queryTokens, Bm25Parameters p, int totalDocs, Func<int, bool> isAllowed, int limit)
        {
            var scores = new Dictionary<int, double>();
            double docs = Math.Max(totalDocs, 1);
            float avgdl = (float)Math.Max(_totalLength / docs, 1.0);
            foreach (var token in new HashSet<string>(queryTokens, StringComparer.Ordinal))
            {
                if (!_postings.TryGetValue(token, out var postings))
                {
                    continue;
                }

                float idf = Bm25.Idf(p.Method, totalDocs, Math.Max(postings.Count, 1));
                foreach (var (id, tf) in postings)
                {
                    if (!isAllowed(id))
                    {
                        continue;
                    }

                    float dl = Math.Max(DocumentLength(id), 1);
                    double s = Bm25.ScoreWithIdf(idf, p.Method, tf, dl, avgdl, p.K1, p.B, p.Delta);
                    scores[id] = scores.GetValueOrDefault(id) + s;
                }
            }

            return scores.Select(kv => (kv.Key, kv.Value))
                         .OrderByDescending(x => x.Value)
                         .ThenBy(x => x.Key)
                         .Take(limit)
                         .ToList();
        }
    }
}
