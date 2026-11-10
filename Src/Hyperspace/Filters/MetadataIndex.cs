// <copyright file="MetadataIndex.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;

    /// <summary>
    /// Inverted (<c>key:value</c> to ids), numeric (sorted by integer value) and prefix (sorted values per key) indexes over item metadata.
    /// </summary>
    internal sealed class MetadataIndex
    {
        private readonly Dictionary<string, IdSet> _inverted = new Dictionary<string, IdSet>(StringComparer.Ordinal);
        private readonly Dictionary<string, SortedDictionary<long, IdSet>> _numeric = new Dictionary<string, SortedDictionary<long, IdSet>>(StringComparer.Ordinal);
        private readonly Dictionary<string, SortedSet<string>> _values = new Dictionary<string, SortedSet<string>>(StringComparer.Ordinal);
        private readonly HashSet<string> _fractionalKeys = new HashSet<string>(StringComparer.Ordinal);

        /// <summary>Gets a value indicating whether any item carries a typed shadow value, which the plain indexes cannot see.</summary>
        public bool HasTyped { get; private set; }

        /// <summary>Gets a value indicating whether any metadata is indexed.</summary>
        public bool HasAny => _inverted.Count > 0;

        public void Add(int id, IReadOnlyDictionary<string, string> metadata)
        {
            foreach (var (key, value) in metadata)
            {
                if (key.StartsWith(FilterExpr.TypedKeyPrefix, StringComparison.Ordinal))
                {
                    HasTyped = true;
                }

                string tag = key + ":" + value;
                if (!_inverted.TryGetValue(tag, out var set))
                {
                    _inverted[tag] = set = new IdSet();
                }

                set.Add(id);

                if (!_values.TryGetValue(key, out var values))
                {
                    _values[key] = values = new SortedSet<string>(StringComparer.Ordinal);
                }

                values.Add(value);

                if (long.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var number))
                {
                    if (!_numeric.TryGetValue(key, out var tree))
                    {
                        _numeric[key] = tree = new SortedDictionary<long, IdSet>();
                    }

                    if (!tree.TryGetValue(number, out var ids))
                    {
                        tree[number] = ids = new IdSet();
                    }

                    ids.Add(id);
                }
                else if (double.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out _))
                {
                    _fractionalKeys.Add(key);
                }
            }
        }

        public void Remove(int id, IReadOnlyDictionary<string, string> metadata)
        {
            foreach (var (key, value) in metadata)
            {
                if (_inverted.TryGetValue(key + ":" + value, out var set))
                {
                    set.Remove(id);
                }

                if (long.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var number)
                    && _numeric.TryGetValue(key, out var tree)
                    && tree.TryGetValue(number, out var ids))
                {
                    ids.Remove(id);
                }
            }
        }

        public IdSet? Match(string key, string value) => _inverted.TryGetValue(key + ":" + value, out var set) ? set : null;

        public IdSet Prefix(string key, string prefix)
        {
            var union = new IdSet();
            if (!_values.TryGetValue(key, out var values))
            {
                return union;
            }

            foreach (var value in values.GetViewBetween(prefix, prefix + '￿'))
            {
                if (!value.StartsWith(prefix, StringComparison.Ordinal))
                {
                    break;
                }

                if (_inverted.TryGetValue(key + ":" + value, out var set))
                {
                    union.UnionWith(set);
                }
            }

            return union;
        }

        /// <summary>The ids whose integer value of <paramref name="key"/> is in range, or <c>null</c> if the index cannot answer exactly (fractional or typed values exist).</summary>
        public IdSet? Range(string key, double? gte, double? lte)
        {
            if (HasTyped || _fractionalKeys.Contains(key))
            {
                return null;
            }

            var result = new IdSet();
            if (!_numeric.TryGetValue(key, out var tree))
            {
                return result;
            }

            long start = gte is null ? long.MinValue : (long)Math.Ceiling(gte.Value);
            long end = lte is null ? long.MaxValue : (long)Math.Floor(lte.Value);
            if (start > end)
            {
                return result;
            }

            foreach (var (number, ids) in tree)
            {
                if (number < start)
                {
                    continue;
                }

                if (number > end)
                {
                    break;
                }

                result.UnionWith(ids);
            }

            return result;
        }
    }
}
