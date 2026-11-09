// <copyright file="FilterExpr.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;
    using System.Linq;
    using System.Text.Json;

    /// <summary>
    /// A filter over item metadata and / or geometry. Compose them with <see cref="And"/>, <see cref="Or"/> and <see cref="Not"/>.
    /// </summary>
    /// <remarks>
    /// Typed metadata (numbers, booleans, strings sent through typed APIs) lives in a shadow key <c>__hs_typed__{key}</c> as the JSON
    /// <c>{"v": value}</c>; filters transparently fall back to it when the plain key is absent.
    /// </remarks>
    public abstract class FilterExpr
    {
        /// <summary>The prefix of the shadow key holding the typed JSON value of a metadata entry.</summary>
        public const string TypedKeyPrefix = "__hs_typed__";

        /// <summary>Items whose metadata <paramref name="key"/> equals <paramref name="value"/>.</summary>
        public static FilterExpr Match(string key, string value) => new MatchFilter(key, value);

        /// <summary>Items whose metadata <paramref name="key"/> starts with <paramref name="prefix"/>.</summary>
        public static FilterExpr Prefix(string key, string prefix) => new PrefixFilter(key, prefix);

        /// <summary>Items whose numeric metadata <paramref name="key"/> is within the (inclusive, optional) bounds.</summary>
        public static FilterExpr Range(string key, double? gte, double? lte) => new RangeFilter(key, gte, lte);

        /// <summary>Items inside the Cartesian product of angular sectors (ConE embeddings, Zhang and Wang 2021).</summary>
        public static FilterExpr InCone(double[] axes, double[] apertures, double cen = 0.0) => new ConeFilter(axes, apertures, cen);

        /// <summary>Items inside an axis-aligned box.</summary>
        public static FilterExpr InBox(double[] minBounds, double[] maxBounds) => new BoxFilter(minBounds, maxBounds);

        /// <summary>Items within <paramref name="radius"/> of <paramref name="center"/> (hyperbolic radius for 33-dimensional Lorentz centers).</summary>
        public static FilterExpr InBall(double[] center, double radius) => new BallFilter(center, radius);

        /// <summary>All of the filters.</summary>
        public static FilterExpr And(params FilterExpr[] filters) => new AndFilter(filters);

        /// <summary>Any of the filters.</summary>
        public static FilterExpr Or(params FilterExpr[] filters) => new OrFilter(filters);

        /// <summary>Items not matching the filter.</summary>
        public static FilterExpr Not(FilterExpr filter) => new NotFilter(filter);

        /// <summary>Evaluates the filter against an item's coordinates and metadata.</summary>
        public abstract bool Check(ReadOnlySpan<double> coordinates, IReadOnlyDictionary<string, string> metadata);

        /// <summary>Evaluates only the metadata part; geometric filters pass.</summary>
        public abstract bool CheckMetadata(IReadOnlyDictionary<string, string> metadata);

        internal static string? GetString(IReadOnlyDictionary<string, string> metadata, string key)
        {
            if (metadata.TryGetValue(key, out var v))
            {
                return v;
            }

            if (!metadata.TryGetValue(TypedKeyPrefix + key, out var raw))
            {
                return null;
            }

            try
            {
                using var doc = JsonDocument.Parse(raw);
                if (!doc.RootElement.TryGetProperty("v", out var el))
                {
                    return null;
                }

                return el.ValueKind switch
                {
                    JsonValueKind.String => el.GetString(),
                    JsonValueKind.Number => el.TryGetInt64(out var l) ? l.ToString(CultureInfo.InvariantCulture) : el.GetDouble().ToString(CultureInfo.InvariantCulture),
                    JsonValueKind.True => "true",
                    JsonValueKind.False => "false",
                    _ => null,
                };
            }
            catch (JsonException)
            {
                return null;
            }
        }

        internal static double? GetNumber(IReadOnlyDictionary<string, string> metadata, string key)
        {
            if (metadata.TryGetValue(key, out var v) && double.TryParse(v, NumberStyles.Float, CultureInfo.InvariantCulture, out var d))
            {
                return d;
            }

            if (!metadata.TryGetValue(TypedKeyPrefix + key, out var raw))
            {
                return null;
            }

            try
            {
                using var doc = JsonDocument.Parse(raw);
                if (doc.RootElement.TryGetProperty("v", out var el) && el.ValueKind == JsonValueKind.Number)
                {
                    return el.GetDouble();
                }
            }
            catch (JsonException)
            {
            }

            return null;
        }

        internal sealed class MatchFilter : FilterExpr
        {
            public MatchFilter(string key, string value)
            {
                Key = key;
                Value = value;
            }

            public string Key { get; }

            public string Value { get; }

            public override bool Check(ReadOnlySpan<double> c, IReadOnlyDictionary<string, string> m) => CheckMetadata(m);

            public override bool CheckMetadata(IReadOnlyDictionary<string, string> m) => string.Equals(GetString(m, Key), Value, StringComparison.Ordinal);
        }

        internal sealed class PrefixFilter : FilterExpr
        {
            public PrefixFilter(string key, string prefix)
            {
                Key = key;
                Value = prefix;
            }

            public string Key { get; }

            public string Value { get; }

            public override bool Check(ReadOnlySpan<double> c, IReadOnlyDictionary<string, string> m) => CheckMetadata(m);

            public override bool CheckMetadata(IReadOnlyDictionary<string, string> m) => GetString(m, Key)?.StartsWith(Value, StringComparison.Ordinal) ?? false;
        }

        internal sealed class RangeFilter : FilterExpr
        {
            public RangeFilter(string key, double? gte, double? lte)
            {
                Key = key;
                Gte = gte;
                Lte = lte;
            }

            public string Key { get; }

            public double? Gte { get; }

            public double? Lte { get; }

            public bool InRange(double value) => (Gte is null || value >= Gte) && (Lte is null || value <= Lte);

            public override bool Check(ReadOnlySpan<double> c, IReadOnlyDictionary<string, string> m) => CheckMetadata(m);

            public override bool CheckMetadata(IReadOnlyDictionary<string, string> m) => GetNumber(m, Key) is double v && InRange(v);
        }

        internal sealed class ConeFilter : FilterExpr
        {
            public ConeFilter(double[] axes, double[] apertures, double cen)
            {
                Region = new ConeRegion(axes, apertures, cen);
            }

            public ConeRegion Region { get; }

            public override bool Check(ReadOnlySpan<double> c, IReadOnlyDictionary<string, string> m) => Region.Contains(c);

            public override bool CheckMetadata(IReadOnlyDictionary<string, string> m) => true;
        }

        internal sealed class BoxFilter : FilterExpr
        {
            public BoxFilter(double[] min, double[] max)
            {
                Region = new BoxRegion(min, max);
            }

            public BoxRegion Region { get; }

            public override bool Check(ReadOnlySpan<double> c, IReadOnlyDictionary<string, string> m) => Region.Contains(c);

            public override bool CheckMetadata(IReadOnlyDictionary<string, string> m) => true;
        }

        internal sealed class BallFilter : FilterExpr
        {
            public BallFilter(double[] center, double radius)
            {
                Region = new BallRegion(center, radius);
            }

            public BallRegion Region { get; }

            public override bool Check(ReadOnlySpan<double> c, IReadOnlyDictionary<string, string> m) => Region.Contains(c);

            public override bool CheckMetadata(IReadOnlyDictionary<string, string> m) => true;
        }

        internal sealed class AndFilter : FilterExpr
        {
            public AndFilter(FilterExpr[] filters)
            {
                Filters = filters;
            }

            public FilterExpr[] Filters { get; }

            public override bool Check(ReadOnlySpan<double> c, IReadOnlyDictionary<string, string> m)
            {
                foreach (var f in Filters)
                {
                    if (!f.Check(c, m))
                    {
                        return false;
                    }
                }

                return true;
            }

            public override bool CheckMetadata(IReadOnlyDictionary<string, string> m) => Filters.All(f => f.CheckMetadata(m));
        }

        internal sealed class OrFilter : FilterExpr
        {
            public OrFilter(FilterExpr[] filters)
            {
                Filters = filters;
            }

            public FilterExpr[] Filters { get; }

            public override bool Check(ReadOnlySpan<double> c, IReadOnlyDictionary<string, string> m)
            {
                foreach (var f in Filters)
                {
                    if (f.Check(c, m))
                    {
                        return true;
                    }
                }

                return false;
            }

            public override bool CheckMetadata(IReadOnlyDictionary<string, string> m) => Filters.Any(f => f.CheckMetadata(m));
        }

        internal sealed class NotFilter : FilterExpr
        {
            public NotFilter(FilterExpr inner)
            {
                Inner = inner;
            }

            public FilterExpr Inner { get; }

            public override bool Check(ReadOnlySpan<double> c, IReadOnlyDictionary<string, string> m) => !Inner.Check(c, m);

            public override bool CheckMetadata(IReadOnlyDictionary<string, string> m) => !Inner.CheckMetadata(m);
        }
    }
}
