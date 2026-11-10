// <copyright file="FloatCodec.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Runtime.InteropServices;

    /// <summary>Raw 32- or 64-bit float storage.</summary>
    internal sealed class FloatCodec : VectorCodec
    {
        private readonly bool _double;
        private readonly bool _fastRouting;

        public FloatCodec(int dimension, DistanceMetric metric, bool doublePrecision, bool fastRouting)
            : base(dimension, metric)
        {
            _double = doublePrecision;
            _fastRouting = fastRouting && metric == DistanceMetric.Poincare;
        }

        public override int EncodedLength => Dimension * (_double ? 8 : 4);

        public override bool SupportsTruncation => true;

        public override byte[] Encode(ReadOnlySpan<float> vector)
        {
            var bytes = new byte[EncodedLength];
            if (_double)
            {
                var dst = MemoryMarshal.Cast<byte, double>(bytes.AsSpan());
                for (int i = 0; i < vector.Length; i++)
                {
                    dst[i] = vector[i];
                }
            }
            else
            {
                MemoryMarshal.Cast<float, byte>(vector).CopyTo(bytes);
            }

            return bytes;
        }

        public override void Decode(ReadOnlySpan<byte> stored, Span<float> destination)
        {
            if (_double)
            {
                var src = MemoryMarshal.Cast<byte, double>(stored);
                for (int i = 0; i < destination.Length; i++)
                {
                    destination[i] = (float)src[i];
                }
            }
            else
            {
                MemoryMarshal.Cast<byte, float>(stored).Slice(0, destination.Length).CopyTo(destination);
            }
        }

        public override double PairDistance(ReadOnlySpan<byte> a, ReadOnlySpan<byte> b)
        {
            if (_double)
            {
                return Metrics.Distance(Metric, MemoryMarshal.Cast<byte, double>(a), MemoryMarshal.Cast<byte, double>(b));
            }

            return Metrics.Distance(Metric, MemoryMarshal.Cast<byte, float>(a), MemoryMarshal.Cast<byte, float>(b));
        }

        public override PreparedQuery Prepare(ReadOnlySpan<float> query, PreparedQuery? reuse = null)
        {
            var q = base.Prepare(query, reuse);
            if (_fastRouting)
            {
                double sq = 0;
                foreach (var x in q.Coords)
                {
                    sq += x * x;
                }

                double f = 2.0 / (1.0 + sq);
                var k = q.Klein is { } existing && existing.Length == q.Coords.Length ? existing : new double[q.Coords.Length];
                for (int i = 0; i < k.Length; i++)
                {
                    k[i] = q.Coords[i] * f;
                }

                q.Klein = k;
            }

            return q;
        }

        public override double Distance(ReadOnlySpan<byte> stored, PreparedQuery query, int truncation = 0)
        {
            int n = truncation > 0 ? Math.Min(truncation, Dimension) : Dimension;
            if (_double)
            {
                var a = MemoryMarshal.Cast<byte, double>(stored).Slice(0, n);
                return Metrics.Distance(Metric, a, query.Coords.AsSpan(0, n));
            }

            var af = MemoryMarshal.Cast<byte, float>(stored).Slice(0, n);
            return Metrics.Distance(Metric, af, query.Coords32.AsSpan(0, n));
        }

        public override double UpperDistance(ReadOnlySpan<byte> stored, PreparedQuery query)
        {
            var qk = query.Klein;
            if (qk is null)
            {
                return Distance(stored, query);
            }

            // Klein chord distance: cheap Euclidean routing in the upper layers, k = 2p / (1 + |p|^2).
            double sq = 0;
            int n = Dimension;
            if (_double)
            {
                var p = MemoryMarshal.Cast<byte, double>(stored);
                for (int i = 0; i < n; i++)
                {
                    sq += p[i] * p[i];
                }

                double f = 2.0 / (1.0 + sq);
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    double d = (p[i] * f) - qk[i];
                    sum += d * d;
                }

                return sum;
            }
            else
            {
                var p = MemoryMarshal.Cast<byte, float>(stored);
                for (int i = 0; i < n; i++)
                {
                    sq += (double)p[i] * p[i];
                }

                double f = 2.0 / (1.0 + sq);
                double sum = 0;
                for (int i = 0; i < n; i++)
                {
                    double d = (p[i] * f) - qk[i];
                    sum += d * d;
                }

                return sum;
            }
        }
    }
}
