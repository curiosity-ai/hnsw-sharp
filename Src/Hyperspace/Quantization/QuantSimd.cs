// <copyright file="QuantSimd.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Numerics;
    using System.Runtime.InteropServices;

    /// <summary>
    /// Kernels shared by the quantized codecs. Byte-per-dimension codes are widened to floats with <see cref="Vector{T}"/> and
    /// multiply-added against the query; codes packed below a byte (nibbles, bit planes) are instead scored through a per-query lookup table
    /// built once per search, which turns a distance into a few table additions per byte.
    /// </summary>
    internal static class QuantSimd
    {
        /// <summary>The sum of <c>codes[i] * query[i]</c>.</summary>
        public static float DotInt8(ReadOnlySpan<sbyte> codes, ReadOnlySpan<float> query)
        {
            int n = Math.Min(codes.Length, query.Length);
            int i = 0;
            float sum = 0;
            if (Vector.IsHardwareAccelerated && n >= Vector<sbyte>.Count)
            {
                int f = Vector<float>.Count;
                var acc0 = Vector<float>.Zero;
                var acc1 = Vector<float>.Zero;
                var acc2 = Vector<float>.Zero;
                var acc3 = Vector<float>.Zero;
                ref sbyte rc = ref MemoryMarshal.GetReference(codes);
                ref float rq = ref MemoryMarshal.GetReference(query);
                int last = n - Vector<sbyte>.Count;
                for (; i <= last; i += Vector<sbyte>.Count)
                {
                    Widen(Vector.LoadUnsafe(ref rc, (nuint)i), out var c0, out var c1, out var c2, out var c3);
                    acc0 = Vector.FusedMultiplyAdd(c0, Vector.LoadUnsafe(ref rq, (nuint)i), acc0);
                    acc1 = Vector.FusedMultiplyAdd(c1, Vector.LoadUnsafe(ref rq, (nuint)(i + f)), acc1);
                    acc2 = Vector.FusedMultiplyAdd(c2, Vector.LoadUnsafe(ref rq, (nuint)(i + (2 * f))), acc2);
                    acc3 = Vector.FusedMultiplyAdd(c3, Vector.LoadUnsafe(ref rq, (nuint)(i + (3 * f))), acc3);
                }

                sum = Vector.Sum((acc0 + acc1) + (acc2 + acc3));
            }

            for (; i < n; i++)
            {
                sum += codes[i] * query[i];
            }

            return sum;
        }

        /// <summary>The sum of <c>(codes[i] * step - query[i])^2</c>.</summary>
        public static float L2Int8(ReadOnlySpan<sbyte> codes, ReadOnlySpan<float> query, float step)
        {
            int n = Math.Min(codes.Length, query.Length);
            int i = 0;
            float sum = 0;
            if (Vector.IsHardwareAccelerated && n >= Vector<sbyte>.Count)
            {
                int f = Vector<float>.Count;
                var stepV = new Vector<float>(step);
                var acc0 = Vector<float>.Zero;
                var acc1 = Vector<float>.Zero;
                var acc2 = Vector<float>.Zero;
                var acc3 = Vector<float>.Zero;
                ref sbyte rc = ref MemoryMarshal.GetReference(codes);
                ref float rq = ref MemoryMarshal.GetReference(query);
                int last = n - Vector<sbyte>.Count;
                for (; i <= last; i += Vector<sbyte>.Count)
                {
                    Widen(Vector.LoadUnsafe(ref rc, (nuint)i), out var c0, out var c1, out var c2, out var c3);
                    var d0 = Vector.FusedMultiplyAdd(c0, stepV, -Vector.LoadUnsafe(ref rq, (nuint)i));
                    var d1 = Vector.FusedMultiplyAdd(c1, stepV, -Vector.LoadUnsafe(ref rq, (nuint)(i + f)));
                    var d2 = Vector.FusedMultiplyAdd(c2, stepV, -Vector.LoadUnsafe(ref rq, (nuint)(i + (2 * f))));
                    var d3 = Vector.FusedMultiplyAdd(c3, stepV, -Vector.LoadUnsafe(ref rq, (nuint)(i + (3 * f))));
                    acc0 = Vector.FusedMultiplyAdd(d0, d0, acc0);
                    acc1 = Vector.FusedMultiplyAdd(d1, d1, acc1);
                    acc2 = Vector.FusedMultiplyAdd(d2, d2, acc2);
                    acc3 = Vector.FusedMultiplyAdd(d3, d3, acc3);
                }

                sum = Vector.Sum((acc0 + acc1) + (acc2 + acc3));
            }

            for (; i < n; i++)
            {
                float d = (codes[i] * step) - query[i];
                sum += d * d;
            }

            return sum;
        }

        /// <summary>Writes <c>codes[i] * step</c> into <paramref name="destination"/>.</summary>
        public static void DecodeInt8(ReadOnlySpan<sbyte> codes, float step, Span<float> destination)
        {
            int n = Math.Min(codes.Length, destination.Length);
            int i = 0;
            if (Vector.IsHardwareAccelerated && n >= Vector<sbyte>.Count)
            {
                int f = Vector<float>.Count;
                var stepV = new Vector<float>(step);
                ref sbyte rc = ref MemoryMarshal.GetReference(codes);
                ref float rd = ref MemoryMarshal.GetReference(destination);
                int last = n - Vector<sbyte>.Count;
                for (; i <= last; i += Vector<sbyte>.Count)
                {
                    Widen(Vector.LoadUnsafe(ref rc, (nuint)i), out var c0, out var c1, out var c2, out var c3);
                    (c0 * stepV).StoreUnsafe(ref rd, (nuint)i);
                    (c1 * stepV).StoreUnsafe(ref rd, (nuint)(i + f));
                    (c2 * stepV).StoreUnsafe(ref rd, (nuint)(i + (2 * f)));
                    (c3 * stepV).StoreUnsafe(ref rd, (nuint)(i + (3 * f)));
                }
            }

            for (; i < n; i++)
            {
                destination[i] = codes[i] * step;
            }
        }

        private static void Widen(Vector<sbyte> v, out Vector<float> f0, out Vector<float> f1, out Vector<float> f2, out Vector<float> f3)
        {
            Vector.Widen(v, out Vector<short> s0, out Vector<short> s1);
            Vector.Widen(s0, out Vector<int> i0, out Vector<int> i1);
            Vector.Widen(s1, out Vector<int> i2, out Vector<int> i3);
            f0 = Vector.ConvertToSingle(i0);
            f1 = Vector.ConvertToSingle(i1);
            f2 = Vector.ConvertToSingle(i2);
            f3 = Vector.ConvertToSingle(i3);
        }

        // ------------------------------------------------------------------ lookup tables

        /// <summary>
        /// A table for codes that hold two 4-bit levels per byte (high nibble = even dimension): <c>table[p * 256 + b]</c> is
        /// <c>levels[b >> 4] * query[2p] + levels[b &amp; 15] * query[2p + 1]</c>, so the dot product with a code is one lookup per byte.
        /// Dimensions beyond the end of <paramref name="query"/> count as 0.
        /// </summary>
        public static float[] BuildNibblePairTable(ReadOnlySpan<float> query, ReadOnlySpan<float> levels, int bytes)
        {
            var table = new float[bytes * 256];
            Span<float> hi = stackalloc float[16];
            Span<float> lo = stackalloc float[16];
            for (int p = 0; p < bytes; p++)
            {
                float q0 = 2 * p < query.Length ? query[2 * p] : 0f;
                float q1 = (2 * p) + 1 < query.Length ? query[(2 * p) + 1] : 0f;
                for (int l = 0; l < 16; l++)
                {
                    hi[l] = levels[l] * q0;
                    lo[l] = levels[l] * q1;
                }

                int baseIndex = p * 256;
                for (int h = 0; h < 16; h++)
                {
                    for (int l = 0; l < 16; l++)
                    {
                        table[baseIndex + (h << 4) + l] = hi[h] + lo[l];
                    }
                }
            }

            return table;
        }

        /// <summary>The sum of <c>table[(first + p) * 256 + codes[p]]</c>, i.e. the dot product of the codes with the query the table was built from.</summary>
        public static float SumTable(ReadOnlySpan<byte> codes, float[] table, int first = 0)
        {
            float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
            int p = 0;
            int baseIndex = first * 256;
            for (; p + 4 <= codes.Length; p += 4)
            {
                a0 += table[baseIndex + ((p) << 8) + codes[p]];
                a1 += table[baseIndex + ((p + 1) << 8) + codes[p + 1]];
                a2 += table[baseIndex + ((p + 2) << 8) + codes[p + 2]];
                a3 += table[baseIndex + ((p + 3) << 8) + codes[p + 3]];
            }

            float sum = (a0 + a1) + (a2 + a3);
            for (; p < codes.Length; p++)
            {
                sum += table[baseIndex + (p << 8) + codes[p]];
            }

            return sum;
        }

        /// <summary>
        /// A table for sign bits, eight dimensions per byte: <c>table[c * 256 + b]</c> is the sum of <c>+query[8c + j]</c> for the set bits j of b
        /// and <c>-query[8c + j]</c> for the clear ones. Dimensions beyond the end of <paramref name="query"/> count as 0.
        /// </summary>
        public static float[] BuildSignTable(ReadOnlySpan<float> query, int bytes)
        {
            var table = new float[bytes * 256];
            Span<float> r = stackalloc float[8];
            for (int c = 0; c < bytes; c++)
            {
                float total = 0;
                for (int j = 0; j < 8; j++)
                {
                    r[j] = (8 * c) + j < query.Length ? query[(8 * c) + j] : 0f;
                    total += r[j];
                }

                int baseIndex = c * 256;
                table[baseIndex] = -total;
                for (int b = 1; b < 256; b++)
                {
                    int low = BitOperations.TrailingZeroCount(b);
                    table[baseIndex + b] = table[baseIndex + (b & (b - 1))] + (2f * r[low]);
                }
            }

            return table;
        }

        /// <summary>
        /// A table for ternary codes held as a non-zero plane and a sign plane (sign bits only where the non-zero bit is set), four dimensions
        /// per group: <c>table[g * 256 + (nz4 &lt;&lt; 4 | sign4)]</c> is the signed sum of the query over the non-zero dimensions of the group.
        /// Only the 81 reachable <c>(nz, sign)</c> combinations of each group are filled in.
        /// </summary>
        public static float[] BuildTernaryTable(ReadOnlySpan<float> query, int groups)
        {
            var table = new float[groups * 256];
            Span<float> r = stackalloc float[4];
            for (int g = 0; g < groups; g++)
            {
                for (int j = 0; j < 4; j++)
                {
                    r[j] = (4 * g) + j < query.Length ? query[(4 * g) + j] : 0f;
                }

                int baseIndex = g * 256;
                for (int nz = 0; nz < 16; nz++)
                {
                    // Every subset of the non-zero bits can carry the sign.
                    int sign = nz;
                    while (true)
                    {
                        float sum = 0;
                        for (int j = 0; j < 4; j++)
                        {
                            if ((nz & (1 << j)) != 0)
                            {
                                sum += (sign & (1 << j)) != 0 ? r[j] : -r[j];
                            }
                        }

                        table[baseIndex + (nz << 4) + sign] = sum;
                        if (sign == 0)
                        {
                            break;
                        }

                        sign = (sign - 1) & nz;
                    }
                }
            }

            return table;
        }
    }
}
