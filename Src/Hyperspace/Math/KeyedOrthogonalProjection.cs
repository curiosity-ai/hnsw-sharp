// <copyright file="KeyedOrthogonalProjection.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Buffers.Binary;
    using System.Security.Cryptography;
    using System.Text;

    /// <summary>
    /// Keyed, exactly isometric projection of embeddings (the "shield" defence against embedding-inversion attacks).
    /// </summary>
    /// <remarks>
    /// <para>
    /// <c>Q(K) = D1(K) . H . P(K) . D0(K)</c> where D0 and D1 are Rademacher sign matrices, P a keyed coordinate permutation and
    /// H the block-diagonal normalized Walsh-Hadamard transform. Because Q is orthogonal, L2 distances, inner products, cosine
    /// similarity and Poincare-ball norms are preserved exactly, so a collection can store projected vectors and search with
    /// projected queries without losing accuracy. For the Lorentz and Hybrid metrics the time coordinate is left untouched, so
    /// points stay on the hyperboloid.
    /// </para>
    /// </remarks>
    public sealed class KeyedOrthogonalProjection
    {
        private readonly byte[] _seed;

        /// <summary>Initializes a new instance of the <see cref="KeyedOrthogonalProjection"/> class.</summary>
        /// <param name="seed">The 32-byte key material, see <see cref="DeriveSeed"/>.</param>
        public KeyedOrthogonalProjection(byte[] seed)
        {
            if (seed is null || seed.Length != 32)
            {
                throw new ArgumentException("The seed must be 32 bytes", nameof(seed));
            }

            _seed = (byte[])seed.Clone();
        }

        /// <summary>Creates a projection from a collection name and secret, hashing them to a seed.</summary>
        public static KeyedOrthogonalProjection FromSecret(string collection, string secretKey) => new KeyedOrthogonalProjection(DeriveSeed(collection, secretKey));

        /// <summary>Derives the 32-byte seed with SHA-256 over <c>HS_SHIELD_V4_2_DOMAIN:collection:secret</c>.</summary>
        public static byte[] DeriveSeed(string collection, string secretKey)
            => SHA256.HashData(Encoding.UTF8.GetBytes("HS_SHIELD_V4_2_DOMAIN:" + collection + ":" + secretKey));

        /// <summary>Applies the forward (or inverse) projection in place.</summary>
        public void Transform(Span<double> vector, DistanceMetric metric = DistanceMetric.L2, bool inverse = false)
        {
            int start = (metric == DistanceMetric.Lorentz || metric == DistanceMetric.Hybrid) ? Math.Min(1, vector.Length) : 0;
            var slice = vector.Slice(start);
            int n = slice.Length;
            if (n <= 1)
            {
                return;
            }

            BuildSchedule(n, out var signs0, out var signs1, out var perm);
            var copy = new double[n];

            if (!inverse)
            {
                for (int i = 0; i < n; i++)
                {
                    slice[i] *= signs0[i];
                }

                slice.CopyTo(copy);
                for (int i = 0; i < n; i++)
                {
                    slice[i] = copy[perm[i]];
                }

                FwhtBlocks(slice);
                for (int i = 0; i < n; i++)
                {
                    slice[i] *= signs1[i];
                }
            }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    slice[i] *= signs1[i];
                }

                FwhtBlocks(slice);
                slice.CopyTo(copy);
                for (int i = 0; i < n; i++)
                {
                    slice[perm[i]] = copy[i];
                }

                for (int i = 0; i < n; i++)
                {
                    slice[i] *= signs0[i];
                }
            }
        }

        /// <summary>Returns a projected copy of a vector.</summary>
        public double[] Project(ReadOnlySpan<double> vector, DistanceMetric metric = DistanceMetric.L2)
        {
            var copy = vector.ToArray();
            Transform(copy, metric, false);
            return copy;
        }

        /// <summary>Returns a projected copy of a vector.</summary>
        public float[] Project(ReadOnlySpan<float> vector, DistanceMetric metric = DistanceMetric.L2)
        {
            var d = new double[vector.Length];
            for (int i = 0; i < d.Length; i++)
            {
                d[i] = vector[i];
            }

            Transform(d, metric, false);
            var result = new float[d.Length];
            for (int i = 0; i < d.Length; i++)
            {
                result[i] = (float)d[i];
            }

            return result;
        }

        /// <summary>Returns the original vector of a projected copy.</summary>
        public double[] Unproject(ReadOnlySpan<double> vector, DistanceMetric metric = DistanceMetric.L2)
        {
            var copy = vector.ToArray();
            Transform(copy, metric, true);
            return copy;
        }

        private void BuildSchedule(int n, out double[] signs0, out double[] signs1, out int[] perm)
        {
            ulong s0 = BinaryPrimitives.ReadUInt64LittleEndian(_seed.AsSpan(0, 8));
            ulong s1 = BinaryPrimitives.ReadUInt64LittleEndian(_seed.AsSpan(8, 8));

            signs0 = new double[n];
            signs1 = new double[n];
            for (int i = 0; i < n; i++)
            {
                signs0[i] = (NextU64(ref s0) & 1) == 0 ? 1.0 : -1.0;
                signs1[i] = (NextU64(ref s1) & 1) == 0 ? 1.0 : -1.0;
            }

            perm = new int[n];
            for (int i = 0; i < n; i++)
            {
                perm[i] = i;
            }

            for (int i = n - 1; i >= 1; i--)
            {
                int j = (int)(NextU64(ref s0) % (ulong)(i + 1));
                (perm[i], perm[j]) = (perm[j], perm[i]);
            }
        }

        private static ulong NextU64(ref ulong state)
        {
            unchecked
            {
                state += 0x9e3779b97f4a7c15UL;
                ulong z = state;
                z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9UL;
                z = (z ^ (z >> 27)) * 0x94d049bb133111ebUL;
                return z ^ (z >> 31);
            }
        }

        private static void FwhtBlocks(Span<double> data)
        {
            int offset = 0;
            int remaining = data.Length;
            while (remaining > 0)
            {
                int blockLen = 1 << (31 - System.Numerics.BitOperations.LeadingZeroCount((uint)remaining));
                if (blockLen > 1)
                {
                    var block = data.Slice(offset, blockLen);
                    for (int len = 1; len < blockLen; len <<= 1)
                    {
                        int step = len << 1;
                        for (int i = 0; i < blockLen; i += step)
                        {
                            for (int j = i; j < i + len; j++)
                            {
                                double u = block[j], v = block[j + len];
                                block[j] = u + v;
                                block[j + len] = u - v;
                            }
                        }
                    }

                    double scale = 1.0 / Math.Sqrt(blockLen);
                    for (int i = 0; i < blockLen; i++)
                    {
                        block[i] *= scale;
                    }
                }

                offset += blockLen;
                remaining -= blockLen;
            }
        }
    }
}
