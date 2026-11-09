// <copyright file="TurboRotation.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;

    /// <summary>
    /// Randomized Fast Walsh-Hadamard rotation: an orthogonal transform in O(D log D) time and O(1) extra memory.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A dimension D is split greedily into power-of-two blocks (768 = 512 + 256), each block is multiplied by deterministic
    /// pseudo-random signs and put through a normalized Walsh-Hadamard butterfly. The result is exactly norm and inner-product
    /// preserving, and exactly invertible with <see cref="Unrotate"/>.
    /// </para>
    /// <para>
    /// The sign stream is the same one HyperspaceDB uses, so rotated vectors are interchangeable between the two implementations.
    /// Binary, ternary, TurboQuant and OPQ codecs rotate vectors with it to spread the energy evenly across coordinates before
    /// quantizing.
    /// </para>
    /// </remarks>
    public static class TurboRotation
    {
        private sealed class Plan
        {
            public Plan((int Offset, int Length)[] blocks, float[] scaledSigns)
            {
                Blocks = blocks;
                ScaledSigns = scaledSigns;
            }

            public (int Offset, int Length)[] Blocks { get; }

            public float[] ScaledSigns { get; }
        }

        private static readonly ConcurrentDictionary<int, Plan> Plans = new ConcurrentDictionary<int, Plan>();

        /// <summary>Decomposes a dimension into disjoint power-of-two blocks, largest first.</summary>
        public static IReadOnlyList<(int Offset, int Length)> DecomposeBlocks(int dimension) => BlocksFor(dimension);

        private static (int Offset, int Length)[] BlocksFor(int dimension)
        {
            var blocks = new List<(int, int)>();
            int offset = 0;
            int remaining = dimension;
            while (remaining > 0)
            {
                int len = 1 << (31 - System.Numerics.BitOperations.LeadingZeroCount((uint)remaining));
                blocks.Add((offset, len));
                offset += len;
                remaining -= len;
            }

            return blocks.ToArray();
        }

        private static Plan GetPlan(int dimension) => Plans.GetOrAdd(dimension, static dim =>
        {
            var blocks = BlocksFor(dim);
            var signs = new float[dim];
            int p = 0;
            foreach (var (offset, len) in blocks)
            {
                float invSqrt = 1.0f / MathF.Sqrt(len);
                ulong state = unchecked(0x517cc1b727220a95UL + ((ulong)offset * 0x9e3779b97f4a7c15UL));
                for (int i = 0; i < len; i++)
                {
                    state = unchecked((state * 6364136223846793005UL) + 1442695040888963407UL);
                    signs[p++] = (state >> 63) == 0 ? invSqrt : -invSqrt;
                }
            }

            return new Plan(blocks, signs);
        });

        /// <summary>In-place unnormalized Walsh-Hadamard butterfly on a power-of-two length span.</summary>
        public static void FastWalshHadamard(Span<float> buffer)
        {
            int n = buffer.Length;
            if (n <= 1)
            {
                return;
            }

            if ((n & (n - 1)) != 0)
            {
                throw new ArgumentException("Length must be a power of two", nameof(buffer));
            }

            for (int len = 1; len < n; len <<= 1)
            {
                int step = len << 1;
                for (int i = 0; i < n; i += step)
                {
                    for (int j = i; j < i + len; j++)
                    {
                        float x = buffer[j];
                        float y = buffer[j + len];
                        buffer[j] = x + y;
                        buffer[j + len] = x - y;
                    }
                }
            }
        }

        /// <summary>Rotates a vector in place: <c>y = FWHT(x * signs / sqrt(blockLength))</c>.</summary>
        public static void Rotate(Span<float> vector)
        {
            int dim = vector.Length;
            if (dim <= 1)
            {
                return;
            }

            var plan = GetPlan(dim);
            var signs = plan.ScaledSigns;
            for (int i = 0; i < dim; i++)
            {
                vector[i] *= signs[i];
            }

            foreach (var (offset, len) in plan.Blocks)
            {
                FastWalshHadamard(vector.Slice(offset, len));
            }
        }

        /// <summary>Exact inverse of <see cref="Rotate"/>, in place.</summary>
        public static void Unrotate(Span<float> vector)
        {
            int dim = vector.Length;
            if (dim <= 1)
            {
                return;
            }

            var plan = GetPlan(dim);
            foreach (var (offset, len) in plan.Blocks)
            {
                FastWalshHadamard(vector.Slice(offset, len));
            }

            var signs = plan.ScaledSigns;
            for (int i = 0; i < dim; i++)
            {
                vector[i] *= signs[i];
            }
        }

        /// <summary>Returns a rotated copy of <paramref name="vector"/>.</summary>
        public static float[] Rotated(ReadOnlySpan<float> vector)
        {
            var copy = vector.ToArray();
            Rotate(copy);
            return copy;
        }

        /// <summary>Returns an un-rotated copy of <paramref name="vector"/>.</summary>
        public static float[] Unrotated(ReadOnlySpan<float> vector)
        {
            var copy = vector.ToArray();
            Unrotate(copy);
            return copy;
        }
    }
}
