// <copyright file="CosineDistance.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Numerics;
    using System.Runtime.CompilerServices;

    /// <summary>
    /// Calculates cosine similarity.
    /// </summary>
    /// <remarks>
    /// Intuition behind selecting float as a carrier.
    ///
    /// 1. In practice we work with vectors of dimensionality 100 and each component has value in range [-1; 1]
    ///    There certainly is a possibility of underflow.
    ///    But we assume that such cases are rare and we can rely on such underflow losses.
    ///
    /// 2. According to the article http://www.ti3.tuhh.de/paper/rump/JeaRu13.pdf
    ///    the floating point rounding error is less then 100 * 2^-24 * sqrt(100) * sqrt(100) &lt; 0.0005960
    ///    We deem such precision is satisfactory for out needs.
    /// </remarks>
    public static class CosineDistance
    {
        /// <summary>
        /// Calculates cosine distance without making any optimizations.
        /// </summary>
        /// <param name="u">Left vector.</param>
        /// <param name="v">Right vector.</param>
        /// <returns>Cosine distance between u and v.</returns>
        public static float NonOptimized(float[] u, float[] v)
        {
            if (u.Length != v.Length)
            {
                throw new ArgumentException("Vectors have non-matching dimensions");
            }

            float dot = 0.0f;
            float nru = 0.0f;
            float nrv = 0.0f;
            for (int i = 0; i < u.Length; ++i)
            {
                dot += u[i] * v[i];
                nru += u[i] * u[i];
                nrv += v[i] * v[i];
            }

            var similarity = dot / (float)(Math.Sqrt(nru) * Math.Sqrt(nrv));
            return 1 - similarity;
        }

        /// <summary>
        /// Calculates cosine distance with assumption that u and v are unit vectors.
        /// </summary>
        /// <param name="u">Left vector.</param>
        /// <param name="v">Right vector.</param>
        /// <returns>Cosine distance between u and v.</returns>
        public static float ForUnits(float[] u, float[] v)
        {
            if (u.Length!= v.Length)
            {
                throw new ArgumentException("Vectors have non-matching dimensions");
            }

            float dot = 0;
            for (int i = 0; i < u.Length; ++i)
            {
                dot += u[i] * v[i];
            }

            return 1 - dot;
        }

        /// <summary>
        /// Calculates cosine distance optimized using SIMD instructions.
        /// </summary>
        /// <param name="u">Left vector.</param>
        /// <param name="v">Right vector.</param>
        /// <returns>Cosine distance between u and v.</returns>
        public static float SIMD(float[] u, float[] v)
        {
            if (!Vector.IsHardwareAccelerated)
            {
                throw new NotSupportedException($"SIMD version of {nameof(CosineDistance)} is not supported");
            }

            if (u.Length != v.Length)
            {
                throw new ArgumentException("Vectors have non-matching dimensions");
            }

            int step = Vector<float>.Count;

            // accumulate element-wise in vector registers; reduce horizontally only once at the end
            var dotAcc = Vector<float>.Zero;
            var normUAcc = Vector<float>.Zero;
            var normVAcc = Vector<float>.Zero;

            int i, to = u.Length - step;
            for (i = 0; i <= to; i += step)
            {
                var ui = new Vector<float>(u, i);
                var vi = new Vector<float>(v, i);
                dotAcc += ui * vi;
                normUAcc += ui * ui;
                normVAcc += vi * vi;
            }

            float dot = Vector.Sum(dotAcc);
            var norm = new Vector2(Vector.Sum(normUAcc), Vector.Sum(normVAcc));

            for (; i < u.Length; ++i)
            {
                dot += u[i] * v[i];
                norm.X += u[i] * u[i];
                norm.Y += v[i] * v[i];
            }

            norm = Vector2.SquareRoot(norm);
            float n = (norm.X * norm.Y);

            if (n == 0)
            {
                return 1f;
            }

            var similarity = dot / n;
            return 1f - similarity;
        }

        /// <summary>
        /// Calculates cosine distance with assumption that u and v are unit vectors using SIMD instructions.
        /// </summary>
        /// <param name="u">Left vector.</param>
        /// <param name="v">Right vector.</param>
        /// <returns>Cosine distance between u and v.</returns>
        public static float SIMDForUnits(float[] u, float[] v)
        {
            return 1f - DotProduct(ref u, ref v);
        }

        private static readonly int _vs1 = Vector<float>.Count;
        private static readonly int _vs2 = 2 * Vector<float>.Count;
        private static readonly int _vs3 = 3 * Vector<float>.Count;
        private static readonly int _vs4 = 4 * Vector<float>.Count;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float DotProduct(ref float[] lhs, ref float[] rhs)
        {
            // Accumulate element-wise products in independent vector registers (to hide FMA latency)
            // and reduce horizontally only once at the end, instead of paying for a horizontal
            // Vector.Dot reduction on every chunk.
            var acc0 = Vector<float>.Zero;
            var acc1 = Vector<float>.Zero;
            var acc2 = Vector<float>.Zero;
            var acc3 = Vector<float>.Zero;

            var count = lhs.Length;
            var offset = 0;

            while (count >= _vs4)
            {
                acc0 += new Vector<float>(lhs, offset) * new Vector<float>(rhs, offset);
                acc1 += new Vector<float>(lhs, offset + _vs1) * new Vector<float>(rhs, offset + _vs1);
                acc2 += new Vector<float>(lhs, offset + _vs2) * new Vector<float>(rhs, offset + _vs2);
                acc3 += new Vector<float>(lhs, offset + _vs3) * new Vector<float>(rhs, offset + _vs3);
                count -= _vs4;
                offset += _vs4;
            }

            if (count >= _vs2)
            {
                acc0 += new Vector<float>(lhs, offset) * new Vector<float>(rhs, offset);
                acc1 += new Vector<float>(lhs, offset + _vs1) * new Vector<float>(rhs, offset + _vs1);
                count -= _vs2;
                offset += _vs2;
            }

            if (count >= _vs1)
            {
                acc2 += new Vector<float>(lhs, offset) * new Vector<float>(rhs, offset);
                count -= _vs1;
                offset += _vs1;
            }

            float result = Vector.Sum((acc0 + acc1) + (acc2 + acc3));

            while (count > 0)
            {
                result += lhs[offset] * rhs[offset];
                offset++; count--;
            }

            return result;
        }
    }
}
