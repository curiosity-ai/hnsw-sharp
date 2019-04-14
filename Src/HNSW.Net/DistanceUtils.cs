// <copyright file="DistanceUtils.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;

    /// <summary>
    /// Auxiliary tools for working with distance variables.
    /// </summary>
    public static class DistanceUtils
    {
        /// <summary>
        /// Distance is Lower Than.
        /// </summary>
        /// <typeparam name="TDistance">The type of the distance.</typeparam>
        /// <param name="x">Left argument.</param>
        /// <param name="y">Right argument.</param>
        /// <returns>True if x &lt; y.</returns>
        public static bool Lt<TDistance>(TDistance x, TDistance y)
            where TDistance : IComparable<TDistance>
        {
            return x.CompareTo(y) < 0;
        }

        /// <summary>
        /// Distance is Greater Than.
        /// </summary>
        /// <typeparam name="TDistance">The type of the distance.</typeparam>
        /// <param name="x">Left argument.</param>
        /// <param name="y">Right argument.</param>
        /// <returns>True if x &gt; y.</returns>
        public static bool Gt<TDistance>(TDistance x, TDistance y)
            where TDistance : IComparable<TDistance>
        {
            return x.CompareTo(y) > 0;
        }

        /// <summary>
        /// Distances are Equal.
        /// </summary>
        /// <typeparam name="TDistance">The type of the distance.</typeparam>
        /// <param name="x">Left argument.</param>
        /// <param name="y">Right argument.</param>
        /// <returns>True if x == y.</returns>
        public static bool DEq<TDistance>(TDistance x, TDistance y)
            where TDistance : IComparable<TDistance>
        {
            return x.CompareTo(y) == 0;
        }
    }
}
