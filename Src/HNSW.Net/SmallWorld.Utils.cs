// <copyright file="SmallWorld.Utils.cs" company="Microsoft">
// Copyright (c) Microsoft. All rights reserved.
// </copyright>

namespace HNSW.Net
{
    using System.Diagnostics.CodeAnalysis;

    /// <content>
    /// The part with the auxiliary tools for hnsw algorithm.
    /// </content>
    public partial class SmallWorld<TItem, TDistance>
    {
        /// <summary>
        /// Distance is Lower Than.
        /// </summary>
        /// <param name="x">Left argument.</param>
        /// <param name="y">Right argument.</param>
        /// <returns>True if x &lt; y.</returns>
        [SuppressMessage("Design", "CA1000:Do not declare static members on generic types", Justification = "By Design")]
        public static bool DLt(TDistance x, TDistance y)
        {
            return x.CompareTo(y) < 0;
        }

        /// <summary>
        /// Distance is Greater Than.
        /// </summary>
        /// <param name="x">Left argument.</param>
        /// <param name="y">Right argument.</param>
        /// <returns>True if x &gt; y.</returns>
        [SuppressMessage("Design", "CA1000:Do not declare static members on generic types", Justification = "By Design")]
        public static bool DGt(TDistance x, TDistance y)
        {
            return x.CompareTo(y) > 0;
        }

        /// <summary>
        /// Distances are Equal.
        /// </summary>
        /// <param name="x">Left argument.</param>
        /// <param name="y">Right argument.</param>
        /// <returns>True if x == y.</returns>
        [SuppressMessage("Design", "CA1000:Do not declare static members on generic types", Justification = "By Design")]
        public static bool DEq(TDistance x, TDistance y)
        {
            return x.CompareTo(y) == 0;
        }
    }
}