// <copyright file="TravelingCosts.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Concurrent;
    using System.Collections.Generic;

    /// <summary>
    /// Implementation of distance calculation from an arbitrary point to the given destination.
    /// </summary>
    /// <typeparam name="TItem">Type of the points.</typeparam>
    /// <typeparam name="TDistance">Type of the diatnce.</typeparam>
    public class TravelingCosts<TItem, TDistance> : IComparer<TItem>
    {
        /// <summary>
        /// Default distance comaprer.
        /// </summary>
        private static readonly Comparer<TDistance> DistanceComparer = Comparer<TDistance>.Default;

        /// <summary>
        /// The distance funciton.
        /// </summary>
        private readonly Func<TItem, TItem, TDistance> distance;

        /// <summary>
        /// The destination point.
        /// </summary>
        private readonly TItem destination;

        /// <summary>
        /// Cached values.
        /// </summary>
        private readonly ConcurrentDictionary<TItem, TDistance> cache;

        /// <summary>
        /// Initializes a new instance of the <see cref="TravelingCosts{TItem, TDistance}"/> class.
        /// </summary>
        /// <param name="distance">The distance function.</param>
        /// <param name="destination">The destination point.</param>
        public TravelingCosts(Func<TItem, TItem, TDistance> distance, TItem destination)
        {
            this.distance = distance;
            this.destination = destination;

            this.cache = new ConcurrentDictionary<TItem, TDistance>();
        }

        /// <summary>
        /// Calculates distance from the departure to the destination.
        /// </summary>
        /// <param name="departure">The point of departure.</param>
        /// <returns>The distance from the departure to the destination.</returns>
        public TDistance From(TItem departure)
        {
            TDistance result;
            if (!this.cache.TryGetValue(departure, out result))
            {
                result = this.distance(departure, this.destination);
                this.cache.TryAdd(departure, result);
            }

            return result;
        }

        /// <summary>
        /// Compares 2 points by the distance from the destination.
        /// </summary>
        /// <param name="x">Left point.</param>
        /// <param name="y">Right point.</param>
        /// <returns>
        /// -1 if x is closer to the destination than y;
        /// 0 if x and y are equally far from the destination;
        /// 1 if x is farther from the destination than y.
        /// </returns>
        public int Compare(TItem x, TItem y)
        {
            var fromX = this.From(x);
            var fromY = this.From(y);
            return DistanceComparer.Compare(fromX, fromY);
        }
    }
}
