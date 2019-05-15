// <copyright file="DistanceCache.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    /// <summary>
    /// Cache for distance between 2 points.
    /// </summary>
    /// <typeparam name="TDistance">The type of the distance</typeparam>
    internal class DistanceCache<TDistance>
        where TDistance : struct
    {
        /// <summary>
        /// https://referencesource.microsoft.com/#mscorlib/system/array.cs,2d2b551eabe74985,references
        /// We use powers of 2 for efficient modulo
        /// 2^28 = 268435456
        /// 2^29 = 536870912
        /// 2^30 = 1073741824
        /// </summary>
        private const int MaxArrayLength = 1073741824; // 0x40000000;

        /// <summary>
        /// The cached values.
        /// </summary>
        private readonly TDistance[] values;

        /// <summary>
        /// The cached keys;
        /// </summary>
        private readonly long[] keys;

        /// <summary>
        /// Initializes a new instance of the <see cref="DistanceCache{TDistance}"/> class.
        /// </summary>
        /// <param name="pointsCount">
        /// The number of points to allocate cache for.
        /// </param>
        internal DistanceCache(int pointsCount)
        {
            long capacity = ((long)pointsCount * (pointsCount + 1)) >> 1;
            capacity = capacity < MaxArrayLength ? capacity : MaxArrayLength;

            this.keys = new long[(int)capacity];
            this.values = new TDistance[(int)capacity];

            // TODO: may be there is a better way to warm up cache and force OS to allocate pages
            for (int i = 0; i < this.keys.Length; ++i)
            {
                this.keys[i] = -1;
                this.values[i] = default;
            }
        }

        /// <summary>
        /// Tries to get value from the cache.
        /// </summary>
        /// <param name="fromId">The 'from' point identifier.</param>
        /// <param name="toId">The 'to' point identifier.</param>
        /// <param name="distance">The buffer for the result.</param>
        /// <returns>True if the distance value is retrieved from the cache.</returns>
        internal bool TryGetValue(int fromId, int toId, out TDistance distance)
        {
            long key = MakeKey(fromId, toId);
            int hash = (int)(key & (MaxArrayLength - 1));

            if (this.keys[hash] == key)
            {
                distance = this.values[hash];
                return true;
            }

            distance = default;
            return false;
        }

        /// <summary>
        /// Caches the distance value.
        /// </summary>
        /// <param name="fromId">The 'from' point identifier.</param>
        /// <param name="toId">The 'to' point identifier.</param>
        /// <param name="distance">The distance value to cache.</param>
        internal void SetValue(int fromId, int toId, TDistance distance)
        {
            long key = MakeKey(fromId, toId);
            int hash = (int)(key & (MaxArrayLength - 1));
            this.keys[hash] = key;
            this.values[hash] = distance;
        }

        /// <summary>
        /// Builds key for the pair of points.
        /// MakeKey(fromId, toId) == MakeKey(toId, fromId)
        /// </summary>
        /// <param name="fromId">The from point identifier.</param>
        /// <param name="toId">The to point identifier.</param>
        /// <returns>Key of the pair.</returns>
        private static long MakeKey(int fromId, int toId)
        {
            return fromId > toId
                ? (((long)fromId * (fromId + 1)) >> 1) + toId
                : (((long)toId * (toId + 1)) >> 1) + fromId;
        }
    }
}
