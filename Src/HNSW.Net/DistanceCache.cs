// <copyright file="DistanceCache.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

using System;

namespace HNSW.Net
{
    internal class DistanceCache<TDistance> where TDistance : struct
    {
        /// <summary>
        /// https://referencesource.microsoft.com/#mscorlib/system/array.cs,2d2b551eabe74985,references
        /// We use powers of 2 for efficient modulo
        /// 2^28 = 268435456
        /// 2^29 = 536870912
        /// 2^30 = 1073741824
        /// </summary>
        private const int MaxArrayLength = 1073741824; // 0x40000000;

        private TDistance[] values;

        private long[] keys;

        internal DistanceCache()
        {
        }

        internal void Resize(int pointsCount)
        {
            long capacity = ((long)pointsCount * (pointsCount + 1)) >> 1;
            capacity = capacity < MaxArrayLength ? capacity : MaxArrayLength;
            int i0 = 0;
            if (keys is null)
            {
                keys = new long[(int)capacity];
                values = new TDistance[(int)capacity];
            }
            else
            {
                i0 = keys.Length;
                Array.Resize(ref keys,   (int)capacity);
                Array.Resize(ref values, (int)capacity);
            }

            // TODO: may be there is a better way to warm up cache and force OS to allocate pages
            for (int i = i0; i < keys.Length; ++i)
            {
                keys[i] = -1;
                values[i] = default;
            }
        }

        internal bool TryGetValue(int fromId, int toId, out TDistance distance)
        {
            long key = MakeKey(fromId, toId);
            int hash = (int)(key & (MaxArrayLength - 1));

            if (keys[hash] == key)
            {
                distance = values[hash];
                return true;
            }

            distance = default;
            return false;
        }

        internal void SetValue(int fromId, int toId, TDistance distance)
        {
            long key = MakeKey(fromId, toId);
            int hash = (int)(key & (MaxArrayLength - 1));
            keys[hash] = key;
            values[hash] = distance;
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
            return fromId > toId ? (((long)fromId * (fromId + 1)) >> 1) + toId : (((long)toId * (toId + 1)) >> 1) + fromId;
        }
    }
}
