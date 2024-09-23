// <copyright file="DistanceCache.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

using System;
using System.Runtime.CompilerServices;

namespace HNSW.Net
{
    public static class DistanceCacheLimits
    {
        /// <summary>
        /// https://referencesource.microsoft.com/#mscorlib/system/array.cs,2d2b551eabe74985,references
        /// We use powers of 2 for efficient modulo
        /// 2^28 = 268435456
        /// 2^29 = 536870912
        /// 2^30 = 1073741824
        /// </summary>
        public static int MaxArrayLength 
        { 
            get { return _maxArrayLength; }
            set { _maxArrayLength = NextPowerOf2((uint)value); }
        }

        private static int NextPowerOf2(uint x)
        {
#if NET7_0_OR_GREATER
            var v = System.Numerics.BitOperations.RoundUpToPowerOf2(x);
            if (v > 0x10000000) return 0x10000000;
            return (int)v;
#else
            return (int)Math.Pow(2, Math.Ceiling(Math.Log(x) / Math.Log(2)));
#endif
        }

        private static int _maxArrayLength = 268_435_456; // 0x10000000;
    }

    internal class DistanceCache<TDistance> where TDistance : struct
    {
        private TDistance[] _values;

        private long[] _keys;

        internal int HitCount;

        internal DistanceCache()
        {
        }

        internal void Resize(int pointsCount, bool overwrite)
        {
            if (pointsCount <= 0) { pointsCount = 1024; }

            long capacity = ((long)pointsCount * (pointsCount + 1)) >> 1;
            
            capacity = capacity < DistanceCacheLimits.MaxArrayLength ? capacity : DistanceCacheLimits.MaxArrayLength;

            if (_keys is null || capacity > _keys.Length || overwrite)
            {
                int i0 = 0;
                if (_keys is null || overwrite)
                {
                    _keys   = new long[(int)capacity];
                    _values = new TDistance[(int)capacity];
                }
                else
                {
                    i0 = _keys.Length;
                    Array.Resize(ref _keys,   (int)capacity);
                    Array.Resize(ref _values, (int)capacity);
                }

                // TODO: may be there is a better way to warm up cache and force OS to allocate pages
                _keys.AsSpan().Slice(i0).Fill(-1);
                _values.AsSpan().Slice(i0).Fill(default);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal TDistance GetOrCacheValue(int fromId, int toId, Func<int,int,TDistance> getter)
        {
            long key = MakeKey(fromId, toId);
            int hash = (int)(key & (_keys.Length - 1));

            if (_keys[hash] == key)
            {
                HitCount++;
                return _values[hash];
            }
            else
            {
                var d = getter(fromId, toId);
                _keys[hash] = key;
                _values[hash] = d;
                return d;
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void SetValue(int fromId, int toId, TDistance distance)
        {
            long key = MakeKey(fromId, toId);
            int hash = (int)(key & (_keys.Length - 1));
            _keys[hash] = key;
            _values[hash] = distance;
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
