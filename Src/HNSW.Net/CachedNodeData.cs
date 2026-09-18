// <copyright file="CachedNodeData.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Threading;

    public class CachedNodeData
    {
        private const int DEFAULT_BUCKET_SIZE = 2 * 1024 * 1024; //8 MB

        /// <summary>
        /// Upper bound for a bucket sized from a capacity hint (256 MB). Beyond this the data is chained
        /// across buckets rather than asking the LOH for a single allocation of unbounded size.
        /// </summary>
        private const int MAX_BUCKET_SIZE = 64 * 1024 * 1024;

        private int[][] _flattenedLayerArrays;
        private readonly int _bucketSize;
        private ushort _currentBucket;
        private int _currentIndexOnBucket;
        private long _extraSize = 0;
        private bool _addNextBucket;

        internal CachedNodeData() : this(0)
        {
        }

        /// <summary>
        /// Creates a cache whose first bucket already fits <paramref name="capacityHint"/> ints, so a load
        /// with a known total does one allocation instead of growing a bucket at a time.
        /// </summary>
        internal CachedNodeData(long capacityHint)
        {
            _bucketSize = (int)Math.Clamp(capacityHint, DEFAULT_BUCKET_SIZE, MAX_BUCKET_SIZE);
            _flattenedLayerArrays = new int[1][];
            _flattenedLayerArrays[0] = GC.AllocateUninitializedArray<int>(_bucketSize, pinned: false); //Initialize first array
        }

        public ReadOnlySpan<int> GetLayer(int bucketIndex, int position, int layerIndex, int maxLayer)
        {
            var bucket = _flattenedLayerArrays[bucketIndex];
            var nodeStart = bucket.AsSpan(position);
            var layerStart = nodeStart[layerIndex] + maxLayer + 1;
            var layerEnd = nodeStart[layerIndex + 1] + maxLayer + 1;
            return nodeStart.Slice(layerStart, layerEnd - layerStart);
        }

        public (int bucketIndex, int position, int maxLayers) Add(List<List<int>> list)
        {
            if (list.Count == 0) return (-1, -1, 0);

            var maxLayer = list.Count;

            var totalSize = 0;
            for (int i = 0; i < maxLayer; i++)
            {
                totalSize += list[i].Count;
            }

            var destination = Reserve(totalSize + maxLayer + 1, out var bucketIndex, out var position);

            int c = 0;
            int j = maxLayer + 1;

            for (int i = 0; i < maxLayer; i++)
            {
                var l = list[i];
                destination[i] = c;

                for (int k = 0; k < l.Count; k++)
                {
                    destination[j++] = l[k];
                }

                c += l.Count;
            }

            destination[maxLayer] = c;

            return (bucketIndex, position, maxLayer);
        }

        public (int bucketIndex, int position, int maxLayers) Add(ReadOnlySpan<int> final, int maxLayer)
        {
            var destination = Reserve(final.Length, out var bucketIndex, out var position);
            final.CopyTo(destination);
            return (bucketIndex, position, maxLayer);
        }

        /// <summary>
        /// Claims <paramref name="length"/> ints of contiguous storage and hands back the span to fill.
        /// </summary>
        /// <remarks>
        /// The span aliases the bucket and is invalidated by the next call, so a caller fills it before
        /// reserving again. This is what lets deserialization read a node's connections straight off the
        /// stream into their final home, with no per-node array in between.
        /// </remarks>
        internal Span<int> Reserve(int length, out int bucketIndex, out int position)
        {
            var (index, bucket) = GetBucketWithCapacityFor(length);
            bucketIndex = index;
            position    = _currentIndexOnBucket;
            _currentIndexOnBucket += length;
            return bucket.AsSpan(position, length);
        }

        private (ushort bucketIndex, int[] bucket) GetBucketWithCapacityFor(int edgeCount)
        {
            //We always resize the last bucket to fit the new edge count, and then create a new bucket for the next call.

            if ((long)_currentIndexOnBucket + edgeCount > _bucketSize)
            {
                if (_addNextBucket || ((long)_currentIndexOnBucket + edgeCount > int.MaxValue))
                {
                    int targetSize = _bucketSize;

                    if (edgeCount > _bucketSize)
                    {
                        targetSize = edgeCount;
                        Interlocked.Add(ref _extraSize, targetSize - _bucketSize);
                        //The entire bucket will be oversized and will contain only edges for this node
                    }

                    _currentBucket++;
                    _currentIndexOnBucket = 0;

                    Array.Resize(ref _flattenedLayerArrays, _currentBucket + 1);

                    _flattenedLayerArrays[_currentBucket] = GC.AllocateUninitializedArray<int>(targetSize, pinned: false);

                    _addNextBucket = false;
                }
                else
                {
                    _addNextBucket = true;

                    var targetResizedSize = _currentIndexOnBucket + edgeCount + 1; //TODO: We probably don't need this +1 here, need to test to make sure
                    Interlocked.Add(ref _extraSize, targetResizedSize - _bucketSize);

                    Array.Resize(ref _flattenedLayerArrays[_currentBucket], targetResizedSize);
                }
            }

            return (_currentBucket, _flattenedLayerArrays[_currentBucket]);
        }

        internal ReadOnlySpan<int> GetAll(int bucketIndex, int position, int maxLayers)
        {
            var bucket = _flattenedLayerArrays[bucketIndex];
            var nodeStart = bucket.AsSpan(position);
            var layerEnd   = nodeStart[maxLayers] + maxLayers + 1;
            return nodeStart.Slice(0, layerEnd);
        }
    }
}
