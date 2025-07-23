// <copyright file="CachedNodeData.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Linq;
    using System.Threading;

    public class CachedNodeData
    {
        private int[][] _flattenedLayerArrays;
        private readonly int _bucketSize;
        private ushort _currentBucket;
        private int _currentIndexOnBucket;
        private long _extraSize = 0;
        private bool _addNextBucket;
        internal CachedNodeData()
        {
            _bucketSize = 2 * 1024 * 1024; //8 MB
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

            var totalSize = list.Sum(static v => v.Count);

            int c = 0;
            int j = 0;
            var maxLayer = list.Count;
            var final = new int[totalSize + maxLayer + 1];
            for (int i = 0; i < maxLayer; i++)
            {
                var l = list[i];
                var d = l.Count;
                final[i] = c;
                foreach (var v in l)
                {
                    final[j + maxLayer + 1] = v;
                    j++;
                }
                c += d;
            }

            final[maxLayer] = c;

            int bucketLength = -1;

            var (bucketIndex, edgesBucket) = GetBucketWithCapacityFor(final.Length);
            bucketLength = edgesBucket.Length;
            final.AsSpan().CopyTo(edgesBucket.AsSpan(_currentIndexOnBucket, final.Length));
            var currentPos = _currentIndexOnBucket;
            _currentIndexOnBucket += final.Length;
            return (bucketIndex, currentPos, maxLayer);
        }

        public (int bucketIndex, int position, int maxLayers) Add(ReadOnlySpan<int> final, int maxLayer)
        {
            int bucketLength = -1;
            var (bucketIndex, edgesBucket) = GetBucketWithCapacityFor(final.Length);
            bucketLength = edgesBucket.Length;
            final.CopyTo(edgesBucket.AsSpan(_currentIndexOnBucket, final.Length));
            var currentPos = _currentIndexOnBucket;
            _currentIndexOnBucket += final.Length;
            return (bucketIndex, currentPos, maxLayer);
        }

        private (ushort bucketIndex, int[] bucket) GetBucketWithCapacityFor(int edgeCount)
        {
            //We always resize the last bucket to fit the new edge count, and then create a new bucket for the next call.

            if (_currentIndexOnBucket + edgeCount > _bucketSize)
            {
                if (_addNextBucket || (_currentIndexOnBucket + edgeCount > int.MaxValue))
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
