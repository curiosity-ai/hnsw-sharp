// <copyright file="BinaryHeap.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics.CodeAnalysis;
    using System.Runtime.InteropServices;

    /// <summary>
    /// Binary heap wrapper around the <see cref="IList{T}"/> It's a max-heap implementation i.e. the maximum element is always on top. But the order of elements can be customized by providing <see cref="IComparer{T}"/> instance.
    /// </summary>
    /// <typeparam name="T">The type of the items in the source list.</typeparam>
    [SuppressMessage("Performance", "CA1815:Override equals and operator equals on value types", Justification = "By design")]
    internal struct BinaryHeap
    {
        internal IComparer<int> Comparer;
        internal List<int> Buffer;
        internal bool Any => Buffer.Count > 0;
        internal BinaryHeap(List<int> buffer) : this(buffer, Comparer<int>.Default) { }
        internal BinaryHeap(List<int> buffer, IComparer<int> comparer)
        {
            Buffer = buffer ?? throw new ArgumentNullException(nameof(buffer));
            Comparer = comparer;
            for (int i = 1; i < Buffer.Count; ++i) { SiftUp(i); }
        }

        internal void Push(int item)
        {
            Buffer.Add(item);
            SiftUp(Buffer.Count - 1);
        }

        internal int Pop()
        {
            var bufferSpan = CollectionsMarshal.AsSpan(Buffer);

            if (bufferSpan.Length > 0)
            {
                var result = bufferSpan[0];

                //This is safe to modify as we don't change the inner collection size
                bufferSpan[0] = bufferSpan[Buffer.Count - 1];

                //Now we change the collection, so bufferSpan can be invalid
                Buffer.RemoveAt(Buffer.Count - 1);

                SiftDown(0);

                return result;
            }

            throw new InvalidOperationException("Heap is empty");
        }

        /// <summary>
        /// Restores the heap property starting from i'th position down to the bottom given that the downstream items fulfill the rule.
        /// </summary>
        /// <param name="i">The position of item where heap property is violated.</param>
        private void SiftDown(int i)
        {
            var bufferSpan = CollectionsMarshal.AsSpan(Buffer);
            while (i < bufferSpan.Length)
            {
                int l = (i << 1) + 1;
                int r = l + 1;
                if (l >= bufferSpan.Length)
                {
                    break;
                }

                int m = r < bufferSpan.Length && Comparer.Compare(bufferSpan[l], bufferSpan[r]) < 0 ? r : l;
                if (Comparer.Compare(bufferSpan[m], bufferSpan[i]) <= 0)
                {
                    break;
                }

                Swap(bufferSpan, i, m);
                i = m;
            }
        }

        /// <summary>
        /// Restores the heap property starting from i'th position up to the head given that the upstream items fulfill the rule.
        /// </summary>
        /// <param name="i">The position of item where heap property is violated.</param>
        private void SiftUp(int i)
        {
            var bufferSpan = CollectionsMarshal.AsSpan(Buffer);

            while (i > 0)
            {
                int p = (i - 1) >> 1;
                if (Comparer.Compare(bufferSpan[i], bufferSpan[p]) <= 0)
                {
                    break;
                }

                Swap(bufferSpan, i, p);
                i = p;
            }
        }

        private static void Swap(Span<int> buffer, int i, int j)
        {
            var temp = buffer[i];
            buffer[i] = buffer[j];
            buffer[j] = temp;
        }
    }
}
