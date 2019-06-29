// <copyright file="BinaryHeap.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Diagnostics.CodeAnalysis;

    /// <summary>
    /// Binary heap wrapper around the <see cref="IList{T}"/>
    /// It's a max-heap implementation i.e. the maximum element is always on top.
    /// But the order of elements can be customized by providing <see cref="IComparer{T}"/> instance.
    /// </summary>
    /// <typeparam name="T">The type of the items in the source list.</typeparam>
    [SuppressMessage("Performance", "CA1815:Override equals and operator equals on value types", Justification = "By design")]
    internal struct BinaryHeap<T>
    {
        private IComparer<T> comparer;
        private IList<T> buffer;

        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryHeap{T}"/> struct.
        /// </summary>
        /// <param name="buffer">The buffer to store heap items.</param>
        internal BinaryHeap(IList<T> buffer) : this(buffer, Comparer<T>.Default)
        {
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="BinaryHeap{T}"/> struct.
        /// </summary>
        /// <param name="buffer">The buffer to store heap items.</param>
        /// <param name="comparer">The comparer which defines order of items.</param>
        internal BinaryHeap(IList<T> buffer, IComparer<T> comparer)
        {
            if (buffer == null)
            {
                throw new ArgumentNullException(nameof(buffer));
            }

            this.buffer = buffer;
            this.comparer = comparer;
            for (int i = 1; i < this.buffer.Count; ++i)
            {
                SiftUp(i);
            }
        }

        internal IComparer<T> Comparer => comparer;

        internal IList<T> Buffer => buffer;

        internal void Push(T item)
        {
            buffer.Add(item);
            SiftUp(buffer.Count - 1);
        }

        internal T Pop()
        {
            if (buffer.Count > 0)
            {
                var result = buffer[0];

                buffer[0] = buffer[buffer.Count - 1];
                buffer.RemoveAt(buffer.Count - 1);
                SiftDown(0);

                return result;
            }

            throw new InvalidOperationException("Heap is empty");
        }

        /// <summary>
        /// Restores the heap property starting from i'th position down to the bottom
        /// given that the downstream items fulfill the rule.
        /// </summary>
        /// <param name="i">The position of item where heap property is violated.</param>
        private void SiftDown(int i)
        {
            while (i < buffer.Count)
            {
                int l = (i << 1) + 1;
                int r = l + 1;
                if (l >= buffer.Count)
                {
                    break;
                }

                int m = r < buffer.Count && comparer.Compare(buffer[l], buffer[r]) < 0 ? r : l;
                if (comparer.Compare(buffer[m], buffer[i]) <= 0)
                {
                    break;
                }

                Swap(i, m);
                i = m;
            }
        }

        /// <summary>
        /// Restores the heap property starting from i'th position up to the head
        /// given that the upstream items fulfill the rule.
        /// </summary>
        /// <param name="i">The position of item where heap property is violated.</param>
        private void SiftUp(int i)
        {
            while (i > 0)
            {
                int p = (i - 1) >> 1;
                if (comparer.Compare(buffer[i], buffer[p]) <= 0)
                {
                    break;
                }

                Swap(i, p);
                i = p;
            }
        }

        private void Swap(int i, int j)
        {
            var temp = buffer[i];
            buffer[i] = buffer[j];
            buffer[j] = temp;
        }
    }
}
