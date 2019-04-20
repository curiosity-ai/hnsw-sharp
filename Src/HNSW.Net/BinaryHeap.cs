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
        internal BinaryHeap(IList<T> buffer)
            : this(buffer, Comparer<T>.Default)
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
                this.SiftUp(i);
            }
        }

        /// <summary>
        /// Gets the heap comparer.
        /// </summary>
        internal IComparer<T> Comparer => this.comparer;

        /// <summary>
        /// Gets the buffer of the heap.
        /// </summary>
        internal IList<T> Buffer => this.buffer;

        /// <summary>
        /// Pushes item to the heap.
        /// </summary>
        /// <param name="item">The item to push.</param>
        internal void Push(T item)
        {
            this.buffer.Add(item);
            this.SiftUp(this.buffer.Count - 1);
        }

        /// <summary>
        /// Pops the item from the heap.
        /// </summary>
        /// <returns>The popped item.</returns>
        internal T Pop()
        {
            if (this.buffer.Count > 0)
            {
                var result = this.buffer[0];

                this.buffer[0] = this.buffer[this.buffer.Count - 1];
                this.buffer.RemoveAt(this.buffer.Count - 1);
                this.SiftDown(0);

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
            while (i < this.buffer.Count)
            {
                int l = (i << 1) + 1;
                int r = l + 1;
                if (l >= this.buffer.Count)
                {
                    break;
                }

                int m = r < this.buffer.Count && this.comparer.Compare(this.buffer[l], this.buffer[r]) < 0 ? r : l;
                if (this.comparer.Compare(this.buffer[m], this.buffer[i]) <= 0)
                {
                    break;
                }

                this.Swap(i, m);
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
                if (this.comparer.Compare(this.buffer[i], this.buffer[p]) <= 0)
                {
                    break;
                }

                this.Swap(i, p);
                i = p;
            }
        }

        /// <summary>
        /// Swaps items with the specified indices.
        /// </summary>
        /// <param name="i">The first index.</param>
        /// <param name="j">The second index.</param>
        private void Swap(int i, int j)
        {
            var temp = this.buffer[i];
            this.buffer[i] = this.buffer[j];
            this.buffer[j] = temp;
        }
    }
}
