// <copyright file="BinaryHeapTests.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net.Tests
{
    using System.Linq;
    using Microsoft.VisualStudio.TestTools.UnitTesting;

    /// <summary>
    /// Tests for <see cref="BinaryHeap{T}"/>
    /// </summary>
    [TestClass]
    public class BinaryHeapTests
    {
        /// <summary>
        /// Tests heap construction.
        /// </summary>
        [TestMethod]
        public void HeapifyTest()
        {
            // Basic tests
            {
                var heap = new BinaryHeap<int>(Enumerable.Empty<int>().ToList());
                Assert.IsFalse(heap.Buffer.Any());

                heap = new BinaryHeap<int>(Enumerable.Range(1, 1).ToList());
                Assert.AreEqual(1, heap.Buffer.Count);
                Assert.AreEqual(1, heap.Buffer.First());

                heap = new BinaryHeap<int>(Enumerable.Range(1, 2).ToList());
                Assert.AreEqual(2, heap.Buffer.Count);
                Assert.AreEqual(2, heap.Buffer.First());
            }

            // Heapify produces correct heap.
            {
                const string input = "Hello, World!";
                var heap = new BinaryHeap<char>(input.ToList());
                AssertMaxHeap(heap);
            }
        }

        /// <summary>
        /// Tests <see cref="BinaryHeap{T}.Push(T)"/> and <see cref="BinaryHeap{T}.Pop()"/>
        /// </summary>
        [TestMethod]
        public void PushPopTest()
        {
            var heap = new BinaryHeap<int>(Enumerable.Empty<int>().ToList());
            for (int i = 0; i < 10; ++i)
            {
                heap.Push(i);
            }

            AssertMaxHeap(heap);

            int top = heap.Buffer.First();
            while (heap.Buffer.Any())
            {
                Assert.AreEqual(top, heap.Pop());
                top = heap.Buffer.FirstOrDefault();
            }
        }

        private void AssertMaxHeap<T>(BinaryHeap<T> heap)
        {
            for (int p = 0; p < heap.Buffer.Count; ++p)
            {
                int l = (2 * p) + 1;
                int r = l + 1;

                var parent = heap.Buffer[p];
                if (l < heap.Buffer.Count)
                {
                    var left = heap.Buffer[l];
                    Assert.IsTrue(heap.Comparer.Compare(parent, left) >= 0);
                }

                if (r < heap.Buffer.Count)
                {
                    var right = heap.Buffer[r];
                    Assert.IsTrue(heap.Comparer.Compare(parent, right) >= 0);
                }
            }
        }
    }
}