// <copyright file="Candidates.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Collections.Generic;
    using System.Runtime.CompilerServices;
    using System.Runtime.InteropServices;

    /// <summary>
    /// A graph node id paired with its (pre-computed) distance to the current search target.
    /// Storing the distance next to the id lets the heaps order candidates without re-invoking
    /// the distance function on every comparison.
    /// </summary>
    /// <typeparam name="TDistance">The type of the distance.</typeparam>
    internal readonly struct Candidate<TDistance> where TDistance : struct, IComparable<TDistance>
    {
        public readonly TDistance Distance;
        public readonly int Id;

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public Candidate(TDistance distance, int id)
        {
            Distance = distance;
            Id = id;
        }
    }

    /// <summary>
    /// Orders candidates by ascending distance.
    /// </summary>
    internal sealed class CandidateComparer<TDistance> : IComparer<Candidate<TDistance>> where TDistance : struct, IComparable<TDistance>
    {
        public static readonly CandidateComparer<TDistance> Instance = new CandidateComparer<TDistance>();

        public int Compare(Candidate<TDistance> x, Candidate<TDistance> y)
        {
            return x.Distance.CompareTo(y.Distance);
        }
    }

    /// <summary>
    /// Binary max-heap over a list of candidates: the candidate with the largest distance is on top.
    /// Used to maintain the dynamic result set of a search (the farthest result is evicted first).
    /// The constructor does not heapify; it expects either an empty buffer or a buffer which already
    /// satisfies the heap invariant.
    /// </summary>
    internal struct CandidateMaxHeap<TDistance> where TDistance : struct, IComparable<TDistance>
    {
        internal readonly List<Candidate<TDistance>> Buffer;

        public CandidateMaxHeap(List<Candidate<TDistance>> buffer)
        {
            Buffer = buffer;
        }

        public int Count
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Buffer.Count;
        }

        public Candidate<TDistance> Top
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Buffer[0];
        }

        public void Push(in Candidate<TDistance> item)
        {
            Buffer.Add(item);
            var span = CollectionsMarshal.AsSpan(Buffer);
            int i = span.Length - 1;
            while (i > 0)
            {
                int p = (i - 1) >> 1;
                if (span[i].Distance.CompareTo(span[p].Distance) <= 0) { break; }
                (span[i], span[p]) = (span[p], span[i]);
                i = p;
            }
        }

        public Candidate<TDistance> Pop()
        {
            var span = CollectionsMarshal.AsSpan(Buffer);
            var result = span[0];
            span[0] = span[span.Length - 1];
            Buffer.RemoveAt(Buffer.Count - 1);
            SiftDown(CollectionsMarshal.AsSpan(Buffer));
            return result;
        }

        private static void SiftDown(Span<Candidate<TDistance>> span)
        {
            int i = 0;
            while (true)
            {
                int l = (i << 1) + 1;
                if (l >= span.Length) { break; }
                int r = l + 1;
                int m = r < span.Length && span[l].Distance.CompareTo(span[r].Distance) < 0 ? r : l;
                if (span[m].Distance.CompareTo(span[i].Distance) <= 0) { break; }
                (span[i], span[m]) = (span[m], span[i]);
                i = m;
            }
        }
    }

    /// <summary>
    /// Binary min-heap over a list of candidates: the candidate with the smallest distance is on top.
    /// Used as the expansion frontier of a search (the closest candidate is expanded first).
    /// </summary>
    internal struct CandidateMinHeap<TDistance> where TDistance : struct, IComparable<TDistance>
    {
        internal readonly List<Candidate<TDistance>> Buffer;

        public CandidateMinHeap(List<Candidate<TDistance>> buffer)
        {
            Buffer = buffer;
        }

        public int Count
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Buffer.Count;
        }

        public Candidate<TDistance> Top
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Buffer[0];
        }

        public void Push(in Candidate<TDistance> item)
        {
            Buffer.Add(item);
            var span = CollectionsMarshal.AsSpan(Buffer);
            int i = span.Length - 1;
            while (i > 0)
            {
                int p = (i - 1) >> 1;
                if (span[i].Distance.CompareTo(span[p].Distance) >= 0) { break; }
                (span[i], span[p]) = (span[p], span[i]);
                i = p;
            }
        }

        public Candidate<TDistance> Pop()
        {
            var span = CollectionsMarshal.AsSpan(Buffer);
            var result = span[0];
            span[0] = span[span.Length - 1];
            Buffer.RemoveAt(Buffer.Count - 1);
            SiftDown(CollectionsMarshal.AsSpan(Buffer));
            return result;
        }

        private static void SiftDown(Span<Candidate<TDistance>> span)
        {
            int i = 0;
            while (true)
            {
                int l = (i << 1) + 1;
                if (l >= span.Length) { break; }
                int r = l + 1;
                int m = r < span.Length && span[l].Distance.CompareTo(span[r].Distance) > 0 ? r : l;
                if (span[m].Distance.CompareTo(span[i].Distance) >= 0) { break; }
                (span[i], span[m]) = (span[m], span[i]);
                i = m;
            }
        }
    }
}
