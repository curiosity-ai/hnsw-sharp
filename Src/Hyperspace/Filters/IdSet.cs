// <copyright file="IdSet.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections;
    using System.Collections.Generic;
    using System.Numerics;

    /// <summary>A growable bitmap of item ids with fast set algebra; what the metadata filters compile to.</summary>
    public sealed class IdSet : IEnumerable<int>
    {
        private ulong[] _words;

        /// <summary>Initializes a new, empty instance of the <see cref="IdSet"/> class.</summary>
        public IdSet()
        {
            _words = Array.Empty<ulong>();
        }

        /// <summary>Initializes a new instance of the <see cref="IdSet"/> class holding the given ids.</summary>
        public IdSet(IEnumerable<int> ids)
            : this()
        {
            foreach (var id in ids)
            {
                Add(id);
            }
        }

        /// <summary>Creates the set <c>{0, ..., count - 1}</c>.</summary>
        public static IdSet Range(int count)
        {
            var set = new IdSet();
            if (count <= 0)
            {
                return set;
            }

            set._words = new ulong[(count + 63) >> 6];
            Array.Fill(set._words, ulong.MaxValue);
            int tail = count & 63;
            if (tail != 0)
            {
                set._words[^1] = (1UL << tail) - 1;
            }

            return set;
        }

        /// <summary>Gets the number of ids in the set.</summary>
        public int Count
        {
            get
            {
                int c = 0;
                foreach (var w in _words)
                {
                    c += BitOperations.PopCount(w);
                }

                return c;
            }
        }

        /// <summary>Gets a value indicating whether the set is empty.</summary>
        public bool IsEmpty
        {
            get
            {
                foreach (var w in _words)
                {
                    if (w != 0)
                    {
                        return false;
                    }
                }

                return true;
            }
        }

        /// <summary>Tests membership.</summary>
        public bool Contains(int id)
        {
            int w = id >> 6;
            return id >= 0 && w < _words.Length && (_words[w] & (1UL << (id & 63))) != 0;
        }

        /// <summary>Adds an id; returns <c>true</c> if it was not already present.</summary>
        public bool Add(int id)
        {
            if (id < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(id));
            }

            int w = id >> 6;
            if (w >= _words.Length)
            {
                Array.Resize(ref _words, Math.Max(w + 1, _words.Length * 2));
            }

            ulong bit = 1UL << (id & 63);
            bool added = (_words[w] & bit) == 0;
            _words[w] |= bit;
            return added;
        }

        /// <summary>Removes an id; returns <c>true</c> if it was present.</summary>
        public bool Remove(int id)
        {
            int w = id >> 6;
            if (id < 0 || w >= _words.Length)
            {
                return false;
            }

            ulong bit = 1UL << (id & 63);
            bool had = (_words[w] & bit) != 0;
            _words[w] &= ~bit;
            return had;
        }

        /// <summary>Returns a copy.</summary>
        public IdSet Clone() => new IdSet { _words = (ulong[])_words.Clone() };

        /// <summary>Keeps only the ids also in <paramref name="other"/>.</summary>
        public void IntersectWith(IdSet other)
        {
            for (int i = 0; i < _words.Length; i++)
            {
                _words[i] &= i < other._words.Length ? other._words[i] : 0;
            }
        }

        /// <summary>Adds all ids of <paramref name="other"/>.</summary>
        public void UnionWith(IdSet other)
        {
            if (other._words.Length > _words.Length)
            {
                Array.Resize(ref _words, other._words.Length);
            }

            for (int i = 0; i < other._words.Length; i++)
            {
                _words[i] |= other._words[i];
            }
        }

        /// <summary>Removes all ids of <paramref name="other"/>.</summary>
        public void ExceptWith(IdSet other)
        {
            int n = Math.Min(_words.Length, other._words.Length);
            for (int i = 0; i < n; i++)
            {
                _words[i] &= ~other._words[i];
            }
        }

        /// <inheritdoc />
        public IEnumerator<int> GetEnumerator()
        {
            for (int i = 0; i < _words.Length; i++)
            {
                ulong w = _words[i];
                while (w != 0)
                {
                    int bit = BitOperations.TrailingZeroCount(w);
                    yield return (i << 6) + bit;
                    w &= w - 1;
                }
            }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        internal ulong[] Words => _words;
    }
}
