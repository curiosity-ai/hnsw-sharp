// <copyright file="SmallWorldTests.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    public class RewindableRandomNumberGenerator : IProvideRandomValues
    {
        public bool IsThreadSafe => false;

        private FastRandom _random = new FastRandom(42);
        private List<Action> _calls = new List<Action>();

        public int Next(int minValue, int maxValue)
        {
            _calls.Add( () => _random.Next(minValue, maxValue) );
            return _random.Next(minValue, maxValue);
        }

        public float NextFloat()
        {
            _calls.Add(() => _random.NextFloat());
            return _random.NextFloat();
        }

        public void NextFloats(Span<float> buffer)
        {
            var len = buffer.Length;
            _calls.Add(() =>
            {
                var b = new float[len];
                _random.NextFloats(b);
            });
            _random.NextFloats(buffer);
        }

        public int GetState() => _calls.Count;

        public void RewindTo(int state)
        {
            _random = new FastRandom(42);
            var toInvoke = _calls.Take(state).ToArray();
            _calls.Clear();
            foreach (var a in toInvoke) 
            {
                a(); 
            }
        }
    }
}