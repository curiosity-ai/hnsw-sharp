// <copyright file="ScopeLatencyTracker.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Diagnostics;

    /// <summary>
    /// Latency tracker for using scope.
    /// TODO: make it ref struct in C# 8.0
    /// </summary>
    internal struct ScopeLatencyTracker : IDisposable
    {
        private long startTimestamp;
        private Action<float> latencyCallback;

        /// <summary>
        /// Initializes a new instance of the <see cref="ScopeLatencyTracker"/> struct.
        /// </summary>
        /// <param name="callback">The latency reporting callback to associate with the scope.</param>
        internal ScopeLatencyTracker(Action<float> callback)
        {
            this.startTimestamp = callback != null ? Stopwatch.GetTimestamp() : 0;
            this.latencyCallback = callback;
        }

        /// <summary>
        /// Reports the time ellsapsed between the tracker creation and this call.
        /// </summary>
        public void Dispose()
        {
            const long ticksPerMicroSecond = TimeSpan.TicksPerMillisecond / 1000;
            if (this.latencyCallback != null)
            {
                long ellapsedMuS = (Stopwatch.GetTimestamp() - this.startTimestamp) / ticksPerMicroSecond;
                this.latencyCallback(ellapsedMuS);
            }
        }
    }
}
