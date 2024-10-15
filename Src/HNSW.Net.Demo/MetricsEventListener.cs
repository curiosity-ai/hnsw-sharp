// <copyright file="Program.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net.Demo;

using System;
using System.Collections.Generic;
using System.Diagnostics.Tracing;
using System.Linq;

public class MetricsEventListener : EventListener
{
    private readonly EventSource eventSource;

    public MetricsEventListener(EventSource eventSource)
    {
        this.eventSource = eventSource;
        EnableEvents(this.eventSource, EventLevel.LogAlways, EventKeywords.All, new Dictionary<string, string> { { "EventCounterIntervalSec", "1" } });
    }

    public override void Dispose()
    {
        DisableEvents(eventSource);
        base.Dispose();
    }

    protected override void OnEventWritten(EventWrittenEventArgs eventData)
    {
        var counterData = eventData.Payload?.FirstOrDefault() as IDictionary<string, object>;
        if (counterData?.Count == 0)
        {
            return;
        }

        const string emptyVal = "N/A";
        var name = counterData.TryGetValue("Name", out object _name) ? _name.ToString() : emptyVal;
        var avg = counterData.TryGetValue("Mean", out object mean) ? mean.ToString() : emptyVal;
        var sd = counterData.TryGetValue("StandardDeviation", out object standardDev) ? standardDev.ToString() : emptyVal;
        var ct = counterData.TryGetValue("Count", out object count) ? count.ToString() : emptyVal;

        Console.WriteLine($"[{name:n1}]: Avg={avg:n1}; SD={sd:n1}; Count={ct}");
    }
}
