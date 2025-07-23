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

        if (counterData["Name"] == "GetDistance.CacheHitRate" || counterData["Name"] == "InsertNode.Latency")
        {
            Console.WriteLine($"[{counterData["Name"]:n1}]: Avg={counterData["Mean"]:n1}; SD={counterData["StandardDeviation"]:n1}; Count={counterData["Count"]}");
        }
    }
}
