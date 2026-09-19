// <copyright file="Graph.Serialization.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Buffers;
    using System.Collections.Generic;
    using System.IO;
    using System.Runtime.InteropServices;

    /// <summary>
    /// The flat graph format ("HNSW2").
    /// </summary>
    /// <remarks>
    /// <para>
    /// The connections of a graph already live in <see cref="CachedNodeData"/> as one flat
    /// <see cref="int"/> run per node - the per-layer start offsets, the total, then the neighbour ids. This
    /// format is that run, written as little-endian <see cref="int"/>s, so writing a graph is a span copy per
    /// node and reading one is a span fill per node.
    /// </para>
    /// <para>
    /// What it replaces (the "HNSW" format, still read by <see cref="DeserializeMessagePack"/>) round-tripped
    /// every node through <c>List&lt;List&lt;int&gt;&gt;</c>: writing hydrated one per node out of the flat
    /// cache - so a save allocated two objects per layer of every node, left the whole graph un-flattened,
    /// and had to re-run <c>Optimize</c> afterwards to undo that - and reading allocated the same lists again
    /// only to flatten and drop them. Neither happens here, and the bytes in flight are a pooled buffer of a
    /// fixed size rather than the whole graph staged in memory.
    /// </para>
    /// <para>
    /// Layout, all values little-endian:
    /// <code>
    /// int32  nodeCount
    /// int32  entryPointId          // -1 when the graph has no entry point
    /// int64  totalRecordInts       // sum of recordLength, used to size the connection storage in one go
    /// per node, in id order:
    ///   int32    id
    ///   int32    maxLayers
    ///   int32    recordLength      // == maxLayers + 1 + neighbourCount
    ///   int32[]  record            // layer start offsets, total, then the neighbour ids
    /// </code>
    /// </para>
    /// </remarks>
    internal partial class Graph<TItem, TDistance>
    {
        /// <summary>
        /// Writes the graph in the flat format. Read-only: unlike the MessagePack path this neither hydrates
        /// nor re-flattens anything, so a caller holding only a read lock is not lying about it.
        /// </summary>
        internal void SerializeFlat(IBufferWriter<byte> writer)
        {
            var nodes = CollectionsMarshal.AsSpan(GraphCore.Nodes);

            long totalRecordInts = 0;
            for (int i = 0; i < nodes.Length; i++)
            {
                totalRecordInts += nodes[i].FlatRecordLength;
            }

            writer.WriteInt32(nodes.Length);
            writer.WriteInt32(EntryPoint?.Id ?? -1);
            writer.WriteInt64(totalRecordInts);

            for (int i = 0; i < nodes.Length; i++)
            {
                nodes[i].WriteTo(writer);
            }
        }

        /// <summary>
        /// Reads a graph written by <see cref="SerializeFlat"/> and assigns <paramref name="items"/> to its
        /// nodes, returning the items the graph does not cover.
        /// </summary>
        internal TItem[] DeserializeFlat(IReadOnlyList<TItem> items, PooledStreamBufferReader reader)
        {
            int  nodeCount       = reader.ReadInt32();
            int  entryPointId    = reader.ReadInt32();
            long totalRecordInts = reader.ReadInt64();

            if (nodeCount < 0 || totalRecordInts < 0 || entryPointId >= nodeCount)
            {
                throw new InvalidDataException($"Invalid HNSW graph header (nodes: {nodeCount}, entry point: {entryPointId}, size: {totalRecordInts})");
            }

            var cache = new CachedNodeData(totalRecordInts);
            var core  = new Core(Distance, Parameters);

            var remainingItems = core.DeserializeFlat(items, reader, nodeCount, totalRecordInts, cache);

            _cachedNodeData = cache;
            GraphCore       = core;
            EntryPoint      = entryPointId >= 0 ? core.Nodes[entryPointId] : (Node?)null;

            return remainingItems;
        }

        internal partial class Core
        {
            internal TItem[] DeserializeFlat(IReadOnlyList<TItem> items, PooledStreamBufferReader reader, int nodeCount, long totalRecordInts, CachedNodeData cache)
            {
                var nodes = new List<Node>(Math.Max(nodeCount, 1));

                long remaining = totalRecordInts;

                for (int i = 0; i < nodeCount; i++)
                {
                    int id           = reader.ReadInt32();
                    int maxLayers    = reader.ReadInt32();
                    int recordLength = reader.ReadInt32();

                    //Nodes are written in id order and read back by position, so the two have to agree
                    if (id != i)
                    {
                        throw new InvalidDataException($"Invalid HNSW node record at index {i}: it carries id {id}");
                    }

                    //Bounded against the total announced in the header, so corrupted data cannot make the
                    //reader ask for an arbitrarily large allocation before it notices. A node without
                    //layers has no record, and a node with layers has at least the offsets and the total.
                    if (maxLayers < 0 || recordLength < 0 || recordLength > remaining || (recordLength == 0) != (maxLayers == 0) || (recordLength > 0 && recordLength < maxLayers + 1))
                    {
                        throw new InvalidDataException($"Invalid HNSW node record at index {i} (layers: {maxLayers}, size: {recordLength})");
                    }

                    remaining -= recordLength;

                    if (recordLength == 0)
                    {
                        nodes.Add(Node.FromCache(id, cache, -1, -1, 0));
                        continue;
                    }

                    var record = cache.Reserve(recordLength, out var bucketIndex, out var position);
                    reader.ReadInt32Span(record);

                    ValidateRecord(record, i, maxLayers, nodeCount);

                    nodes.Add(Node.FromCache(id, cache, bucketIndex, position, maxLayers));
                }

                Nodes = nodes;

                return AssignItems(items, nodeCount);
            }

            /// <summary>
            /// Checks a node record describes a graph the searcher can walk: monotonic layer offsets covering
            /// exactly the record, and neighbour ids that name a node.
            /// </summary>
            /// <remarks>
            /// One pass over data that was just read from disk anyway, and it is what turns a corrupted file
            /// into an <see cref="InvalidDataException"/> the caller can rebuild from, instead of an
            /// index-out-of-range thrown from inside a search hours later.
            /// </remarks>
            private static void ValidateRecord(ReadOnlySpan<int> record, int index, int maxLayers, int nodeCount)
            {
                int neighbours = record.Length - maxLayers - 1;

                if (record[maxLayers] != neighbours)
                {
                    throw new InvalidDataException($"Invalid HNSW node record at index {index}: the layer offsets do not describe {neighbours} neighbours");
                }

                for (int layer = 0; layer < maxLayers; layer++)
                {
                    if (record[layer] < 0 || record[layer] > record[layer + 1])
                    {
                        throw new InvalidDataException($"Invalid HNSW node record at index {index}: layer {layer} starts at {record[layer]} but layer {layer + 1} starts at {record[layer + 1]}");
                    }
                }

                var connections = record.Slice(maxLayers + 1);

                for (int i = 0; i < connections.Length; i++)
                {
                    if ((uint)connections[i] >= (uint)nodeCount)
                    {
                        throw new InvalidDataException($"Invalid HNSW node record at index {index}: neighbour {connections[i]} is not one of the {nodeCount} nodes in the graph");
                    }
                }
            }

            /// <summary>
            /// Takes the first <paramref name="nodeCount"/> items as the graph's own and returns the rest,
            /// which the caller re-adds (they are the vectors a crash left outside the persisted graph).
            /// </summary>
            internal TItem[] AssignItems(IReadOnlyList<TItem> items, int nodeCount)
            {
                int covered = Math.Min(nodeCount, items.Count);

                Items.EnsureCapacity(covered);
                for (int i = 0; i < covered; i++)
                {
                    Items.Add(items[i]);
                }

                var remainingItems = new TItem[items.Count - covered];
                for (int i = 0; i < remainingItems.Length; i++)
                {
                    remainingItems[i] = items[covered + i];
                }

                return remainingItems;
            }
        }
    }
}
