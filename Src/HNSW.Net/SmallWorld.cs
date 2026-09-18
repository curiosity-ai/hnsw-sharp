// <copyright file="SmallWorld.cs" company="Microsoft">
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace HNSW.Net
{
    using System;
    using System.Buffers;
    using System.Collections.Generic;
    using System.Diagnostics;
    using System.Diagnostics.CodeAnalysis;
    using System.IO;
    using System.Linq;
    using System.Threading;
    using MessagePack;
    using MessagePackCompat;

    /// <summary>
    /// The Hierarchical Navigable Small World Graphs. https://arxiv.org/abs/1603.09320
    /// </summary>
    /// <typeparam name="TItem">The type of items to connect into small world.</typeparam>
    /// <typeparam name="TDistance">The type of distance between items (expect any numeric type: float, double, decimal, int, ...).</typeparam>
    public partial class SmallWorld<TItem, TDistance> where TDistance : struct, IComparable<TDistance>
    {
        /// <summary>The header of the legacy MessagePack format, which is still read.</summary>
        private const string SERIALIZATION_HEADER = "HNSW";

        /// <summary>The header of the flat format - see <c>Graph.Serialization.cs</c> - which is what is written.</summary>
        private const string SERIALIZATION_HEADER_V2 = "HNSW2";

        /// <summary>Sanity bound on the serialized <see cref="SmallWorldParameters"/> block (it is a couple of hundred bytes).</summary>
        private const int MAXIMUM_PARAMETERS_SIZE = 1024 * 1024;

        private readonly Func<TItem, TItem, TDistance> Distance;

        private Graph<TItem, TDistance> Graph;

        /// <summary>
        /// Gets the parameters of the small world.
        /// </summary>
        public SmallWorldParameters Parameters => Graph.Parameters;

        private IProvideRandomValues Generator;

        private ReaderWriterLockSlim _rwLock;

        /// <summary>
        /// Gets the list of items currently held by the SmallWorld graph. 
        /// The list is not protected by any locks, and should only be used when it is known the graph won't change
        /// </summary>
        public IReadOnlyList<TItem> UnsafeItems => Graph?.GraphCore?.Items;

        /// <summary>
        /// Gets a copy of the list of items currently held by the SmallWorld graph. 
        /// This call is protected by a read-lock and is safe to be called from multiple threads.
        /// </summary>
        public IReadOnlyList<TItem> Items
        {
            get
            {
                if (_rwLock is object)
                {
                    _rwLock.EnterReadLock();
                    try
                    {
                        return Graph.GraphCore.Items.ToList();
                    }
                    finally
                    {
                        _rwLock.ExitReadLock();
                    }
                }
                else
                {
                    return Graph?.GraphCore?.Items;
                }
            }
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="SmallWorld{TItem, TDistance}"/> class.
        /// </summary>
        /// <param name="distance">The distance function to use in the small world.</param>
        /// <param name="generator">The random number generator for building graph.</param>
        /// <param name="parameters">Parameters of the algorithm.</param>
        public SmallWorld(Func<TItem, TItem, TDistance> distance, IProvideRandomValues generator, SmallWorldParameters parameters, bool threadSafe = true)
        {
            Distance = distance;
            Graph = new Graph<TItem, TDistance>(Distance, parameters);
            Generator = generator;
            _rwLock = threadSafe ? new ReaderWriterLockSlim() : null;
        }

        /// <summary>
        /// Builds hnsw graph from the items.
        /// </summary>
        /// <param name="items">The items to connect into the graph.</param>

        public IReadOnlyList<int> AddItems(IReadOnlyList<TItem> items, IProgressReporter progressReporter = null)
        {
            _rwLock?.EnterWriteLock();
            try
            {
               return Graph.AddItems(items, Generator, progressReporter);
            }
            finally
            {
                _rwLock?.ExitWriteLock();
            }
        }

        /// <summary>
        /// Run knn search for a given item.
        /// </summary>
        /// <param name="item">The item to search nearest neighbours.</param>
        /// <param name="k">The number of nearest neighbours.</param>
        /// <param name="filterItem">Filter results by ID that should be kept (return true to keep, false to exclude from results)</param>
        /// <param name="cancellationToken">Cancellation Token for stopping the search when filtering is active</param>
        /// <returns>The list of found nearest neighbours.</returns>
        public IList<KNNSearchResult> KNNSearch(TItem item, int k, Func<TItem, bool> filterItem = null, CancellationToken cancellationToken = default)
        {
            _rwLock?.EnterReadLock();
            try
            {
                return Graph.KNearest(item, k, filterItem, cancellationToken);
            }
            finally
            {
                _rwLock?.ExitReadLock();
            }
        }

        /// <summary>
        /// Get the item with the index
        /// </summary>
        /// <param name="index">The index of the item</param>
        public TItem GetItem(int index)
        {
            _rwLock?.EnterReadLock();
            try
            {
                return Items[index];
            }
            finally
            {
                _rwLock?.ExitReadLock();
            }
        }

        /// <summary>
        /// Serializes the graph WITHOUT linked items.
        /// </summary>
        /// <remarks>
        /// The bytes travel through a buffer rented from <see cref="ArrayPool{T}.Shared"/> and are flushed to
        /// <paramref name="stream"/> as they are produced, so peak memory does not scale with the graph.
        /// </remarks>
        public void SerializeGraph(Stream stream)
        {
            if (Graph == null)
            {
                throw new InvalidOperationException("The graph does not exist");
            }
            _rwLock?.EnterReadLock();
            try
            {
                using (var writer = new PooledStreamBufferWriter(stream))
                {
                    WriteGraph(writer);
                }
            }
            finally
            {
                _rwLock?.ExitReadLock();
            }
        }

        /// <summary>
        /// Serializes the graph WITHOUT linked items, straight into a caller-owned
        /// <see cref="IBufferWriter{T}"/>.
        /// </summary>
        /// <remarks>
        /// For a caller that already has a pooled writer and a destination taking a contiguous span (a
        /// key-value store's put, an encryption call), this skips staging the graph in a stream first.
        /// </remarks>
        public void SerializeGraph(IBufferWriter<byte> writer)
        {
            if (Graph == null)
            {
                throw new InvalidOperationException("The graph does not exist");
            }
            _rwLock?.EnterReadLock();
            try
            {
                WriteGraph(writer);
            }
            finally
            {
                _rwLock?.ExitReadLock();
            }
        }

        private void WriteGraph(IBufferWriter<byte> writer)
        {
            var messagePackWriter = new MessagePackWriter(writer);
            messagePackWriter.Write(SERIALIZATION_HEADER_V2);
            messagePackWriter.Flush();

            //Length-prefixed so the reader never has to rewind the stream - see PooledStreamBufferReader.ReadMessagePackBlock
            var parameters = new ArrayBufferWriter<byte>(256);
            MessagePackSerializer.Serialize(parameters, Graph.Parameters);

            writer.WriteInt32(parameters.WrittenCount);
            writer.Write(parameters.WrittenSpan);

            Graph.SerializeFlat(writer);
        }

        /// <summary>
        /// Serializes the graph in the legacy MessagePack format. Only used to produce data for the
        /// backwards-compatibility tests - <see cref="SerializeGraph(Stream)"/> is the one to call.
        /// </summary>
        internal void SerializeGraphLegacy(Stream stream)
        {
            if (Graph == null)
            {
                throw new InvalidOperationException("The graph does not exist");
            }
            _rwLock?.EnterReadLock();
            try
            {
                MessagePackBinary.WriteString(stream, SERIALIZATION_HEADER);
                MessagePackSerializer.Serialize(stream, Graph.Parameters);
                Graph.SerializeMessagePack(stream);
            }
            finally
            {
                _rwLock?.ExitReadLock();
            }
        }

        /// <summary>
        /// Deserializes the graph from byte array.
        /// </summary>
        /// <param name="items">The items to assign to the graph's verticies.</param>
        /// <param name="bytes">The serialized parameters and edges.</param>
        public static (SmallWorld<TItem, TDistance> Graph, TItem[] ItemsNotInGraph) DeserializeGraph(IReadOnlyList<TItem> items, Func<TItem, TItem, TDistance> distance, IProvideRandomValues generator, Stream stream, bool threadSafe = true)
        {
            var p0 = stream.CanSeek ? stream.Position : 0; //Only used to rewind a stream that turns out not to be a graph
            string hnswHeader;
            try
            {
                hnswHeader = MessagePackBinary.ReadString(stream);
            }
            catch(Exception E)
            {
                if(stream.CanSeek) { stream.Position = p0; } //Resets the stream to original position
                throw new InvalidDataException($"Invalid header found in stream, data is corrupted or invalid", E);
            }

            if (hnswHeader != SERIALIZATION_HEADER && hnswHeader != SERIALIZATION_HEADER_V2)
            {
                if (stream.CanSeek) { stream.Position = p0; } //Resets the stream to original position
                throw new InvalidDataException($"Invalid header found in stream, data is corrupted or invalid");
            }

            // readStrict: true -> removed, as not available anymore on MessagePack 2.0 - also probably not necessary anymore
            //                     see https://github.com/neuecc/MessagePack-CSharp/pull/663

            if (hnswHeader == SERIALIZATION_HEADER_V2)
            {
                //One reader for the whole payload: the parameters are length-prefixed, so nothing reads past
                //what it needs and the stream does not have to be seekable.
                using (var reader = new PooledStreamBufferReader(stream))
                {
                    var flatParameters = reader.ReadMessagePackBlock<SmallWorldParameters>(MAXIMUM_PARAMETERS_SIZE);
                    flatParameters.InitialDistanceCacheSize = 0;

                    var flatWorld = new SmallWorld<TItem, TDistance>(distance, generator, flatParameters, threadSafe: threadSafe);
                    return (flatWorld, flatWorld.Graph.DeserializeFlat(items, reader));
                }
            }

            var parameters = MessagePackSerializer.Deserialize<SmallWorldParameters>(stream);

            //Overwrite previous InitialDistanceCacheSize parameter, so we don't waste time/memory allocating a distance cache for an already existing graph
            parameters.InitialDistanceCacheSize = 0;

            var world = new SmallWorld<TItem, TDistance>(distance, generator, parameters, threadSafe: threadSafe);

            return (world, world.Graph.DeserializeMessagePack(items, stream));
        }

        /// <summary>
        /// Prints edges of the graph. Mostly for debug and test purposes.
        /// </summary>
        /// <returns>String representation of the graph's edges.</returns>
        public string Print()
        {
            return Graph.Print();
        }

        /// <summary>
        /// Ensure all layer connections are cached in a flatten memory representation.
        /// </summary>
        public void OptimizeIfNeeded(bool force = false)
        {
            Graph.OptimizeIfNeeded(force);
        }


        /// <summary>
        /// Frees the memory used by the Distance Cache
        /// </summary>
        public void DisableDistanceCache()
        {
            Graph.GraphCore.ResizeDistanceCache(-1);
        }

        /// <summary>
        /// Resizes the distance cache used for caching embedding distances
        /// </summary>
        public void ResizeDistanceCache(int newSize)
        {
            Graph.GraphCore.ResizeDistanceCache(Math.Max(0, newSize));
        }

        public class KNNSearchResult
        {
            internal KNNSearchResult(int id, TItem item, TDistance distance)
            {
                Id = id;
                Item = item;
                Distance = distance;
            }

            public int Id { get; }

            public TItem Item { get; }

            public TDistance Distance { get; }

            public override string ToString()
            {
                return $"I:{Id} Dist:{Distance:n2} [{Item}]";
            }
        }
    }
    [MessagePackObject(keyAsPropertyName:true)]
    public class SmallWorldParameters
    {
        public SmallWorldParameters()
        {
            M = 10;
            LevelLambda = 1 / Math.Log(M);
            NeighbourHeuristic = NeighbourSelectionHeuristic.SelectSimple;
            ConstructionPruning = 200;
            EfSearch = 50;
            ExpandBestSelection = false;
            KeepPrunedConnections = false;
            EnableDistanceCacheForConstruction = false;
            InitialDistanceCacheSize = 1024 * 1024;
            InitialItemsSize = 1024;
            OptimizeForFiltering = false;
            Gamma = 1;
            Mb = 10;
            EnableEarlyTermination = false;
            EarlyTerminationSaturationThreshold = 0.95;
            EarlyTerminationPatience = 0;
        }

        /// <summary>
        /// Gets or sets a value indicating whether the layer-0 search should stop early once the result set has
        /// stopped improving ("patience" based early termination, see https://manticoresearch.com/blog/knn-early-termination/
        /// and "Patience in Proximity", Teofili &amp; Lin, ECIR 2025).
        /// While traversing the graph the saturation of the result set is tracked on every hop: once the fraction of
        /// the top-k that stayed unchanged stays at or above <see cref="EarlyTerminationSaturationThreshold"/> for
        /// <see cref="EarlyTerminationPatience"/> consecutive hops, the search stops. This trades a small amount of
        /// recall for fewer distance computations, with the largest savings as k and efSearch grow.
        /// Disabled by default to preserve the exact, exhaustive search behaviour.
        /// </summary>
        public bool EnableEarlyTermination { get; set; }

        /// <summary>
        /// Gets or sets the saturation ratio (in the range (0, 1]) that a search hop must reach to be counted as
        /// "non-improving" when <see cref="EnableEarlyTermination"/> is enabled. The saturation of a hop is the
        /// fraction of the current top-k results that were left unchanged by that hop, so a value closer to 1 makes
        /// early termination more conservative (higher recall, less speed-up). Defaults to 0.95.
        /// </summary>
        public double EarlyTerminationSaturationThreshold { get; set; }

        /// <summary>
        /// Gets or sets the number of consecutive non-improving (saturated) hops that must be observed before the
        /// search terminates early when <see cref="EnableEarlyTermination"/> is enabled. A value of 0 (the default)
        /// selects an adaptive patience that scales inversely with efSearch (≈9 at low ef down to 6 at very high ef),
        /// matching the behaviour described in the Manticore/ECIR work. Larger values are more conservative.
        /// </summary>
        public int EarlyTerminationPatience { get; set; }

        /// <summary>
        /// Gets or sets whether the graph should be constructed for filtering, according to ACORN (https://arxiv.org/html/2403.04871v1).
        /// </summary>
        public bool OptimizeForFiltering { get; set; }

        /// <summary>
        /// Gets or sets the neighbor expansion factor for the ACORN-γ index.
        /// </summary>
        public int Gamma { get; set; }

        /// <summary>
        /// Gets or sets the compression parameter for ACORN's layer 0.
        /// </summary>
        public int Mb { get; set; }

        /// <summary>
        /// Gets or sets the parameter which defines the maximum number of neighbors in the zero and above-zero layers.
        /// The maximum number of neighbors for the zero layer is 2 * M.
        /// The maximum number of neighbors for higher layers is M.
        /// </summary>
        public int M { get; set; }

        /// <summary>
        /// Gets or sets the max level decay parameter. https://en.wikipedia.org/wiki/Exponential_distribution See 'mL' parameter in the HNSW article.
        /// </summary>
        public double LevelLambda { get; set; }

        /// <summary>
        /// Gets or sets parameter which specifies the type of heuristic to use for best neighbours selection.
        /// </summary>
        public NeighbourSelectionHeuristic NeighbourHeuristic { get; set; }

        /// <summary>
        /// Gets or sets the number of candidates to consider as neighbours for a given node at the graph construction phase. See 'efConstruction' parameter in the article.
        /// </summary>
        public int ConstructionPruning { get; set; }

        /// <summary>
        /// Gets or sets the number of candidates to keep during search at layer 0. See 'efSearch' parameter in the article.
        /// </summary>
        public int EfSearch { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to expand candidates if <see cref="NeighbourSelectionHeuristic.SelectHeuristic"/> is used. See 'extendCandidates' parameter in the article.
        /// </summary>
        public bool ExpandBestSelection { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to keep pruned candidates if <see cref="NeighbourSelectionHeuristic.SelectHeuristic"/> is used. See 'keepPrunedConnections' parameter in the article.
        /// </summary>
        public bool KeepPrunedConnections { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether to cache calculated distances at graph construction time.
        /// Since the search now computes the distance to each visited node exactly once, the cache rarely pays
        /// for its memory footprint and lookup cost, so it is disabled by default.
        /// </summary>
        public bool EnableDistanceCacheForConstruction { get; set; }

        /// <summary>
        /// Gets or sets a the initial distance cache size. 
        /// Note: This value is reset to 0 on deserialization to avoid allocating the distance cache for pre-built graphs.
        /// </summary>
        public int InitialDistanceCacheSize { get; set; }

        /// <summary>
        /// Gets or sets a the initial size of the Items list
        /// </summary>
        public int InitialItemsSize { get; set; }
    }

}
