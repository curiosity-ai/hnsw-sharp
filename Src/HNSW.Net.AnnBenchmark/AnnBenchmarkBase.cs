using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BenchmarkDotNet.Attributes;
using HNSW.Net;

namespace HNSW.Net.AnnBenchmark
{
    /// <summary>
    /// Shared driver for the ANN-Benchmarks datasets: downloads the HDF5 file, builds an HNSW graph
    /// over the base items and measures query throughput and recall against the precomputed ground truth.
    /// One concrete subclass exists per distance type, each pinned to the smallest dataset of that type.
    /// </summary>
    public abstract class AnnBenchmarkBase<TItem>
    {
        // Loaded data and built graphs are cached across the different EfSearch parameter values so that
        // the (expensive) download/parse/build only happens once per dataset and per (M, EfConstruction).
        private static readonly ConcurrentDictionary<string, (TItem[] baseItems, TItem[] queries, int[][] groundTruth)> DataCache = new();
        private static readonly ConcurrentDictionary<string, SmallWorld<TItem, float>> GraphCache = new();

        private TItem[] _queries;
        private int[][] _groundTruth;
        private SmallWorld<TItem, float> _graph;

        [Params(16)]
        public int M { get; set; }

        [Params(200)]
        public int EfConstruction { get; set; }

        [Params(50, 100, 200)]
        public int EfSearch { get; set; }

        /// <summary>Human-readable dataset id used as a cache key and in log output.</summary>
        protected abstract string DatasetName { get; }

        /// <summary>Local file name the HDF5 dataset is stored under.</summary>
        protected abstract string FileName { get; }

        /// <summary>ANN-Benchmarks download URL for the HDF5 dataset.</summary>
        protected abstract string DownloadUrl { get; }

        /// <summary>Distance function matching the dataset's distance type.</summary>
        protected abstract Func<TItem, TItem, float> Distance { get; }

        /// <summary>Loads base items, query items and ground-truth neighbour indices from the HDF5 file.</summary>
        protected abstract (TItem[] baseItems, TItem[] queries, int[][] groundTruth) Load(string path);

        [GlobalSetup]
        public void Setup()
        {
            string workingDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data");
            if (!Directory.Exists(workingDir)) Directory.CreateDirectory(workingDir);

            string path = AnnDataset.EnsureDownloaded(workingDir, FileName, DownloadUrl);

            var data = DataCache.GetOrAdd(DatasetName, _ =>
            {
                Console.WriteLine($"Reading {DatasetName} ...");
                var loaded = Load(path);
                Console.WriteLine($"Loaded {loaded.baseItems.Length} base items and {loaded.queries.Length} queries.");
                return loaded;
            });

            _queries = data.queries;
            _groundTruth = data.groundTruth;

            string graphKey = $"{DatasetName}|M={M}|efc={EfConstruction}";
            _graph = GraphCache.GetOrAdd(graphKey, _ =>
            {
                var parameters = new SmallWorldParameters
                {
                    M = M,
                    LevelLambda = 1 / Math.Log(M),
                    ConstructionPruning = EfConstruction,
                    EfSearch = EfSearch,
                    EnableDistanceCacheForConstruction = true
                };

                Console.WriteLine($"Building {DatasetName} graph with M={M}, EfConstruction={EfConstruction} ...");
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var graph = new SmallWorld<TItem, float>(Distance, DefaultRandomGenerator.Instance, parameters);
                graph.AddItems(data.baseItems);
                sw.Stop();
                Console.WriteLine($"Graph built in {sw.Elapsed.TotalSeconds:N2}s.");
                return graph;
            });

            _graph.Parameters.EfSearch = EfSearch;
        }

        [Benchmark]
        public void Search()
        {
            for (int i = 0; i < _queries.Length; i++)
            {
                _graph.KNNSearch(_queries[i], 10);
            }
        }

        [IterationCleanup]
        public void ReportRecall()
        {
            int total = _queries.Length;
            double sumRecall1 = 0;
            double sumRecall10 = 0;

            var truth10 = new HashSet<int>();
            for (int i = 0; i < total; i++)
            {
                // KNNSearch returns the k nearest neighbours but does not guarantee ascending
                // distance order, so sort before evaluating the top-1 result.
                var results = _graph.KNNSearch(_queries[i], 10)
                    .OrderBy(r => r.Distance)
                    .ToList();

                // Recall@1: the nearest returned item matches the true nearest neighbour.
                if (results.Count > 0 && results[0].Id == _groundTruth[i][0])
                {
                    sumRecall1 += 1;
                }

                // Recall@10: fraction of the true top-10 neighbours found among the 10 returned.
                truth10.Clear();
                int k = Math.Min(10, _groundTruth[i].Length);
                for (int t = 0; t < k; t++) truth10.Add(_groundTruth[i][t]);

                int hits = 0;
                foreach (var r in results)
                {
                    if (truth10.Contains(r.Id)) hits++;
                }
                sumRecall10 += (double)hits / k;
            }

            double recall1 = sumRecall1 / total;
            double recall10 = sumRecall10 / total;
            Console.WriteLine($"[{DatasetName} M={M}, EfConstruction={EfConstruction}, EfSearch={EfSearch}] Recall@1: {recall1:P2}, Recall@10: {recall10:P2}");
        }
    }
}
