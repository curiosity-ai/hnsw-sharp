using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using BenchmarkDotNet.Attributes;
using BenchmarkDotNet.Columns;
using BenchmarkDotNet.Configs;
using BenchmarkDotNet.Reports;
using BenchmarkDotNet.Running;
using HNSW.Net;

namespace HNSW.Net.SiftBenchmark
{
    public class SiftBenchmarkTurboQuant
    {
        private static float[][] _baseVectors;
        private static float[][] _queryVectors;
        private static int[][] _groundTruth;

        private static SmallWorldTurboQuant _cachedGraph;
        private static (int M, int EfConstruction) _cachedGraphParams;

        private SmallWorldTurboQuant _graph;

        [Params(32)]
        public int M { get; set; }

        [Params(200)]
        public int EfConstruction { get; set; }

        [Params(50, 100, 200)]
        public int EfSearch { get; set; }

        public bool QuickTest { get; set; } = false;

        [GlobalSetup]
        public void Setup()
        {
            string workingDir = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "data");
            if (!Directory.Exists(workingDir)) Directory.CreateDirectory(workingDir);

            Dataset.DownloadAndExtractAsync(workingDir).GetAwaiter().GetResult();

            string siftDir = Path.Combine(workingDir, "sift");

            if (_baseVectors == null)
            {
                Console.WriteLine("Reading dataset...");
                _baseVectors = Dataset.ReadFvecs(Path.Combine(siftDir, "sift_base.fvecs"));
                _queryVectors = Dataset.ReadFvecs(Path.Combine(siftDir, "sift_query.fvecs"));
                _groundTruth = Dataset.ReadIvecs(Path.Combine(siftDir, "sift_groundtruth.ivecs"));

                if (QuickTest)
                {
                    _queryVectors = _queryVectors.Take(100).ToArray();
                    _groundTruth  = _groundTruth.Take(100).ToArray();
                    var indexesToKeep = _groundTruth.SelectMany(i => i).ToHashSet();
                    var newData = _baseVectors.Select((d, i) => (d, i)).Where(v => indexesToKeep.Contains(v.i)).ToDictionary(v => v.i, v => v.d);
                    _baseVectors = newData.OrderBy(kv => kv.Key).Select(kv => kv.Value).ToArray();
                    var indexMap = newData.OrderBy(kv => kv.Key).Select((kv, i) => (original: kv.Key, newIndex: i)).ToDictionary(d => d.original, d => d.newIndex);
                    _groundTruth = _groundTruth.Select(d => d.Select(i => indexMap[i]).ToArray()).ToArray();
                }

                Console.WriteLine($"Loaded {_baseVectors.Length} base vectors");
                Console.WriteLine($"Loaded {_queryVectors.Length} query vectors");

                Console.WriteLine("Normalizing dataset for TurboQuant...");
                for (int i = 0; i < _baseVectors.Length; i++)
                {
                    float norm = 0;
                    for (int j = 0; j < _baseVectors[i].Length; j++)
                    {
                        norm += _baseVectors[i][j] * _baseVectors[i][j];
                    }
                    norm = (float)Math.Sqrt(norm);
                    if (norm > 0)
                    {
                        for (int j = 0; j < _baseVectors[i].Length; j++)
                        {
                            _baseVectors[i][j] /= norm;
                        }
                    }
                }
                for (int i = 0; i < _queryVectors.Length; i++)
                {
                    float norm = 0;
                    for (int j = 0; j < _queryVectors[i].Length; j++)
                    {
                        norm += _queryVectors[i][j] * _queryVectors[i][j];
                    }
                    norm = (float)Math.Sqrt(norm);
                    if (norm > 0)
                    {
                        for (int j = 0; j < _queryVectors[i].Length; j++)
                        {
                            _queryVectors[i][j] /= norm;
                        }
                    }
                }
            }

            if (_cachedGraph == null || _cachedGraphParams.M != M || _cachedGraphParams.EfConstruction != EfConstruction)
            {
                var parameters = new SmallWorldParameters
                {
                    M = M,
                    LevelLambda = 1 / Math.Log(M),
                    ConstructionPruning = EfConstruction,
                    EfSearch = EfSearch,
                    EnableDistanceCacheForConstruction = true
                };

                Console.WriteLine($"Building TurboQuant graph with M={M}, EfConstruction={EfConstruction}...");
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var quantizer = TurboQuant.Create(_baseVectors[0].Length, DefaultRandomGenerator.Instance);
                var graph = new SmallWorldTurboQuant(quantizer, DefaultRandomGenerator.Instance, parameters, false);
                graph.AddItems(_baseVectors, new ConsoleProgress());
                sw.Stop();
                Console.WriteLine($"TurboQuant Graph built in {sw.Elapsed.TotalSeconds:N2}s.");

                _cachedGraph = graph;
                _cachedGraphParams = (M, EfConstruction);
            }

            _graph = _cachedGraph;
            _graph.Parameters.EfSearch = EfSearch;
        }

        [Benchmark]
        public void Search()
        {
            for (int i = 0; i < _queryVectors.Length; i++)
            {
                _graph.KNNSearch(_queryVectors[i], 1);
            }
        }

        [IterationCleanup]
        public void Cleanup()
        {
            int correct1 = 0;
            int correct10 = 0;
            int total = _queryVectors.Length;

            for (int i = 0; i < total; i++)
            {
                var results1 = _graph.KNNSearch(_queryVectors[i], 1);
                if (results1.Count > 0 && results1[0].Id == _groundTruth[i][0])
                {
                    correct1++;
                }

                var results10 = _graph.KNNSearch(_queryVectors[i], 10);
                if (results10.Any(r => r.Id == _groundTruth[i][0]))
                {
                    correct10++;
                }
            }

            double recall1  = (double)correct1 / total;
            double recall10 = (double)correct10 / total;
            Console.WriteLine($"[TurboQuant] [Configuration M={M}, EfConstruction={EfConstruction}, EfSearch={EfSearch}] Recall@1: {recall1:P2}, Recall@10: {recall10:P2}");
        }
    }
}
