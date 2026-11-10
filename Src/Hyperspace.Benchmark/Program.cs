// Compares Hyperspace with the HNSW package on identical data: build time, recall@10, queries per second and memory.
//
//   dotnet run -c Release --project Src/Hyperspace.Benchmark -- [--n 20000] [--dim 128] [--queries 500] [--only name,name] [--out results.md]
//
// The data is synthetic (unit-norm Gaussian clusters), so absolute numbers say little about your embeddings; the comparison between rows is the point.
using System.Diagnostics;
using System.Runtime;
using System.Text;
using HNSW.Net;
using Hyperspace;

if (args.Contains("--micro")) { Micro.Run(args); return; }
if (args.Contains("--ab")) { AB.Run(args); return; }
#if DIAG
if (args.Contains("--evals")) { Evals.Run(args); return; }
#endif

int n = 20000, dim = 128, nq = 500, k = 10;
string? only = null, outPath = null;
for (int i = 0; i < args.Length - 1; i++)
{
    switch (args[i])
    {
        case "--n": n = int.Parse(args[i + 1]); break;
        case "--dim": dim = int.Parse(args[i + 1]); break;
        case "--queries": nq = int.Parse(args[i + 1]); break;
        case "--only": only = args[i + 1]; break;
        case "--out": outPath = args[i + 1]; break;
    }
}

const int M = 16, EfConstruction = 100;
int[] efs = { 16, 32, 64, 128, 256 };
int threads = Environment.ProcessorCount;

Console.WriteLine($"n={n} dim={dim} queries={nq} k={k} M={M} efConstruction={EfConstruction} threads={threads} {System.Runtime.InteropServices.RuntimeInformation.OSDescription} {Environment.Version}");

// ---- data: unit vectors drawn around 100 cluster centres, queries from the same distribution
float[][] Generate(int count, int seed, float[][] centres, double noise)
{
    var r = new Random(seed);
    var data = new float[count][];
    for (int i = 0; i < count; i++)
    {
        var c = centres[r.Next(centres.Length)];
        var v = new float[dim];
        for (int d = 0; d < dim; d++) v[d] = c[d] + (float)(Gauss(r) * noise);
        Normalize(v);
        data[i] = v;
    }
    return data;
}
static double Gauss(Random r) => Math.Sqrt(-2 * Math.Log(1 - r.NextDouble())) * Math.Cos(2 * Math.PI * r.NextDouble());
static void Normalize(float[] v) { double s = 0; foreach (var x in v) s += (double)x * x; float inv = (float)(1 / Math.Sqrt(s)); for (int i = 0; i < v.Length; i++) v[i] *= inv; }

var rc = new Random(1);
var centres = Enumerable.Range(0, 100).Select(_ => { var c = new float[dim]; for (int d = 0; d < dim; d++) c[d] = (float)Gauss(rc); Normalize(c); return c; }).ToArray();
double noise = 0.5 / Math.Sqrt(dim) * 1.6;
var data = Generate(n, 2, centres, noise);
var queries = Generate(nq, 3, centres, noise);

// ---- exact ground truth by brute force (cosine on unit vectors = max dot product)
var truth = new int[nq][];
Parallel.For(0, nq, qi =>
{
    var q = queries[qi];
    var best = new PriorityQueue<int, float>();
    for (int i = 0; i < n; i++)
    {
        float dot = 0; var v = data[i]; for (int d = 0; d < dim; d++) dot += q[d] * v[d];
        if (best.Count < k) best.Enqueue(i, dot); else if (best.TryPeek(out _, out var w) && dot > w) { best.Dequeue(); best.Enqueue(i, dot); }
    }
    var ids = new int[k]; int p = k; while (best.TryDequeue(out var id, out _)) ids[--p] = id; truth[qi] = ids;
});

double Recall(Func<float[], int[]> search)
{
    int hit = 0;
    for (int qi = 0; qi < nq; qi++) { var t = new HashSet<int>(truth[qi]); foreach (var id in search(queries[qi])) if (t.Contains(id)) hit++; }
    return hit / (double)(nq * k);
}

long Heap()
{
    GCSettings.LargeObjectHeapCompactionMode = GCLargeObjectHeapCompactionMode.CompactOnce;
    GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
    GC.WaitForPendingFinalizers();
    GC.Collect(2, GCCollectionMode.Forced, blocking: true, compacting: true);
    return GC.GetTotalMemory(true);
}

var table = new StringBuilder();
table.AppendLine($"n={n}, dim={dim}, {nq} queries, k={k}, M={M}, efConstruction={EfConstruction}, {threads} cores. Raw vectors: {n * (long)dim * 4 / 1048576.0:F1} MiB.\n");
table.AppendLine("| Index | Build | Heap | B/vec | ef | Recall@10 | QPS 1 thread | QPS " + threads + " threads |");
table.AppendLine("|---|---:|---:|---:|---:|---:|---:|---:|");

void Report(string name, double buildSeconds, long heapDelta, long extraVectors, Func<int, Func<float[], int[]>> searcherForEf)
{
    long footprint = heapDelta + extraVectors;
    bool first = true;
    foreach (int ef in efs)
    {
        var search = searcherForEf(ef);
        for (int w = 0; w < Math.Min(nq, 200); w++) search(queries[w]);                // warm-up
        double recall = Recall(search);
        var sw = Stopwatch.StartNew();
        int rounds = 0;
        do { for (int qi = 0; qi < nq; qi++) search(queries[qi]); rounds++; } while (sw.Elapsed.TotalSeconds < 1.5);
        double qps1 = rounds * nq / sw.Elapsed.TotalSeconds;
        sw.Restart(); rounds = 0;
        do { Parallel.For(0, nq, new ParallelOptions { MaxDegreeOfParallelism = threads }, qi => search(queries[qi])); rounds++; } while (sw.Elapsed.TotalSeconds < 1.5);
        double qpsN = rounds * nq / sw.Elapsed.TotalSeconds;
        var line = $"| {(first ? name : "")} | {(first ? $"{buildSeconds:F1} s" : "")} | {(first ? $"{footprint / 1048576.0:F1} MiB" : "")} | {(first ? $"{footprint / (double)n:F0}" : "")} | {ef} | {recall:F4} | {qps1:N0} | {qpsN:N0} |";
        Console.WriteLine(line); table.AppendLine(line); first = false;
    }
}

bool Wanted(string name) => only is null || only.Split(',').Any(o => name.Contains(o, StringComparison.OrdinalIgnoreCase));

// ---- HNSW package (items are held by reference, so the heap delta excludes the vectors: add them for a fair footprint)
foreach (var (label, heuristic) in new[] { ("HNSW.Net (SelectSimple)", NeighbourSelectionHeuristic.SelectSimple), ("HNSW.Net (SelectHeuristic)", NeighbourSelectionHeuristic.SelectHeuristic) })
{
    if (!Wanted(label)) continue;
    long before = Heap();
    var sw = Stopwatch.StartNew();
    var parameters = new SmallWorldParameters { M = M, LevelLambda = 1 / Math.Log(M), ConstructionPruning = EfConstruction, EfSearch = 50, NeighbourHeuristic = heuristic };
    var graph = new SmallWorld<float[], float>(CosineDistance.SIMDForUnits, DefaultRandomGenerator.Instance, parameters);
    graph.AddItems(data);
    graph.OptimizeIfNeeded();
    graph.DisableDistanceCache();   // frees the construction-time cache so the heap figure is the steady state
    double build = sw.Elapsed.TotalSeconds;
    long after = Heap();
    Report(label, build, after - before, (long)n * dim * 4, ef => { graph.Parameters.EfSearch = ef; return q => graph.KNNSearch(q, k).Select(r => r.Id).ToArray(); });
    GC.KeepAlive(graph);
}

// ---- Hyperspace configurations (they store their own copy of the vectors, so the heap delta is the whole footprint)
var configs = new List<(string Name, Action<HyperspaceOptions> Configure, bool Pq, int Rerank)>
{
    ("Hyperspace Float32", o => o.Quantization = QuantizationMode.Float32, false, 1),
    ("Hyperspace Float32 (level p=0.5, as HyperspaceDB)", o => { o.Quantization = QuantizationMode.Float32; o.LevelProbability = 0.5; }, false, 1),
    ("Hyperspace Float64 (default)", o => o.Quantization = QuantizationMode.None, false, 1),
    ("Hyperspace Int8", o => o.Quantization = QuantizationMode.ScalarInt8, false, 1),
    ("Hyperspace Int4", o => o.Quantization = QuantizationMode.ScalarInt4, false, 1),
    ("Hyperspace Turbo", o => o.Quantization = QuantizationMode.Turbo, false, 1),
    ("Hyperspace Ternary", o => o.Quantization = QuantizationMode.Ternary, false, 1),
    ("Hyperspace Ternary + rerank x4", o => { o.Quantization = QuantizationMode.Ternary; o.KeepOriginalVectors = true; }, false, 4),
    ("Hyperspace Binary", o => o.Quantization = QuantizationMode.Binary, false, 1),
    ("Hyperspace Binary + rerank x4", o => { o.Quantization = QuantizationMode.Binary; o.KeepOriginalVectors = true; }, false, 4),
    ("Hyperspace PQ", o => o.Quantization = QuantizationMode.ProductQuantization, true, 1),
    ("Hyperspace PQ + rerank x4", o => { o.Quantization = QuantizationMode.ProductQuantization; o.KeepOriginalVectors = true; }, true, 4),
    ("Hyperspace OPQ", o => o.Quantization = QuantizationMode.OPQ, true, 1),
    ("Hyperspace Int8 + rerank x4", o => { o.Quantization = QuantizationMode.ScalarInt8; o.KeepOriginalVectors = true; }, false, 4),
};

foreach (var (name, configure, pq, rerank) in configs)
{
    if (!Wanted(name)) continue;
    long before = Heap();
    var sw = Stopwatch.StartNew();
    var options = new HyperspaceOptions { Dimension = dim, Metric = DistanceMetric.Cosine, M = M, EfConstruction = EfConstruction, EfSearch = 50, Seed = 1, ValidateVectors = false };
    configure(options);
    var index = new HyperspaceIndex(options);
    if (pq) index.TrainProductQuantizer(data.Take(Math.Min(n, 20000)).ToArray(), 8);
    index.AddRange(data);
    index.FreezeFlatLayer0();
    double build = sw.Elapsed.TotalSeconds;
    long after = Heap();
    Report(name, build, after - before, 0, ef =>
    {
        var so = new SearchOptions { Ef = ef, RerankOversample = rerank };
        return q => { var hits = index.Search(q, k, so); var ids = new int[hits.Length]; for (int i = 0; i < ids.Length; i++) ids[i] = hits[i].Id; return ids; };
    });
    GC.KeepAlive(index);
}

if (outPath is not null) File.WriteAllText(outPath, table.ToString());

// ---- --micro: cost of one distance evaluation per storage mode (the traversal around it is not measured)
static class Micro
{
    public static void Run(string[] args)
    {
        int dim = 128, count = 4096;
        for (int i = 0; i < args.Length - 1; i++) if (args[i] == "--dim") dim = int.Parse(args[i + 1]);
        var rnd = new Random(1);
        float[] Unit() { var v = new float[dim]; double s = 0; for (int d = 0; d < dim; d++) { v[d] = (float)(Math.Sqrt(-2 * Math.Log(1 - rnd.NextDouble())) * Math.Cos(2 * Math.PI * rnd.NextDouble())); s += v[d] * v[d]; } for (int d = 0; d < dim; d++) v[d] /= (float)Math.Sqrt(s); return v; }
        var vectors = Enumerable.Range(0, count).Select(_ => Unit()).ToArray();
        var query = Unit();
        // The traversal around a distance: one priority-queue push + pop at the size a search keeps (ef = 64), as the search does for every candidate.
        {
            var queue = new PriorityQueue<int, double>();
            for (int i = 0; i < 64; i++) queue.Enqueue(i, rnd.NextDouble());
            var keys = Enumerable.Range(0, 4096).Select(_ => rnd.NextDouble()).ToArray();
            double sum = 0;
            for (int w = 0; w < 200000; w++) { queue.Enqueue(w, keys[w & 4095]); queue.TryDequeue(out _, out var pr); sum += pr; }
            var clock = Stopwatch.StartNew(); long ops = 0;
            while (clock.Elapsed.TotalMilliseconds < 400) { for (int w = 0; w < 100000; w++) { queue.Enqueue(w, keys[w & 4095]); queue.TryDequeue(out _, out var pr); sum += pr; } ops += 100000; }
            Console.WriteLine($"PriorityQueue<int, double> push + pop at size 64: {clock.Elapsed.TotalMilliseconds * 1e6 / ops:F0} ns  ({sum > 0})\n");
        }

        Console.WriteLine($"dim={dim}: ns per Distance call over {count} stored vectors (cache-resident), and the decode-then-metric reference");
        Console.WriteLine("| mode | stored bytes | query-to-vector ns | decode + metric ns | vector-to-vector ns (graph construction) |");
        Console.WriteLine("|---|---:|---:|---:|---:|");
        foreach (var mode in new[] { QuantizationMode.Float32, QuantizationMode.ScalarInt8, QuantizationMode.ScalarInt4, QuantizationMode.Turbo, QuantizationMode.Ternary, QuantizationMode.Binary, QuantizationMode.ProductQuantization, QuantizationMode.OPQ })
        {
            var o = new HyperspaceOptions { Dimension = dim, Metric = DistanceMetric.Cosine, Quantization = mode, NormalizeVectors = false, ValidateVectors = false };
            var codec = VectorCodec.Create(o);
            if (codec is ProductQuantizationCodec pqCodec)
            {
                pqCodec.Quantizer.Train(vectors, 4);
            }

            var stored = vectors.Select(v => codec.Encode(v)).ToArray();
            var prepared = codec.Prepare(query);
            var decoded = new float[dim];
            double sink = 0;
            double Time(Func<byte[], double> f)
            {
                for (int w = 0; w < 3; w++) foreach (var s in stored) sink += f(s);
                var sw = Stopwatch.StartNew(); int calls = 0;
                while (sw.Elapsed.TotalMilliseconds < 400) { foreach (var s in stored) sink += f(s); calls += stored.Length; }
                return sw.Elapsed.TotalMilliseconds * 1e6 / calls;
            }
            double fast = Time(s => codec.Distance(s, prepared));
            double reference = Time(s => { codec.Decode(s, decoded); return Metrics.Distance(DistanceMetric.Cosine, (ReadOnlySpan<float>)decoded, query); });
            int partner = 0;
            double pair = Time(s => codec.PairDistance(s, stored[partner++ & (stored.Length - 1)]));
            Console.WriteLine($"| {mode} | {codec.EncodedLength} | {fast:F0} | {reference:F0} | {pair:F0} |");
            GC.KeepAlive(sink);
        }
    }
}

#if DIAG
// ---- --evals (build with -p:DefineConstants=DIAG): distance evaluations per query, to separate traversal length from the cost of one distance
static class Evals
{
    public static void Run(string[] args)
    {
        int n = 20000, dim = 128;
        string? modes = null;
        for (int i = 0; i < args.Length - 1; i++)
        {
            if (args[i] == "--n") n = int.Parse(args[i + 1]);
            if (args[i] == "--only") modes = args[i + 1];
        }

        var rnd = new Random(1);
        double G() => Math.Sqrt(-2 * Math.Log(1 - rnd.NextDouble())) * Math.Cos(2 * Math.PI * rnd.NextDouble());
        var centres = Enumerable.Range(0, 100).Select(_ => Enumerable.Range(0, dim).Select(__ => (float)G()).ToArray()).ToArray();
        float[][] Gen(int c) => Enumerable.Range(0, c).Select(_ => { var cc = centres[rnd.Next(100)]; var v = cc.Select(x => x + (float)G() * 0.15f).ToArray(); var s = Math.Sqrt(v.Sum(x => (double)x * x)); return v.Select(x => (float)(x / s)).ToArray(); }).ToArray();
        var data = Gen(n); var queries = Gen(500);
        Console.WriteLine("| mode | ef | evaluations / query | us / query | ns / evaluation |");
        Console.WriteLine("|---|---:|---:|---:|---:|");
        Console.WriteLine($"n = {n}");
        foreach (var mode in new[] { QuantizationMode.Float32, QuantizationMode.ScalarInt8, QuantizationMode.ScalarInt4, QuantizationMode.Turbo, QuantizationMode.Ternary, QuantizationMode.Binary })
        {
            if (modes is not null && !modes.Split(',').Contains(mode.ToString())) continue;
            var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = dim, Metric = DistanceMetric.Cosine, Quantization = mode, M = 16, EfConstruction = 100, Seed = 1, ValidateVectors = false });
            index.AddRange(data);
            index.FreezeFlatLayer0();
            var results = new SearchResult[10];
            foreach (int ef in new[] { 64 })
            {
                var so = new SearchOptions { Ef = ef };
                for (int w = 0; w < 100; w++) index.Search(queries[w], results, so);
                HyperspaceIndex.DiagEvaluations = 0;
                for (int q = 0; q < queries.Length; q++) index.Search(queries[q], results, so);
                double evals = HyperspaceIndex.DiagEvaluations / (double)queries.Length;

                // The least-disturbed of several rounds is the best estimate of what the code costs; the median shows how noisy the machine was.
                var rounds = new List<double>();
                for (int r = 0; r < 9; r++)
                {
                    var sw = Stopwatch.StartNew();
                    for (int q = 0; q < queries.Length; q++) index.Search(queries[q], results, so);
                    rounds.Add(sw.Elapsed.TotalMilliseconds * 1000 / queries.Length);
                }

                rounds.Sort();
                double us = rounds[0], median = rounds[rounds.Count / 2];
                Console.WriteLine($"| {mode} | {ef} | {evals:F0} | {us:F0} (median {median:F0}) | {us * 1000 / evals:F0} |");
            }
        }
    }
}
#endif

// ---- --ab: alternate two settings of the same index within one process, so machine noise hits both equally
static class AB
{
    public static void Run(string[] args)
    {
        int n = 20000, dim = 128;
        for (int i = 0; i < args.Length - 1; i++) { if (args[i] == "--n") n = int.Parse(args[i + 1]); if (args[i] == "--dim") dim = int.Parse(args[i + 1]); }
        var rnd = new Random(1);
        double G() => Math.Sqrt(-2 * Math.Log(1 - rnd.NextDouble())) * Math.Cos(2 * Math.PI * rnd.NextDouble());
        var centres = Enumerable.Range(0, 100).Select(_ => Enumerable.Range(0, dim).Select(__ => (float)G()).ToArray()).ToArray();
        float[][] Gen(int c) => Enumerable.Range(0, c).Select(_ => { var cc = centres[rnd.Next(100)]; var v = cc.Select(x => x + (float)G() * 0.15f).ToArray(); var s = Math.Sqrt(v.Sum(x => (double)x * x)); return v.Select(x => (float)(x / s)).ToArray(); }).ToArray();
        var data = Gen(n); var queries = Gen(500);
        var index = new HyperspaceIndex(new HyperspaceOptions { Dimension = dim, Metric = DistanceMetric.Cosine, Quantization = QuantizationMode.Float32, M = 16, EfConstruction = 100, Seed = 1, ValidateVectors = false });
        index.AddRange(data);
        index.FreezeFlatLayer0();
        var results = new SearchResult[10];
        var so = new SearchOptions { Ef = 64 };
        var settings = new[] { 0, 4 };
        var times = settings.ToDictionary(x => x, _ => new List<double>());
        for (int round = 0; round < 25; round++)
        {
            foreach (int lines in settings)
            {
                HyperspaceIndex.PrefetchLines = lines;
                var sw = Stopwatch.StartNew();
                for (int q = 0; q < queries.Length; q++) index.Search(queries[q], results, so);
                times[lines].Add(sw.Elapsed.TotalMilliseconds * 1000 / queries.Length);
            }
        }

        Console.WriteLine($"n = {n}, dim = {dim}, ef = 64, Float32; us per query over 25 interleaved rounds");
        Console.WriteLine("| prefetch lines (0 = original single-pass loop) | min | median | p25 |");
        Console.WriteLine("|---:|---:|---:|---:|");
        foreach (int lines in settings) { var t = times[lines]; t.Sort(); Console.WriteLine($"| {lines} | {t[0]:F1} | {t[t.Count / 2]:F1} | {t[t.Count / 4]:F1} |"); }
    }
}
