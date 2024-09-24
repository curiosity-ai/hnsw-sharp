using HNSW.Net;
using MessagePack;
using System.Diagnostics;

int SampleSize = 10_000;
int SampleIncrSize = 100;
int Dimensions = 8;

var world = new SmallWorld<VectorID, float>(VectorID.Distance, DefaultRandomGenerator.Instance, new () { EnableDistanceCacheForConstruction = true, InitialDistanceCacheSize = SampleSize, NeighbourHeuristic = NeighbourSelectionHeuristic.SelectHeuristic, KeepPrunedConnections = true, ExpandBestSelection = true }, threadSafe: false);

Console.WriteLine($"Creating {SampleSize:n0} vectors");
List<VectorID> vectors;
var fileName = Path.Combine(Directory.GetCurrentDirectory(), $"data-{Dimensions}-{SampleSize}.bin");
if (File.Exists(fileName))
{
    Console.WriteLine("Loading HNSW graph... ");
    using (var f = File.OpenRead(fileName.Replace(".bin", ".vec")))
    {
        vectors = MessagePackSerializer.Deserialize<List<VectorID>>(f);
    }

    using (var f = File.OpenRead(fileName))
    {
        world = SmallWorld<VectorID, float>.DeserializeGraph(vectors, VectorID.Distance, DefaultRandomGenerator.Instance, f, false);
    }
}
else
{
    vectors = RandomVectors(Dimensions, SampleSize);
    Console.WriteLine("Building HNSW graph... ");
    //using (var listener = new HNSW.Net.Demo.MetricsEventListener(EventSources.GraphBuildEventSource.Instance))
    {
        var clock = Stopwatch.StartNew();
        for (int i = 0; i < (SampleSize / SampleIncrSize); i++)
        {
            clock.Restart();
            world.AddItems(vectors.Skip(i * SampleIncrSize).Take(SampleIncrSize).ToArray());
            Console.WriteLine($"Indexing vectors, at {(i + 1) * SampleIncrSize:n0} of {SampleSize:n0}, at {(double)clock.ElapsedMilliseconds / SampleIncrSize:n1}ms/vector");
        }
        Console.WriteLine("Done building HNSW graph");
    }

    using (var f = File.OpenWrite(fileName.Replace(".bin", ".vec")))
    {
        MessagePackSerializer.Serialize(f, vectors);
    }

    using (var f = File.OpenWrite(fileName))
    {
        world.SerializeGraph(f);
    }
}

var times = new TimeSpan[11];
int sample = 100;

for (int repeat = 0; repeat < 2; repeat++)
{
    Console.WriteLine($"----------------------------\nRun: {repeat}\n----------------------------");
    for (int p = 0; p <= 10; p++)
    {
        Console.Write($"Testing {p * 10}% out... ");
        var sw = Stopwatch.StartNew();
        foreach (var v in vectors.Take(sample))
        {
            using (var cts = new CancellationTokenSource())
            {
                //cts.CancelAfter(TimeSpan.FromMilliseconds(1));
                var results = world.KNNSearch(v, 50, v => v.ID % 100 < (100 - p * 10), cts.Token);
            }
        }
        sw.Stop();
        times[p] = sw.Elapsed;
        Console.WriteLine($"{sw.Elapsed.TotalMilliseconds / sample:n2}ms / call");
    }

    Console.WriteLine();
}
Console.WriteLine($"----------------------------\nResults\n----------------------------");
Console.WriteLine(string.Join("\n", times.Select((t, i) => $"Exclude {i*10}%\t{t.TotalMilliseconds / sample:n2}ms / call")));


List<VectorID> RandomVectors(int vectorSize, int vectorsCount)
{
    var vectors = new List<VectorID>();

    for (int i = 0; i < vectorsCount; i++)
    {
        var vector = new float[vectorSize];
        DefaultRandomGenerator.Instance.NextFloats(vector);
        VectorUtils.NormalizeSIMD(vector);
        vectors.Add(new VectorID(vector, i));
    }

    return vectors;
}

[MessagePackObject]
public struct VectorID
{
    [Key(0)] public float[] Vector;
    [Key(1)] public int ID;

    public VectorID(float[] vector, int iD)
    {
        Vector = vector;
        ID = iD;
    }

    internal static float Distance(VectorID a, VectorID b)
    {
        return CosineDistance.SIMDForUnits(a.Vector, b.Vector);
    }
}
