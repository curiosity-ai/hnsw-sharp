using System;
using HNSW.Net;
using System.Linq;

class Program
{
    static void Main()
    {
        int dim = 128;
        var tq = new TurboQuant(dim, 42);
        
        var rnd = new Random(42);
        
        float totalDiff = 0;
        int trials = 1000;
        float trueSum = 0;
        float approxSum = 0;

        for (int t = 0; t < trials; t++)
        {
            float[] v1 = new float[dim];
            float[] v2 = new float[dim];
            for (int i=0; i<dim; i++)
            {
                v1[i] = (float)rnd.NextDouble() * 100f;
                v2[i] = (float)rnd.NextDouble() * 100f;
            }
            float l2True = 0;
            for (int i=0; i<dim; i++)
            {
                l2True += (v1[i] - v2[i]) * (v1[i] - v2[i]);
            }
            var e1 = tq.Encode(v1);
            var e2 = tq.Encode(v2);
            var dist = new TurboQuantDistance(tq);
            float l2Approx = dist.GetDistance(e1, e2);
            trueSum += l2True;
            approxSum += l2Approx;
            totalDiff += (l2True - l2Approx) * (l2True - l2Approx);
        }
        Console.WriteLine($"RMSE: {Math.Sqrt(totalDiff/trials)}, Avg True: {trueSum/trials}, Avg Approx: {approxSum/trials}");
    }
}
