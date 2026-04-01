using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Numerics;
using HNSW.Net;

class Program
{
    static void Main()
    {
        int dim = 128;
        var rnd = new Random(42);
        
        var tq = TurboQuant.Create(dim, DefaultRandomGenerator.Instance, bits: 3, residualProjections: 128);

        var v1 = new float[dim];
        for (int i=0; i<dim; i++) v1[i] = (float)rnd.NextDouble() * 100f;
        var e1 = tq.Encode(v1);

        Console.WriteLine($"Norm: {e1.Norm}");
    }
}
