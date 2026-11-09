namespace Hyperspace.Tests
{
    using System;
    using System.Collections.Generic;
    using System.Linq;

    internal static class TestData
    {
        public static float Gaussian(Random r)
        {
            double u1 = 1.0 - r.NextDouble(), u2 = r.NextDouble();
            return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }

        /// <summary>Gaussian clusters, the shape of real embeddings: neighbours exist, unlike in uniform noise.</summary>
        public static float[][] Clustered(int count, int dim, int seed, int clusters = 12)
        {
            var r = new Random(seed);
            var centers = Enumerable.Range(0, clusters).Select(_ => Enumerable.Range(0, dim).Select(__ => Gaussian(r) * 2f).ToArray()).ToArray();
            var data = new float[count][];
            for (int i = 0; i < count; i++)
            {
                var c = centers[r.Next(clusters)];
                data[i] = new float[dim];
                for (int d = 0; d < dim; d++)
                {
                    data[i][d] = c[d] + (Gaussian(r) * 0.6f);
                }
            }

            return data;
        }

        public static float[][] Poincare(int count, int dim, int seed)
        {
            var r = new Random(seed);
            var data = new float[count][];
            for (int i = 0; i < count; i++)
            {
                var v = Enumerable.Range(0, dim).Select(_ => Gaussian(r)).ToArray();
                double norm = Math.Sqrt(v.Sum(x => (double)x * x));
                double radius = 0.05 + (0.85 * r.NextDouble());
                data[i] = v.Select(x => (float)(x / norm * radius)).ToArray();
            }

            return data;
        }

        public static float[][] Lorentz(int count, int dim, int seed)
        {
            var r = new Random(seed);
            var data = new float[count][];
            for (int i = 0; i < count; i++)
            {
                var spatial = Enumerable.Range(0, dim - 1).Select(_ => Gaussian(r) * 0.6).ToArray();
                var x = new float[dim];
                x[0] = (float)Math.Sqrt(1.0 + spatial.Sum(s => s * s));
                for (int d = 0; d < spatial.Length; d++)
                {
                    x[d + 1] = (float)spatial[d];
                }

                data[i] = x;
            }

            return data;
        }

        public static float[][] Hybrid(int count, int dim, int seed)
        {
            var lorentz = Lorentz(count, Metrics.HybridLorentzDimensions, seed);
            var euclid = Clustered(count, dim - Metrics.HybridLorentzDimensions, seed + 1);
            return lorentz.Select((l, i) => l.Concat(euclid[i].Select(x => x * 0.3f)).ToArray()).ToArray();
        }

        /// <summary>Exact nearest neighbours by brute force, in the metric of the index.</summary>
        public static int[] Exact(float[][] data, float[] query, int k, DistanceMetric metric, bool normalize = false)
        {
            float[] q = query;
            if (normalize)
            {
                q = (float[])query.Clone();
                Metrics.Normalize(q);
            }

            return Enumerable.Range(0, data.Length)
                .Select(i =>
                {
                    float[] v = data[i];
                    if (normalize)
                    {
                        v = (float[])v.Clone();
                        Metrics.Normalize(v);
                    }

                    return (i, d: Metrics.Distance(metric, (ReadOnlySpan<float>)q, v));
                })
                .OrderBy(x => x.d)
                .Take(k)
                .Select(x => x.i)
                .ToArray();
        }

        public static double Recall(HyperspaceIndex index, float[][] data, float[][] queries, int k, DistanceMetric metric, bool normalize, SearchOptions? options = null)
        {
            int hit = 0;
            foreach (var q in queries)
            {
                var truth = new HashSet<int>(Exact(data, q, k, metric, normalize));
                hit += index.Search(q, k, options).Count(r => truth.Contains(r.Id));
            }

            return hit / (double)(queries.Length * k);
        }

        public static HyperspaceIndex Build(float[][] data, HyperspaceOptions options)
        {
            var index = new HyperspaceIndex(options);
            index.AddRange(data);
            return index;
        }
    }
}
