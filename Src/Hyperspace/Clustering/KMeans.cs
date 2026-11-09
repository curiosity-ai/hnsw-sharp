// <copyright file="KMeans.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;
    using System.Threading.Tasks;

    /// <summary>The result of a clustering run.</summary>
    /// <param name="Centroids">The cluster centers.</param>
    /// <param name="Assignments">For every input vector, the index of its cluster (empty for streaming runs).</param>
    /// <param name="Iterations">The number of Lloyd / mini-batch iterations performed.</param>
    /// <param name="Inertia">The sum of squared distances to the assigned centers.</param>
    public sealed record KMeansResult(float[][] Centroids, int[] Assignments, int Iterations, double Inertia);

    /// <summary>K-means clustering: k-means++ seeding, parallel Lloyd iterations with early-abandon distances, and an out-of-core mini-batch variant.</summary>
    public static class KMeans
    {
        /// <summary>Clusters <paramref name="data"/> into <paramref name="k"/> groups.</summary>
        /// <param name="data">The vectors, all of the same dimension.</param>
        /// <param name="k">The number of clusters (clamped to the number of vectors).</param>
        /// <param name="maxIterations">The maximum number of Lloyd iterations.</param>
        /// <param name="tolerance">Stop when the relative drop of the inertia falls below this.</param>
        /// <param name="seed">The random seed.</param>
        public static KMeansResult Fit(IReadOnlyList<float[]> data, int k, int maxIterations = 25, double tolerance = 1e-4, int seed = 42)
        {
            if (data.Count == 0)
            {
                throw new ArgumentException("No data to cluster", nameof(data));
            }

            if (k <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(k));
            }

            k = Math.Min(k, data.Count);
            int dim = data[0].Length;
            var rng = new Random(seed);
            var centroids = SeedPlusPlus(data, k, rng);
            var assignments = new int[data.Count];
            double inertia = double.MaxValue;
            int iterations = 0;

            for (int iter = 0; iter < maxIterations; iter++)
            {
                iterations = iter + 1;
                var dist = new double[data.Count];
                Parallel.For(0, data.Count, i =>
                {
                    assignments[i] = Nearest(data[i], centroids, out dist[i]);
                });

                double newInertia = 0;
                foreach (var d in dist)
                {
                    newInertia += d;
                }

                var sums = new double[k][];
                var counts = new int[k];
                for (int c = 0; c < k; c++)
                {
                    sums[c] = new double[dim];
                }

                for (int i = 0; i < data.Count; i++)
                {
                    int c = assignments[i];
                    counts[c]++;
                    var v = data[i];
                    var s = sums[c];
                    for (int d = 0; d < dim; d++)
                    {
                        s[d] += v[d];
                    }
                }

                for (int c = 0; c < k; c++)
                {
                    if (counts[c] == 0)
                    {
                        // Re-seed an empty cluster on the point farthest from its centre.
                        int far = 0;
                        for (int i = 1; i < dist.Length; i++)
                        {
                            if (dist[i] > dist[far])
                            {
                                far = i;
                            }
                        }

                        centroids[c] = (float[])data[far].Clone();
                        dist[far] = 0;
                        continue;
                    }

                    for (int d = 0; d < dim; d++)
                    {
                        centroids[c][d] = (float)(sums[c][d] / counts[c]);
                    }
                }

                bool converged = inertia < double.MaxValue && (inertia - newInertia) <= tolerance * Math.Max(inertia, 1e-12);
                inertia = newInertia;
                if (converged)
                {
                    break;
                }
            }

            // Final assignment against the last centroids.
            double final = 0;
            for (int i = 0; i < data.Count; i++)
            {
                assignments[i] = Nearest(data[i], centroids, out var d);
                final += d;
            }

            return new KMeansResult(centroids, assignments, iterations, final);
        }

        /// <summary>
        /// Out-of-core mini-batch k-means: the data is read as batches from <paramref name="batches"/> and never held in memory at once.
        /// A first pass draws a reservoir sample of the whole stream to seed the centres (seeding from the first batch alone would put every centre
        /// in whatever region the stream starts in); then each epoch moves every centre towards its batch points with a step of 1 / (points seen).
        /// </summary>
        /// <param name="batches">Produces the stream; it is enumerated <c>epochs + 2</c> times (sampling, training, inertia), so it must be repeatable.</param>
        /// <param name="k">The number of clusters.</param>
        /// <param name="epochs">The number of passes over the data.</param>
        /// <param name="seed">The random seed.</param>
        public static KMeansResult FitStreaming(Func<IEnumerable<IReadOnlyList<float[]>>> batches, int k, int epochs = 3, int seed = 42)
        {
            if (k <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(k));
            }

            var rng = new Random(seed);
            int reservoirSize = Math.Max(k * 50, 1000);
            var reservoir = new List<float[]>(reservoirSize);
            long seenInSample = 0;
            foreach (var batch in batches())
            {
                foreach (var v in batch)
                {
                    seenInSample++;
                    if (reservoir.Count < reservoirSize)
                    {
                        reservoir.Add(v);
                    }
                    else
                    {
                        long slot = rng.NextInt64(seenInSample);
                        if (slot < reservoirSize)
                        {
                            reservoir[(int)slot] = v;
                        }
                    }
                }
            }

            if (reservoir.Count == 0)
            {
                throw new ArgumentException("No data to cluster", nameof(batches));
            }

            var seeds = SeedPlusPlus(reservoir, Math.Min(k, reservoir.Count), rng);
            var centroids = new float[k][];
            for (int c = 0; c < k; c++)
            {
                centroids[c] = (float[])seeds[c % seeds.Length].Clone();
            }

            int dim = centroids[0].Length;
            long[] seen = new long[k];
            int iterations = 0;
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach (var batch in batches())
                {
                    foreach (var v in batch)
                    {
                        int c = Nearest(v, centroids, out _);
                        seen[c]++;
                        float eta = 1f / seen[c];
                        var centre = centroids[c];
                        for (int d = 0; d < dim; d++)
                        {
                            centre[d] += eta * (v[d] - centre[d]);
                        }
                    }

                    iterations++;
                }
            }

            double inertia = 0;
            foreach (var batch in batches())
            {
                foreach (var v in batch)
                {
                    Nearest(v, centroids, out var d);
                    inertia += d;
                }
            }

            return new KMeansResult(centroids, Array.Empty<int>(), iterations, inertia);
        }

        /// <summary>Index of the nearest centroid (squared L2).</summary>
        public static int Nearest(ReadOnlySpan<float> vector, float[][] centroids, out double distance)
        {
            int best = 0;
            double bestDist = double.MaxValue;
            for (int c = 0; c < centroids.Length; c++)
            {
                // Early abandon: stop accumulating once this centroid is already worse than the best.
                var centre = centroids[c];
                double sum = 0;
                int d = 0;
                for (; d + 8 <= vector.Length; d += 8)
                {
                    for (int j = 0; j < 8; j++)
                    {
                        double diff = vector[d + j] - centre[d + j];
                        sum += diff * diff;
                    }

                    if (sum >= bestDist)
                    {
                        break;
                    }
                }

                if (sum < bestDist)
                {
                    for (; d < vector.Length; d++)
                    {
                        double diff = vector[d] - centre[d];
                        sum += diff * diff;
                    }
                }

                if (sum < bestDist)
                {
                    bestDist = sum;
                    best = c;
                }
            }

            distance = bestDist;
            return best;
        }

        private static float[][] SeedPlusPlus(IReadOnlyList<float[]> data, int k, Random rng)
        {
            var centroids = new float[k][];
            centroids[0] = (float[])data[rng.Next(data.Count)].Clone();
            var nearest = new double[data.Count];
            for (int i = 0; i < nearest.Length; i++)
            {
                nearest[i] = Dist(data[i], centroids[0]);
            }

            for (int c = 1; c < k; c++)
            {
                double total = 0;
                foreach (var d in nearest)
                {
                    total += d;
                }

                int pick;
                if (total <= 0)
                {
                    pick = rng.Next(data.Count);
                }
                else
                {
                    double r = rng.NextDouble() * total;
                    pick = 0;
                    double acc = 0;
                    for (int i = 0; i < nearest.Length; i++)
                    {
                        acc += nearest[i];
                        if (acc >= r)
                        {
                            pick = i;
                            break;
                        }
                    }
                }

                centroids[c] = (float[])data[pick].Clone();
                for (int i = 0; i < nearest.Length; i++)
                {
                    nearest[i] = Math.Min(nearest[i], Dist(data[i], centroids[c]));
                }
            }

            return centroids;
        }

        private static double Dist(float[] a, float[] b) => Metrics.L2Squared<float>(a, b);
    }
}
