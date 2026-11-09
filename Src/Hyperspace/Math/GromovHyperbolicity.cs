// <copyright file="GromovHyperbolicity.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;

    /// <summary>The result of <see cref="GromovHyperbolicity.Analyze"/>.</summary>
    /// <param name="Delta">The largest sampled Gromov delta (small values mean tree-like geometry).</param>
    /// <param name="RecommendedMetric">The metric that best suits the data.</param>
    public readonly record struct GeometryRecommendation(double Delta, DistanceMetric RecommendedMetric);

    /// <summary>Estimates how hyperbolic a dataset is using Gromov's four-point condition, to pick a metric.</summary>
    public static class GromovHyperbolicity
    {
        /// <summary>
        /// Samples random 4-tuples and keeps the largest half-gap between the two largest pair-sums; recommends Lorentz for
        /// delta below 0.15, Poincare below 0.30, otherwise Cosine for unit-norm data and L2 for the rest.
        /// </summary>
        public static GeometryRecommendation Analyze(IReadOnlyList<double[]> vectors, int numSamples = 1000, Random? random = null)
        {
            if (vectors.Count < 4)
            {
                return new GeometryRecommendation(0.0, DistanceMetric.L2);
            }

            random ??= new Random(42);
            double maxDelta = 0;
            for (int s = 0; s < numSamples; s++)
            {
                int i = random.Next(vectors.Count), j = random.Next(vectors.Count), k = random.Next(vectors.Count), l = random.Next(vectors.Count);
                if (i == j || i == k || i == l || j == k || j == l || k == l)
                {
                    continue;
                }

                double s1 = Dist(vectors[i], vectors[j]) + Dist(vectors[k], vectors[l]);
                double s2 = Dist(vectors[i], vectors[k]) + Dist(vectors[j], vectors[l]);
                double s3 = Dist(vectors[i], vectors[l]) + Dist(vectors[j], vectors[k]);

                double first, second;
                if (s1 >= s2)
                {
                    if (s1 >= s3)
                    {
                        (first, second) = (s1, Math.Max(s2, s3));
                    }
                    else
                    {
                        (first, second) = (s3, s1);
                    }
                }
                else if (s2 >= s3)
                {
                    (first, second) = (s2, Math.Max(s1, s3));
                }
                else
                {
                    (first, second) = (s3, s2);
                }

                double delta = (first - second) * 0.5;
                if (delta > maxDelta)
                {
                    maxDelta = delta;
                }
            }

            bool normalized = true;
            for (int i = 0; i < Math.Min(20, vectors.Count); i++)
            {
                double n = 0;
                foreach (var x in vectors[i])
                {
                    n += x * x;
                }

                if (Math.Abs(Math.Sqrt(n) - 1.0) > 1e-2)
                {
                    normalized = false;
                    break;
                }
            }

            var metric = maxDelta < 0.15 ? DistanceMetric.Lorentz
                       : maxDelta < 0.30 ? DistanceMetric.Poincare
                       : normalized ? DistanceMetric.Cosine
                       : DistanceMetric.L2;
            return new GeometryRecommendation(maxDelta, metric);
        }

        private static double Dist(double[] a, double[] b) => Math.Sqrt(Metrics.L2Squared<double>(a, b));
    }
}
