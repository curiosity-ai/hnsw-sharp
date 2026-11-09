// <copyright file="MemoryReconsolidator.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// "AI sleep mode": nudges vectors towards an attractor with a Riemannian SGD step and keeps them inside the Poincare ball.
    /// </summary>
    public sealed class MemoryReconsolidator
    {
        /// <summary>Initializes a new instance of the <see cref="MemoryReconsolidator"/> class.</summary>
        /// <param name="learningRate">The fraction of the distance to the target covered by one step, in (0, 1].</param>
        public MemoryReconsolidator(double learningRate)
        {
            if (learningRate <= 0)
            {
                throw new ArgumentOutOfRangeException(nameof(learningRate));
            }

            LearningRate = learningRate;
        }

        /// <summary>Gets the step size.</summary>
        public double LearningRate { get; }

        /// <summary>Moves <paramref name="source"/> towards <paramref name="target"/> in place, projecting back inside the ball if needed.</summary>
        public void StepTowardsPoincareInPlace(Span<double> source, ReadOnlySpan<double> target)
        {
            int len = Math.Min(source.Length, target.Length);
            double normSq = 0;
            for (int i = 0; i < len; i++)
            {
                source[i] += LearningRate * (target[i] - source[i]);
                normSq += source[i] * source[i];
            }

            if (normSq >= 1.0)
            {
                double scale = 0.999_999 / Math.Sqrt(normSq);
                for (int i = 0; i < len; i++)
                {
                    source[i] *= scale;
                }
            }
        }

        /// <summary>Returns a copy of <paramref name="source"/> moved one step towards <paramref name="target"/>.</summary>
        public double[] StepTowardsPoincare(ReadOnlySpan<double> source, ReadOnlySpan<double> target)
        {
            var copy = source.ToArray();
            StepTowardsPoincareInPlace(copy, target);
            return copy;
        }

        /// <summary>Moves every vector one step towards the target, in place.</summary>
        public void Reconsolidate(IReadOnlyList<double[]> vectors, ReadOnlySpan<double> target)
        {
            foreach (var v in vectors)
            {
                StepTowardsPoincareInPlace(v, target);
            }
        }
    }
}
