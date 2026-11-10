// <copyright file="FuzzyLogic.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;

    /// <summary>A triangular norm: fuzzy intersection.</summary>
    public enum TNorm
    {
        /// <summary>Godel: <c>min(a, b)</c>.</summary>
        Min,

        /// <summary>Product: <c>a * b</c>.</summary>
        Product,

        /// <summary>Lukasiewicz: <c>max(a + b - 1, 0)</c>.</summary>
        Lukasiewicz,
    }

    /// <summary>A triangular conorm: fuzzy union.</summary>
    public enum TConorm
    {
        /// <summary>Godel: <c>max(a, b)</c>.</summary>
        Max,

        /// <summary>Probabilistic: <c>a + b - a * b</c>.</summary>
        Probabilistic,

        /// <summary>Lukasiewicz: <c>min(a + b, 1)</c>.</summary>
        Lukasiewicz,
    }

    /// <summary>Fuzzy set operators (FuzzQE, Chen et al., AAAI 2022).</summary>
    public static class FuzzyOps
    {
        /// <summary>Applies a t-norm.</summary>
        public static float Apply(this TNorm norm, float a, float b) => norm switch
        {
            TNorm.Min => Math.Min(a, b),
            TNorm.Product => a * b,
            TNorm.Lukasiewicz => Math.Max(a + b - 1f, 0f),
            _ => throw new ArgumentOutOfRangeException(nameof(norm)),
        };

        /// <summary>Applies a t-conorm.</summary>
        public static float Apply(this TConorm conorm, float a, float b) => conorm switch
        {
            TConorm.Max => Math.Max(a, b),
            TConorm.Probabilistic => a + b - (a * b),
            TConorm.Lukasiewicz => Math.Min(a + b, 1f),
            _ => throw new ArgumentOutOfRangeException(nameof(conorm)),
        };

        /// <summary>The De Morgan dual conorm of a t-norm.</summary>
        public static TConorm Dual(this TNorm norm) => norm switch
        {
            TNorm.Min => TConorm.Max,
            TNorm.Product => TConorm.Probabilistic,
            _ => TConorm.Lukasiewicz,
        };

        /// <summary>The De Morgan dual norm of a t-conorm.</summary>
        public static TNorm Dual(this TConorm conorm) => conorm switch
        {
            TConorm.Max => TNorm.Min,
            TConorm.Probabilistic => TNorm.Product,
            _ => TNorm.Lukasiewicz,
        };

        /// <summary>Standard fuzzy negation <c>1 - a</c>.</summary>
        public static float Negate(float a) => 1f - a;
    }

    /// <summary>A logical query over the vector space: leaves are vectors, nodes are fuzzy AND / OR / NOT.</summary>
    public abstract class FuzzyQuery
    {
        /// <summary>A single target vector.</summary>
        public static FuzzyQuery Vector(float[] vector) => new VectorQuery(vector);

        /// <summary>Fuzzy intersection of two queries.</summary>
        public static FuzzyQuery And(FuzzyQuery left, FuzzyQuery right, TNorm norm = TNorm.Product) => new AndQuery(left, right, norm);

        /// <summary>Fuzzy union of two queries.</summary>
        public static FuzzyQuery Or(FuzzyQuery left, FuzzyQuery right, TConorm conorm = TConorm.Probabilistic) => new OrQuery(left, right, conorm);

        /// <summary>Fuzzy complement.</summary>
        public static FuzzyQuery Not(FuzzyQuery inner) => new NotQuery(inner);

        /// <summary>Collects the leaf vectors in evaluation order.</summary>
        public abstract void ExtractVectors(List<float[]> output);

        /// <summary>Gets a value indicating whether the query contains a <c>NOT</c>, in which case items far from every vector can match.</summary>
        public abstract bool HasNegation { get; }

        /// <summary>
        /// Evaluates the query for one item, giving a membership in [0, 1]. <paramref name="distance"/> returns the distance from the item to
        /// the leaf vector with the given index (evaluation order); a leaf's membership is <c>exp(-distance)</c>.
        /// </summary>
        public abstract float Evaluate(ref int counter, Func<int, float> distance);

        private sealed class VectorQuery : FuzzyQuery
        {
            private readonly float[] _vector;

            public VectorQuery(float[] vector)
            {
                _vector = vector;
            }

            public override bool HasNegation => false;

            public override void ExtractVectors(List<float[]> output) => output.Add(_vector);

            public override float Evaluate(ref int counter, Func<int, float> distance) => MathF.Exp(-distance(counter++));
        }

        private sealed class AndQuery : FuzzyQuery
        {
            private readonly FuzzyQuery _l, _r;
            private readonly TNorm _norm;

            public AndQuery(FuzzyQuery l, FuzzyQuery r, TNorm norm)
            {
                _l = l;
                _r = r;
                _norm = norm;
            }

            public override bool HasNegation => _l.HasNegation || _r.HasNegation;

            public override void ExtractVectors(List<float[]> output)
            {
                _l.ExtractVectors(output);
                _r.ExtractVectors(output);
            }

            public override float Evaluate(ref int counter, Func<int, float> distance)
            {
                float a = _l.Evaluate(ref counter, distance);
                float b = _r.Evaluate(ref counter, distance);
                return _norm.Apply(a, b);
            }
        }

        private sealed class OrQuery : FuzzyQuery
        {
            private readonly FuzzyQuery _l, _r;
            private readonly TConorm _conorm;

            public OrQuery(FuzzyQuery l, FuzzyQuery r, TConorm conorm)
            {
                _l = l;
                _r = r;
                _conorm = conorm;
            }

            public override bool HasNegation => _l.HasNegation || _r.HasNegation;

            public override void ExtractVectors(List<float[]> output)
            {
                _l.ExtractVectors(output);
                _r.ExtractVectors(output);
            }

            public override float Evaluate(ref int counter, Func<int, float> distance)
            {
                float a = _l.Evaluate(ref counter, distance);
                float b = _r.Evaluate(ref counter, distance);
                return _conorm.Apply(a, b);
            }
        }

        private sealed class NotQuery : FuzzyQuery
        {
            private readonly FuzzyQuery _inner;

            public NotQuery(FuzzyQuery inner)
            {
                _inner = inner;
            }

            public override bool HasNegation => true;

            public override void ExtractVectors(List<float[]> output) => _inner.ExtractVectors(output);

            public override float Evaluate(ref int counter, Func<int, float> distance) => FuzzyOps.Negate(_inner.Evaluate(ref counter, distance));
        }
    }
}
