// <copyright file="Bm25.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;

    /// <summary>The BM25 family variants (same definitions as the Python <c>bm25s</c> library that HyperspaceDB follows).</summary>
    public enum Bm25Method
    {
        /// <summary>Robertson-Sparck Jones: <c>ln((N - df + 0.5) / (df + 0.5))</c>, can go negative for very common terms.</summary>
        Robertson,

        /// <summary>Lucene: <c>ln(1 + (N - df + 0.5) / (df + 0.5))</c>, never negative.</summary>
        Lucene,

        /// <summary>ATIRE: <c>ln(N / df)</c>.</summary>
        Atire,

        /// <summary>BM25L, which lifts the term frequency by <c>delta</c> to stop over-penalizing long documents.</summary>
        Bm25l,

        /// <summary>BM25+, which adds <c>delta</c> to the saturated term frequency (the default).</summary>
        Bm25Plus,
    }

    /// <summary>The parameters of lexical scoring.</summary>
    public sealed class Bm25Parameters
    {
        /// <summary>Gets or sets the scoring variant.</summary>
        public Bm25Method Method { get; set; } = Bm25Method.Bm25Plus;

        /// <summary>Gets or sets the term-frequency saturation.</summary>
        public float K1 { get; set; } = 1.2f;

        /// <summary>Gets or sets the length normalization strength in [0, 1].</summary>
        public float B { get; set; } = 0.75f;

        /// <summary>Gets or sets the lower-bound shift of BM25L / BM25+.</summary>
        public float Delta { get; set; } = 0.5f;

        /// <summary>Gets or sets the language of the tokenizer (stop words, stemming). Only <c>english</c> has built-in resources.</summary>
        public string Language { get; set; } = "english";

        /// <summary>Gets or sets the longest word n-gram added to the unigrams (1 = unigrams only). N-grams are joined with <c>_</c>.</summary>
        public int NGrams { get; set; } = 1;

        /// <summary>Creates a copy.</summary>
        public Bm25Parameters Clone() => (Bm25Parameters)MemberwiseClone();
    }

    /// <summary>BM25 scoring functions.</summary>
    public static class Bm25
    {
        /// <summary>The inverse document frequency of a term.</summary>
        public static float Idf(Bm25Method method, int numDocs, int docFreq)
        {
            double n = numDocs, df = docFreq;
            double v = method switch
            {
                Bm25Method.Robertson => Math.Log((n - df + 0.5) / (df + 0.5)),
                Bm25Method.Lucene => Math.Log(1.0 + ((n - df + 0.5) / (df + 0.5))),
                Bm25Method.Atire => Math.Log(n / df),
                Bm25Method.Bm25l => Math.Log((n + 1.0) / (df + 0.5)),
                Bm25Method.Bm25Plus => Math.Log((n + 1.0) / df),
                _ => throw new ArgumentOutOfRangeException(nameof(method)),
            };
            return (float)v;
        }

        /// <summary>The length-normalized, saturated term-frequency component.</summary>
        public static float TermFrequencyComponent(Bm25Method method, float tf, float docLength, float averageDocLength, float k1, float b, float delta)
        {
            double ratio = docLength / (double)averageDocLength;
            double norm = 1.0 - b + (b * ratio);
            double v = method switch
            {
                Bm25Method.Robertson or Bm25Method.Lucene => tf / ((k1 * norm) + tf),
                Bm25Method.Atire => tf * (k1 + 1.0) / (tf + (k1 * norm)),
                Bm25Method.Bm25l => (k1 + 1.0) * (tf + delta) / (k1 + tf + delta),
                Bm25Method.Bm25Plus => ((k1 + 1.0) * tf / ((k1 * norm) + tf)) + delta,
                _ => throw new ArgumentOutOfRangeException(nameof(method)),
            };
            return (float)v;
        }

        /// <summary>The score of one term in one document, given a precomputed IDF.</summary>
        public static float ScoreWithIdf(float idf, Bm25Method method, float tf, float docLength, float averageDocLength, float k1, float b, float delta)
            => (float)((double)idf * TermFrequencyComponent(method, tf, docLength, averageDocLength, k1, b, delta));

        /// <summary>The score of one term in one document.</summary>
        public static float Score(Bm25Method method, float tf, float docLength, float averageDocLength, int numDocs, int docFreq, float k1, float b, float delta)
            => ScoreWithIdf(Idf(method, numDocs, docFreq), method, tf, docLength, averageDocLength, k1, b, delta);
    }
}
