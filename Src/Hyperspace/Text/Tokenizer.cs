// <copyright file="Tokenizer.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;
    using System.Globalization;

    /// <summary>
    /// Splits text into lowercase word tokens, drops stop words and applies a light stemmer; the building block of the lexical half of
    /// hybrid search. Plug in your own with <see cref="HyperspaceOptions.Tokenizer"/>.
    /// </summary>
    public sealed class Tokenizer
    {
        private static readonly HashSet<string> EnglishStopWords = new HashSet<string>(StringComparer.Ordinal)
        {
            "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been",
            "before", "being", "below", "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down", "during", "each", "few",
            "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "him", "his", "how", "i", "if", "in",
            "into", "is", "it", "its", "just", "me", "more", "most", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
            "or", "other", "our", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their",
            "them", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we",
            "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "you", "your",
        };

        private readonly HashSet<string> _stopWords;
        private readonly Func<string, string>? _stemmer;
        private readonly Func<string, IEnumerable<string>>? _custom;

        /// <summary>Initializes a new instance of the <see cref="Tokenizer"/> class.</summary>
        /// <param name="stopWords">The words to drop, or <c>null</c> for none.</param>
        /// <param name="stemmer">Maps a token to its stem, or <c>null</c> to keep tokens as they are.</param>
        public Tokenizer(IEnumerable<string>? stopWords = null, Func<string, string>? stemmer = null)
        {
            _stopWords = stopWords is null ? new HashSet<string>() : new HashSet<string>(stopWords, StringComparer.Ordinal);
            _stemmer = stemmer;
        }

        private Tokenizer(Func<string, IEnumerable<string>> custom)
        {
            _stopWords = new HashSet<string>();
            _custom = custom;
        }

        /// <summary>Gets a tokenizer that does not filter or stem.</summary>
        public static Tokenizer Plain { get; } = new Tokenizer();

        /// <summary>Gets the English tokenizer: English stop words and <see cref="LightEnglishStem"/>.</summary>
        public static Tokenizer English { get; } = new Tokenizer(EnglishStopWords, LightEnglishStem);

        /// <summary>Wraps an arbitrary splitting function.</summary>
        public static Tokenizer FromFunction(Func<string, IEnumerable<string>> split) => new Tokenizer(split ?? throw new ArgumentNullException(nameof(split)));

        /// <summary>Picks the built-in tokenizer for a language name (<c>english</c> has stop words and a stemmer, others only split).</summary>
        public static Tokenizer ForLanguage(string? language)
            => string.Equals(language, "english", StringComparison.OrdinalIgnoreCase) || string.IsNullOrEmpty(language) ? English : Plain;

        /// <summary>Tokenizes a text.</summary>
        public List<string> Tokenize(string text)
        {
            var result = new List<string>();
            if (string.IsNullOrEmpty(text))
            {
                return result;
            }

            if (_custom is not null)
            {
                result.AddRange(_custom(text));
                return result;
            }

            int start = -1;
            for (int i = 0; i <= text.Length; i++)
            {
                bool word = i < text.Length && (char.IsLetterOrDigit(text[i]) || text[i] == '\'');
                if (word)
                {
                    if (start < 0)
                    {
                        start = i;
                    }

                    continue;
                }

                if (start >= 0)
                {
                    var token = text.Substring(start, i - start).Trim('\'').ToLower(CultureInfo.InvariantCulture);
                    start = -1;
                    if (token.Length == 0 || _stopWords.Contains(token))
                    {
                        continue;
                    }

                    result.Add(_stemmer is null ? token : _stemmer(token));
                }
            }

            return result;
        }

        /// <summary>
        /// Tokenizes and appends the word n-grams (joined with <c>_</c>) up to <paramref name="ngrams"/> words, in the order
        /// unigrams, bigrams, trigrams.
        /// </summary>
        public List<string> Tokenize(string text, int ngrams)
        {
            var tokens = Tokenize(text);
            if (ngrams > 1)
            {
                int unigrams = tokens.Count;
                for (int n = 2; n <= ngrams; n++)
                {
                    for (int i = 0; i + n <= unigrams; i++)
                    {
                        tokens.Add(string.Join("_", tokens.GetRange(i, n)));
                    }
                }
            }

            return tokens;
        }

        /// <summary>A conservative suffix stripper: plural and gerund / past endings of words longer than four letters.</summary>
        public static string LightEnglishStem(string word)
        {
            if (word.Length <= 3 || !IsAscii(word))
            {
                return word;
            }

            if (word.EndsWith("ies", StringComparison.Ordinal) && word.Length > 4)
            {
                return word.Substring(0, word.Length - 3) + "y";
            }

            if (word.EndsWith("sses", StringComparison.Ordinal) || word.EndsWith("xes", StringComparison.Ordinal) || word.EndsWith("zes", StringComparison.Ordinal)
                || word.EndsWith("ches", StringComparison.Ordinal) || word.EndsWith("shes", StringComparison.Ordinal))
            {
                return word.Substring(0, word.Length - 2);
            }

            if (word.EndsWith("ing", StringComparison.Ordinal) && word.Length > 5)
            {
                return Undouble(word.Substring(0, word.Length - 3));
            }

            if (word.EndsWith("ed", StringComparison.Ordinal) && word.Length > 4)
            {
                return Undouble(word.Substring(0, word.Length - 2));
            }

            if (word.EndsWith("s", StringComparison.Ordinal) && !word.EndsWith("ss", StringComparison.Ordinal) && !word.EndsWith("us", StringComparison.Ordinal))
            {
                return word.Substring(0, word.Length - 1);
            }

            return word;
        }

        private static string Undouble(string stem)
            => stem.Length > 2 && stem[^1] == stem[^2] && "bdgmnprt".IndexOf(stem[^1]) >= 0 ? stem.Substring(0, stem.Length - 1) : stem;

        private static bool IsAscii(string s)
        {
            foreach (var c in s)
            {
                if (c > 127)
                {
                    return false;
                }
            }

            return true;
        }
    }
}
