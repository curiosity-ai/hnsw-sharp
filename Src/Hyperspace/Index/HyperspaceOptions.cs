// <copyright file="HyperspaceOptions.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Collections.Generic;
    using System.Threading;

    /// <summary>The configuration of a <see cref="HyperspaceIndex"/>. The part that shapes the stored data is persisted with the index.</summary>
    public sealed class HyperspaceOptions
    {
        /// <summary>Gets or sets the number of dimensions of every vector. Required.</summary>
        public int Dimension { get; set; }

        /// <summary>Gets or sets the distance metric.</summary>
        public DistanceMetric Metric { get; set; } = DistanceMetric.Cosine;

        /// <summary>Gets or sets how vectors are stored.</summary>
        public QuantizationMode Quantization { get; set; } = QuantizationMode.None;

        /// <summary>Gets or sets the number of links per node on the upper layers (layer 0 keeps twice as many).</summary>
        public int M { get; set; } = 16;

        /// <summary>Gets or sets the size of the candidate list while building the graph.</summary>
        public int EfConstruction { get; set; } = 100;

        /// <summary>Gets or sets the default size of the candidate list while searching.</summary>
        public int EfSearch { get; set; } = 100;

        /// <summary>
        /// Gets or sets the probability of a node climbing one more layer. <c>null</c> (the default) uses <c>1 / M</c>, the setting of the HNSW paper
        /// (<c>mL = 1 / ln M</c>), which gives about <c>log_M(n)</c> layers. HyperspaceDB uses 0.5, which stacks many more, sparser layers: on 20,000
        /// vectors that doubled the build time and added 14 % memory with no gain in recall.
        /// </summary>
        public double? LevelProbability { get; set; }

        /// <summary>Gets or sets the highest number of layers.</summary>
        public int MaxLayers { get; set; } = 16;

        /// <summary>
        /// Gets or sets a value indicating whether the upper layers are routed through the Klein model (cheap Euclidean chords)
        /// instead of the exact Poincare distance. Only for the Poincare metric with <see cref="QuantizationMode.None"/> or <see cref="QuantizationMode.Float32"/>.
        /// </summary>
        public bool FastRouting { get; set; }

        /// <summary>Gets or sets a value indicating whether nodes in sparse regions (average neighbour distance above 1.95) get half the links.</summary>
        public bool DensityPruning { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether vectors and queries are scaled to unit length. <c>null</c> (the default) turns this on for
        /// <see cref="DistanceMetric.Cosine"/> and <see cref="DistanceMetric.Angular"/> only.
        /// </summary>
        public bool? NormalizeVectors { get; set; }

        /// <summary>Gets or sets a value indicating whether 8-bit quantization of Poincare vectors (up to 128 dimensions) refines the codes to minimize the error along the vector.</summary>
        public bool AnisotropicRefinement { get; set; } = true;

        /// <summary>Gets or sets the ternary threshold as a fraction of the mean absolute coordinate.</summary>
        public float TernaryThreshold { get; set; } = 0.67f;

        /// <summary>
        /// Gets or sets the number of leading dimensions the 4-bit, TurboQuant, ternary and hybrid codecs keep in float32.
        /// <c>null</c> keeps 33 for <see cref="DistanceMetric.Hybrid"/> and none otherwise.
        /// </summary>
        public int? HeadDimensions { get; set; }

        /// <summary>Gets or sets trained product-quantization codebooks for <see cref="QuantizationMode.ProductQuantization"/> / <see cref="QuantizationMode.OPQ"/> (defaults to untrained deterministic ones).</summary>
        public ProductQuantizer? ProductQuantizer { get; set; }

        /// <summary>Gets or sets the lexical scoring parameters of hybrid search.</summary>
        public Bm25Parameters Bm25 { get; set; } = new Bm25Parameters();

        /// <summary>Gets or sets a custom tokenizer for the lexical index. It is not persisted: set it again after loading.</summary>
        public Tokenizer? Tokenizer { get; set; }

        /// <summary>Gets or sets the default fusion of hybrid search, <c>rrf</c> (reciprocal rank) or <c>weighted</c>.</summary>
        public string FusionMethod { get; set; } = "rrf";

        /// <summary>Gets or sets the configuration of the property graph.</summary>
        public GraphConfig Graph { get; set; } = new GraphConfig();

        /// <summary>Gets or sets the seed of the level generator, for reproducible graphs. <c>null</c> seeds randomly.</summary>
        public int? Seed { get; set; }

        /// <summary>Gets or sets a value indicating whether the index guards itself with a reader-writer lock.</summary>
        public bool ThreadSafe { get; set; } = true;

        /// <summary>Gets or sets a value indicating whether added vectors are validated against the metric (finite, inside the ball...).</summary>
        public bool ValidateVectors { get; set; } = true;

        /// <summary>
        /// Gets or sets a value indicating whether the original float32 vectors are kept next to the quantized ones, so searches can re-rank
        /// their candidates exactly (two-pass search). Costs <c>4 * Dimension</c> bytes per vector, but turns the low single-pass recall of the
        /// aggressive codecs (binary, ternary, product quantization) into near-exact results.
        /// </summary>
        public bool KeepOriginalVectors { get; set; }

        /// <summary>
        /// Gets or sets how many times more candidates than requested the first pass collects before the exact re-ranking pass, when
        /// <see cref="KeepOriginalVectors"/> is on. 1 turns re-ranking off. Can be overridden per query.
        /// </summary>
        public int RerankOversample { get; set; } = 4;

        /// <summary>Gets or sets a value indicating whether layer 0 is frozen into contiguous arrays right after loading.</summary>
        public bool FreezeOnLoad { get; set; } = true;

        internal double ResolvedLevelProbability => LevelProbability ?? 1.0 / M;

        internal int ResolvedHeadDimensions => Math.Clamp(HeadDimensions ?? (Metric == DistanceMetric.Hybrid ? Metrics.HybridLorentzDimensions : 0), 0, Dimension);

        internal bool ResolvedNormalize => NormalizeVectors ?? (Metric == DistanceMetric.Cosine || Metric == DistanceMetric.Angular);

        /// <summary>Creates a copy.</summary>
        public HyperspaceOptions Clone()
        {
            var copy = (HyperspaceOptions)MemberwiseClone();
            copy.Bm25 = Bm25.Clone();
            copy.Graph = new GraphConfig
            {
                Enabled = Graph.Enabled,
                StorageLayout = Graph.StorageLayout,
                MaxHopsLimit = Graph.MaxHopsLimit,
                MaxFanoutPerNode = Graph.MaxFanoutPerNode,
            };
            return copy;
        }

        internal void Validate()
        {
            if (Dimension <= 0)
            {
                throw new ArgumentException("Dimension must be positive", nameof(Dimension));
            }

            if (M < 2)
            {
                throw new ArgumentException("M must be at least 2", nameof(M));
            }

            if (EfConstruction < 1 || EfSearch < 1)
            {
                throw new ArgumentException("EfConstruction and EfSearch must be positive");
            }

            if (ResolvedLevelProbability <= 0 || ResolvedLevelProbability >= 1)
            {
                throw new ArgumentException("LevelProbability must be in (0, 1)", nameof(LevelProbability));
            }

            if (MaxLayers < 1)
            {
                throw new ArgumentException("MaxLayers must be positive", nameof(MaxLayers));
            }

            if (RerankOversample < 1)
            {
                throw new ArgumentException("RerankOversample must be at least 1", nameof(RerankOversample));
            }

            if (Metric == DistanceMetric.Lorentz && Dimension < 2)
            {
                throw new ArgumentException("The Lorentz metric needs at least 2 dimensions");
            }

            if (Metric == DistanceMetric.Hybrid && Dimension <= Metrics.HybridLorentzDimensions)
            {
                throw new ArgumentException($"The hybrid metric needs more than {Metrics.HybridLorentzDimensions} dimensions");
            }

            if (FastRouting && (Metric != DistanceMetric.Poincare || (Quantization != QuantizationMode.None && Quantization != QuantizationMode.Float32)))
            {
                throw new ArgumentException("FastRouting needs the Poincare metric with None or Float32 storage");
            }

            if (!string.Equals(FusionMethod, "rrf", StringComparison.OrdinalIgnoreCase) && !string.Equals(FusionMethod, "weighted", StringComparison.OrdinalIgnoreCase))
            {
                throw new ArgumentException("FusionMethod must be 'rrf' or 'weighted'", nameof(FusionMethod));
            }
        }
    }

    /// <summary>Restricts a search to the items reachable in the property graph from some seeds.</summary>
    /// <param name="Seeds">The ids the traversal starts from.</param>
    /// <param name="Hops">The maximum number of hops (the graph's limit still applies).</param>
    /// <param name="Relations">Follow only edges of these relations, or all if <c>null</c>.</param>
    public sealed record GraphFilter(IReadOnlyList<uint> Seeds, int Hops = 1, IReadOnlyCollection<ushort>? Relations = null);

    /// <summary>Per-query settings of <see cref="HyperspaceIndex.Search(ReadOnlySpan{float}, int, SearchOptions)"/>.</summary>
    public sealed class SearchOptions
    {
        /// <summary>Gets or sets the size of the candidate list; the index default when <c>null</c>. Always at least <c>k</c>.</summary>
        public int? Ef { get; set; }

        /// <summary>Gets or sets a metadata / geometry filter.</summary>
        public FilterExpr? Filter { get; set; }

        /// <summary>Gets or sets metadata keys that must equal the given values (all of them).</summary>
        public IReadOnlyDictionary<string, string>? ExactMetadata { get; set; }

        /// <summary>Gets or sets a property-graph reachability restriction.</summary>
        public GraphFilter? Graph { get; set; }

        /// <summary>
        /// Gets or sets the number of candidates, as a multiple of <c>k</c>, to collect before re-ranking them exactly against the original vectors
        /// (see <see cref="HyperspaceOptions.KeepOriginalVectors"/>); the index default when <c>null</c>, 1 to skip re-ranking for this query.
        /// </summary>
        public int? RerankOversample { get; set; }

        /// <summary>Gets or sets a value indicating whether results carry their metadata.</summary>
        public bool IncludeMetadata { get; set; }

        /// <summary>Gets or sets the number of leading dimensions to compare (matryoshka truncation); needs a codec that supports it.</summary>
        public int? TruncateDimensions { get; set; }

        /// <summary>Gets or sets the text of a hybrid query. When set, vector and BM25 results are fused.</summary>
        public string? HybridQuery { get; set; }

        /// <summary>Gets or sets the fusion constant of <c>rrf</c> (default 60) or the vector weight in [0, 1] of <c>weighted</c> (default 0.5).</summary>
        public float? HybridAlpha { get; set; }

        /// <summary>Gets or sets the fusion method for this query, <c>rrf</c> or <c>weighted</c>.</summary>
        public string? FusionMethod { get; set; }

        /// <summary>Gets or sets BM25 parameters for this query.</summary>
        public Bm25Parameters? Bm25 { get; set; }

        /// <summary>Gets or sets a token to cancel a long search.</summary>
        public CancellationToken CancellationToken { get; set; }
    }

    /// <summary>One search hit.</summary>
    /// <param name="Id">The item id.</param>
    /// <param name="Distance">The distance to the query (smaller is closer). For hybrid search it is <c>10 - fusedScore</c>.</param>
    /// <param name="Metadata">The item metadata, if requested.</param>
    /// <param name="Score">For hybrid search, the fused score (larger is better).</param>
    public readonly record struct SearchResult(int Id, double Distance, IReadOnlyDictionary<string, string>? Metadata = null, double? Score = null);

    /// <summary>A vector with its metadata; the unit <see cref="HyperspaceIndex"/> graphs are rebuilt from.</summary>
    /// <param name="Vector">The embedding.</param>
    /// <param name="Metadata">The metadata, or <c>null</c>.</param>
    public sealed record HyperspaceItem(float[] Vector, IReadOnlyDictionary<string, string>? Metadata = null)
    {
        /// <summary>Wraps plain vectors.</summary>
        public static IReadOnlyList<HyperspaceItem> FromVectors(IEnumerable<float[]> vectors)
        {
            var list = new List<HyperspaceItem>();
            foreach (var v in vectors)
            {
                list.Add(new HyperspaceItem(v));
            }

            return list;
        }
    }
}
