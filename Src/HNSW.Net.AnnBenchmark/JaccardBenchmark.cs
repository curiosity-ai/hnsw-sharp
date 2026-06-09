using System;

namespace HNSW.Net.AnnBenchmark
{
    /// <summary>
    /// Jaccard distance benchmark over integer sets. Smallest Jaccard dataset in ANN-Benchmarks:
    /// Kosarak (74,962 train sets, 33MB). Items are sets of integer ids rather than dense vectors.
    /// </summary>
    public class JaccardBenchmark : AnnBenchmarkBase<int[]>
    {
        protected override string DatasetName => "kosarak-jaccard";
        protected override string FileName => "kosarak-jaccard.hdf5";
        protected override string DownloadUrl => "http://ann-benchmarks.com/kosarak-jaccard.hdf5";
        protected override Func<int[], int[], float> Distance => JaccardDistance.SortedSets;

        protected override (int[][] baseItems, int[][] queries, int[][] groundTruth) Load(string path)
            => AnnDataset.ReadSparse(path);
    }
}
