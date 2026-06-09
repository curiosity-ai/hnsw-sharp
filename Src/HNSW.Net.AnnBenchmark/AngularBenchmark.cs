using System;
using HNSW.Net;

namespace HNSW.Net.AnnBenchmark
{
    /// <summary>
    /// Angular (cosine) distance benchmark. Smallest Angular dataset in ANN-Benchmarks: GloVe-25
    /// (25 dims, 1,183,514 train, 121MB).
    /// </summary>
    public class AngularBenchmark : AnnBenchmarkBase<float[]>
    {
        protected override string DatasetName => "glove-25-angular";
        protected override string FileName => "glove-25-angular.hdf5";
        protected override string DownloadUrl => "http://ann-benchmarks.com/glove-25-angular.hdf5";
        protected override Func<float[], float[], float> Distance => CosineDistance.SIMD;

        protected override (float[][] baseItems, float[][] queries, int[][] groundTruth) Load(string path)
            => AnnDataset.ReadDense(path);
    }
}
