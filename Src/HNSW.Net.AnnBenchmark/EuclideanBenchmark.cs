using System;

namespace HNSW.Net.AnnBenchmark
{
    /// <summary>
    /// Euclidean distance benchmark. Smallest Euclidean dataset in ANN-Benchmarks: Fashion-MNIST
    /// (784 dims, 60,000 train, 217MB) — tied in size with MNIST; Fashion-MNIST is the modern
    /// drop-in replacement and the more common ANN benchmark.
    /// </summary>
    public class EuclideanBenchmark : AnnBenchmarkBase<float[]>
    {
        protected override string DatasetName => "fashion-mnist-784-euclidean";
        protected override string FileName => "fashion-mnist-784-euclidean.hdf5";
        protected override string DownloadUrl => "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5";
        protected override Func<float[], float[], float> Distance => L2Distance.SIMD;

        protected override (float[][] baseItems, float[][] queries, int[][] groundTruth) Load(string path)
            => AnnDataset.ReadDense(path);
    }
}
