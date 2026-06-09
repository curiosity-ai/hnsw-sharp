using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using PureHDF;

namespace HNSW.Net.AnnBenchmark
{
    /// <summary>
    /// Downloads and reads the precomputed ANN-Benchmarks datasets
    /// (https://github.com/erikbern/ann-benchmarks) distributed as HDF5 files.
    ///
    /// Every dataset exposes the following datasets:
    ///   - train      : the base vectors to index
    ///   - test       : the query vectors
    ///   - neighbors  : [test, 100] ground-truth nearest-neighbour indices into "train"
    ///   - distances  : [test, 100] ground-truth distances
    ///
    /// Dense datasets (Euclidean / Angular) store train/test as 2-D float arrays.
    /// Sparse datasets (Jaccard) store train/test as a single flat integer array of
    /// concatenated sets, together with "size_train"/"size_test" giving the length of
    /// each individual set.
    /// </summary>
    public static class AnnDataset
    {
        /// <summary>
        /// Ensures the HDF5 file for <paramref name="fileName"/> exists locally,
        /// downloading it from <paramref name="url"/> if needed.
        /// </summary>
        public static string EnsureDownloaded(string workingDir, string fileName, string url)
        {
            string path = Path.Combine(workingDir, fileName);
            if (!File.Exists(path))
            {
                Console.WriteLine($"Downloading {fileName} from {url} ...");
                var sw = System.Diagnostics.Stopwatch.StartNew();
                DownloadAsync(url, path).GetAwaiter().GetResult();
                sw.Stop();
                Console.WriteLine($"Downloaded {new FileInfo(path).Length / (1024 * 1024)} MB in {sw.Elapsed.TotalSeconds:N1}s.");
            }
            return path;
        }

        private static async Task DownloadAsync(string url, string path)
        {
            using var client = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
            using var response = await client.GetAsync(url, HttpCompletionOption.ResponseHeadersRead);
            response.EnsureSuccessStatusCode();

            string tmp = path + ".part";
            await using (var src = await response.Content.ReadAsStreamAsync())
            await using (var dst = File.Create(tmp))
            {
                await src.CopyToAsync(dst);
            }
            File.Move(tmp, path, overwrite: true);
        }

        /// <summary>
        /// Reads a dense dataset, returning the base/query vectors and ground-truth neighbour indices.
        /// </summary>
        public static (float[][] baseVectors, float[][] queryVectors, int[][] groundTruth) ReadDense(string path)
        {
            using var file = H5File.OpenRead(path);
            var baseVectors = ReadFloatMatrix(file.Dataset("train"));
            var queryVectors = ReadFloatMatrix(file.Dataset("test"));
            var groundTruth = ReadIntMatrix(file.Dataset("neighbors"));
            return (baseVectors, queryVectors, groundTruth);
        }

        /// <summary>
        /// Reads a sparse (Jaccard) dataset, returning the base/query sets and ground-truth neighbour indices.
        /// Each set is returned as a sorted array of the integer member ids.
        /// </summary>
        public static (int[][] baseSets, int[][] querySets, int[][] groundTruth) ReadSparse(string path)
        {
            using var file = H5File.OpenRead(path);
            var baseSets = ReadSets(file.Dataset("train"), file.Dataset("size_train"));
            var querySets = ReadSets(file.Dataset("test"), file.Dataset("size_test"));
            var groundTruth = ReadIntMatrix(file.Dataset("neighbors"));
            return (baseSets, querySets, groundTruth);
        }

        private static float[][] ReadFloatMatrix(IH5Dataset dataset)
        {
            var dims = dataset.Space.Dimensions;
            int rows = (int)dims[0];
            int cols = (int)dims[1];
            var flat = dataset.Read<float[]>();

            var result = new float[rows][];
            for (int r = 0; r < rows; r++)
            {
                var row = new float[cols];
                Array.Copy(flat, (long)r * cols, row, 0, cols);
                result[r] = row;
            }
            return result;
        }

        private static int[][] ReadIntMatrix(IH5Dataset dataset)
        {
            var dims = dataset.Space.Dimensions;
            int rows = (int)dims[0];
            int cols = (int)dims[1];
            var flat = dataset.Read<int[]>();

            var result = new int[rows][];
            for (int r = 0; r < rows; r++)
            {
                var row = new int[cols];
                Array.Copy(flat, (long)r * cols, row, 0, cols);
                result[r] = row;
            }
            return result;
        }

        private static int[][] ReadSets(IH5Dataset valuesDataset, IH5Dataset sizesDataset)
        {
            // Sparse member ids are stored as 64-bit integers in the ANN-Benchmarks files.
            var values = valuesDataset.Read<long[]>();
            var sizes = sizesDataset.Read<int[]>();

            var sets = new int[sizes.Length][];
            int offset = 0;
            for (int s = 0; s < sizes.Length; s++)
            {
                int len = sizes[s];
                var set = new int[len];
                for (int i = 0; i < len; i++)
                {
                    set[i] = (int)values[offset + i];
                }
                Array.Sort(set); // JaccardDistance requires sorted sets (the files already store them sorted).
                sets[s] = set;
                offset += len;
            }
            return sets;
        }
    }
}
