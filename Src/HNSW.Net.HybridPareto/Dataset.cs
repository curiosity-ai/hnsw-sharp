using System;
using System.Collections.Generic;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;
using PureHDF;

namespace HNSW.Net.HybridPareto
{
    public static class Dataset
    {
        // Smallest Euclidean dataset in ANN-Benchmarks (https://github.com/erikbern/ann-benchmarks):
        // Fashion-MNIST (784 dims, 60,000 base / 10,000 query). Used as the dense vector source for
        // the attribute-filtered hybrid search benchmark (which generates its own attributes/ground truth).
        private const string FileName = "fashion-mnist-784-euclidean.hdf5";
        private const string DownloadUrl = "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5";

        public static int[] GenerateRandomAttributes(int count, int seed = 42)
        {
            var random = new Random(seed);
            var attributes = new int[count];
            for (int i = 0; i < count; i++)
            {
                attributes[i] = random.Next(1, 13); // Range 1-12 inclusive
            }
            return attributes;
        }

        public static int[][] ComputeHybridGroundTruth(float[][] baseVectors, int[] baseAttributes, float[][] queryVectors, int[] queryAttributes, int k)
        {
            Console.WriteLine("Computing hybrid ground truth...");
            var groundTruth = new int[queryVectors.Length][];

            // To make this much faster, let's use Parallel.For
            Parallel.For(0, queryVectors.Length, i =>
            {
                var queryVector = queryVectors[i];
                var queryAttribute = queryAttributes[i];
                var distances = new List<(int Id, float Distance)>();

                for (int j = 0; j < baseVectors.Length; j++)
                {
                    if (baseAttributes[j] == queryAttribute)
                    {
                        float dist = L2Distance.SIMD(queryVector, baseVectors[j]);
                        distances.Add((j, dist));
                    }
                }

                distances.Sort((a, b) => a.Distance.CompareTo(b.Distance));
                groundTruth[i] = new int[Math.Min(k, distances.Count)];
                for (int j = 0; j < groundTruth[i].Length; j++)
                {
                    groundTruth[i][j] = distances[j].Id;
                }

                if (i > 0 && i % 1000 == 0)
                {
                    Console.WriteLine($"Computed ground truth for {i} queries");
                }
            });

            Console.WriteLine("Finished computing hybrid ground truth.");
            return groundTruth;
        }
        /// <summary>Reads the base ("train") and query ("test") vectors from the HDF5 dataset.</summary>
        public static (float[][] baseVectors, float[][] queryVectors) ReadVectors(string path)
        {
            using var file = H5File.OpenRead(path);
            var baseVectors = ReadFloatMatrix(file.Dataset("train"));
            var queryVectors = ReadFloatMatrix(file.Dataset("test"));
            return (baseVectors, queryVectors);
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

        public static void SaveGroundTruth(int[][] groundTruth, string path)
        {
            using var stream = File.Create(path);
            using var writer = new BinaryWriter(stream);
            writer.Write(groundTruth.Length);
            for (int i = 0; i < groundTruth.Length; i++)
            {
                var row = groundTruth[i];
                writer.Write(row.Length);
                for (int j = 0; j < row.Length; j++)
                {
                    writer.Write(row[j]);
                }
            }
        }

        public static int[][] LoadGroundTruth(string path)
        {
            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);
            int length = reader.ReadInt32();
            var groundTruth = new int[length][];
            for (int i = 0; i < length; i++)
            {
                int rowLength = reader.ReadInt32();
                var row = new int[rowLength];
                for (int j = 0; j < rowLength; j++)
                {
                    row[j] = reader.ReadInt32();
                }
                groundTruth[i] = row;
            }
            return groundTruth;
        }

        /// <summary>Ensures the HDF5 dataset exists locally (downloading it if needed) and returns its path.</summary>
        public static string EnsureDownloaded(string workingDir)
        {
            string path = Path.Combine(workingDir, FileName);
            if (!File.Exists(path))
            {
                Console.WriteLine($"Downloading {FileName} from {DownloadUrl} ...");
                DownloadAsync(DownloadUrl, path).GetAwaiter().GetResult();
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
    }
}
