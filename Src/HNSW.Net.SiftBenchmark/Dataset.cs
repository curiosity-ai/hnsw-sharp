using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Formats.Tar;
using System.IO;
using System.IO.Compression;
using System.Threading.Tasks;

namespace HNSW.Net.SiftBenchmark
{
    public static class Dataset
    {
        public static float[][] ReadFvecs(string path)
        {
            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);
            var results = new List<float[]>();
            while (stream.Position < stream.Length)
            {
                int dim = reader.ReadInt32();
                var vector = new float[dim];
                for (int i = 0; i < dim; i++)
                {
                    vector[i] = reader.ReadSingle();
                }
                results.Add(vector);
            }
            return results.ToArray();
        }

        public static int[][] ReadIvecs(string path)
        {
            using var stream = File.OpenRead(path);
            using var reader = new BinaryReader(stream);
            var results = new List<int[]>();
            while (stream.Position < stream.Length)
            {
                int dim = reader.ReadInt32();
                var vector = new int[dim];
                for (int i = 0; i < dim; i++)
                {
                    vector[i] = reader.ReadInt32();
                }
                results.Add(vector);
            }
            return results.ToArray();
        }

        public static async Task DownloadAndExtractAsync(string workingDir)
        {
            string tarPath = Path.Combine(workingDir, "sift.tar.gz");
            if (!File.Exists(tarPath))
            {
                Console.WriteLine("Downloading Sift1M dataset via curl...");
                var psi = new ProcessStartInfo("curl", $"-L ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz -o \"{tarPath}\"")
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false
                };
                var process = Process.Start(psi);
                if (process == null) throw new Exception("Failed to start curl");
                await process.WaitForExitAsync();
                if (process.ExitCode != 0)
                {
                    string error = await process.StandardError.ReadToEndAsync();
                    throw new Exception($"curl failed with exit code {process.ExitCode}: {error}");
                }
            }

            // Check if extracted files already exist to avoid re-extracting
            string baseFile = Path.Combine(workingDir, "sift", "sift_base.fvecs");
            if (!File.Exists(baseFile))
            {
                Console.WriteLine("Extracting Sift1M dataset...");
                using var fsIn = File.OpenRead(tarPath);
                using var gzipStream = new GZipStream(fsIn, CompressionMode.Decompress);
                TarFile.ExtractToDirectory(gzipStream, workingDir, true);
            }
        }
    }
}
