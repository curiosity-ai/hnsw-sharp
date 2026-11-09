// <copyright file="ProductQuantizer.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;
    using System.Buffers.Binary;
    using System.Collections.Generic;
    using System.IO;
    using System.Threading.Tasks;

    /// <summary>
    /// Product quantization (and OPQ-by-rotation): a vector is cut into sub-vectors, each replaced by the index of its nearest of
    /// 256 centroids, so a D-dimensional vector costs one byte per sub-vector. Distances use asymmetric lookup tables.
    /// </summary>
    public sealed class ProductQuantizer
    {
        /// <summary>The magic of a serialized codebook.</summary>
        public static readonly byte[] Magic = { (byte)'P', (byte)'Q', (byte)'C', (byte)'B' };

        private readonly float[] _centroids;

        /// <summary>Initializes a new instance of the <see cref="ProductQuantizer"/> class with deterministic pseudo-Gaussian codebooks.</summary>
        /// <param name="dimension">The vector dimension.</param>
        /// <param name="numSubvectors">The number of sub-vectors; the dimension is rounded up to a multiple of it.</param>
        /// <param name="isOpq">Rotate vectors with the Walsh-Hadamard transform before quantizing, which balances the energy between sub-vectors.</param>
        public ProductQuantizer(int dimension, int numSubvectors, bool isOpq)
        {
            if (dimension <= 0 || numSubvectors <= 0)
            {
                throw new ArgumentException("Dimension and number of sub-vectors must be positive");
            }

            Dimension = dimension;
            NumSubvectors = numSubvectors;
            SubvectorDimension = (dimension + numSubvectors - 1) / numSubvectors;
            IsOpq = isOpq;
            _centroids = new float[checked(numSubvectors * 256 * SubvectorDimension)];

            float scale = 1.0f / Math.Max(MathF.Sqrt(dimension), 1e-4f);
            for (int sub = 0; sub < numSubvectors; sub++)
            {
                for (int k = 0; k < 256; k++)
                {
                    int offset = ((sub * 256) + k) * SubvectorDimension;
                    for (int d = 0; d < SubvectorDimension; d++)
                    {
                        ulong seed = ((((ulong)sub * 256) + (ulong)k) * (ulong)SubvectorDimension) + (ulong)d + 1;
                        ulong x = unchecked(seed * 0x517cc1b727220a95UL);
                        x ^= x >> 32;
                        float u1 = (((x & 0xffff) + 1.0f) / 65537.0f);
                        float u2 = ((((x >> 16) & 0xffff) + 1.0f) / 65537.0f);
                        _centroids[offset + d] = MathF.Sqrt(-2.0f * MathF.Log(u1)) * MathF.Cos(2.0f * MathF.PI * u2) * scale;
                    }
                }
            }
        }

        private ProductQuantizer(int dimension, int numSubvectors, bool isOpq, float[] centroids)
        {
            Dimension = dimension;
            NumSubvectors = numSubvectors;
            SubvectorDimension = (dimension + numSubvectors - 1) / numSubvectors;
            IsOpq = isOpq;
            _centroids = centroids;
        }

        /// <summary>Gets the vector dimension.</summary>
        public int Dimension { get; }

        /// <summary>Gets the number of sub-vectors (and the length in bytes of a code).</summary>
        public int NumSubvectors { get; }

        /// <summary>Gets the dimension of one sub-vector.</summary>
        public int SubvectorDimension { get; }

        /// <summary>Gets a value indicating whether vectors are rotated before quantization.</summary>
        public bool IsOpq { get; }

        /// <summary>The default number of sub-vectors for a dimension, targeting sub-vectors of 16 dimensions.</summary>
        public static int DefaultSubvectorCount(int dimension)
        {
            if (dimension <= 0)
            {
                return 1;
            }

            if (dimension % 16 == 0)
            {
                return dimension / 16;
            }

            if (dimension % 8 == 0)
            {
                return dimension / 8;
            }

            return dimension % 4 == 0 ? dimension / 4 : dimension;
        }

        private Span<float> Centroid(int sub, int k) => _centroids.AsSpan((((sub * 256) + k) * SubvectorDimension), SubvectorDimension);

        /// <summary>Trains the codebooks with Lloyd's k-means on each sub-space.</summary>
        public void Train(IReadOnlyList<float[]> samples, int iterations = 10)
        {
            if (samples.Count == 0)
            {
                return;
            }

            var data = new float[samples.Count][];
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = samples[i].AsSpan(0, Math.Min(samples[i].Length, Dimension)).ToArray();
                if (data[i].Length < Dimension)
                {
                    Array.Resize(ref data[i], Dimension);
                }

                if (IsOpq)
                {
                    TurboRotation.Rotate(data[i]);
                }
            }

            int n = data.Length;
            int subDim = SubvectorDimension;

            Parallel.For(0, NumSubvectors, sub =>
            {
                int subOffset = sub * subDim;
                for (int k = 0; k < 256; k++)
                {
                    var sample = data[n > 256 ? (k * n) / 256 : k % n];
                    var c = Centroid(sub, k);
                    for (int d = 0; d < subDim; d++)
                    {
                        int idx = subOffset + d;
                        c[d] = idx < sample.Length ? sample[idx] : 0f;
                    }
                }

                var assignments = new byte[n];
                var counts = new int[256];
                var sums = new float[256 * subDim];
                for (int iter = 0; iter < iterations; iter++)
                {
                    for (int s = 0; s < n; s++)
                    {
                        var v = data[s];
                        byte best = 0;
                        float bestDist = float.MaxValue;
                        for (int k = 0; k < 256; k++)
                        {
                            var c = Centroid(sub, k);
                            float dist = 0;
                            for (int d = 0; d < subDim; d++)
                            {
                                int idx = subOffset + d;
                                float diff = (idx < v.Length ? v[idx] : 0f) - c[d];
                                dist += diff * diff;
                            }

                            if (dist < bestDist)
                            {
                                bestDist = dist;
                                best = (byte)k;
                            }
                        }

                        assignments[s] = best;
                    }

                    Array.Clear(counts);
                    Array.Clear(sums);
                    for (int s = 0; s < n; s++)
                    {
                        int k = assignments[s];
                        counts[k]++;
                        var v = data[s];
                        for (int d = 0; d < subDim; d++)
                        {
                            int idx = subOffset + d;
                            sums[(k * subDim) + d] += idx < v.Length ? v[idx] : 0f;
                        }
                    }

                    for (int k = 0; k < 256; k++)
                    {
                        if (counts[k] > 0)
                        {
                            var c = Centroid(sub, k);
                            float inv = 1.0f / counts[k];
                            for (int d = 0; d < subDim; d++)
                            {
                                c[d] = sums[(k * subDim) + d] * inv;
                            }
                        }
                    }
                }
            });
        }

        /// <summary>Encodes a vector to <see cref="NumSubvectors"/> bytes.</summary>
        public byte[] Encode(ReadOnlySpan<float> vector)
        {
            var work = Prepare(vector);
            var codes = new byte[NumSubvectors];
            int subDim = SubvectorDimension;
            for (int sub = 0; sub < NumSubvectors; sub++)
            {
                int subOffset = sub * subDim;
                byte best = 0;
                float bestDist = float.MaxValue;
                for (int k = 0; k < 256; k++)
                {
                    var c = Centroid(sub, k);
                    float dist = 0;
                    for (int d = 0; d < subDim; d++)
                    {
                        int idx = subOffset + d;
                        float diff = (idx < work.Length ? work[idx] : 0f) - c[d];
                        dist += diff * diff;
                    }

                    if (dist < bestDist)
                    {
                        bestDist = dist;
                        best = (byte)k;
                    }
                }

                codes[sub] = best;
            }

            return codes;
        }

        /// <summary>Reconstructs a vector from its codes.</summary>
        public float[] Decode(ReadOnlySpan<byte> codes)
        {
            var output = new float[Dimension];
            DecodeInto(codes, output, unrotate: true);
            return output;
        }

        /// <summary>
        /// Reconstructs a vector into <paramref name="output"/>. With <paramref name="unrotate"/> off an OPQ vector stays in the rotated
        /// domain, which is enough (and cheaper) to compare two codes by L2 distance.
        /// </summary>
        public void DecodeInto(ReadOnlySpan<byte> codes, Span<float> output, bool unrotate)
        {
            output.Slice(0, Dimension).Clear();
            int subDim = SubvectorDimension;
            for (int sub = 0; sub < codes.Length && sub < NumSubvectors; sub++)
            {
                var c = Centroid(sub, codes[sub]);
                for (int d = 0; d < subDim; d++)
                {
                    int idx = (sub * subDim) + d;
                    if (idx < Dimension)
                    {
                        output[idx] = c[d];
                    }
                }
            }

            if (IsOpq && unrotate)
            {
                TurboRotation.Unrotate(output.Slice(0, Dimension));
            }
        }

        /// <summary>Precomputes the table of squared distances from the query to every centroid.</summary>
        public LookupTable ComputeLookupTable(ReadOnlySpan<float> query)
        {
            var work = Prepare(query);
            var table = new float[NumSubvectors * 256];
            int subDim = SubvectorDimension;
            for (int sub = 0; sub < NumSubvectors; sub++)
            {
                int subOffset = sub * subDim;
                for (int k = 0; k < 256; k++)
                {
                    var c = Centroid(sub, k);
                    float dist = 0;
                    for (int d = 0; d < subDim; d++)
                    {
                        int idx = subOffset + d;
                        float diff = (idx < work.Length ? work[idx] : 0f) - c[d];
                        dist += diff * diff;
                    }

                    table[(sub * 256) + k] = dist;
                }
            }

            return new LookupTable(NumSubvectors, table);
        }

        private float[] Prepare(ReadOnlySpan<float> vector)
        {
            var work = new float[Dimension];
            vector.Slice(0, Math.Min(vector.Length, Dimension)).CopyTo(work);
            if (IsOpq)
            {
                TurboRotation.Rotate(work);
            }

            return work;
        }

        // ------------------------------------------------------------------ serialization ("PQCB")

        /// <summary>
        /// Writes the codebooks as <c>"PQCB" | u32 dimension | u32 subvectors | u32 isOpq | u32 centroidCount | f32[] centroids</c>,
        /// little-endian (the same file HyperspaceDB reads and writes).
        /// </summary>
        public void Save(Stream stream)
        {
            Span<byte> header = stackalloc byte[20];
            Magic.CopyTo(header);
            BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(4), (uint)Dimension);
            BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(8), (uint)NumSubvectors);
            BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(12), IsOpq ? 1u : 0u);
            BinaryPrimitives.WriteUInt32LittleEndian(header.Slice(16), (uint)_centroids.Length);
            stream.Write(header);

            var buffer = new byte[4096];
            int pos = 0;
            while (pos < _centroids.Length)
            {
                int count = Math.Min(_centroids.Length - pos, buffer.Length / 4);
                for (int i = 0; i < count; i++)
                {
                    BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(i * 4, 4), _centroids[pos + i]);
                }

                stream.Write(buffer, 0, count * 4);
                pos += count;
            }
        }

        /// <summary>Saves the codebooks to a file.</summary>
        public void Save(string path)
        {
            using var fs = File.Create(path);
            Save(fs);
        }

        /// <summary>Reads codebooks written by <see cref="Save(Stream)"/>.</summary>
        public static ProductQuantizer Load(Stream stream)
        {
            Span<byte> header = stackalloc byte[20];
            stream.ReadExactly(header);
            if (!header.Slice(0, 4).SequenceEqual(Magic))
            {
                throw new InvalidDataException("Invalid PQ codebook magic");
            }

            int dimension = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(4)));
            int subvectors = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(8)));
            bool opq = BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(12)) != 0;
            int count = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(header.Slice(16)));
            if (dimension <= 0 || subvectors <= 0)
            {
                throw new InvalidDataException("Invalid PQ codebook header");
            }

            int expected = checked(subvectors * 256 * ((dimension + subvectors - 1) / subvectors));
            if (count != expected)
            {
                throw new InvalidDataException($"PQ codebook declares {count} centroid values but its shape needs {expected}");
            }

            var centroids = new float[count];
            var buffer = new byte[4096];
            int pos = 0;
            while (pos < count)
            {
                int n = Math.Min(count - pos, buffer.Length / 4);
                stream.ReadExactly(buffer, 0, n * 4);
                for (int i = 0; i < n; i++)
                {
                    centroids[pos + i] = BinaryPrimitives.ReadSingleLittleEndian(buffer.AsSpan(i * 4, 4));
                }

                pos += n;
            }

            return new ProductQuantizer(dimension, subvectors, opq, centroids);
        }

        /// <summary>Loads codebooks from a file.</summary>
        public static ProductQuantizer Load(string path)
        {
            using var fs = File.OpenRead(path);
            return Load(fs);
        }

        /// <summary>A query-to-centroid table giving each distance in <see cref="NumSubvectors"/> additions.</summary>
        public sealed class LookupTable
        {
            private readonly float[] _table;

            internal LookupTable(int numSubvectors, float[] table)
            {
                NumSubvectors = numSubvectors;
                _table = table;
            }

            /// <summary>Gets the number of sub-vectors.</summary>
            public int NumSubvectors { get; }

            /// <summary>The squared distance from the query to the vector whose codes are given.</summary>
            public float Distance(ReadOnlySpan<byte> codes)
            {
                int m = Math.Min(NumSubvectors, codes.Length);
                float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
                int i = 0;
                for (; i + 4 <= m; i += 4)
                {
                    a0 += _table[(i * 256) + codes[i]];
                    a1 += _table[((i + 1) * 256) + codes[i + 1]];
                    a2 += _table[((i + 2) * 256) + codes[i + 2]];
                    a3 += _table[((i + 3) * 256) + codes[i + 3]];
                }

                float sum = (a0 + a1) + (a2 + a3);
                for (; i < m; i++)
                {
                    sum += _table[(i * 256) + codes[i]];
                }

                return sum;
            }
        }
    }

    /// <summary>Product quantization storage; distances are asymmetric squared L2 whatever the index metric.</summary>
    internal sealed class ProductQuantizationCodec : VectorCodec
    {
        private readonly ProductQuantizer _pq;

        public ProductQuantizationCodec(int dimension, DistanceMetric metric, ProductQuantizer pq)
            : base(dimension, metric)
        {
            _pq = pq;
        }

        public ProductQuantizer Quantizer => _pq;

        public override DistanceMetric PairMetric => DistanceMetric.L2;

        protected override void DecodeForPair(ReadOnlySpan<byte> stored, Span<float> destination) => _pq.DecodeInto(stored, destination, unrotate: false);

        public override int EncodedLength => _pq.NumSubvectors;

        public override byte[] Encode(ReadOnlySpan<float> vector) => _pq.Encode(vector);

        public override void Decode(ReadOnlySpan<byte> stored, Span<float> destination) => _pq.Decode(stored).AsSpan(0, destination.Length).CopyTo(destination);

        public override PreparedQuery Prepare(ReadOnlySpan<float> query)
        {
            var q = base.Prepare(query);
            q.Lookup = _pq.ComputeLookupTable(query);
            return q;
        }

        public override double Distance(ReadOnlySpan<byte> stored, PreparedQuery query, int truncation = 0) => query.Lookup!.Distance(stored);
    }
}
