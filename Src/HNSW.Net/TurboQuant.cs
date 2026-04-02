// Copyright (c) 2025 Relatude.DB - Proventus AS
// Licensed under the MIT License. https://github.com/Relatude/Relatude.DB/blob/main/LICENSE.txt

using System.Collections.Generic;
using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
namespace HNSW.Net;

public sealed class TurboQuant
{
    private readonly int _dimension;
    private readonly int _bits;
    private readonly int _residualProjections;

    // Rotation config: diagonal Rademacher signs + Walsh-Hadamard
    private readonly float[] _rotationSigns;

    // Scalar quantizer
    private readonly float[] _codebook;
    private readonly float[] _thresholds;

    // Fast ApproxDot Lookup Table (256x256)
    private readonly float[] _dotTable;

    // Residual sign sketch projection matrix (dense Gaussian)
    private readonly float[][] _residualProjection;

    private TurboQuant(int dimension, int bits, int residualProjections, float[] rotationSigns, float[] codebook, float[] thresholds, float[][] residualProjection)
    {
        _dimension = dimension;
        _bits = bits;
        _residualProjections = residualProjections;
        _rotationSigns = rotationSigns;
        _codebook = codebook;
        _thresholds = thresholds;
        _residualProjection = residualProjection;

        // Initialize fast lookup table for ApproxDot
        _dotTable = new float[256 * 256];
        int maxIdx = codebook.Length - 1;
        for (int i = 0; i < 256; i++)
        {
            for (int j = 0; j < 256; j++)
            {
                // Safely cap index to the codebook size if out-of-range bytes are ever encountered
                int ci = i > maxIdx ? maxIdx : i;
                int cj = j > maxIdx ? maxIdx : j;
                _dotTable[(i << 8) | j] = codebook[ci] * codebook[cj];
            }
        }
    }

    public static TurboQuant Create(int dimension, IProvideRandomValues randomValuesGenerator, int bits = 3, int residualProjections = 0, int lloydMaxTrainingSamples = 200_000, int lloydMaxIterations = 30)
    {
        if (dimension <= 0 || (dimension & (dimension - 1)) != 0)
            throw new ArgumentException("Dimension must be a positive power of 2 for Walsh-Hadamard rotation.");

        if (bits <= 0 || bits > 8)
            throw new ArgumentException("bits must be between 1 and 8.");

        if (residualProjections < 0)
            throw new ArgumentException("residualProjections must be >= 0.");


        var rotationSigns = new float[dimension];

        for (int i = 0; i < dimension; i++)
            rotationSigns[i] = randomValuesGenerator.NextFloat() < 0.5f ? -1f : 1f;

        var training = SampleSphereCoordinates(dimension, lloydMaxTrainingSamples, randomValuesGenerator);
        
        var codebook = TrainLloydMax(training, 1 << bits, lloydMaxIterations);

        Array.Sort(codebook);

        var thresholds = new float[codebook.Length - 1];
        for (int i = 0; i < thresholds.Length; i++)
            thresholds[i] = 0.5f * (codebook[i] + codebook[i + 1]);

        float[][] residualProjection = null;

        if (residualProjections > 0)
        {
            residualProjection = CreateGaussianProjection(rows: residualProjections, cols: dimension, randomValuesGenerator);
        }

        return new TurboQuant(dimension, bits, residualProjections, rotationSigns, codebook, thresholds, residualProjection);
    }

    public EncodedVector Encode(ReadOnlySpan<float> vector, bool isQuery = false)
    {
        if (vector.Length != _dimension)
            throw new ArgumentException($"Expected vector length {_dimension}.");

        var norm = L2Norm(vector);

        if (norm == 0f)
        {
            return new EncodedVector(0f, new byte[_dimension], _residualProjections > 0 ? new byte[(_residualProjections + 7) / 8] : null);
        }

        var normalized = new float[_dimension];
        for (int i = 0; i < _dimension; i++)
        {
            normalized[i] = vector[i] / norm;
        }

        var rotated = ApplyRandomHadamard(normalized);

        var indices = new byte[_dimension];
        var reconstructedRotated = new float[_dimension];

        for (int i = 0; i < _dimension; i++)
        {
            int idx = QuantizeIndex(rotated[i]);
            indices[i] = checked((byte)idx);
            reconstructedRotated[i] = _codebook[idx];
        }

        byte[] residualBits = null;
        float[] queryProjections = null;

        if (_residualProjections > 0 && _residualProjection is not null)
        {
            if (isQuery)
            {
                queryProjections = new float[_residualProjections];
                for (int r = 0; r < _residualProjections; r++)
                {
                    float dot = 0f;
                    var row = _residualProjection[r];
                    for (int c = 0; c < _dimension; c++)
                        dot += row[c] * rotated[c];
                    queryProjections[r] = dot;
                }
            }
            else
            {
                var residual = new float[_dimension];
                for (int i = 0; i < _dimension; i++)
                {
                    residual[i] = rotated[i] - reconstructedRotated[i];
                }
                residualBits = ProjectToSignBits(residual, _residualProjection);
            }
        }

        return new EncodedVector(norm, indices, residualBits, queryProjections);
    }

    public float[] Decode(EncodedVector encoded)
    {
        var rotatedRecon = new float[_dimension];
        for (int i = 0; i < _dimension; i++)
        {
            rotatedRecon[i] = _codebook[encoded.Indices[i]];
        }

        var unitRecon = ApplyInverseRandomHadamard(rotatedRecon);

        var output = new float[_dimension];
        for (int i = 0; i < _dimension; i++)
        {
            output[i] = unitRecon[i] * encoded.Norm;
        }

        return output;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public float ApproxDot(EncodedVector a, EncodedVector b)
    {
        int dim = _dimension;

        if (a.Indices.Length < dim || b.Indices.Length < dim)
            throw new ArgumentException("Encoded vectors do not match dimension.");

        float baseDot = 0f;

        // Stage 1: Vectorized, Unrolled LUT lookups (Zero Bounds Checking)
        ref byte pA = ref MemoryMarshal.GetArrayDataReference(a.Indices);
        ref byte pB = ref MemoryMarshal.GetArrayDataReference(b.Indices);
        ref float pLut = ref MemoryMarshal.GetArrayDataReference(_dotTable);

        int i = 0;
        int dim8 = dim - (dim % 8);

        for (; i < dim8; i += 8)
        {
            baseDot += Unsafe.Add(ref pLut, (Unsafe.Add(ref pA, i) << 8) | Unsafe.Add(ref pB, i))
                     + Unsafe.Add(ref pLut, (Unsafe.Add(ref pA, i + 1) << 8) | Unsafe.Add(ref pB, i + 1))
                     + Unsafe.Add(ref pLut, (Unsafe.Add(ref pA, i + 2) << 8) | Unsafe.Add(ref pB, i + 2))
                     + Unsafe.Add(ref pLut, (Unsafe.Add(ref pA, i + 3) << 8) | Unsafe.Add(ref pB, i + 3))
                     + Unsafe.Add(ref pLut, (Unsafe.Add(ref pA, i + 4) << 8) | Unsafe.Add(ref pB, i + 4))
                     + Unsafe.Add(ref pLut, (Unsafe.Add(ref pA, i + 5) << 8) | Unsafe.Add(ref pB, i + 5))
                     + Unsafe.Add(ref pLut, (Unsafe.Add(ref pA, i + 6) << 8) | Unsafe.Add(ref pB, i + 6))
                     + Unsafe.Add(ref pLut, (Unsafe.Add(ref pA, i + 7) << 8) | Unsafe.Add(ref pB, i + 7));
        }

        for (; i < dim; i++)
        {
            baseDot += Unsafe.Add(ref pLut, (Unsafe.Add(ref pA, i) << 8) | Unsafe.Add(ref pB, i));
        }

        float result = a.Norm * b.Norm * baseDot;

        // Stage 2: QJL asymmetric error correction
        if (_residualProjections > 0)
        {
            if (a.QueryProjections != null && b.ResidualBits != null)
            {
                float correction = 0f;
                for (int r = 0; r < _residualProjections; r++)
                {
                    bool sign = (b.ResidualBits[r >> 3] & (1 << (r & 7))) != 0;
                    float b_K = sign ? 1f : -1f;
                    correction += a.QueryProjections[r] * b_K;
                }
                correction /= _residualProjections;

                // For QJL with 1-bit sign, the unbiased estimator scales by sqrt(pi/2) * ||e||.
                // Assuming roughly constant residual norm for 3-bit quantization.
                // MSE is roughly 0.0425, so norm is sqrt(0.0425) approx 0.206.
                // sqrt(pi/2) * 0.206 approx 0.258.
                // We'll scale the correction appropriately.
                float scale = MathF.Sqrt(MathF.PI / 2f) * 0.206f;
                result += a.Norm * b.Norm * correction * scale;
            }
            else if (b.QueryProjections != null && a.ResidualBits != null)
            {
                float correction = 0f;
                for (int r = 0; r < _residualProjections; r++)
                {
                    bool sign = (a.ResidualBits[r >> 3] & (1 << (r & 7))) != 0;
                    float b_K = sign ? 1f : -1f;
                    correction += b.QueryProjections[r] * b_K;
                }
                correction /= _residualProjections;

                float scale = MathF.Sqrt(MathF.PI / 2f) * 0.206f;
                result += a.Norm * b.Norm * correction * scale;
            }
        }

        return result;
    }

    public float ReconstructionMse(IEnumerable<KeyValuePair<int, float[]>> vectors)
    {
        float total = 0f;
        long count = 0;

        foreach (var v in vectors)
        {
            var enc = Encode(v.Value);
            var dec = Decode(enc);
            for (int i = 0; i < _dimension; i++)
            {
                float diff = v.Value[i] - dec[i];
                total += diff * diff;
                count++;
            }
        }

        return count == 0 ? 0f : (total / count);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    private int QuantizeIndex(float x)
    {
        int lo = 0;
        int hi = _thresholds.Length;

        while (lo < hi)
        {
            int mid = lo + ((hi - lo) >> 1);
            if (x <= _thresholds[mid])
            {
                hi = mid;
            }
            else
            {
                lo = mid + 1;
            }
        }

        return lo;
    }

    private float[] ApplyRandomHadamard(ReadOnlySpan<float> input)
    {
        var buffer = new float[_dimension];
        for (int i = 0; i < _dimension; i++)
            buffer[i] = input[i] * _rotationSigns[i];

        FastWalshHadamard(buffer);

        float scale = 1f / MathF.Sqrt(_dimension);
        for (int i = 0; i < _dimension; i++)
        {
            buffer[i] *= scale;
        }

        return buffer;
    }

    private float[] ApplyInverseRandomHadamard(ReadOnlySpan<float> input)
    {
        var buffer = input.ToArray();
        FastWalshHadamard(buffer);

        float scale = 1f / MathF.Sqrt(_dimension);
        for (int i = 0; i < _dimension; i++)
        {
            buffer[i] *= scale * _rotationSigns[i];
        }

        return buffer;
    }

    private static void FastWalshHadamard(float[] data)
    {
        int n = data.Length;
        ref float ptr = ref MemoryMarshal.GetArrayDataReference(data);
        int vecCount = Vector<float>.Count;

        for (int len = 1; 2 * len <= n; len <<= 1)
        {
            // Once the stride length is large enough, switch to hardware-accelerated SIMD
            if (Vector.IsHardwareAccelerated && len >= vecCount)
            {
                for (int i = 0; i < n; i += (len << 1))
                {
                    ref float ptrU = ref Unsafe.Add(ref ptr, i);
                    ref float ptrV = ref Unsafe.Add(ref ptr, i + len);

                    for (int j = 0; j < len; j += vecCount)
                    {
                        // Load multiple floats at once
                        var uVec = Vector.LoadUnsafe(ref ptrU, (nuint)j);
                        var vVec = Vector.LoadUnsafe(ref ptrV, (nuint)j);

                        // Add/Sub in a single CPU cycle, then store back
                        Vector.StoreUnsafe(uVec + vVec, ref ptrU, (nuint)j);
                        Vector.StoreUnsafe(uVec - vVec, ref ptrV, (nuint)j);
                    }
                }
            }
            else
            {
                // Scalar fallback for the first few iterations (len = 1, 2, 4...)
                for (int i = 0; i < n; i += (len << 1))
                {
                    for (int j = 0; j < len; j++)
                    {
                        float u = Unsafe.Add(ref ptr, i + j);
                        float v = Unsafe.Add(ref ptr, i + j + len);
                        Unsafe.Add(ref ptr, i + j) = u + v;
                        Unsafe.Add(ref ptr, i + j + len) = u - v;
                    }
                }
            }
        }
    }

    private static float L2Norm(ReadOnlySpan<float> v)
    {
        float sum = 0f;
        int i = 0;
        ref float ptr = ref MemoryMarshal.GetReference(v);

        if (Vector.IsHardwareAccelerated && v.Length >= Vector<float>.Count)
        {
            Vector<float> sumVec = Vector<float>.Zero;
            int step = Vector<float>.Count;
            int limit = v.Length - (v.Length % step);

            // Unroll into vector accumulators
            for (; i < limit; i += step)
            {
                var vec = Vector.LoadUnsafe(ref ptr, (nuint)i);
                sumVec += vec * vec; // SIMD multiply and add
            }

            // Sum the lanes of the vector together
            sum += Vector.Sum(sumVec);
        }

        // Mop up any remaining elements if the length wasn't a perfect multiple of the vector size
        for (; i < v.Length; i++)
        {
            float val = Unsafe.Add(ref ptr, i);
            sum += val * val;
        }

        return MathF.Sqrt(sum);
    }
    private static float[] SampleSphereCoordinates(int dimension, int sampleCount, IProvideRandomValues randomValuesGenerator)
    {
        var result = new float[sampleCount];

        for (int s = 0; s < sampleCount; s++)
        {
            var vec = new float[dimension];

            float norm2 = 0;

            for (int i = 0; i < dimension; i++)
            {
                float g = NextGaussian(randomValuesGenerator);
                vec[i] = g;
                norm2 += g * g;
            }

            float invNorm = 1f / MathF.Sqrt(norm2);

            for (int i = 0; i < dimension; i++)
            {
                vec[i] *= invNorm;
            }

            result[s] = vec[randomValuesGenerator.Next(0, dimension)];
        }

        return result;
    }

    private static float[] TrainLloydMax(float[] samples, int levels, int iterations)
    {
        if (levels < 2) throw new ArgumentException("levels must be >= 2.");

        float min = samples.Min();
        float max = samples.Max();

        var centers = new float[levels];

        for (int i = 0; i < levels; i++)
        {
            float t = (float)i / (levels - 1);
            centers[i] = min + t * (max - min);
        }

        for (int iter = 0; iter < iterations; iter++)
        {
            var sums   = new float[levels];
            var counts = new int[levels];

            var thresholds = new float[levels - 1];
            for (int i = 0; i < thresholds.Length; i++)
            {
                thresholds[i] = 0.5f * (centers[i] + centers[i + 1]);
            }

            foreach (var x in samples)
            {
                int idx = 0;
                while (idx < thresholds.Length && x > thresholds[idx])
                {
                    idx++;
                }

                sums[idx] += x;
                counts[idx]++;
            }

            for (int i = 0; i < levels; i++)
            {
                if (counts[i] > 0)
                {
                    centers[i] = sums[i] / counts[i];
                }
            }
        }

        return centers;
    }

    private static float[][] CreateGaussianProjection(int rows, int cols, IProvideRandomValues randomValuesGenerator)
    {
        var proj = new float[rows][];

        float scale = 1f / MathF.Sqrt(rows);

        for (int r = 0; r < rows; r++)
        {
            proj[r] = new float[cols];
            for (int c = 0; c < cols; c++)
                proj[r][c] = NextGaussian(randomValuesGenerator) * scale;
        }

        return proj;
    }

    private static byte[] ProjectToSignBits(ReadOnlySpan<float> vector, float[][] projection)
    {
        int rows = projection.Length;
        byte[] bits = new byte[(rows + 7) / 8];

        for (int r = 0; r < rows; r++)
        {
            float dot = 0f;
            var row = projection[r];
            for (int c = 0; c < vector.Length; c++)
                dot += row[c] * vector[c];

            bool sign = dot >= 0f;
            if (sign)
                bits[r >> 3] |= (byte)(1 << (r & 7));
        }

        return bits;
    }

    private static float NextGaussian(IProvideRandomValues rng)
    {
        float u1 = 1f - rng.NextFloat();
        float u2 = 1f - rng.NextFloat();
        return (float)(MathF.Sqrt(-2f * MathF.Log(u1)) * MathF.Cos(2f * MathF.PI * u2));
    }

    public byte[] ToByteArray()
    {
        var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true);

        const int formatVersion = 1;
        bw.Write(formatVersion);

        bw.Write(_dimension);
        bw.Write(_bits);
        bw.Write(_residualProjections);

        bw.Write(_rotationSigns.Length);
        for (int i = 0; i < _rotationSigns.Length; i++)
            bw.Write(_rotationSigns[i]);

        bw.Write(_codebook.Length);
        for (int i = 0; i < _codebook.Length; i++)
            bw.Write(_codebook[i]);

        bw.Write(_thresholds.Length);
        for (int i = 0; i < _thresholds.Length; i++)
            bw.Write(_thresholds[i]);

        bool hasResidualProjection = _residualProjection is not null;
        bw.Write(hasResidualProjection);
        if (hasResidualProjection)
        {
            bw.Write(_residualProjection!.Length);
            for (int r = 0; r < _residualProjection.Length; r++)
            {
                var row = _residualProjection[r];
                bw.Write(row.Length);
                for (int c = 0; c < row.Length; c++)
                    bw.Write(row[c]);
            }
        }

        bw.Flush();
        return ms.ToArray();
    }

    public static TurboQuant FromByteArray(byte[] bytes)
    {
        if (bytes is null) throw new ArgumentNullException(nameof(bytes));

        using var ms = new MemoryStream(bytes);
        using var br = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: false);

        const int expectedFormatVersion = 1;
        int formatVersion = br.ReadInt32();
        if (formatVersion != expectedFormatVersion)
            throw new InvalidDataException($"Unsupported TurboQuant format version: {formatVersion}.");

        int dimension = br.ReadInt32();
        int bits = br.ReadInt32();
        int residualProjections = br.ReadInt32();

        int rotationLength = br.ReadInt32();
        var rotationSigns = new float[rotationLength];
        for (int i = 0; i < rotationLength; i++)
            rotationSigns[i] = br.ReadSingle();

        int codebookLength = br.ReadInt32();
        var codebook = new float[codebookLength];
        for (int i = 0; i < codebookLength; i++)
            codebook[i] = br.ReadSingle();

        int thresholdsLength = br.ReadInt32();
        var thresholds = new float[thresholdsLength];
        for (int i = 0; i < thresholdsLength; i++)
            thresholds[i] = br.ReadSingle();

        float[][] residualProjection = null;
        bool hasResidualProjection = br.ReadBoolean();
        if (hasResidualProjection)
        {
            int rowCount = br.ReadInt32();
            residualProjection = new float[rowCount][];
            for (int r = 0; r < rowCount; r++)
            {
                int colCount = br.ReadInt32();
                var row = new float[colCount];
                for (int c = 0; c < colCount; c++)
                    row[c] = br.ReadSingle();
                residualProjection[r] = row;
            }
        }

        return new TurboQuant(
            dimension,
            bits,
            residualProjections,
            rotationSigns,
            codebook,
            thresholds,
            residualProjection);
    }
}

public sealed class EncodedVector
{
    public float Norm { get; }
    public byte[] Indices { get; }
    public byte[] ResidualBits { get; }
    public float[] QueryProjections { get; }
    public EncodedVector(float norm, byte[] indices, byte[] residualBits, float[] queryProjections = null)
    {
        Norm = norm;
        Indices = indices;
        ResidualBits = residualBits;
        QueryProjections = queryProjections;
    }

    public int ApproxCompressedBytes => sizeof(float) + sizeof(int) + Indices.Length + (ResidualBits?.Length ?? 0) + (QueryProjections?.Length * sizeof(float) ?? 0);

    public byte[] ToByteArray()
    {
        int residualLength = ResidualBits?.Length ?? 0;
        byte[] data = new byte[8 + Indices.Length + residualLength];
        BitConverter.GetBytes(Norm).CopyTo(data, 0);
        BitConverter.GetBytes(Indices.Length).CopyTo(data, 4);
        Indices.CopyTo(data, 8);
        if (ResidualBits != null)
            ResidualBits.CopyTo(data, 8 + Indices.Length);
        return data;
    }
    public static EncodedVector FromByteArray(byte[] data)
    {
        float norm = BitConverter.ToSingle(data, 0);
        int dimension = BitConverter.ToInt32(data, 4);
        byte[] indices = new byte[dimension];
        Array.Copy(data, 8, indices, 0, dimension);
        byte[] residualBits = null;
        if (data.Length > 8 + dimension)
        {
            int residualLength = data.Length - 8 - dimension;
            residualBits = new byte[residualLength];
            Array.Copy(data, 8 + dimension, residualBits, 0, residualLength);
        }
        return new EncodedVector(norm, indices, residualBits);
    }
}