import re

content = """// Copyright (c) 2025 Relatude.DB - Proventus AS
// Licensed under the MIT License. https://github.com/Relatude/Relatude.DB/blob/main/LICENSE.txt

using System.Collections.Generic;
using System;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace HNSW.Net;

public sealed class EncodedVector
{
    public float Norm { get; }
    public byte[] Payload { get; }

    public EncodedVector(float norm, byte[] payload)
    {
        Norm = norm;
        Payload = payload;
    }

    public int ApproxCompressedBytes => sizeof(float) + sizeof(int) + Payload.Length;

    public byte[] ToByteArray()
    {
        byte[] data = new byte[8 + Payload.Length];
        BitConverter.GetBytes(Norm).CopyTo(data, 0);
        BitConverter.GetBytes(Payload.Length).CopyTo(data, 4);
        Payload.CopyTo(data, 8);
        return data;
    }

    public static EncodedVector FromByteArray(byte[] data)
    {
        float norm = BitConverter.ToSingle(data, 0);
        int payloadLength = BitConverter.ToInt32(data, 4);
        byte[] payload = new byte[payloadLength];
        Array.Copy(data, 8, payload, 0, payloadLength);
        return new EncodedVector(norm, payload);
    }
}

internal class RotationOperator
{
    private const ulong RNG_MULTIPLIER = 1103515245;
    private const ulong RNG_INCREMENT = 12345;

    public int Dim { get; }
    public uint Seed { get; }
    public float[] Matrix { get; }
    public float[] MatrixT { get; }

    private static ulong NextRng(ref ulong seed)
    {
        seed = unchecked(seed * RNG_MULTIPLIER + RNG_INCREMENT);
        return seed;
    }

    private static float RandF32(ref ulong seed)
    {
        long masked = (long)(NextRng(ref seed) % (1UL << 31));
        float val = masked / (float)(1UL << 31);
        return val == 0f ? 0.00001f : val;
    }

    private static float RandGaussian(ref ulong seed)
    {
        float v1 = RandF32(ref seed);
        float v2 = RandF32(ref seed);
        return MathF.Sqrt(-2.0f * MathF.Log(v1)) * MathF.Cos(2.0f * MathF.PI * v2);
    }

    private static float GaussianCoeff(ulong seed, int row, int col)
    {
        ulong rngState = unchecked(seed + (ulong)(row * 31 + col));
        return RandGaussian(ref rngState);
    }

    public RotationOperator(int dim, uint seed)
    {
        Dim = dim;
        Seed = seed;
        Matrix = new float[dim * dim];
        MatrixT = new float[dim * dim];

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                Matrix[i * dim + j] = GaussianCoeff(seed, i, j);
            }
        }

        Orthogonalize(Matrix, dim);

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                MatrixT[j * dim + i] = Matrix[i * dim + j];
            }
        }
    }

    private static void Orthogonalize(float[] matrix, int dim)
    {
        double[] cols = new double[dim * dim];
        for (int col = 0; col < dim; col++)
        {
            for (int row = 0; row < dim; row++)
            {
                cols[col * dim + row] = matrix[row * dim + col];
            }
        }

        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < i; j++)
            {
                double dot = 0;
                for (int row = 0; row < dim; row++)
                {
                    dot += cols[i * dim + row] * cols[j * dim + row];
                }
                for (int row = 0; row < dim; row++)
                {
                    cols[i * dim + row] -= dot * cols[j * dim + row];
                }
            }
            double colNorm = 0;
            for (int row = 0; row < dim; row++)
            {
                double v = cols[i * dim + row];
                colNorm += v * v;
            }
            colNorm = Math.Sqrt(colNorm);
            if (colNorm > 0)
            {
                double inv = 1.0 / colNorm;
                for (int row = 0; row < dim; row++)
                {
                    cols[i * dim + row] *= inv;
                }
            }
        }

        for (int col = 0; col < dim; col++)
        {
            for (int row = 0; row < dim; row++)
            {
                matrix[row * dim + col] = (float)cols[col * dim + row];
            }
        }
    }

    public void MatVecMul(ReadOnlySpan<float> input, Span<float> output)
    {
        int d = Dim;
        for (int i = 0; i < d; i++)
        {
            output[i] = RowDot(Matrix.AsSpan(i * d, d), input);
        }
    }

    public void MatVecMulTransposed(ReadOnlySpan<float> input, Span<float> output)
    {
        int d = Dim;
        for (int i = 0; i < d; i++)
        {
            output[i] = RowDot(MatrixT.AsSpan(i * d, d), input);
        }
    }

    public void Rotate(ReadOnlySpan<float> input, Span<float> output)
    {
        MatVecMul(input, output);
    }

    private static float RowDot(ReadOnlySpan<float> matrixRow, ReadOnlySpan<float> input)
    {
        float total = 0;
        int d = matrixRow.Length;
        int i = 0;
        if (Vector.IsHardwareAccelerated)
        {
            int step = Vector<float>.Count;
            int limit = d - (d % step);
            Vector<float> sumVec = Vector<float>.Zero;
            ref float pM = ref MemoryMarshal.GetReference(matrixRow);
            ref float pI = ref MemoryMarshal.GetReference(input);
            for (; i < limit; i += step)
            {
                var m = Vector.LoadUnsafe(ref pM, (nuint)i);
                var v = Vector.LoadUnsafe(ref pI, (nuint)i);
                sumVec += m * v;
            }
            total += Vector.Sum(sumVec);
        }
        for (; i < d; i++)
        {
            total += matrixRow[i] * input[i];
        }
        return total;
    }
}

internal static class Polar
{
    private const int R_BITS = 4;
    private const int THETA_BITS = 3;
    private const int BITS_PER_PAIR = R_BITS + THETA_BITS;
    private const float R_LEVELS = 15.0f;
    private const float THETA_LEVELS = 7.0f;
    private const int ANGLE_BUCKETS = 8;
    private const float PI = MathF.PI;
    private const float TWO_PI = 2.0f * PI;

    private static readonly float[] polar_sin_table;
    private static readonly float[] polar_cos_table;
    private static readonly float[][] direction_vectors;

    static Polar()
    {
        polar_sin_table = new float[ANGLE_BUCKETS];
        polar_cos_table = new float[ANGLE_BUCKETS];
        direction_vectors = new float[ANGLE_BUCKETS][];

        for (int i = 0; i < ANGLE_BUCKETS; i++)
        {
            float theta = (float)i / THETA_LEVELS * TWO_PI - PI;
            polar_sin_table[i] = MathF.Sin(theta);
            polar_cos_table[i] = MathF.Cos(theta);
            direction_vectors[i] = new float[] { MathF.Cos(theta), MathF.Sin(theta) };
        }
    }

    private static int FindNearestAngleBucket(float x, float y)
    {
        int bestBucket = 0;
        float bestDot = -1.0f;
        for (int i = 0; i < ANGLE_BUCKETS; i++)
        {
            float dot = x * direction_vectors[i][0] + y * direction_vectors[i][1];
            if (dot > bestDot)
            {
                bestDot = dot;
                bestBucket = i;
            }
        }
        return bestBucket;
    }

    public static byte[] Encode(ReadOnlySpan<float> rotated, float maxR)
    {
        int dim = rotated.Length;
        if (dim == 0 || dim % 2 != 0) throw new ArgumentException("Invalid Dimension");

        int numPairs = dim / 2;
        int polarBits = numPairs * BITS_PER_PAIR;
        int polarBytes = (polarBits + 7) / 8;

        byte[] result = new byte[polarBytes];

        int bitPos = 0;
        for (int i = 0; i < numPairs; i++)
        {
            float x = rotated[i * 2];
            float y = rotated[i * 2 + 1];
            float r = MathF.Sqrt(x * x + y * y);

            int rQuant = (int)(r / maxR * R_LEVELS);
            if (rQuant > 15) rQuant = 15;
            int thetaBucket = FindNearestAngleBucket(x, y);
            int combined = (rQuant << THETA_BITS) | thetaBucket;

            for (int j = 0; j < BITS_PER_PAIR; j++)
            {
                int bit = (combined >> (BITS_PER_PAIR - 1 - j)) & 1;
                if (bit == 1)
                {
                    result[bitPos / 8] |= (byte)(1 << (bitPos % 8));
                }
                bitPos++;
            }
        }
        return result;
    }

    public struct Pair
    {
        public float dx;
        public float dy;
    }

    private static (float r, int bucket) UnpackOne(ReadOnlySpan<byte> compressed, int bitPos)
    {
        int combined = 0;
        for (int j = 0; j < BITS_PER_PAIR; j++)
        {
            int bit = (compressed[(bitPos + j) / 8] >> ((bitPos + j) % 8)) & 1;
            combined = (combined << 1) | bit;
        }
        float r = (float)((combined >> THETA_BITS) & 0xF) / R_LEVELS;
        int bucket = combined & 0x7;
        return (r, bucket);
    }

    public static Pair ReconstructPair(ReadOnlySpan<byte> compressed, int bitPos, float maxR)
    {
        var unpacked = UnpackOne(compressed, bitPos);
        float r = unpacked.r * maxR;
        int bucket = unpacked.bucket;
        return new Pair
        {
            dx = r * polar_cos_table[bucket],
            dy = r * polar_sin_table[bucket]
        };
    }

    public static void DecodeInto(Span<float> outVec, ReadOnlySpan<byte> compressed, float maxR)
    {
        int dim = outVec.Length;
        if (dim == 0 || dim % 2 != 0) throw new ArgumentException("Invalid Dimension");

        int numPairs = dim / 2;
        int bitPos = 0;

        for (int i = 0; i < numPairs; i++)
        {
            var p = ReconstructPair(compressed, bitPos, maxR);
            outVec[i * 2] = p.dx;
            outVec[i * 2 + 1] = p.dy;
            bitPos += BITS_PER_PAIR;
        }
    }

    public static float DotProduct(ReadOnlySpan<float> q, ReadOnlySpan<byte> compressed, float maxR)
    {
        int dim = q.Length;
        if (dim == 0 || dim % 2 != 0) return 0f;

        int numPairs = dim / 2;
        float sum = 0f;
        int bitPos = 0;

        for (int i = 0; i < numPairs; i++)
        {
            var p = ReconstructPair(compressed, bitPos, maxR);
            sum += q[i * 2] * p.dx + q[i * 2 + 1] * p.dy;
            bitPos += BITS_PER_PAIR;
        }

        return sum;
    }
}

internal class QjlWorkspace
{
    public float[] Projected { get; }
    public float[] SignVec { get; }
    public float[] StSign { get; }

    public QjlWorkspace(int dim)
    {
        Projected = new float[dim];
        SignVec = new float[dim];
        StSign = new float[dim];
    }
}

internal static class Qjl
{
    private const float SQRT_PI_OVER_2 = 1.2533141373155003f;

    public static byte[] EncodeWithWorkspace(ReadOnlySpan<float> residual, RotationOperator rotOp, QjlWorkspace workspace)
    {
        int d = residual.Length;
        if (d == 0) throw new ArgumentException("Invalid Dimension");

        rotOp.MatVecMul(residual, workspace.Projected);

        int bitsBytes = (d + 7) / 8;
        byte[] result = new byte[bitsBytes];

        int i = 0;
        for (; i + 8 <= d; i += 8)
        {
            byte b = 0;
            for (int bit = 0; bit < 8; bit++)
            {
                if (workspace.Projected[i + bit] > 0)
                {
                    b |= (byte)(1 << bit);
                }
            }
            result[i / 8] = b;
        }
        for (; i < d; i++)
        {
            if (workspace.Projected[i] > 0)
            {
                result[i / 8] |= (byte)(1 << (i % 8));
            }
        }
        return result;
    }

    public static void SignToVector(ReadOnlySpan<byte> signBits, int dim, Span<float> output)
    {
        for (int i = 0; i < dim; i++)
        {
            int bit = (signBits[i / 8] >> (i % 8)) & 1;
            output[i] = bit == 1 ? 1.0f : -1.0f;
        }
    }

    public static void DecodeIntoRotated(Span<float> outVec, ReadOnlySpan<byte> qjlBits, float gamma, QjlWorkspace workspace)
    {
        int dim = outVec.Length;
        if (dim == 0) return;

        SignToVector(qjlBits, dim, workspace.SignVec);

        float scaleVal = SQRT_PI_OVER_2 / (float)dim * gamma;

        for (int i = 0; i < dim; i++)
        {
            outVec[i] = workspace.SignVec[i] * scaleVal;
        }
    }

    public static float EstimateDotWithWorkspace(ReadOnlySpan<float> rotatedQ, ReadOnlySpan<byte> qjlBits, float gamma, QjlWorkspace workspace)
    {
        int d = rotatedQ.Length;
        if (d == 0) return 0f;

        SignToVector(qjlBits, d, workspace.SignVec);

        float dotSum = 0f;
        for (int i = 0; i < d; i++)
        {
            dotSum += rotatedQ[i] * workspace.SignVec[i];
        }

        float scaleVal = SQRT_PI_OVER_2 / (float)d * gamma;
        return dotSum * scaleVal;
    }
}

public sealed class TurboQuant
{
    private readonly int _dimension;
    private readonly uint _seed;
    private readonly RotationOperator _rotOp;

    public TurboQuant(int dimension, uint seed)
    {
        if (dimension == 0 || dimension % 2 != 0)
            throw new ArgumentException("Dimension must be positive and even.");

        _dimension = dimension;
        _seed = seed;
        _rotOp = new RotationOperator(dimension, seed);
    }

    public static TurboQuant Create(int dimension, IProvideRandomValues randomValuesGenerator, int bits = 3, int residualProjections = 0, int lloydMaxTrainingSamples = 200_000, int lloydMaxIterations = 30)
    {
        uint seed = (uint)randomValuesGenerator.Next(0, int.MaxValue);
        return new TurboQuant(dimension, seed);
    }

    public EncodedVector Encode(ReadOnlySpan<float> vector)
    {
        if (vector.Length != _dimension)
            throw new ArgumentException($"Expected vector length {_dimension}.");

        float trueNorm = L2Norm(vector);

        float[] scratchRotated = new float[_dimension];
        _rotOp.Rotate(vector, scratchRotated);

        float maxR = 0f;
        for (int i = 0; i < _dimension / 2; i++)
        {
            float x = scratchRotated[i * 2];
            float y = scratchRotated[i * 2 + 1];
            float r = MathF.Sqrt(x * x + y * y);
            if (r > maxR) maxR = r;
        }
        if (maxR == 0) maxR = 1.0f;

        byte[] polarEncoded = Polar.Encode(scratchRotated, maxR);

        float[] scratchResidual = new float[_dimension];
        int bitPos = 0;
        for (int i = 0; i < _dimension / 2; i++)
        {
            var pair = Polar.ReconstructPair(polarEncoded, bitPos, maxR);
            bitPos += 7;

            scratchResidual[i * 2] = scratchRotated[i * 2] - pair.dx;
            scratchResidual[i * 2 + 1] = scratchRotated[i * 2 + 1] - pair.dy;
        }

        float gamma = L2Norm(scratchResidual);

        var ws = new QjlWorkspace(_dimension);
        byte[] qjlEncoded = Qjl.EncodeWithWorkspace(scratchResidual, _rotOp, ws);

        int totalSize = 22 + polarEncoded.Length + qjlEncoded.Length;
        byte[] payload = new byte[totalSize];

        payload[0] = 1; // version
        BitConverter.GetBytes(_dimension).CopyTo(payload, 1);
        payload[5] = 0; // reserved
        BitConverter.GetBytes(polarEncoded.Length).CopyTo(payload, 6);
        BitConverter.GetBytes(qjlEncoded.Length).CopyTo(payload, 10);
        BitConverter.GetBytes(maxR).CopyTo(payload, 14);
        BitConverter.GetBytes(gamma).CopyTo(payload, 18);

        Array.Copy(polarEncoded, 0, payload, 22, polarEncoded.Length);
        Array.Copy(qjlEncoded, 0, payload, 22 + polarEncoded.Length, qjlEncoded.Length);

        return new EncodedVector(trueNorm, payload);
    }

    public float[] Decode(EncodedVector encoded)
    {
        byte[] compressed = encoded.Payload;
        if (compressed.Length < 22) throw new InvalidDataException("Invalid Payload");

        int dim = BitConverter.ToInt32(compressed, 1);
        if (dim != _dimension) throw new InvalidDataException("Invalid Payload");

        int polarBytes = BitConverter.ToInt32(compressed, 6);
        int qjlBytes = BitConverter.ToInt32(compressed, 10);
        float maxR = BitConverter.ToSingle(compressed, 14);
        float gamma = BitConverter.ToSingle(compressed, 18);

        ReadOnlySpan<byte> polarEncoded = new ReadOnlySpan<byte>(compressed, 22, polarBytes);
        ReadOnlySpan<byte> qjlEncoded = new ReadOnlySpan<byte>(compressed, 22 + polarBytes, qjlBytes);

        float[] scratchPolarDecoded = new float[_dimension];
        Polar.DecodeInto(scratchPolarDecoded, polarEncoded, maxR);

        float[] scratchQjlDecoded = new float[_dimension];
        var ws = new QjlWorkspace(_dimension);
        Qjl.DecodeIntoRotated(scratchQjlDecoded, qjlEncoded, gamma, ws);

        for (int i = 0; i < _dimension; i++)
        {
            scratchPolarDecoded[i] += scratchQjlDecoded[i];
        }

        float[] result = new float[_dimension];
        _rotOp.MatVecMulTransposed(scratchPolarDecoded, result);

        return result;
    }

    public float ApproxDot(EncodedVector a, EncodedVector b)
    {
        throw new NotSupportedException("TurboQuant v2 uses query vs encoded distance calculation. ApproxDot(a, b) not fully analogous.");
    }

    public float ApproxDot(ReadOnlySpan<float> q, EncodedVector encoded)
    {
        byte[] compressed = encoded.Payload;
        if (compressed.Length < 22) return 0;
        int dim = BitConverter.ToInt32(compressed, 1);
        if (q.Length != _dimension || dim != _dimension) return 0;

        int polarBytes = BitConverter.ToInt32(compressed, 6);
        int qjlBytes = BitConverter.ToInt32(compressed, 10);
        float maxR = BitConverter.ToSingle(compressed, 14);
        float gamma = BitConverter.ToSingle(compressed, 18);

        ReadOnlySpan<byte> polarEncoded = new ReadOnlySpan<byte>(compressed, 22, polarBytes);
        ReadOnlySpan<byte> qjlEncoded = new ReadOnlySpan<byte>(compressed, 22 + polarBytes, qjlBytes);

        float[] scratchRotated = new float[_dimension];
        _rotOp.Rotate(q, scratchRotated);

        float polarSum = Polar.DotProduct(scratchRotated, polarEncoded, maxR);
        var ws = new QjlWorkspace(_dimension);
        float qjlSum = Qjl.EstimateDotWithWorkspace(scratchRotated, qjlEncoded, gamma, ws);

        return polarSum + qjlSum;
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

            for (; i < limit; i += step)
            {
                var vec = Vector.LoadUnsafe(ref ptr, (nuint)i);
                sumVec += vec * vec;
            }

            sum += Vector.Sum(sumVec);
        }

        for (; i < v.Length; i++)
        {
            float val = Unsafe.Add(ref ptr, i);
            sum += val * val;
        }

        return MathF.Sqrt(sum);
    }

    public byte[] ToByteArray()
    {
        using var ms = new MemoryStream();
        using var bw = new BinaryWriter(ms, System.Text.Encoding.UTF8, leaveOpen: true);

        const int formatVersion = 2;
        bw.Write(formatVersion);
        bw.Write(_dimension);
        bw.Write(_seed);

        bw.Flush();
        return ms.ToArray();
    }

    public static TurboQuant FromByteArray(byte[] bytes)
    {
        if (bytes is null) throw new ArgumentNullException(nameof(bytes));

        using var ms = new MemoryStream(bytes);
        using var br = new BinaryReader(ms, System.Text.Encoding.UTF8, leaveOpen: false);

        int formatVersion = br.ReadInt32();
        if (formatVersion != 2)
            throw new InvalidDataException($"Unsupported TurboQuant format version: {formatVersion}.");

        int dimension = br.ReadInt32();
        uint seed = br.ReadUInt32();

        return new TurboQuant(dimension, seed);
    }
}
"""

with open("Src/HNSW.Net/TurboQuant.cs", "w") as f:
    f.write(content)
