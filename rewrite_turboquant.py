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
"""

with open("Src/HNSW.Net/TurboQuant.cs", "w") as f:
    f.write(content)
