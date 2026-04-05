import re

with open("Src/HNSW.Net/TurboQuant.cs", "r") as f:
    content = f.read()

turboquant_class = """public sealed class TurboQuant
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
content = re.sub(r'public sealed class TurboQuant.*?^\}', turboquant_class, content, flags=re.MULTILINE | re.DOTALL)

with open("Src/HNSW.Net/TurboQuant.cs", "w") as f:
    f.write(content)
