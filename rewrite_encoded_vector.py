import re

with open("Src/HNSW.Net/TurboQuant.cs", "r") as f:
    content = f.read()

# Replace EncodedVector logic
old_encoded_vector = r"""public sealed class EncodedVector
\{
    public float Norm \{ get; \}
    public byte\[\] Indices \{ get; \}
    public byte\[\] ResidualBits \{ get; \}
    public EncodedVector\(float norm, byte\[\] indices, byte\[\] residualBits\)
    \{
        Norm = norm;
        Indices = indices;
        ResidualBits = residualBits;
    \}

    public int ApproxCompressedBytes => sizeof\(float\) \+ sizeof\(int\) \+ Indices\.Length \+ \(ResidualBits\?\.Length \?\? 0\);

    public byte\[\] ToByteArray\(\)
    \{
        int residualLength = ResidualBits\?\.Length \?\? 0;
        byte\[\] data = new byte\[8 \+ Indices\.Length \+ residualLength\];
        BitConverter\.GetBytes\(Norm\)\.CopyTo\(data, 0\);
        BitConverter\.GetBytes\(Indices\.Length\)\.CopyTo\(data, 4\);
        Indices\.CopyTo\(data, 8\);
        if \(ResidualBits != null\)
            ResidualBits\.CopyTo\(data, 8 \+ Indices\.Length\);
        return data;
    \}
    public static EncodedVector FromByteArray\(byte\[\] data\)
    \{
        float norm = BitConverter\.ToSingle\(data, 0\);
        int dimension = BitConverter\.ToInt32\(data, 4\);
        byte\[\] indices = new byte\[dimension\];
        Array\.Copy\(data, 8, indices, 0, dimension\);
        byte\[\] residualBits = null;
        if \(data\.Length > 8 \+ dimension\)
        \{
            int residualLength = data\.Length - 8 - dimension;
            residualBits = new byte\[residualLength\];
            Array\.Copy\(data, 8 \+ dimension, residualBits, 0, residualLength\);
        \}
        return new EncodedVector\(norm, indices, residualBits\);
    \}
\}"""

new_encoded_vector = """public sealed class EncodedVector
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
}"""

content = re.sub(old_encoded_vector, new_encoded_vector, content)

with open("Src/HNSW.Net/TurboQuant.cs", "w") as f:
    f.write(content)
