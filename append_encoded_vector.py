content = """
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
"""
with open("Src/HNSW.Net/TurboQuant.cs", "a") as f:
    f.write(content)
