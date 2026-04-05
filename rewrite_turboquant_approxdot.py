import re

with open("Src/HNSW.Net/TurboQuant.cs", "r") as f:
    content = f.read()

# We need a symmetric distance method ApproxDot(EncodedVector a, EncodedVector b) to allow graph building.
# In the original TurboQuant v2 WASM, they only implemented asymmetric search (query vs encoded).
# To compute ApproxDot between two EncodedVectors, we can fully decode one of them and then do the asymmetric dot product.
# This is slow, but correct. Alternatively we decode both and do a standard dot.

symmetric_dot = """    public float ApproxDot(EncodedVector a, EncodedVector b)
    {
        float[] decA = Decode(a);
        return ApproxDot(decA, b);
    }"""

content = re.sub(r'public float ApproxDot\(EncodedVector a, EncodedVector b\)\s*\{\s*throw new NotSupportedException\(".*?"\);\s*\}', symmetric_dot, content)

with open("Src/HNSW.Net/TurboQuant.cs", "w") as f:
    f.write(content)
