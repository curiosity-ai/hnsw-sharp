import re

with open("Src/HNSW.Net/TurboQuantDistance.cs", "r") as f:
    content = f.read()

new_get_distance = """        public float GetDistance(EncodedVector u, EncodedVector v)
        {
            float dot = _quantizer.ApproxDot(u, v);
            float dist = u.Norm * u.Norm + v.Norm * v.Norm - 2f * dot;
            return dist < 0f ? 0f : dist;
        }"""

content = re.sub(r'public float GetDistance\(EncodedVector u, EncodedVector v\).*?return dist;\s*\}', new_get_distance, content, flags=re.DOTALL)

with open("Src/HNSW.Net/TurboQuantDistance.cs", "w") as f:
    f.write(content)
