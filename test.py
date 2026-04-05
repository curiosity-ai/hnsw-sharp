with open("Src/HNSW.Net/TurboQuant.cs", "r") as f:
    text = f.read()
if "public sealed class TurboQuant" not in text:
    print("TurboQuant class missing")
