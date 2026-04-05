import re

with open("Src/HNSW.Net/SmallWorldTurboQuant.cs", "r") as f:
    content = f.read()

# I need to update the Searcher distance callback in SmallWorldTurboQuant to use GetDistance(float[], EncodedVector)
# The inner graph is typed `SmallWorld<EncodedVector, float>` and takes `Func<EncodedVector, EncodedVector, float> distance`.
# If I change TurboQuantDistance to throw on (EncodedVector, EncodedVector), building graph won't work because HNSW computes distance between two items in the graph during AddItems.
