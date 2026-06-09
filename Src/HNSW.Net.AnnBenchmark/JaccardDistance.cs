using System;

namespace HNSW.Net.AnnBenchmark
{
    /// <summary>
    /// Jaccard distance between two sets represented as sorted arrays of distinct integer ids.
    ///
    /// distance = 1 - |A ∩ B| / |A ∪ B|
    /// </summary>
    public static class JaccardDistance
    {
        public static float SortedSets(int[] u, int[] v)
        {
            if (u.Length == 0 && v.Length == 0)
            {
                return 0f;
            }

            int intersection = 0;
            int i = 0, j = 0;
            while (i < u.Length && j < v.Length)
            {
                int a = u[i];
                int b = v[j];
                if (a == b)
                {
                    intersection++;
                    i++;
                    j++;
                }
                else if (a < b)
                {
                    i++;
                }
                else
                {
                    j++;
                }
            }

            // |A ∪ B| = |A| + |B| - |A ∩ B|
            int union = u.Length + v.Length - intersection;
            if (union == 0)
            {
                return 0f;
            }

            return 1f - (float)intersection / union;
        }
    }
}
