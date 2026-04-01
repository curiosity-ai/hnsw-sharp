// Copyright (c) 2025 Relatude.DB - Proventus AS
// Licensed under the MIT License.

namespace HNSW.Net
{
    /// <summary>
    /// Calculates distance using TurboQuant approximation.
    /// </summary>
    public class TurboQuantDistance
    {
        private readonly TurboQuant _quantizer;

        public TurboQuantDistance(TurboQuant quantizer)
        {
            _quantizer = quantizer;
        }

        /// <summary>
        /// Calculates approximate distance.
        /// </summary>
        /// <param name="u">Left encoded vector.</param>
        /// <param name="v">Right encoded vector.</param>
        /// <returns>Distance between u and v.</returns>
        public float GetDistance(EncodedVector u, EncodedVector v)
        {
            return 1f - _quantizer.ApproxDot(u, v);
        }
    }
}
