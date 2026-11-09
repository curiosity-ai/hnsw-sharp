// <copyright file="Regions.cs" company="Curiosity GmbH">
// Copyright (c) Curiosity GmbH. All rights reserved.
// Licensed under the MIT License.
// </copyright>

namespace Hyperspace
{
    using System;

    /// <summary>An axis-aligned box.</summary>
    public sealed class BoxRegion
    {
        /// <summary>Initializes a new instance of the <see cref="BoxRegion"/> class.</summary>
        public BoxRegion(double[] minBounds, double[] maxBounds)
        {
            MinBounds = minBounds ?? throw new ArgumentNullException(nameof(minBounds));
            MaxBounds = maxBounds ?? throw new ArgumentNullException(nameof(maxBounds));
        }

        /// <summary>Gets the lower corner.</summary>
        public double[] MinBounds { get; }

        /// <summary>Gets the upper corner.</summary>
        public double[] MaxBounds { get; }

        /// <summary>Tests whether a point is inside the box (inclusive); only the dimensions present in both bounds are tested.</summary>
        public bool Contains(ReadOnlySpan<double> coords)
        {
            int n = Math.Min(Math.Min(MinBounds.Length, MaxBounds.Length), coords.Length);
            for (int i = 0; i < n; i++)
            {
                if (coords[i] < MinBounds[i] || coords[i] > MaxBounds[i])
                {
                    return false;
                }
            }

            return true;
        }
    }

    /// <summary>
    /// A ConE cone region: the Cartesian product of per-dimension angular sectors with an axis, an aperture and a tolerance.
    /// A coordinate <c>x</c> is inside when <c>|sin((x - axis) / 2)| &lt;= |sin(aperture / 2)| + cen</c>.
    /// </summary>
    public sealed class ConeRegion
    {
        private readonly double[] _thresholds;

        /// <summary>Initializes a new instance of the <see cref="ConeRegion"/> class.</summary>
        public ConeRegion(double[] axes, double[] apertures, double cen)
        {
            Axes = axes ?? throw new ArgumentNullException(nameof(axes));
            Apertures = apertures ?? throw new ArgumentNullException(nameof(apertures));
            Cen = cen;
            _thresholds = new double[apertures.Length];
            for (int i = 0; i < apertures.Length; i++)
            {
                _thresholds[i] = Math.Abs(Math.Sin(apertures[i] * 0.5)) + cen;
            }
        }

        /// <summary>Gets the axes (centers of the sectors).</summary>
        public double[] Axes { get; }

        /// <summary>Gets the apertures (widths of the sectors).</summary>
        public double[] Apertures { get; }

        /// <summary>Gets the tolerance added to every sector.</summary>
        public double Cen { get; }

        /// <summary>Tests whether a point is inside the cone.</summary>
        public bool Contains(ReadOnlySpan<double> coords)
        {
            int n = Math.Min(Math.Min(Axes.Length, coords.Length), _thresholds.Length);
            for (int d = 0; d < n; d++)
            {
                if (Math.Abs(Math.Sin((coords[d] - Axes[d]) * 0.5)) > _thresholds[d])
                {
                    return false;
                }
            }

            return true;
        }
    }

    /// <summary>A ball: Euclidean for most dimensions, hyperbolic when the center has 33 coordinates (a Lorentz point).</summary>
    public sealed class BallRegion
    {
        /// <summary>Initializes a new instance of the <see cref="BallRegion"/> class.</summary>
        public BallRegion(double[] center, double radius)
        {
            Center = center ?? throw new ArgumentNullException(nameof(center));
            Radius = radius;
        }

        /// <summary>Gets the center.</summary>
        public double[] Center { get; }

        /// <summary>Gets the radius.</summary>
        public double Radius { get; }

        /// <summary>Tests whether a point is inside the ball.</summary>
        public bool Contains(ReadOnlySpan<double> coords)
        {
            if (Radius < 0)
            {
                return false;
            }

            if (Center.Length == Metrics.HybridLorentzDimensions)
            {
                if (coords.Length < Metrics.HybridLorentzDimensions)
                {
                    return false;
                }

                double inner = -Center[0] * coords[0];
                for (int i = 1; i < Metrics.HybridLorentzDimensions; i++)
                {
                    inner += Center[i] * coords[i];
                }

                return Math.Max(-inner, 1.0 + 1e-12) <= Math.Cosh(Radius);
            }

            int n = Math.Min(Center.Length, coords.Length);
            double sq = 0;
            for (int i = 0; i < n; i++)
            {
                double d = coords[i] - Center[i];
                sq += d * d;
            }

            return sq <= Radius * Radius;
        }
    }
}
