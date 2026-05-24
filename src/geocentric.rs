// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Sun geocentric longitude (`Θ`) and latitude (`β`). Section 3.3.

use crate::helper::limit_degrees;

/// `Θ = L + 180°`, wrapped into `[0°, 360°)`. Equation 13.
#[inline]
#[must_use]
pub const fn geocentric_longitude(heliocentric_longitude: f64) -> f64 {
    limit_degrees(heliocentric_longitude + 180.0)
}

/// `β = -B`. Equation 14.
#[inline]
#[must_use]
pub const fn geocentric_latitude(heliocentric_latitude: f64) -> f64 {
    -heliocentric_latitude
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{geocentric_latitude, geocentric_longitude};
    use crate::heliocentric::{earth_heliocentric_latitude, earth_heliocentric_longitude};
    use crate::test_fixtures::reference_jme;

    #[test]
    fn geocentric_longitude_matches_table_a5_1() {
        let l = earth_heliocentric_longitude(reference_jme());
        let theta = geocentric_longitude(l);
        assert!((theta - 204.018_261_691_7).abs() < 1e-6);
    }

    #[test]
    fn geocentric_latitude_matches_table_a5_1() {
        let b = earth_heliocentric_latitude(reference_jme());
        let beta = geocentric_latitude(b);
        assert!((beta - 0.000_101_121_9).abs() < 1e-9);
        assert!(beta > 0.0);
    }

    #[test]
    fn geocentric_longitude_wraps_into_zero_360() {
        for &l in &[0.0_f64, 90.0, 179.999, 180.0, 181.0, 359.999, -45.0, 720.5] {
            let theta = geocentric_longitude(l);
            assert!((0.0..360.0).contains(&theta));
        }
    }

    #[test]
    fn geocentric_longitude_offsets_input_by_180_modulo_360() {
        for &l in &[0.0_f64, 30.0, 179.999, 180.0, 200.0, 350.0] {
            let theta = geocentric_longitude(l);
            let residue = (theta - l - 180.0).rem_euclid(360.0);
            let distance_to_zero = residue.min(360.0 - residue);
            assert!(distance_to_zero < 1e-12);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn geocentric_latitude_negates_input() {
        for &b in &[-1.0_f64, -1e-6, -0.0, 0.0, 1e-6, 1.0] {
            assert_eq!(geocentric_latitude(b), -b);
        }
    }
}
