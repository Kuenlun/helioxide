// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Mean (`ε₀`) and true (`ε`) obliquity of the ecliptic. Section 3.5.

/// Coefficients of `ε₀` (arc seconds) in `U = JME / 10`, constant first.
/// Equation 24.
const MEAN_OBLIQUITY_COEFFICIENTS: [f64; 11] = [
    84_381.448, -4_680.93, -1.55, 1_999.25, -51.38, -249.67, -39.05, 7.12, 27.87, 5.79, 2.45,
];

/// Mean obliquity of the ecliptic `ε₀` (arc seconds). Equation 24.
#[must_use]
pub fn mean_obliquity_of_ecliptic_arcseconds(jme: f64) -> f64 {
    let u = jme / 10.0;
    MEAN_OBLIQUITY_COEFFICIENTS
        .iter()
        .rev()
        .fold(0.0_f64, |acc, &c| acc.mul_add(u, c))
}

/// True obliquity of the ecliptic `ε = ε₀ / 3600 + Δε` (degrees). Equation 25.
#[must_use]
pub fn true_obliquity_of_ecliptic(jme: f64, delta_epsilon: f64) -> f64 {
    mean_obliquity_of_ecliptic_arcseconds(jme) / 3600.0 + delta_epsilon
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{mean_obliquity_of_ecliptic_arcseconds, true_obliquity_of_ecliptic};
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::test_fixtures::{reference_jce, reference_jme};

    #[test]
    fn true_obliquity_matches_table_a5_1() {
        let (_, delta_epsilon) = nutation_in_longitude_and_obliquity(reference_jce());
        let epsilon = true_obliquity_of_ecliptic(reference_jme(), delta_epsilon);
        assert!((epsilon - 23.440_465).abs() < 1e-6);
    }

    #[test]
    fn true_obliquity_at_j2000_collapses_to_constant() {
        let epsilon = true_obliquity_of_ecliptic(0.0, 0.0);
        assert!((epsilon - 84_381.448 / 3600.0).abs() < f64::EPSILON);
    }

    #[test]
    fn true_obliquity_offsets_by_delta_epsilon() {
        let jme = reference_jme();
        let baseline = true_obliquity_of_ecliptic(jme, 0.0);
        for &delta in &[-0.5_f64, -1e-3, 1e-6, 0.5] {
            let shifted = true_obliquity_of_ecliptic(jme, delta);
            assert!((shifted - baseline - delta).abs() < 1e-13);
        }
    }

    #[test]
    fn mean_obliquity_arcseconds_matches_table_a5_1_via_round_trip() {
        let (_, delta_epsilon) = nutation_in_longitude_and_obliquity(reference_jce());
        let from_fn = mean_obliquity_of_ecliptic_arcseconds(reference_jme());
        let from_round_trip = (23.440_465_f64 - delta_epsilon) * 3600.0;
        assert!((from_fn - from_round_trip).abs() < 0.1);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn mean_obliquity_arcseconds_at_j2000_collapses_to_constant() {
        assert_eq!(mean_obliquity_of_ecliptic_arcseconds(0.0), 84_381.448);
    }
}
