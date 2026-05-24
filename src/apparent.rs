// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Apparent sun longitude (`λ`) and aberration correction (`Δτ`). Sections 3.6 and 3.7.

/// Aberration constant `20.4898"` of equation 26.
const ABERRATION_CONSTANT_ARCSECONDS: f64 = 20.4898;

/// `Δτ = -20.4898" / (3600 · R)` (degrees). Equation 26.
#[inline]
#[must_use]
pub const fn aberration_correction(earth_radius_vector: f64) -> f64 {
    -ABERRATION_CONSTANT_ARCSECONDS / (3600.0 * earth_radius_vector)
}

/// `λ = Θ + Δψ + Δτ` (degrees). Equation 27.
#[inline]
#[must_use]
pub const fn apparent_sun_longitude(
    geocentric_longitude: f64,
    delta_psi: f64,
    delta_tau: f64,
) -> f64 {
    geocentric_longitude + delta_psi + delta_tau
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{aberration_correction, apparent_sun_longitude};
    use crate::geocentric::geocentric_longitude;
    use crate::heliocentric::{earth_heliocentric_longitude, earth_radius_vector};
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::test_fixtures::{reference_jce, reference_jme};

    #[test]
    fn apparent_sun_longitude_matches_table_a5_1() {
        let jme = reference_jme();
        let theta = geocentric_longitude(earth_heliocentric_longitude(jme));
        let (delta_psi, _) = nutation_in_longitude_and_obliquity(reference_jce());
        let delta_tau = aberration_correction(earth_radius_vector(jme));
        let lambda = apparent_sun_longitude(theta, delta_psi, delta_tau);
        assert!((lambda - 204.008_551_928_1).abs() < 1e-6);
    }

    #[test]
    fn aberration_correction_at_unit_distance() {
        let dt = aberration_correction(1.0);
        assert!((dt - -20.4898 / 3600.0).abs() < f64::EPSILON);
    }

    #[test]
    fn aberration_correction_is_inversely_proportional_to_r() {
        let dt_unit = aberration_correction(1.0);
        for &r in &[0.5_f64, 0.95, 1.05, 2.0, 10.0] {
            let dt = aberration_correction(r);
            assert!(dt.mul_add(r, -dt_unit).abs() < 1e-15);
        }
    }

    #[test]
    fn apparent_sun_longitude_shifts_linearly() {
        let delta_tau = aberration_correction(1.0);
        let baseline = apparent_sun_longitude(0.0, 0.0, delta_tau);
        for &delta in &[-1.0_f64, -1e-3, 1e-6, 0.5] {
            let shifted_theta = apparent_sun_longitude(delta, 0.0, delta_tau);
            let shifted_psi = apparent_sun_longitude(0.0, delta, delta_tau);
            assert!((shifted_theta - baseline - delta).abs() < 1e-13);
            assert!((shifted_psi - baseline - delta).abs() < 1e-13);
        }
    }
}
