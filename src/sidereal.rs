// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Mean (`ν₀`) and apparent (`ν`) sidereal time at Greenwich. Section 3.8.

use crate::helper::limit_degrees;

/// `ν₀` (degrees), wrapped into `[0°, 360°)`. Equation 28.
#[must_use]
pub fn mean_sidereal_time(julian_day: f64) -> f64 {
    let days_since_j2000 = julian_day - 2_451_545.0;
    let jc = days_since_j2000 / 36_525.0;

    // `JC² · (C - D · JC)` fuses the two coupled JC corrections.
    let jc_polynomial = (jc * jc).mul_add(
        (-1.0_f64 / 38_710_000.0).mul_add(jc, 0.000_387_933),
        280.460_618_37,
    );
    let nu0 = 360.985_647_366_29_f64.mul_add(days_since_j2000, jc_polynomial);

    limit_degrees(nu0)
}

/// `ν = ν₀ + Δψ · cos(ε)` (degrees, signed, not wrapped). Equation 29.
#[inline]
#[must_use]
pub fn apparent_sidereal_time(mean_sidereal_time: f64, delta_psi: f64, epsilon: f64) -> f64 {
    delta_psi.mul_add(epsilon.to_radians().cos(), mean_sidereal_time)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{apparent_sidereal_time, mean_sidereal_time};
    use crate::helper::limit_degrees;
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::obliquity::true_obliquity_of_ecliptic;
    use crate::test_fixtures::{reference_jce, reference_jd, reference_jme};

    // ν₀ and ν are not printed in Table A5.1; the expected values below are
    // the NREL reference implementation's output for the same instant
    // (ν is also derivable from the table as H − σ + α).
    #[test]
    fn mean_sidereal_time_matches_reference_implementation() {
        let nu0 = mean_sidereal_time(reference_jd());
        assert!((nu0 - 318.515_578).abs() < 1e-4);
    }

    #[test]
    fn apparent_sidereal_time_matches_reference_implementation() {
        let (delta_psi, delta_epsilon) = nutation_in_longitude_and_obliquity(reference_jce());
        let epsilon = true_obliquity_of_ecliptic(reference_jme(), delta_epsilon);
        let nu = apparent_sidereal_time(mean_sidereal_time(reference_jd()), delta_psi, epsilon);
        assert!((nu - 318.511_910).abs() < 1e-4);
    }

    #[test]
    fn mean_sidereal_time_at_j2000_collapses_to_constant() {
        let nu0 = mean_sidereal_time(2_451_545.0);
        assert!((nu0 - 280.460_618_37).abs() < 1e-12);
    }

    #[test]
    fn mean_sidereal_time_diurnal_rate() {
        let shift = mean_sidereal_time(2_451_546.0) - mean_sidereal_time(2_451_545.0);
        assert!((shift - 0.985_647_366_29).abs() < 1e-9);
    }

    #[test]
    fn mean_sidereal_time_higher_order_terms_at_jc_100() {
        let jc = 100.0_f64;
        let jd = jc.mul_add(36_525.0, 2_451_545.0);
        let full = mean_sidereal_time(jd);
        let linear_only =
            limit_degrees(360.985_647_366_29_f64.mul_add(jd - 2_451_545.0, 280.460_618_37));
        let higher_order = limit_degrees(full - linear_only);
        assert!((higher_order - 3.853_496_881_94).abs() < 5e-6);
    }

    #[test]
    fn mean_sidereal_time_wraps_into_zero_360() {
        for &jd in &[
            -1_000_000.0_f64,
            0.0,
            2_451_545.0,
            2_452_930.312_847,
            5_000_000.0,
        ] {
            let nu0 = mean_sidereal_time(jd);
            assert!((0.0..360.0).contains(&nu0));
        }
    }

    #[test]
    fn apparent_sidereal_time_shifts_linearly_with_delta_psi() {
        let nu0 = 100.0_f64;
        let epsilon = 23.440_465_f64;
        let cos_eps = epsilon.to_radians().cos();
        let baseline = apparent_sidereal_time(nu0, 0.0, epsilon);
        for &dp in &[-1e-2_f64, -1e-4, 1e-6, 5e-3] {
            let shifted = apparent_sidereal_time(nu0, dp, epsilon);
            assert!(dp.mul_add(-cos_eps, shifted - baseline).abs() < 1e-13);
        }
    }

    #[test]
    fn apparent_sidereal_time_correction_vanishes_at_quarter_turn_obliquity() {
        for &dp in &[-1.0_f64, 0.0, 1.0] {
            let nu = apparent_sidereal_time(217.5, dp, 90.0);
            assert!((nu - 217.5).abs() < 1e-13);
        }
    }
}
