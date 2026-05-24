// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Sun mean longitude (`M`) and equation of time (`E`). Appendix A.1.

use crate::helper::limit_degrees;

/// Empirical `0.0057183°` correction of equation A1.
const APPARENT_MEAN_LONGITUDE_CORRECTION_DEGREES: f64 = 0.005_718_3;

/// `1° ↔ 4 min` (Earth rotates `360°` in `1440` min).
const MINUTES_OF_TIME_PER_DEGREE: f64 = 4.0;
const MINUTES_OF_TIME_PER_REVOLUTION: f64 = 1440.0;

/// Appendix bound `|E| ≤ 20 min`. Any larger value is the residual `±1440`
/// ambiguity left over by `M` and `α` being independently wrapped.
const EQUATION_OF_TIME_BOUND_MINUTES: f64 = 20.0;

/// Coefficients of `M` (degrees) in `JME`, constant first. Equation A2.
const SUN_MEAN_LONGITUDE_COEFFICIENTS: [f64; 6] = [
    280.466_456_7,
    360_007.698_277_9,
    0.030_320_28,
    1.0 / 49_931.0,
    -1.0 / 15_300.0,
    -1.0 / 2_000_000.0,
];

/// Sun mean longitude `M` (degrees), wrapped into `[0°, 360°)`. Equation A2.
#[must_use]
pub fn sun_mean_longitude(jme: f64) -> f64 {
    let raw = SUN_MEAN_LONGITUDE_COEFFICIENTS
        .iter()
        .rev()
        .fold(0.0_f64, |acc, &c| acc.mul_add(jme, c));
    limit_degrees(raw)
}

/// Equation of time `E` (minutes of time). Equation A1.
#[inline]
#[must_use]
pub fn equation_of_time(
    sun_mean_longitude: f64,
    geocentric_right_ascension: f64,
    delta_psi: f64,
    true_obliquity: f64,
) -> f64 {
    let cos_epsilon = true_obliquity.to_radians().cos();
    let nutation_correction =
        delta_psi.mul_add(cos_epsilon, -APPARENT_MEAN_LONGITUDE_CORRECTION_DEGREES);
    let degrees = sun_mean_longitude - geocentric_right_ascension + nutation_correction;
    let minutes = degrees * MINUTES_OF_TIME_PER_DEGREE;

    if minutes.abs() > EQUATION_OF_TIME_BOUND_MINUTES {
        minutes - MINUTES_OF_TIME_PER_REVOLUTION.copysign(minutes)
    } else {
        minutes
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{APPARENT_MEAN_LONGITUDE_CORRECTION_DEGREES, equation_of_time, sun_mean_longitude};
    use crate::apparent::{aberration_correction, apparent_sun_longitude};
    use crate::equatorial::geocentric_right_ascension;
    use crate::geocentric::{geocentric_latitude, geocentric_longitude};
    use crate::heliocentric::{
        earth_heliocentric_latitude, earth_heliocentric_longitude, earth_radius_vector,
    };
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::obliquity::true_obliquity_of_ecliptic;
    use crate::test_fixtures::{reference_jce, reference_jme};

    fn reference_inputs() -> (f64, f64, f64, f64) {
        let jce = reference_jce();
        let jme = reference_jme();

        let m = sun_mean_longitude(jme);
        let (delta_psi, delta_epsilon) = nutation_in_longitude_and_obliquity(jce);
        let epsilon = true_obliquity_of_ecliptic(jme, delta_epsilon);

        let theta = geocentric_longitude(earth_heliocentric_longitude(jme));
        let beta = geocentric_latitude(earth_heliocentric_latitude(jme));
        let delta_tau = aberration_correction(earth_radius_vector(jme));
        let lambda = apparent_sun_longitude(theta, delta_psi, delta_tau);
        let alpha = geocentric_right_ascension(lambda, beta, epsilon);

        (m, alpha, delta_psi, epsilon)
    }

    #[test]
    fn sun_mean_longitude_matches_table_a5_1() {
        let m = sun_mean_longitude(reference_jme());
        assert!((m - 205.897_172_251_6).abs() < 1e-6);
    }

    #[test]
    fn sun_mean_longitude_at_j2000_collapses_to_constant() {
        let m = sun_mean_longitude(0.0);
        assert!((m - 280.466_456_7).abs() < f64::EPSILON);
    }

    #[test]
    fn sun_mean_longitude_wraps_into_zero_360() {
        for &jme in &[-4.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 4.0] {
            let m = sun_mean_longitude(jme);
            assert!((0.0..360.0).contains(&m));
        }
    }

    #[test]
    fn equation_of_time_matches_table_a5_1() {
        let (m, alpha, delta_psi, epsilon) = reference_inputs();
        let e = equation_of_time(m, alpha, delta_psi, epsilon);
        assert!((e - 14.641_503).abs() < 1e-4);
    }

    #[test]
    fn equation_of_time_collapses_to_correction_when_m_equals_alpha() {
        let expected = -APPARENT_MEAN_LONGITUDE_CORRECTION_DEGREES * 4.0;
        for &v in &[0.0_f64, 90.0, 200.0] {
            let e = equation_of_time(v, v, 0.0, 23.44);
            assert!((e - expected).abs() < 1e-13);
        }
    }

    #[test]
    fn equation_of_time_is_linear_in_m_and_alpha() {
        let baseline = equation_of_time(100.0, 99.0, 0.0, 23.44);
        for &d in &[-1.0_f64, -1e-3, 1e-6, 0.5] {
            let from_m = equation_of_time(100.0 + d, 99.0, 0.0, 23.44);
            let from_alpha = equation_of_time(100.0, 99.0 + d, 0.0, 23.44);
            assert!(4.0_f64.mul_add(-d, from_m - baseline).abs() < 1e-12);
            assert!(4.0_f64.mul_add(d, from_alpha - baseline).abs() < 1e-12);
        }
    }

    #[test]
    fn equation_of_time_scales_delta_psi_by_cos_epsilon() {
        for &eps in &[0.0_f64, 23.44, 60.0] {
            let baseline = equation_of_time(100.0, 99.0, 0.0, eps);
            let cos_eps = eps.to_radians().cos();
            for &d in &[-1e-3_f64, 1e-6, 0.5] {
                let shifted = equation_of_time(100.0, 99.0, d, eps);
                assert!((4.0 * d).mul_add(-cos_eps, shifted - baseline).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn equation_of_time_clamps_overshoot_by_one_revolution() {
        let positive = equation_of_time(358.0, 1.0, 0.0, 23.44);
        let positive_raw = 4.0 * (358.0 - 1.0 - APPARENT_MEAN_LONGITUDE_CORRECTION_DEGREES);
        assert!(positive_raw.abs() > 20.0);
        assert!((positive - (positive_raw - 1440.0)).abs() < 1e-12);

        let negative = equation_of_time(1.0, 358.0, 0.0, 23.44);
        let negative_raw = 4.0 * (1.0 - 358.0 - APPARENT_MEAN_LONGITUDE_CORRECTION_DEGREES);
        assert!(negative_raw.abs() > 20.0);
        assert!((negative - (negative_raw + 1440.0)).abs() < 1e-12);
    }
}
