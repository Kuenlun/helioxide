// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Sun geocentric right ascension (`α`) and declination (`δ`). Sections 3.9 and 3.10.

use crate::helper::limit_degrees;

/// `α = atan2(sin λ · cos ε − tan β · sin ε, cos λ)` (degrees, wrapped into
/// `[0°, 360°)`). Equation 30.
#[inline]
#[must_use]
pub fn geocentric_right_ascension(
    apparent_sun_longitude: f64,
    geocentric_latitude: f64,
    true_obliquity: f64,
) -> f64 {
    let (sin_lambda, cos_lambda) = apparent_sun_longitude.to_radians().sin_cos();
    let (sin_epsilon, cos_epsilon) = true_obliquity.to_radians().sin_cos();
    let tan_beta = geocentric_latitude.to_radians().tan();

    let numerator = sin_lambda.mul_add(cos_epsilon, -(tan_beta * sin_epsilon));
    limit_degrees(numerator.atan2(cos_lambda).to_degrees())
}

/// `δ = arcsin(sin β · cos ε + cos β · sin ε · sin λ)` (degrees, signed,
/// in `[-90°, 90°]`). Equation 31.
#[inline]
#[must_use]
pub fn geocentric_declination(
    apparent_sun_longitude: f64,
    geocentric_latitude: f64,
    true_obliquity: f64,
) -> f64 {
    let (sin_beta, cos_beta) = geocentric_latitude.to_radians().sin_cos();
    let (sin_epsilon, cos_epsilon) = true_obliquity.to_radians().sin_cos();
    let sin_lambda = apparent_sun_longitude.to_radians().sin();

    (cos_beta * sin_epsilon)
        .mul_add(sin_lambda, sin_beta * cos_epsilon)
        .asin()
        .to_degrees()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{geocentric_declination, geocentric_right_ascension};
    use crate::apparent::{aberration_correction, apparent_sun_longitude};
    use crate::geocentric::{geocentric_latitude, geocentric_longitude};
    use crate::heliocentric::{
        earth_heliocentric_latitude, earth_heliocentric_longitude, earth_radius_vector,
    };
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::obliquity::true_obliquity_of_ecliptic;
    use crate::test_fixtures::{reference_jce, reference_jme};

    fn reference_lambda_beta_epsilon() -> (f64, f64, f64) {
        let jme = reference_jme();
        let theta = geocentric_longitude(earth_heliocentric_longitude(jme));
        let beta = geocentric_latitude(earth_heliocentric_latitude(jme));
        let (delta_psi, delta_epsilon) = nutation_in_longitude_and_obliquity(reference_jce());
        let epsilon = true_obliquity_of_ecliptic(jme, delta_epsilon);
        let delta_tau = aberration_correction(earth_radius_vector(jme));
        let lambda = apparent_sun_longitude(theta, delta_psi, delta_tau);
        (lambda, beta, epsilon)
    }

    #[test]
    fn geocentric_right_ascension_matches_table_a5_1() {
        let (lambda, beta, epsilon) = reference_lambda_beta_epsilon();
        let alpha = geocentric_right_ascension(lambda, beta, epsilon);
        assert!((alpha - 202.227_41).abs() < 1e-4_f64);
    }

    #[test]
    fn geocentric_declination_matches_table_a5_1() {
        let (lambda, beta, epsilon) = reference_lambda_beta_epsilon();
        let delta = geocentric_declination(lambda, beta, epsilon);
        assert!((delta - -9.314_34).abs() < 1e-4_f64);
    }

    #[test]
    fn right_ascension_collapses_to_longitude_when_beta_and_epsilon_zero() {
        for &lambda in &[
            15.0_f64, 75.0_f64, 165.0_f64, 200.0_f64, 269.5_f64, 350.0_f64, -45.0_f64, 720.5_f64,
        ] {
            let alpha = geocentric_right_ascension(lambda, 0.0, 0.0);
            assert!((alpha - lambda.rem_euclid(360.0)).abs() < 1e-10_f64);
        }
    }

    #[test]
    fn right_ascension_resolves_atan2_quadrant() {
        for &(lambda, expected) in &[(180.0_f64, 180.0_f64), (270.0_f64, 270.0_f64)] {
            let alpha = geocentric_right_ascension(lambda, 0.0, 23.44);
            assert!((alpha - expected).abs() < 1e-10_f64);
        }
    }

    #[test]
    fn right_ascension_pins_tan_beta_correction() {
        for &(beta, eps) in &[
            (0.001_f64, 23.0_f64),
            (-0.001_f64, 23.0_f64),
            (0.5_f64, 45.0_f64),
            (-0.5_f64, 45.0_f64),
        ] {
            let alpha = geocentric_right_ascension(0.0, beta, eps);
            let expected = (-beta.to_radians().tan() * eps.to_radians().sin())
                .atan()
                .to_degrees()
                .rem_euclid(360.0);
            assert!((alpha - expected).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn right_ascension_wraps_into_zero_360() {
        for &lambda in &[-720.0_f64, -10.0_f64, 250.0_f64, 1_000.0_f64] {
            let alpha = geocentric_right_ascension(lambda, 0.000_1, 23.44);
            assert!((0.0_f64..360.0_f64).contains(&alpha));
        }
    }

    #[test]
    fn declination_collapses_to_beta_when_epsilon_zero() {
        for &beta in &[-30.0_f64, -1e-3_f64, 1e-3_f64, 30.0_f64, 89.999_f64] {
            for &lambda in &[0.0_f64, 90.0_f64, 180.0_f64, 270.0_f64] {
                let delta = geocentric_declination(lambda, beta, 0.0);
                assert!((delta - beta).abs() < 1e-10_f64);
            }
        }
    }

    #[test]
    fn declination_pins_sin_eps_sin_lambda_when_beta_zero() {
        for &lambda in &[15.0_f64, 90.0_f64, 150.0_f64, 210.0_f64, 300.0_f64] {
            for &eps in &[5.0_f64, 23.44_f64, 45.0_f64, 89.0_f64] {
                let delta = geocentric_declination(lambda, 0.0, eps);
                let expected = (eps.to_radians().sin() * lambda.to_radians().sin())
                    .asin()
                    .to_degrees();
                assert!((delta - expected).abs() < 1e-12_f64);
            }
        }
    }
}
