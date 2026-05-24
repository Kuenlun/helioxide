// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Topocentric elevation, refraction, zenith and azimuth angles.
//! Sections 3.14 and 3.15.

use crate::helper::limit_degrees;

/// `1010 mbar` (equation 42 pressure-ratio denominator).
pub const STANDARD_PRESSURE_MILLIBARS: f64 = 1010.0;

/// `283 K` (equation 42 temperature-ratio numerator, `10 °C`).
pub const REFERENCE_TEMPERATURE_KELVIN: f64 = 283.0;

/// `273` of equation 42's `273 + T` (the paper drops the `0.15` of strict IAU).
pub const KELVIN_OFFSET_FROM_CELSIUS: f64 = 273.0;

/// `-0.8333°`: sun-disk radius plus horizon-level refraction (`0.26667° + 0.5667°`,
/// rounded per appendix A.2). Below this elevation the upper limb is at or
/// below the geometric horizon, so equation 42's "below the horizon" branch
/// collapses `Δe` to zero.
const HORIZON_CUTOFF_ELEVATION_DEGREES: f64 = -0.8333;

/// `e₀ = arcsin(sin φ · sin δ' + cos φ · cos δ' · cos H')` (degrees,
/// in `[-90°, 90°]`). Equation 41.
#[inline]
#[must_use]
pub fn topocentric_elevation_without_refraction(
    observer_latitude: f64,
    topocentric_declination: f64,
    topocentric_local_hour_angle: f64,
) -> f64 {
    let (sin_phi, cos_phi) = observer_latitude.to_radians().sin_cos();
    let (sin_delta_prime, cos_delta_prime) = topocentric_declination.to_radians().sin_cos();
    let cos_h_prime = topocentric_local_hour_angle.to_radians().cos();

    (cos_phi * cos_delta_prime)
        .mul_add(cos_h_prime, sin_phi * sin_delta_prime)
        .asin()
        .to_degrees()
}

/// `Δe = (P/1010) · (283/(273+T)) · 1.02 / (60 · tan(e₀ + 10.3/(e₀+5.11)))`
/// (degrees), or zero below the visible horizon. Equation 42.
#[inline]
#[must_use]
pub fn atmospheric_refraction(
    elevation_without_refraction: f64,
    pressure_millibars: f64,
    temperature_celsius: f64,
) -> f64 {
    if elevation_without_refraction < HORIZON_CUTOFF_ELEVATION_DEGREES {
        return 0.0;
    }

    let pressure_ratio = pressure_millibars / STANDARD_PRESSURE_MILLIBARS;
    let temperature_ratio =
        REFERENCE_TEMPERATURE_KELVIN / (KELVIN_OFFSET_FROM_CELSIUS + temperature_celsius);
    let aux = elevation_without_refraction + 10.3 / (elevation_without_refraction + 5.11);

    pressure_ratio * temperature_ratio * 1.02 / (60.0 * aux.to_radians().tan())
}

/// `e = e₀ + Δe` (degrees). Equation 43.
#[inline]
#[must_use]
pub const fn topocentric_elevation_corrected(
    elevation_without_refraction: f64,
    refraction: f64,
) -> f64 {
    elevation_without_refraction + refraction
}

/// `θ = 90° − (e₀ + Δe)` (degrees). Equations 43 and 44 fused.
#[inline]
#[must_use]
pub const fn topocentric_zenith_angle(elevation_without_refraction: f64, refraction: f64) -> f64 {
    90.0 - (elevation_without_refraction + refraction)
}

/// `Γ = atan2(sin H', cos H' · sin φ − tan δ' · cos φ)` (degrees, wrapped
/// into `[0°, 360°)`, measured westward from south). Equation 45.
#[inline]
#[must_use]
pub fn astronomers_azimuth(
    topocentric_local_hour_angle: f64,
    observer_latitude: f64,
    topocentric_declination: f64,
) -> f64 {
    let (sin_h_prime, cos_h_prime) = topocentric_local_hour_angle.to_radians().sin_cos();
    let (sin_phi, cos_phi) = observer_latitude.to_radians().sin_cos();
    let tan_delta_prime = topocentric_declination.to_radians().tan();

    let denominator = cos_h_prime.mul_add(sin_phi, -(tan_delta_prime * cos_phi));
    limit_degrees(sin_h_prime.atan2(denominator).to_degrees())
}

/// `Φ = Γ + 180°` (degrees, wrapped, measured eastward from north).
/// Equation 46.
#[inline]
#[must_use]
pub const fn topocentric_azimuth_angle(astronomers_azimuth: f64) -> f64 {
    limit_degrees(astronomers_azimuth + 180.0)
}

/// Re-express `Γ` in the signed range `(-180°, 180°]`. Negative is east of
/// south, positive is west. Pass only `Γ ∈ [0°, 360°)`.
#[inline]
#[must_use]
pub const fn astronomers_azimuth_signed(astronomers_azimuth_zero_360: f64) -> f64 {
    if astronomers_azimuth_zero_360 > 180.0 {
        astronomers_azimuth_zero_360 - 360.0
    } else {
        astronomers_azimuth_zero_360
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{
        HORIZON_CUTOFF_ELEVATION_DEGREES, REFERENCE_TEMPERATURE_KELVIN,
        STANDARD_PRESSURE_MILLIBARS, astronomers_azimuth, astronomers_azimuth_signed,
        atmospheric_refraction, topocentric_azimuth_angle, topocentric_elevation_corrected,
        topocentric_elevation_without_refraction, topocentric_zenith_angle,
    };
    use crate::test_fixtures::{
        REFERENCE_LATITUDE_DEGREES, REFERENCE_PRESSURE_MILLIBARS, REFERENCE_TEMPERATURE_CELSIUS,
        reference_delta_prime_and_h_prime, reference_elevation_without_refraction,
    };

    #[test]
    fn topocentric_zenith_angle_matches_table_a5_1() {
        let e0 = reference_elevation_without_refraction();
        let delta_e = atmospheric_refraction(
            e0,
            REFERENCE_PRESSURE_MILLIBARS,
            REFERENCE_TEMPERATURE_CELSIUS,
        );
        let theta = topocentric_zenith_angle(e0, delta_e);
        assert!((theta - 50.111_62).abs() < 1e-4);
    }

    #[test]
    fn astronomers_azimuth_matches_published_intermediates() {
        let gamma = astronomers_azimuth(11.106_29, 39.742_476, -9.316_179);
        assert!((gamma - 14.340_24).abs() < 1e-4);
    }

    #[test]
    fn topocentric_azimuth_angle_matches_table_a5_1() {
        let (delta_prime, h_prime) = reference_delta_prime_and_h_prime();
        let gamma = astronomers_azimuth(h_prime, REFERENCE_LATITUDE_DEGREES, delta_prime);
        let phi = topocentric_azimuth_angle(gamma);
        assert!((phi - 194.340_24).abs() < 1e-4);
    }

    #[test]
    fn elevation_collapses_to_complement_of_latitude_at_meridian() {
        for &phi in &[-89.0_f64, -45.0, 0.0, 30.0, 60.0, 89.0] {
            let e0 = topocentric_elevation_without_refraction(phi, 0.0, 0.0);
            assert!((e0 - (90.0 - phi.abs())).abs() < 1e-12);
        }
    }

    #[test]
    fn elevation_collapses_to_arcsin_cos_h_prime_on_equator() {
        for &h_prime in &[-90.0_f64, -45.0, 0.0, 30.0, 60.0, 90.0] {
            let e0 = topocentric_elevation_without_refraction(0.0, 0.0, h_prime);
            let expected = h_prime.to_radians().cos().asin().to_degrees();
            assert!((e0 - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn elevation_reaches_zenith_when_latitude_equals_declination_on_meridian() {
        for &phi in &[-60.0_f64, -23.5, 0.0, 23.5, 60.0] {
            let e0 = topocentric_elevation_without_refraction(phi, phi, 0.0);
            assert!((e0 - 90.0).abs() < 1e-12);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn refraction_vanishes_below_horizon_cutoff() {
        for &e0 in &[
            -90.0_f64,
            -10.0,
            -1.0,
            HORIZON_CUTOFF_ELEVATION_DEGREES - f64::EPSILON,
        ] {
            assert_eq!(atmospheric_refraction(e0, 1010.0, 10.0), 0.0);
        }
    }

    #[test]
    fn refraction_at_standard_atmosphere() {
        let p = STANDARD_PRESSURE_MILLIBARS;
        let t = REFERENCE_TEMPERATURE_KELVIN - 273.0;
        for &e0 in &[HORIZON_CUTOFF_ELEVATION_DEGREES, 0.0, 10.0, 45.0, 89.0] {
            let actual = atmospheric_refraction(e0, p, t);
            let aux = e0 + 10.3 / (e0 + 5.11);
            let expected = 1.02 / (60.0 * aux.to_radians().tan());
            assert!((actual - expected).abs() < 1e-15);
        }
    }

    #[test]
    fn refraction_is_linear_in_pressure() {
        let baseline = atmospheric_refraction(20.0, 500.0, 10.0);
        let doubled = atmospheric_refraction(20.0, 1000.0, 10.0);
        assert!(2.0_f64.mul_add(-baseline, doubled).abs() < 1e-15);
    }

    #[test]
    fn refraction_inverse_temperature_ratio_holds() {
        let cold = atmospheric_refraction(20.0, 1010.0, -30.0);
        let hot = atmospheric_refraction(20.0, 1010.0, 40.0);
        let expected = (273.0 + -30.0_f64) / (273.0 + 40.0);
        assert!((hot / cold - expected).abs() < 1e-12);
    }

    #[test]
    fn zenith_angle_is_linear_in_each_input() {
        let baseline = topocentric_zenith_angle(45.0, 0.01);
        for &d in &[-1.0_f64, -1e-4, 1e-6, 0.5] {
            assert!((topocentric_zenith_angle(45.0 + d, 0.01) - baseline + d).abs() < 1e-13);
            assert!((topocentric_zenith_angle(45.0, 0.01 + d) - baseline + d).abs() < 1e-13);
        }
    }

    #[test]
    fn astronomers_azimuth_wraps_into_zero_360() {
        for &(h, phi, delta) in &[
            (0.0_f64, 0.0, 30.0),
            (-30.0, 39.742_476, -9.316_179),
            (-90.0, 39.742_476, -9.316_179),
            (-179.999, 0.0, 0.0),
        ] {
            let gamma = astronomers_azimuth(h, phi, delta);
            assert!((0.0..360.0).contains(&gamma));
        }
    }

    #[test]
    fn astronomers_azimuth_is_odd_in_h_prime_on_equator() {
        for &h in &[1.0_f64, 30.0, 89.0, 175.0] {
            let plus = astronomers_azimuth(h, 0.0, 10.0);
            let minus = astronomers_azimuth(-h, 0.0, 10.0);
            let sum = (plus + minus).rem_euclid(360.0);
            assert!(sum < 1e-12 || (sum - 360.0).abs() < 1e-12);
        }
    }

    #[test]
    fn topocentric_azimuth_wraps_both_halves() {
        for &(gamma, expected) in &[
            (0.0_f64, 180.0),
            (90.0, 270.0),
            (179.999, 359.999),
            (180.0, 0.0),
            (270.0, 90.0),
            (359.999, 179.999),
        ] {
            let phi = topocentric_azimuth_angle(gamma);
            assert!((phi - expected).abs() < 1e-12);
            assert!((0.0..360.0).contains(&phi));
        }
    }

    #[test]
    fn elevation_corrected_is_linear_in_each_input() {
        let baseline = topocentric_elevation_corrected(45.0, 0.01);
        for &d in &[-1.0_f64, -1e-4, 1e-6, 0.5] {
            assert!((topocentric_elevation_corrected(45.0 + d, 0.01) - baseline - d).abs() < 1e-13);
            assert!((topocentric_elevation_corrected(45.0, 0.01 + d) - baseline - d).abs() < 1e-13);
        }
    }

    #[test]
    fn elevation_corrected_is_complement_of_zenith() {
        for &(e0, de) in &[(0.0_f64, 0.0), (39.872, 0.016_33), (-1.0, 0.0), (89.5, 0.5)] {
            let e = topocentric_elevation_corrected(e0, de);
            let theta = topocentric_zenith_angle(e0, de);
            assert!((e + theta - 90.0).abs() < 1e-13);
        }
    }

    #[test]
    fn signed_azimuth_passes_lower_and_shifts_upper_half() {
        for &(input, expected) in &[
            (0.0_f64, 0.0),
            (90.0, 90.0),
            (179.999, 179.999),
            (180.0, 180.0),
            (180.001, -179.999),
            (270.0, -90.0),
            (359.999, -0.001),
        ] {
            let signed = astronomers_azimuth_signed(input);
            assert!((signed - expected).abs() < 1e-12);
            assert!(signed > -180.0 && signed <= 180.0);
        }
    }
}
