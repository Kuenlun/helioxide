// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Equatorial horizontal parallax (`ξ`) and topocentric equatorial
//! coordinates (`α'`, `δ'`). Section 3.12.

/// `1 − f` of equation 34 (Earth flattening factor).
const EARTH_POLAR_RADIUS_FACTOR: f64 = 0.996_647_19;

/// Earth equatorial radius (metres) of equations 35 and 36.
const EARTH_EQUATORIAL_RADIUS_METRES: f64 = 6_378_140.0;

/// `8.794"`, the equatorial horizontal parallax at one AU. Equation 33.
const SOLAR_HORIZONTAL_PARALLAX_AT_UNIT_DISTANCE_ARCSECONDS: f64 = 8.794;

/// `ξ = 8.794" / (3600 · R)` (degrees). Equation 33.
#[inline]
#[must_use]
pub const fn equatorial_horizontal_parallax(earth_radius_vector: f64) -> f64 {
    SOLAR_HORIZONTAL_PARALLAX_AT_UNIT_DISTANCE_ARCSECONDS / (3600.0 * earth_radius_vector)
}

/// Topocentric `α'`, `δ'` and the shared `Δα`. Equations 34 to 39 share the
/// `(u, x, y)` reduction and the `cos δ − x·sin ξ·cos H` denominator, so a
/// single call evaluates the whole bundle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopocentricEquatorialCoordinates {
    /// `α'` (degrees), equation 38.
    pub right_ascension: f64,
    /// `δ'` (degrees, in `[-90°, 90°]`), equation 39.
    pub declination: f64,
    /// `Δα` (degrees, signed), equation 37. Required by step 3.13.
    pub parallax_in_right_ascension: f64,
}

/// Run equations 34 to 39 together.
#[inline]
#[must_use]
pub fn topocentric_equatorial_coordinates(
    geocentric_right_ascension: f64,
    geocentric_declination: f64,
    observer_local_hour_angle: f64,
    equatorial_horizontal_parallax: f64,
    observer_latitude: f64,
    observer_elevation: f64,
) -> TopocentricEquatorialCoordinates {
    // Equations 34 to 36: geodetic-to-geocentric reduction.
    let phi_rad = observer_latitude.to_radians();
    let (sin_phi, cos_phi) = phi_rad.sin_cos();
    let (sin_u, cos_u) = (EARTH_POLAR_RADIUS_FACTOR * phi_rad.tan()).atan().sin_cos();
    let elevation_normalised = observer_elevation / EARTH_EQUATORIAL_RADIUS_METRES;
    let x = elevation_normalised.mul_add(cos_phi, cos_u);
    let y = elevation_normalised.mul_add(sin_phi, EARTH_POLAR_RADIUS_FACTOR * sin_u);

    // Equations 37 and 39 share the `cos δ − x·sin ξ·cos H` denominator.
    let (sin_h, cos_h) = observer_local_hour_angle.to_radians().sin_cos();
    let (sin_delta, cos_delta) = geocentric_declination.to_radians().sin_cos();
    let sin_xi = equatorial_horizontal_parallax.to_radians().sin();
    let x_sin_xi = x * sin_xi;
    let denominator = (-x_sin_xi).mul_add(cos_h, cos_delta);
    let delta_alpha_rad = (-x_sin_xi * sin_h).atan2(denominator);
    let delta_alpha = delta_alpha_rad.to_degrees();

    let topocentric_declination_numerator = (-y).mul_add(sin_xi, sin_delta) * delta_alpha_rad.cos();
    let delta_prime = topocentric_declination_numerator
        .atan2(denominator)
        .to_degrees();

    TopocentricEquatorialCoordinates {
        // Equation 38. `|Δα| ≤ ξ` is arc-second order, so the wrapped `α`
        // stays inside `[0°, 360°)`.
        right_ascension: geocentric_right_ascension + delta_alpha,
        declination: delta_prime,
        parallax_in_right_ascension: delta_alpha,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{
        EARTH_EQUATORIAL_RADIUS_METRES, SOLAR_HORIZONTAL_PARALLAX_AT_UNIT_DISTANCE_ARCSECONDS,
        equatorial_horizontal_parallax, topocentric_equatorial_coordinates,
    };
    use crate::apparent::{aberration_correction, apparent_sun_longitude};
    use crate::equatorial::{geocentric_declination, geocentric_right_ascension};
    use crate::geocentric::{geocentric_latitude, geocentric_longitude};
    use crate::heliocentric::{
        earth_heliocentric_latitude, earth_heliocentric_longitude, earth_radius_vector,
    };
    use crate::hour_angle::observer_local_hour_angle;
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::obliquity::true_obliquity_of_ecliptic;
    use crate::sidereal::{apparent_sidereal_time, mean_sidereal_time};
    use crate::test_fixtures::{
        REFERENCE_ELEVATION_METRES, REFERENCE_LATITUDE_DEGREES, REFERENCE_LONGITUDE_DEGREES,
        reference_jce, reference_jd, reference_jme,
    };

    fn reference_alpha_delta_h_xi() -> (f64, f64, f64, f64) {
        let jd = reference_jd();
        let jce = reference_jce();
        let jme = reference_jme();

        let (delta_psi, delta_epsilon) = nutation_in_longitude_and_obliquity(jce);
        let epsilon = true_obliquity_of_ecliptic(jme, delta_epsilon);
        let nu = apparent_sidereal_time(mean_sidereal_time(jd), delta_psi, epsilon);

        let theta = geocentric_longitude(earth_heliocentric_longitude(jme));
        let beta = geocentric_latitude(earth_heliocentric_latitude(jme));
        let r = earth_radius_vector(jme);
        let delta_tau = aberration_correction(r);
        let lambda = apparent_sun_longitude(theta, delta_psi, delta_tau);

        let alpha = geocentric_right_ascension(lambda, beta, epsilon);
        let delta = geocentric_declination(lambda, beta, epsilon);
        let h = observer_local_hour_angle(nu, REFERENCE_LONGITUDE_DEGREES, alpha);
        let xi = equatorial_horizontal_parallax(r);

        (alpha, delta, h, xi)
    }

    #[test]
    fn topocentric_right_ascension_matches_table_a5_1() {
        let (alpha, delta, h, xi) = reference_alpha_delta_h_xi();
        let coords = topocentric_equatorial_coordinates(
            alpha,
            delta,
            h,
            xi,
            REFERENCE_LATITUDE_DEGREES,
            REFERENCE_ELEVATION_METRES,
        );
        assert!((coords.right_ascension - 202.227_04).abs() < 1e-4);
    }

    #[test]
    fn topocentric_declination_matches_table_a5_1() {
        let (alpha, delta, h, xi) = reference_alpha_delta_h_xi();
        let coords = topocentric_equatorial_coordinates(
            alpha,
            delta,
            h,
            xi,
            REFERENCE_LATITUDE_DEGREES,
            REFERENCE_ELEVATION_METRES,
        );
        assert!((coords.declination - -9.316_179).abs() < 1e-4);
    }

    #[test]
    fn equatorial_horizontal_parallax_at_unit_distance() {
        let xi = equatorial_horizontal_parallax(1.0);
        let expected = SOLAR_HORIZONTAL_PARALLAX_AT_UNIT_DISTANCE_ARCSECONDS / 3600.0;
        assert!((xi - expected).abs() < f64::EPSILON);
    }

    #[test]
    fn equatorial_horizontal_parallax_inverse_in_r() {
        let xi_unit = equatorial_horizontal_parallax(1.0);
        for &r in &[0.5_f64, 0.95, 1.05, 2.0, 10.0] {
            let xi = equatorial_horizontal_parallax(r);
            assert!(xi.mul_add(r, -xi_unit).abs() < 1e-15);
        }
    }

    #[test]
    fn corrections_vanish_at_zero_parallax() {
        for &(alpha, delta, h, phi, elevation) in &[
            (10.0_f64, -30.0, 0.0, 0.0, 0.0),
            (200.0, 5.5, 90.0, 39.742_476, 1830.14),
            (350.0, -85.0, -45.0, -60.0, 5_000.0),
        ] {
            let coords = topocentric_equatorial_coordinates(alpha, delta, h, 0.0, phi, elevation);
            assert!(coords.parallax_in_right_ascension.abs() < 1e-15);
            assert!((coords.right_ascension - alpha).abs() < 1e-13);
            assert!((coords.declination - delta).abs() < 1e-13);
        }
    }

    #[test]
    fn parallax_in_right_ascension_vanishes_at_meridian_transit() {
        let xi = equatorial_horizontal_parallax(1.0);
        for &(delta, phi, elevation) in &[
            (0.0_f64, 0.0, 0.0),
            (-9.314_34, 39.742_476, 1830.14),
            (45.0, -45.0, 0.0),
            (-89.0, 89.0, 5_000.0),
        ] {
            let c = topocentric_equatorial_coordinates(123.456, delta, 0.0, xi, phi, elevation);
            assert!(c.parallax_in_right_ascension.abs() < 1e-15);
        }
    }

    #[test]
    fn parallax_in_right_ascension_is_odd_in_h() {
        let xi = equatorial_horizontal_parallax(0.996_542_297_4);
        for &(delta, h, phi, elevation) in &[
            (0.0_f64, 30.0, 0.0, 0.0),
            (-9.314_34, 11.105_900, 39.742_476, 1830.14),
            (45.0, 60.0, -45.0, 500.0),
            (-30.0, 89.999, 60.0, 1_500.0),
        ] {
            let plus = topocentric_equatorial_coordinates(100.0, delta, h, xi, phi, elevation);
            let minus = topocentric_equatorial_coordinates(100.0, delta, -h, xi, phi, elevation);
            let sum = plus.parallax_in_right_ascension + minus.parallax_in_right_ascension;
            assert!(sum.abs() < 1e-13);
        }
    }

    #[test]
    fn topocentric_right_ascension_shifts_with_alpha() {
        let xi = equatorial_horizontal_parallax(1.0);
        let baseline =
            topocentric_equatorial_coordinates(100.0, 10.0, 30.0, xi, 39.742_476, 1830.14);
        for &kappa in &[-30.0_f64, -1e-3, 1e-6, 0.5, 50.0] {
            let shifted = topocentric_equatorial_coordinates(
                100.0 + kappa,
                10.0,
                30.0,
                xi,
                39.742_476,
                1830.14,
            );
            assert!((shifted.right_ascension - baseline.right_ascension - kappa).abs() < 1e-13);
            assert!(
                (shifted.parallax_in_right_ascension - baseline.parallax_in_right_ascension).abs()
                    < 1e-15,
            );
            assert!((shifted.declination - baseline.declination).abs() < 1e-15);
        }
    }

    #[test]
    fn parallax_doubles_when_elevation_equals_earth_radius_at_equator() {
        let xi = equatorial_horizontal_parallax(1.0);
        let sea_level = topocentric_equatorial_coordinates(100.0, 0.0, 90.0, xi, 0.0, 0.0);
        let one_radius_up = topocentric_equatorial_coordinates(
            100.0,
            0.0,
            90.0,
            xi,
            0.0,
            EARTH_EQUATORIAL_RADIUS_METRES,
        );
        let ratio =
            one_radius_up.parallax_in_right_ascension / sea_level.parallax_in_right_ascension;
        assert!((ratio - 2.0).abs() < 1e-7);
    }
}
