// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Observer (`H`) and topocentric (`H'`) local hour angles. Sections 3.11 and 3.13.

use crate::helper::limit_degrees;

/// `H = ν + σ − α` (degrees, wrapped into `[0°, 360°)`), measured westward
/// from south. Equation 32. `σ` is positive east of Greenwich.
#[inline]
#[must_use]
pub const fn observer_local_hour_angle(
    apparent_sidereal_time: f64,
    observer_longitude: f64,
    geocentric_right_ascension: f64,
) -> f64 {
    limit_degrees(apparent_sidereal_time + observer_longitude - geocentric_right_ascension)
}

/// `H' = H − Δα` (degrees, signed, not wrapped). Equation 40.
///
/// Downstream uses (equations 41 and 45) read only `sin H'` and `cos H'`,
/// so excursions outside `[0°, 360°)` are absorbed by trig periodicity.
/// `|Δα| ≤ ξ` is arc-second order, so `H'` stays within a few arc seconds
/// of the wrapped `H`.
#[inline]
#[must_use]
pub const fn topocentric_local_hour_angle(
    observer_local_hour_angle: f64,
    parallax_in_right_ascension: f64,
) -> f64 {
    observer_local_hour_angle - parallax_in_right_ascension
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{observer_local_hour_angle, topocentric_local_hour_angle};
    use crate::apparent::{aberration_correction, apparent_sun_longitude};
    use crate::equatorial::{geocentric_declination, geocentric_right_ascension};
    use crate::geocentric::{geocentric_latitude, geocentric_longitude};
    use crate::heliocentric::{
        earth_heliocentric_latitude, earth_heliocentric_longitude, earth_radius_vector,
    };
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::obliquity::true_obliquity_of_ecliptic;
    use crate::parallax::{equatorial_horizontal_parallax, topocentric_equatorial_coordinates};
    use crate::sidereal::{apparent_sidereal_time, mean_sidereal_time};
    use crate::test_fixtures::{
        REFERENCE_ELEVATION_METRES, REFERENCE_LATITUDE_DEGREES, REFERENCE_LONGITUDE_DEGREES,
        reference_jce, reference_jd, reference_jme,
    };

    fn reference_nu_and_alpha() -> (f64, f64) {
        let jd = reference_jd();
        let jce = reference_jce();
        let jme = reference_jme();

        let (delta_psi, delta_epsilon) = nutation_in_longitude_and_obliquity(jce);
        let epsilon = true_obliquity_of_ecliptic(jme, delta_epsilon);
        let nu = apparent_sidereal_time(mean_sidereal_time(jd), delta_psi, epsilon);

        let theta = geocentric_longitude(earth_heliocentric_longitude(jme));
        let beta = geocentric_latitude(earth_heliocentric_latitude(jme));
        let delta_tau = aberration_correction(earth_radius_vector(jme));
        let lambda = apparent_sun_longitude(theta, delta_psi, delta_tau);
        let alpha = geocentric_right_ascension(lambda, beta, epsilon);

        (nu, alpha)
    }

    fn reference_h_and_delta_alpha() -> (f64, f64) {
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
        let topocentric = topocentric_equatorial_coordinates(
            alpha,
            delta,
            h,
            xi,
            REFERENCE_LATITUDE_DEGREES,
            REFERENCE_ELEVATION_METRES,
        );

        (h, topocentric.parallax_in_right_ascension)
    }

    #[test]
    fn observer_local_hour_angle_matches_table_a5_1() {
        let (nu, alpha) = reference_nu_and_alpha();
        let h = observer_local_hour_angle(nu, -105.1786, alpha);
        assert!((h - 11.105_900).abs() < 1e-4);
    }

    #[test]
    fn observer_local_hour_angle_treats_longitude_as_positive_east() {
        let (nu, alpha) = (100.0, 50.0);
        for &sigma in &[10.0_f64, 45.0, 105.1786, 179.0] {
            let east = observer_local_hour_angle(nu, sigma, alpha);
            let west = observer_local_hour_angle(nu, -sigma, alpha);
            let actual = (east - west).rem_euclid(360.0);
            let expected = (2.0 * sigma).rem_euclid(360.0);
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn observer_local_hour_angle_wraps_into_zero_360() {
        for &(nu, sigma, alpha) in &[
            (0.0_f64, -180.0, 0.0),
            (10.0, -180.0, 50.0),
            (350.0, 180.0, 0.0),
            (720.0, 0.0, -100.0),
            (-100.0, 0.0, 0.0),
        ] {
            let h = observer_local_hour_angle(nu, sigma, alpha);
            assert!((0.0..360.0).contains(&h));
        }
    }

    #[test]
    fn observer_local_hour_angle_is_linear_in_each_input() {
        let baseline = observer_local_hour_angle(200.0, 50.0, 100.0);
        for &d in &[-1.0_f64, -1e-3, 1e-6, 0.5] {
            assert!(
                (observer_local_hour_angle(200.0 + d, 50.0, 100.0) - baseline - d).abs() < 1e-13
            );
            assert!(
                (observer_local_hour_angle(200.0, 50.0 + d, 100.0) - baseline - d).abs() < 1e-13
            );
            assert!(
                (observer_local_hour_angle(200.0, 50.0, 100.0 + d) - baseline + d).abs() < 1e-13
            );
        }
    }

    #[test]
    fn topocentric_local_hour_angle_matches_table_a5_1() {
        let (h, delta_alpha) = reference_h_and_delta_alpha();
        let h_prime = topocentric_local_hour_angle(h, delta_alpha);
        assert!((h_prime - 11.106_29).abs() < 1e-4);
    }

    #[test]
    fn topocentric_local_hour_angle_is_linear() {
        let baseline = topocentric_local_hour_angle(100.0, 1e-3);
        for &d in &[-1.0_f64, -1e-4, 1e-6, 0.5] {
            assert!((topocentric_local_hour_angle(100.0 + d, 1e-3) - baseline - d).abs() < 1e-13);
            assert!((topocentric_local_hour_angle(100.0, 1e-3 + d) - baseline + d).abs() < 1e-13);
        }
    }

    #[test]
    fn topocentric_local_hour_angle_does_not_wrap() {
        for &(h, da, expected) in &[(1e-4_f64, 1.0, 1e-4 - 1.0), (359.9, -1.0, 360.9)] {
            assert!((topocentric_local_hour_angle(h, da) - expected).abs() < 1e-13);
        }
    }
}
