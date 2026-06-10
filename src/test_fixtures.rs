// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Shared fixtures for the per-module unit tests.

use crate::SpaDateTime;
use crate::apparent::{aberration_correction, apparent_sun_longitude};
use crate::equatorial::{geocentric_declination, geocentric_right_ascension};
use crate::geocentric::{geocentric_latitude, geocentric_longitude};
use crate::heliocentric::{
    earth_heliocentric_latitude, earth_heliocentric_longitude, earth_radius_vector,
};
use crate::horizontal::{
    astronomers_azimuth, atmospheric_refraction, topocentric_elevation_without_refraction,
    topocentric_zenith_angle,
};
use crate::hour_angle::{observer_local_hour_angle, topocentric_local_hour_angle};
use crate::julian::{
    julian_day, julian_ephemeris_century, julian_ephemeris_day, julian_ephemeris_millennium,
};
use crate::nutation::nutation_in_longitude_and_obliquity;
use crate::obliquity::true_obliquity_of_ecliptic;
use crate::parallax::{equatorial_horizontal_parallax, topocentric_equatorial_coordinates};
use crate::sidereal::{apparent_sidereal_time, mean_sidereal_time};
use chrono::{TimeZone, Utc};

/// JD for Table A5.1: 2003-10-17 12:30:30 LST at TZ = -7 h (19:30:30 UT).
/// Reconstructed from the civil instant rather than the report's six-decimal
/// printed value, which loses ~2e-7 d that `L1 ≈ 6e11` amplifies into the
/// trailing decimals.
pub fn reference_jd() -> f64 {
    let utc = Utc.with_ymd_and_hms(2003, 10, 17, 19, 30, 30).unwrap();
    julian_day(&SpaDateTime::new(utc))
}

/// JCE for Table A5.1 with ΔT = 67 s.
pub fn reference_jce() -> f64 {
    let jde = julian_ephemeris_day(reference_jd(), 67.0);
    julian_ephemeris_century(jde)
}

/// JME for Table A5.1.
pub fn reference_jme() -> f64 {
    julian_ephemeris_millennium(reference_jce())
}

/// Reference observer site from appendix A.5.
pub const REFERENCE_LATITUDE_DEGREES: f64 = 39.742_476;
pub const REFERENCE_LONGITUDE_DEGREES: f64 = -105.178_6;
pub const REFERENCE_ELEVATION_METRES: f64 = 1830.14;
pub const REFERENCE_PRESSURE_MILLIBARS: f64 = 820.0;
pub const REFERENCE_TEMPERATURE_CELSIUS: f64 = 11.0;

/// `(δ', H')` at the Table A5.1 reference instant.
pub fn reference_delta_prime_and_h_prime() -> (f64, f64) {
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
    let h_prime = topocentric_local_hour_angle(h, topocentric.parallax_in_right_ascension);

    (topocentric.declination, h_prime)
}

/// `e₀` at the Table A5.1 reference instant.
pub fn reference_elevation_without_refraction() -> f64 {
    let (delta_prime, h_prime) = reference_delta_prime_and_h_prime();
    topocentric_elevation_without_refraction(REFERENCE_LATITUDE_DEGREES, delta_prime, h_prime)
}

/// `(θ, Γ)` at the Table A5.1 reference instant.
pub fn reference_theta_and_gamma() -> (f64, f64) {
    let (delta_prime, h_prime) = reference_delta_prime_and_h_prime();
    let e0 =
        topocentric_elevation_without_refraction(REFERENCE_LATITUDE_DEGREES, delta_prime, h_prime);
    let delta_e = atmospheric_refraction(
        e0,
        REFERENCE_PRESSURE_MILLIBARS,
        REFERENCE_TEMPERATURE_CELSIUS,
    );
    let theta = topocentric_zenith_angle(e0, delta_e);
    let gamma = astronomers_azimuth(h_prime, REFERENCE_LATITUDE_DEGREES, delta_prime);

    (theta, gamma)
}
