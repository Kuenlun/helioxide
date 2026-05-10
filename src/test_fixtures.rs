/*!
helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
Copyright (C) 2026  Juan Luis Leal Contreras (Kuenlun)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

//! Shared fixtures for the per-module unit tests.
//!
//! Anything reused by more than one `#[cfg(test)] mod tests { … }` lives here
//! to keep the individual modules from drifting apart.

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
    calculate_julian_day, calculate_julian_ephemeris_century, calculate_julian_ephemeris_day,
    calculate_julian_ephemeris_millennium,
};
use crate::nutation::nutation_in_longitude_and_obliquity;
use crate::obliquity::true_obliquity_of_ecliptic;
use crate::parallax::{equatorial_horizontal_parallax, topocentric_equatorial_coordinates};
use crate::sidereal::{apparent_sidereal_time, mean_sidereal_time};
use chrono::{TimeZone, Utc};

/// JD for the Table A5.1 worked example: 2003-10-17 12:30:30 LST, TZ = -7 h
/// (i.e. 19:30:30 UT). The JD is reconstructed from the civil instant
/// rather than the report's six-decimal printed value, because the latter
/// loses ~2·10⁻⁷ d that `L1 ≈ 6·10¹¹` amplifies into the trailing decimals
/// of every published subseries total. The diurnal coefficient of
/// equation 28 (≈ 360.99°/d) would also amplify it into ~8·10⁻⁵° on `ν₀`.
pub fn reference_jd() -> f64 {
    let utc = Utc
        .with_ymd_and_hms(2003, 10, 17, 19, 30, 30)
        .single()
        .expect("non-ambiguous reference instant");
    calculate_julian_day(&SpaDateTime::new(utc))
}

/// JCE for the Table A5.1 worked example with ΔT = 67 s. See
/// [`reference_jd`] for the civil instant the chain originates from.
pub fn reference_jce() -> f64 {
    let jde = calculate_julian_ephemeris_day(reference_jd(), 67.0);
    calculate_julian_ephemeris_century(jde)
}

/// JME for the Table A5.1 worked example. See [`reference_jd`] for the
/// civil instant the chain originates from.
pub fn reference_jme() -> f64 {
    calculate_julian_ephemeris_millennium(reference_jce())
}

/// Reference observer site from Appendix A.5: latitude `φ` (positive
/// north of the equator per section 3.12.2), longitude `σ` (positive
/// east of Greenwich per section 3.11), elevation `E` above sea level
/// per section 3.12.3. Reused by every reference-instant test so the
/// published civil instant feeds a single, canonical observer.
pub const REFERENCE_LATITUDE_DEGREES: f64 = 39.742_476;
pub const REFERENCE_LONGITUDE_DEGREES: f64 = -105.178_6;
pub const REFERENCE_ELEVATION_METRES: f64 = 1830.14;

/// Reference observer pressure for the Table A5.1 worked example
/// (millibars), per section A.5. Feeds equation 42's pressure ratio.
pub const REFERENCE_PRESSURE_MILLIBARS: f64 = 820.0;
/// Reference observer temperature for the Table A5.1 worked example
/// (degrees Celsius), per section A.5. Feeds equation 42's temperature
/// ratio.
pub const REFERENCE_TEMPERATURE_CELSIUS: f64 = 11.0;

/// Drives the full upstream chain (sections 3.2 through 3.13) to produce
/// `(δ', H')` at the Table A5.1 reference instant. Reused by every
/// section-3.14+ reference test so a single integration regression
/// upstream surfaces in exactly one place rather than corrupting
/// independent reference checks.
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

/// Drives the full upstream chain plus equation 41 to produce `e₀` at the
/// Table A5.1 reference instant. Reused by the refraction, zenith, and
/// downstream reference tests so a single integration regression upstream
/// surfaces in exactly one place.
pub fn reference_elevation_without_refraction() -> f64 {
    let (delta_prime, h_prime) = reference_delta_prime_and_h_prime();
    topocentric_elevation_without_refraction(REFERENCE_LATITUDE_DEGREES, delta_prime, h_prime)
}

/// Drives the full upstream chain (sections 3.2 through 3.15) to produce
/// `(θ, Γ)` at the Table A5.1 reference instant. Reused by section-3.16+
/// reference tests so a single integration regression upstream surfaces
/// in exactly one place.
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
