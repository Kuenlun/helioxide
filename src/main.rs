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

// Allow `#[coverage(off)]` on test modules under `--cfg coverage_nightly` (nightly-only).
#![cfg_attr(all(test, coverage_nightly), feature(coverage_attribute))]

use chrono::Utc;
use chrono_tz::Tz;
use helioxide::{
    SpaDateTime, apparent, equatorial, geocentric, heliocentric, horizontal, hour_angle, incidence,
    julian, nutation, obliquity, parallax, sidereal,
};
use log::{debug, info};

fn main() {
    // Approximate ΔT value in seconds for years around 2026.
    // Update this value as needed for more accurate calculations.
    const DELTA_T: f64 = 69.5;

    // Example observer site: Alicante. Longitude is signed positive east of
    // Greenwich per section 3.11; latitude is signed positive north of the
    // equator per section 3.12.2; elevation is above sea level per section 3.12.3.
    const OBSERVER_LONGITUDE_DEG: f64 = -0.490_68;
    const OBSERVER_LATITUDE_DEG: f64 = 38.346_02;
    const OBSERVER_ELEVATION_M: f64 = 3.0;
    // Annual averages used by equation 42's atmospheric refraction model.
    const OBSERVER_PRESSURE_MBAR: f64 = 1015.0;
    const OBSERVER_TEMPERATURE_C: f64 = 18.0;
    // Example surface orientation for section 3.16: a fixed-tilt solar panel
    // tilted at the observer's latitude, facing due south (γ = 0°).
    const SURFACE_SLOPE_DEG: f64 = OBSERVER_LATITUDE_DEG;
    const SURFACE_AZIMUTH_ROTATION_DEG: f64 = 0.0;

    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    let now = SpaDateTime::new(Utc::now().with_timezone(&Tz::Europe__Madrid));
    info!("Now: {now:?}");

    let jd = julian::calculate_julian_day(&now);
    let jde = julian::calculate_julian_ephemeris_day(jd, DELTA_T);
    let jc = julian::calculate_julian_century(jd);
    let jce = julian::calculate_julian_ephemeris_century(jde);
    let jme = julian::calculate_julian_ephemeris_millennium(jce);
    debug!("Julian Day: {jd}");
    debug!("Julian Ephemeris Day: {jde}");
    debug!("Julian Century: {jc}");
    debug!("Julian Ephemeris Century: {jce}");
    debug!("Julian Ephemeris Millennium: {jme}");

    let l = heliocentric::earth_heliocentric_longitude(jme);
    let b = heliocentric::earth_heliocentric_latitude(jme);
    let r = heliocentric::earth_radius_vector(jme);
    debug!("Earth heliocentric longitude L: {l}°");
    debug!("Earth heliocentric latitude B: {b}°");
    debug!("Earth radius vector R: {r} AU");

    let theta = geocentric::geocentric_longitude(l);
    let beta = geocentric::geocentric_latitude(b);
    debug!("Sun geocentric longitude Θ: {theta}°");
    debug!("Sun geocentric latitude β: {beta}°");

    let (delta_psi, delta_epsilon) = nutation::nutation_in_longitude_and_obliquity(jce);
    debug!("Nutation in longitude Δψ: {delta_psi}°");
    debug!("Nutation in obliquity Δε: {delta_epsilon}°");

    let epsilon = obliquity::true_obliquity_of_ecliptic(jme, delta_epsilon);
    debug!("True obliquity of the ecliptic ε: {epsilon}°");

    let delta_tau = apparent::aberration_correction(r);
    debug!("Aberration correction Δτ: {delta_tau}°");
    let lambda = apparent::apparent_sun_longitude(theta, delta_psi, delta_tau);
    debug!("Apparent sun longitude λ: {lambda}°");

    let nu0 = sidereal::mean_sidereal_time(jd);
    debug!("Mean sidereal time at Greenwich ν₀: {nu0}°");
    let nu = sidereal::apparent_sidereal_time(nu0, delta_psi, epsilon);
    debug!("Apparent sidereal time at Greenwich ν: {nu}°");

    let alpha = equatorial::geocentric_right_ascension(lambda, beta, epsilon);
    debug!("Sun geocentric right ascension α: {alpha}°");
    let delta = equatorial::geocentric_declination(lambda, beta, epsilon);
    debug!("Sun geocentric declination δ: {delta}°");

    let h = hour_angle::observer_local_hour_angle(nu, OBSERVER_LONGITUDE_DEG, alpha);
    debug!("Observer local hour angle H: {h}°");

    let xi = parallax::equatorial_horizontal_parallax(r);
    debug!("Sun equatorial horizontal parallax ξ: {xi}°");
    let topocentric = parallax::topocentric_equatorial_coordinates(
        alpha,
        delta,
        h,
        xi,
        OBSERVER_LATITUDE_DEG,
        OBSERVER_ELEVATION_M,
    );
    debug!(
        "Sun right ascension parallax Δα: {}°",
        topocentric.parallax_in_right_ascension,
    );
    debug!(
        "Topocentric sun right ascension α': {}°",
        topocentric.right_ascension,
    );
    debug!(
        "Topocentric sun declination δ': {}°",
        topocentric.declination,
    );

    let h_prime =
        hour_angle::topocentric_local_hour_angle(h, topocentric.parallax_in_right_ascension);
    debug!("Topocentric local hour angle H': {h_prime}°");

    let e0 = horizontal::topocentric_elevation_without_refraction(
        OBSERVER_LATITUDE_DEG,
        topocentric.declination,
        h_prime,
    );
    debug!("Topocentric elevation without refraction e₀: {e0}°");
    let delta_e =
        horizontal::atmospheric_refraction(e0, OBSERVER_PRESSURE_MBAR, OBSERVER_TEMPERATURE_C);
    debug!("Atmospheric refraction Δe: {delta_e}°");
    let zenith = horizontal::topocentric_zenith_angle(e0, delta_e);
    info!("Topocentric zenith angle θ: {zenith}°");
    let gamma =
        horizontal::astronomers_azimuth(h_prime, OBSERVER_LATITUDE_DEG, topocentric.declination);
    debug!("Topocentric astronomers' azimuth Γ: {gamma}°");
    let azimuth = horizontal::topocentric_azimuth_angle(gamma);
    info!("Topocentric azimuth Φ: {azimuth}°");

    let incidence_angle = incidence::surface_incidence_angle(
        zenith,
        gamma,
        SURFACE_SLOPE_DEG,
        SURFACE_AZIMUTH_ROTATION_DEG,
    );
    info!("Surface incidence angle I: {incidence_angle}°");
}
