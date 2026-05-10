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
    SpaDateTime, apparent, geocentric, heliocentric, julian, nutation, obliquity, sidereal,
};
use log::{debug, info};

fn main() {
    // Approximate ΔT value in seconds for years around 2026.
    // Update this value as needed for more accurate calculations.
    const DELTA_T: f64 = 69.5;

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
}
