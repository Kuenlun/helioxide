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
use crate::julian::{
    calculate_julian_day, calculate_julian_ephemeris_century, calculate_julian_ephemeris_day,
    calculate_julian_ephemeris_millennium,
};
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
