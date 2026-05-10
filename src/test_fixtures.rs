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

/// JCE for the Table A5.1 worked example: 2003-10-17 12:30:30 LST, TZ = -7 h
/// (i.e. 19:30:30 UT) with ΔT = 67 s. The JD is reconstructed from the
/// civil instant rather than the report's six-decimal printed value, because
/// the latter loses ~2·10⁻⁷ d that `L1 ≈ 6·10¹¹` amplifies into the trailing
/// decimals of every published subseries total.
pub fn reference_jce() -> f64 {
    let utc = Utc
        .with_ymd_and_hms(2003, 10, 17, 19, 30, 30)
        .single()
        .expect("non-ambiguous reference instant");
    let jd = calculate_julian_day(&SpaDateTime::new(utc));
    let jde = calculate_julian_ephemeris_day(jd, 67.0);
    calculate_julian_ephemeris_century(jde)
}

/// JME for the Table A5.1 worked example. See [`reference_jce`] for the
/// civil instant the chain originates from.
pub fn reference_jme() -> f64 {
    calculate_julian_ephemeris_millennium(reference_jce())
}
