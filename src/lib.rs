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
#![feature(const_trait_impl)]

pub mod apparent;
pub mod equation_of_time;
pub mod equatorial;
pub mod error;
pub mod geocentric;
pub mod heliocentric;
pub mod helper;
pub mod horizontal;
pub mod hour_angle;
pub mod incidence;
pub mod julian;
pub mod nutation;
pub mod obliquity;
pub mod parallax;
pub mod sidereal;
pub mod spa;
pub mod time;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod test_fixtures;

pub use spa::{Observer, SolarPosition, Surface};
pub use time::{SpaDateTime, SpaTimeError};
