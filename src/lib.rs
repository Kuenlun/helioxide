// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

// Allow `#[coverage(off)]` on test modules under `--cfg coverage_nightly` (nightly-only).
#![cfg_attr(all(test, coverage_nightly), feature(coverage_attribute))]
#![feature(const_trait_impl)]

pub mod apparent;
pub mod delta_t;
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
pub mod solar_time;
pub mod spa;
pub mod time;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod test_fixtures;

pub use solar_time::SolarDay;
pub use spa::{Observer, SolarPosition, Surface};
pub use time::{SpaDateTime, SpaTimeError};
