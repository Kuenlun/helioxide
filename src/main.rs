// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

// Allow `#[coverage(off)]` on `main` under `--cfg coverage_nightly`: the
// constructor `Err` arms guarded by `?` below are statically unreachable
// from the hard-coded constants, so they would otherwise drop coverage.
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use chrono::Utc;
use chrono_tz::Tz;
use helioxide::delta_t::approximate_delta_t_seconds_for_datetime;
use helioxide::{Observer, SolarDay, SolarPosition, SpaDateTime, Surface};

#[cfg_attr(coverage_nightly, coverage(off))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let observer = Observer::try_new(38.346_02, -0.490_68, 3.0, 1015.0, 18.0)?;
    // Fixed-tilt collector facing due south at the observer's own latitude.
    let surface = Surface::try_new(observer.latitude(), 0.0)?;
    // The input timezone selects both the local civil day that drives
    // `SolarDay::compute` and the wall clock used to render its output.
    let now = SpaDateTime::new(Utc::now().with_timezone(&Tz::Europe__Madrid));
    // Espenak-Meeus piecewise polynomial: tracks the current IERS ΔT
    // within a handful of seconds across `2005-2050` without any
    // bulletin lookup, and degrades gracefully outside that window.
    let delta_t = approximate_delta_t_seconds_for_datetime(now.datetime());
    let position = SolarPosition::compute(&now, delta_t, observer, surface);
    let day = SolarDay::compute(&now, delta_t, observer);
    println!("Now: {now:?}");
    println!("ΔT: {delta_t:.3} s");
    println!("{position}");
    println!("{day}");
    Ok(())
}
