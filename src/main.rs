// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use chrono::Utc;
use chrono_tz::Tz;
use helioxide::delta_t::delta_t_seconds_for_datetime;
use helioxide::{Observer, SolarDay, SolarPosition, SpaDateTime, Surface};

#[cfg_attr(coverage_nightly, coverage(off))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let observer = Observer::try_new(38.346_02, -0.490_68, 3.0, 1015.0, 18.0)?;
    let surface = Surface::try_new(observer.latitude(), 0.0)?;
    let now = SpaDateTime::new(Utc::now().with_timezone(&Tz::Europe__Madrid));
    let delta_t = delta_t_seconds_for_datetime(now.datetime());
    let position = SolarPosition::compute(&now, delta_t, observer, surface);
    let day = SolarDay::compute(&now, delta_t, observer);
    println!("Now: {now:?}");
    println!("ΔT: {delta_t:.3} s");
    println!("{position}");
    println!("{day}");
    Ok(())
}
