// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use chrono::Utc;
use chrono_tz::Tz;
use helioxide::{Observer, SolarDay, SolarPosition, SpaDateTime, Surface};

#[cfg_attr(coverage_nightly, coverage(off))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let observer = Observer::try_new(38.346_02, -0.490_68, 3.0, 1015.0, 18.0)?;
    let surface = Surface::try_new(observer.latitude(), 0.0)?;
    let now = SpaDateTime::new(Utc::now().with_timezone(&Tz::Europe__Madrid));
    let position = SolarPosition::compute(&now, observer);
    let day = SolarDay::compute(&now, observer);
    println!("Now: {now:?}");
    println!("{position}");
    println!(
        "Surface incidence angle I: {}°",
        position.surface_incidence(surface)
    );
    println!("{day}");
    Ok(())
}
