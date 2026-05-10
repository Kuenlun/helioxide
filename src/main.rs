// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

use chrono::Utc;
use chrono_tz::Tz;
use helioxide::{Observer, SolarPosition, SpaDateTime, Surface};

/// Approximate ΔT value in seconds for years around 2026. Update as
/// needed for more accurate calculations (consult IERS bulletins).
const DELTA_T: f64 = 69.5;

/// Example observer site: Alicante.
const OBSERVER: Observer = Observer {
    longitude: -0.490_68,
    latitude: 38.346_02,
    elevation: 3.0,
    pressure: 1015.0,
    temperature: 18.0,
};

/// Example surface: a fixed-tilt panel at the observer's latitude, facing
/// due south.
const SURFACE: Surface = Surface {
    slope: OBSERVER.latitude,
    azimuth_rotation: 0.0,
};

fn main() {
    let now = SpaDateTime::new(Utc::now().with_timezone(&Tz::Europe__Madrid));
    let position = SolarPosition::compute(&now, DELTA_T, OBSERVER, SURFACE);
    println!("Now: {now:?}");
    println!("{position}");
}
