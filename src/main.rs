// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

use chrono::Utc;
use chrono_tz::Tz;
use helioxide::delta_t::approximate_delta_t_seconds_for_datetime;
use helioxide::{Observer, SolarDay, SolarPosition, SpaDateTime, Surface};

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
    // The input timezone selects both the local civil day that drives
    // `SolarDay::compute` and the wall clock used to render its output.
    // Alicante is the observer, so Europe/Madrid keeps sunrise, transit
    // and sunset on the same calendar date as Alicante's clock.
    let now = SpaDateTime::new(Utc::now().with_timezone(&Tz::Europe__Madrid));
    // Espenak-Meeus piecewise polynomial: tracks the current IERS ΔT
    // within a handful of seconds across `2005-2050` without any
    // bulletin lookup, and degrades gracefully outside that window.
    let delta_t = approximate_delta_t_seconds_for_datetime(now.datetime());
    let position = SolarPosition::compute(&now, delta_t, OBSERVER, SURFACE);
    let day = SolarDay::compute(&now, delta_t, OBSERVER);
    println!("Now: {now:?}");
    println!("ΔT: {delta_t:.3} s");
    println!("{position}");
    println!("{day}");
}
