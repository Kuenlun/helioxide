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
