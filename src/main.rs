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
use helioxide::{DateTimeWithDUT1, julian::calculate_julian_day};
use log::{debug, info};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    let now = DateTimeWithDUT1::new(Utc::now().with_timezone(&Tz::Europe__Madrid));
    info!("Now: {now:?}");

    let julian_day = calculate_julian_day(&now);
    debug!("Julian Day: {julian_day}");
}
