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

mod error;

use chrono::Utc;
use chrono_tz::Tz;
use log::info;

use crate::error::HelioxideError;

fn main() -> Result<(), HelioxideError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    let tz: Tz = "Europe/Madrid".parse()?;
    let now = Utc::now().with_timezone(&tz);
    info!("Now: {now:?}");

    Ok(())
}
