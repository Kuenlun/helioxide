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

#![cfg_attr(test, feature(coverage_attribute))]
#![feature(const_trait_impl)]

pub mod error;
pub mod helper;
pub mod julian;

use chrono::DateTime;
use chrono_tz::Tz;

#[derive(Debug)]
pub struct DateTimeWithDUT1 {
    datetime: DateTime<Tz>,
    dut1: f64,
}

impl DateTimeWithDUT1 {
    #[must_use]
    pub const fn new(datetime: DateTime<Tz>) -> Self {
        Self {
            datetime,
            dut1: 0.0,
        }
    }
}
