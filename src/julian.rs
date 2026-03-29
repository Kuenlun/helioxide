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

use chrono::{DateTime, Datelike, TimeZone, Timelike};
use chrono_tz::{OffsetComponents, Tz};

use crate::DateTimeWithDUT1;
use crate::helper::int;

/// Computes the Julian Day for the provided datetime, accounting for DUT1 and timezone corrections.
///
/// Refer to section 3.1.1.
///
/// # Examples
///
/// ```
/// use chrono::Utc;
/// use chrono_tz::Tz;
/// use helioxide::{DateTimeWithDUT1, julian::calculate_julian_day};
///
/// let now = DateTimeWithDUT1::new(Utc::now().with_timezone(&Tz::Europe__Madrid));
/// println!("Julian Day for now: {}", calculate_julian_day(&now));
/// ```
#[must_use]
pub fn calculate_julian_day(datetime: &DateTimeWithDUT1) -> f64 {
    let dt = &datetime.datetime;

    let (year, month) = if dt.month() < 3 {
        (dt.year() - 1, dt.month() + 12)
    } else {
        (dt.year(), dt.month())
    };

    let seconds = f64::from(dt.second()) + (f64::from(dt.nanosecond()) / 1_000_000_000.0);
    let tz_offset_s = dt.offset().base_utc_offset().as_seconds_f64();
    let day_decimal = f64::from(dt.day())
        + (f64::from(dt.hour())
            + (f64::from(dt.minute()) + (seconds + datetime.dut1 - tz_offset_s) / 60.0) / 60.0)
            / 24.0;

    // Compute JD using equation 4
    let julian_day: f64 = int::<f64>(365.25 * f64::from(year + 4716))
        + int::<f64>(30.6001 * f64::from(month + 1))
        + day_decimal
        - 1524.5;

    if julian_day > 2_299_160.0 {
        // Gregorian calendar correction
        let b = {
            let a = int::<f64>(f64::from(year) / 100.0);
            2.0 - a + int::<f64>(a / 4.0)
        };
        julian_day + b
    } else {
        // No correction for Julian calendar dates
        julian_day
    }
}

/// Computes the calendar date corresponding to a given Julian Day, accounting for timezone corrections.
///
/// Refer to section A.3.
///
/// # Examples
///
/// ```
/// use chrono::Utc;
/// use chrono_tz::Tz;
/// use helioxide::julian::calculate_calendar_date_from_julian_day;
///
/// let julian_day = 2_461_128.0;
/// let datetime = calculate_calendar_date_from_julian_day(julian_day, Tz::Europe__Madrid).unwrap();
/// println!("Calendar date for Julian Day {}: {}", julian_day, datetime);
/// ```
#[must_use]
#[allow(clippy::many_single_char_names)] // Keep single-letter names to mirror NREL SPA notation
pub fn calculate_calendar_date_from_julian_day(
    julian_day: f64,
    tz: Tz,
) -> chrono::MappedLocalTime<DateTime<Tz>> {
    // A.3.1. Add 0.5 to the Julian Day (JD), then record the integer of the result as Z, and the fraction decimal as F.
    let tz_offset_days = tz
        .offset_from_utc_datetime(&chrono::DateTime::UNIX_EPOCH.naive_utc())
        .base_utc_offset()
        .as_seconds_f64()
        / 86_400.0;
    let jd = julian_day + 0.5 + tz_offset_days;
    let z = int::<f64>(jd);
    let f = jd - z;

    // A.3.2. If Z is less than 2299161, then record A equals Z. Else, calculate the term B
    let a = if z < 2_299_161.0 {
        z
    } else {
        let b = int::<f64>((z - 1_867_216.25) / 36_524.25); // A15
        z + 1.0 + b - int::<f64>(b / 4.0) // A16
    };

    // A.3.3. Calculate the term C
    let c = a + 1524.0; // A17

    // A.3.4. Calculate the term D
    let d = int::<f64>((c - 122.1) / 365.25); // A18

    // A.3.5. Calculate the term G
    let g = int::<f64>(365.25 * d); // A19

    // A.3.6. Calculate the term I
    let i = int::<f64>((c - g) / 30.6001); // A20

    // A.3.7. Calculate the day number of the month with decimals
    let day_decimal = c - g - int::<f64>(30.6001 * i) + f; // A21

    // A.3.8. Calculate the month number (A22)
    let month = if int::<i32>(i) < 14 {
        int::<i32>(i).cast_unsigned() - 1
    } else {
        int::<i32>(i).cast_unsigned() - 13
    };

    // A.3.8. Calculate the month number (A23)
    let year = if month > 2 {
        int::<i32>(d) - 4716
    } else {
        int::<i32>(d) - 4715
    };

    let day = int::<i32>(day_decimal).cast_unsigned();
    let day_fraction = day_decimal - int::<f64>(day_decimal);

    // Convert the fractional day into total seconds.
    // Rounding prevents floating-point precision issues from dropping a second.
    let total_seconds = int::<i32>((day_fraction * 86_400.0).round()).cast_unsigned();

    // Extract time components using standard integer arithmetic
    let hour = total_seconds / 3600;
    let minute = (total_seconds % 3600) / 60;
    let second = total_seconds % 60;

    tz.with_ymd_and_hms(year, month, day, hour, minute, second)
}

/// Computes the Julian Ephemeris Day (JDE) from the given Julian Day (JD)
/// and ΔT (`delta_t`) expressed in seconds.
///
/// Refer to section 3.1.2.
#[must_use]
pub const fn calculate_julian_ephemeris_day(julian_day: f64, delta_t: f64) -> f64 {
    julian_day + delta_t / 86400.0 // Equation 5
}

/// Computes the Julian Century (JC) from the given Julian Day (JD)
/// for the 2000 standard epoch.
///
/// Refer to section 3.1.3.
#[must_use]
pub const fn calculate_julian_century(julian_day: f64) -> f64 {
    (julian_day - 2_451_545.0) / 36_525.0 // Equation 6
}

/// Computes the Julian Ephemeris Century (JCE) from the given Julian Ephemeris Day (JDE)
/// for the 2000 standard epoch.
///
/// Refer to section 3.1.3.
#[must_use]
pub const fn calculate_julian_ephemeris_century(julian_ephemeris_day: f64) -> f64 {
    (julian_ephemeris_day - 2_451_545.0) / 36_525.0 // Equation 7
}

/// Computes the Julian Ephemeris Millennium (JME) from the given Julian Ephemeris Century (JCE) for the 2000 standard epoch.
///
/// Refer to section 3.1.4.
#[must_use]
pub const fn calculate_julian_ephemeris_millennium(julian_ephemeris_century: f64) -> f64 {
    julian_ephemeris_century / 10.0 // Equation 8
}

#[cfg(test)]
#[coverage(off)]
mod tests {
    use super::*;

    /// Reference data from Table A4.1.
    /// Each entry: (year, month, day, hour, minute, second, `expected_julian_day`).
    const TABLE_A4_1: [(i32, u32, u32, u32, u32, u32, f64); 15] = [
        (2000, 1, 1, 12, 0, 0, 2_451_545.0),
        (1999, 1, 1, 0, 0, 0, 2_451_179.5),
        (1987, 1, 27, 0, 0, 0, 2_446_822.5),
        (1987, 6, 19, 12, 0, 0, 2_446_966.0),
        (1988, 1, 27, 0, 0, 0, 2_447_187.5),
        (1988, 6, 19, 12, 0, 0, 2_447_332.0),
        (1900, 1, 1, 0, 0, 0, 2_415_020.5),
        (1600, 1, 1, 0, 0, 0, 2_305_447.5),
        (1600, 12, 31, 0, 0, 0, 2_305_812.5),
        (837, 4, 10, 7, 12, 0, 2_026_871.8),
        (-123, 12, 31, 0, 0, 0, 1_676_496.5),
        (-122, 1, 1, 0, 0, 0, 1_676_497.5),
        (-1000, 7, 12, 12, 0, 0, 1_356_001.0),
        // (-1000, 2, 29, 0, 0, 0, 1_355_866.5), // Skipped: year -1000 is not a leap year in chrono's proleptic Gregorian calendar
        (-1001, 8, 17, 21, 36, 0, 1_355_671.4),
        (-4712, 1, 1, 12, 0, 0, 0.0),
    ];

    /// Builds `(DateTime, expected_julian_day)` pairs from [`TABLE_A4_1`].
    fn table_a4_1_cases() -> Vec<(DateTime<Tz>, f64)> {
        TABLE_A4_1
            .iter()
            .map(|&(y, m, d, h, min, s, expected_jd)| {
                let dt = chrono_tz::UTC
                    .with_ymd_and_hms(y, m, d, h, min, s)
                    .single()
                    .unwrap_or_else(|| {
                        panic!("Invalid test data: {y:04}-{m:02}-{d:02} {h:02}:{min:02}:{s:02}");
                    });
                (dt, expected_jd)
            })
            .collect()
    }

    /// Verifies that [`calculate_julian_day`] produces the expected Julian Day
    /// for every entry in Table A4.1.
    #[test]
    fn julian_day_from_datetime_matches_table_a4_1() {
        for (dt, expected) in table_a4_1_cases() {
            let actual = calculate_julian_day(&DateTimeWithDUT1::new(dt));

            assert!(
                (actual - expected).abs() < f64::EPSILON,
                "Julian day mismatch for {dt}. Expected: {expected}, Got: {actual}"
            );
        }
    }

    /// Verifies that [`calculate_calendar_date_from_julian_day`] recovers the
    /// original calendar date for every Julian Day in Table A4.1.
    #[test]
    fn calendar_date_round_trip_matches_table_a4_1() {
        for (dt, expected_jd) in table_a4_1_cases() {
            let recovered =
                calculate_calendar_date_from_julian_day(expected_jd, chrono_tz::UTC).unwrap();

            assert!(
                (recovered - dt).as_seconds_f64().abs() < f64::EPSILON,
                "Calendar date mismatch for JD {expected_jd}. Expected: {dt}, Got: {recovered}"
            );
        }
    }
}
