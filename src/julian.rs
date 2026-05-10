// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Julian Day (`JD`) and its derived time scales.
//!
//! Implements section 3.1 and section A.3 of NREL/TP-560-34302. Section 3.1
//! maps a UT calendar instant onto the Julian Day (`JD`, equation 4), the
//! Julian Ephemeris Day (`JDE`, equation 5), the Julian Century (`JC`,
//! equation 6), the Julian Ephemeris Century (`JCE`, equation 7) and the
//! Julian Ephemeris Millennium (`JME`, equation 8). Section A.3 inverts
//! equation 4 via equations A15–A23 to recover the calendar date and time
//! of day from a `JD`.

use chrono::{
    DateTime, Datelike, NaiveDate, NaiveDateTime, NaiveTime, TimeDelta, TimeZone, Timelike, Utc,
};
use chrono_tz::Tz;

use crate::SpaDateTime;
use crate::helper::int;

/// Julian Day (computed without the Gregorian-correction term `B`) of
/// 1582-10-04 12:00 UT, the last Julian-calendar instant before the reform.
/// Equation 4 takes the Gregorian branch (`B = 2 − A + INT(A/4)`) once the
/// uncorrected JD strictly exceeds this value, and the Julian branch
/// (`B = 0`) at or below it.
const GREGORIAN_REFORM_JD_NO_B: f64 = 2_299_160.0;

/// `Z = INT(JD + 0.5)` of midnight UT on 1582-10-15, the first
/// Gregorian-calendar day. Step A.3.2 takes the Gregorian branch
/// (equations A15 and A16) once `Z` reaches this value, and the Julian
/// branch (`A = Z`) below it.
const GREGORIAN_REFORM_Z: f64 = 2_299_161.0;

/// Julian Day from `datetime`, with its UT instant computed as `UTC + DUT1`.
///
/// Refer to section 3.1.1, equation 4. The Julian Day is defined in UT, so
/// the input is collapsed to UTC and the calendar fields are read off the
/// UTC instant; the DUT1 offset (UT1 − UTC) carried by `datetime` is added
/// to the seconds-of-minute, propagating naturally into the minute, hour
/// and day decomposition.
///
/// # Examples
///
/// ```
/// use chrono::Utc;
/// use chrono_tz::Tz;
/// use helioxide::{SpaDateTime, julian::calculate_julian_day};
///
/// let now = SpaDateTime::new(Utc::now().with_timezone(&Tz::Europe__Madrid));
/// println!("Julian Day for now: {}", calculate_julian_day(&now));
/// ```
#[must_use]
pub fn calculate_julian_day<Tz: TimeZone>(datetime: &SpaDateTime<Tz>) -> f64 {
    let dt = datetime.datetime().naive_utc();

    let seconds_of_minute =
        f64::from(dt.second()) + f64::from(dt.nanosecond()) / 1.0e9 + datetime.dut1();
    let day_decimal = f64::from(dt.day())
        + (f64::from(dt.hour()) + (f64::from(dt.minute()) + seconds_of_minute / 60.0) / 60.0)
            / 24.0;

    // Treat January and February as months 13 and 14 of the previous year.
    let (year, month) = if dt.month() < 3 {
        (f64::from(dt.year() - 1), f64::from(dt.month() + 12))
    } else {
        (f64::from(dt.year()), f64::from(dt.month()))
    };

    // Equation 4. The Gregorian-correction term `B` only kicks in past the
    // 1582-10-15 reform; the Julian branch sets it to zero.
    let julian_day = int::<f64>(365.25 * (year + 4716.0))
        + int::<f64>(30.6001 * (month + 1.0))
        + (day_decimal - 1524.5);
    if julian_day > GREGORIAN_REFORM_JD_NO_B {
        let a = int::<f64>(year / 100.0);
        julian_day + (2.0 - a + int::<f64>(a / 4.0))
    } else {
        julian_day
    }
}

/// Calendar date corresponding to a given Julian Day, rendered in `tz`.
///
/// `julian_day` is the Julian Day in UT, as produced by
/// [`calculate_julian_day`]. `tz` is the [`chrono_tz::Tz`] the result is
/// projected onto: the JD is interpreted as a UTC instant first, so the
/// round-trip through [`calculate_julian_day`] preserves the wall-clock
/// instant regardless of the timezone offset or DST state.
///
/// Refer to section A.3, equations A15–A23. Step A.3.1 splits `JD + 0.5`
/// into the integer part `Z` and fractional part `F`. Step A.3.2 takes the
/// Gregorian branch (equations A15 and A16) when `Z ≥ 2_299_161` and
/// otherwise sets `A = Z` for the Julian branch. Steps A.3.3 to A.3.6
/// (equations A17 to A20) chain `A → C → D → G → I`. Step A.3.7
/// (equation A21) recovers the decimal day, step A.3.8 (equation A22)
/// the month, and step A.3.9 (equation A23) the year. The fractional day
/// is rounded to whole seconds and added to midnight as a [`TimeDelta`],
/// so a half-second-or-more residue at the last second of the day rolls
/// naturally into the next day, month and year instead of producing the
/// invalid `hour = 24`.
///
/// Returns [`None`] if the resulting `(year, month, day)` is not a
/// proleptic-Gregorian date as understood by chrono. The only triggers in
/// the SPA-validated range are Julian leap days that fall outside chrono's
/// proleptic-Gregorian leap rule, the canonical example being
/// `-1000-02-29` (Table A4.1, JD `1_355_866.5`).
///
/// # Examples
///
/// ```
/// use chrono_tz::Tz;
/// use helioxide::julian::calculate_calendar_date_from_julian_day;
///
/// // Table A4.1: 2000-01-01 12:00:00 UT.
/// let dt = calculate_calendar_date_from_julian_day(2_451_545.0, Tz::UTC).unwrap();
/// assert_eq!(dt.to_string(), "2000-01-01 12:00:00 UTC");
/// ```
///
/// [`calculate_julian_day`]: self::calculate_julian_day
#[must_use]
#[allow(clippy::many_single_char_names)] // Single-letter names mirror SPA's notation.
pub fn calculate_calendar_date_from_julian_day(julian_day: f64, tz: Tz) -> Option<DateTime<Tz>> {
    // Step A.3.1: split JD + 0.5 into Z (integer part) and F (fractional part in [0, 1)).
    let jd_plus_half = julian_day + 0.5;
    let z = int::<f64>(jd_plus_half);
    let f = jd_plus_half - z;

    // Step A.3.2: Gregorian branch (A15, A16) past the 1582-10-15 reform.
    let a = if z < GREGORIAN_REFORM_Z {
        z
    } else {
        let b = int::<f64>((z - 1_867_216.25) / 36_524.25); // A15
        z + 1.0 + b - int::<f64>(b / 4.0) // A16
    };

    let c = a + 1524.0; // A17, step A.3.3
    let d = int::<f64>((c - 122.1) / 365.25); // A18, step A.3.4
    let g = int::<f64>(365.25 * d); // A19, step A.3.5
    let i = int::<f64>((c - g) / 30.6001); // A20, step A.3.6
    let day_decimal = c - g - int::<f64>(30.6001 * i) + f; // A21, step A.3.7

    // Step A.3.8 (A22): the algorithm guarantees I ∈ [4, 15], so `month` lands in [1, 12].
    let i_int = int::<i32>(i);
    let month = if i_int < 14 { i_int - 1 } else { i_int - 13 };

    // Step A.3.9 (A23): January and February belong to the previous astronomical year.
    let year = if month > 2 {
        int::<i32>(d) - 4716
    } else {
        int::<i32>(d) - 4715
    };

    // Round the time of day to whole seconds (matching the precision of
    // the SPA reference tables) and add it to midnight as a `TimeDelta`.
    // A fractional day that rounds to a full 86 400 s collapses to
    // 00:00:00 of the next day, cascading through month and year as
    // needed instead of producing the invalid `hour = 24`.
    let day_int = int::<i32>(day_decimal);
    let day_fraction = day_decimal - f64::from(day_int);
    let seconds_into_day = i64::from(int::<i32>((day_fraction * 86_400.0).round()));

    NaiveDate::from_ymd_opt(year, month.cast_unsigned(), day_int.cast_unsigned()).map(|date| {
        let naive = NaiveDateTime::new(date, NaiveTime::MIN) + TimeDelta::seconds(seconds_into_day);
        Utc.from_utc_datetime(&naive).with_timezone(&tz)
    })
}

/// Computes the Julian Ephemeris Day (JDE) from the given Julian Day (JD)
/// and ΔT (`delta_t`) expressed in seconds.
///
/// Refer to section 3.1.2, equation 5: `JDE = JD + ΔT / 86 400`.
#[must_use]
pub const fn calculate_julian_ephemeris_day(julian_day: f64, delta_t: f64) -> f64 {
    julian_day + delta_t / 86_400.0
}

/// Computes the Julian Century (JC) from the given Julian Day (JD) for the
/// J2000.0 standard epoch.
///
/// Refer to section 3.1.3, equation 6: `JC = (JD − 2_451_545) / 36_525`.
#[must_use]
pub const fn calculate_julian_century(julian_day: f64) -> f64 {
    (julian_day - 2_451_545.0) / 36_525.0
}

/// Computes the Julian Ephemeris Century (JCE) from the given Julian
/// Ephemeris Day (JDE) for the J2000.0 standard epoch.
///
/// Refer to section 3.1.3, equation 7: `JCE = (JDE − 2_451_545) / 36_525`.
#[must_use]
pub const fn calculate_julian_ephemeris_century(julian_ephemeris_day: f64) -> f64 {
    (julian_ephemeris_day - 2_451_545.0) / 36_525.0
}

/// Computes the Julian Ephemeris Millennium (JME) from the given Julian
/// Ephemeris Century (JCE) for the J2000.0 standard epoch.
///
/// Refer to section 3.1.4, equation 8: `JME = JCE / 10`.
#[must_use]
pub const fn calculate_julian_ephemeris_millennium(julian_ephemeris_century: f64) -> f64 {
    julian_ephemeris_century / 10.0
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use chrono::Offset;

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
        // (-1000, 2, 29, 0, 0, 0, 1_355_866.5) is exercised separately:
        // year -1000 is not a leap year in chrono's proleptic Gregorian
        // calendar, so the round-trip surfaces as `Option::None`.
        (-1001, 8, 17, 21, 36, 0, 1_355_671.4),
        (-4712, 1, 1, 12, 0, 0, 0.0),
    ];

    /// Reference local datetimes that span different effective UTC offsets.
    /// Each entry: (year, month, day, hour, minute, second).
    const LOCAL_TIME_CASES: [(i32, u32, u32, u32, u32, u32); 2] =
        [(2026, 3, 15, 23, 41, 0), (2026, 3, 30, 23, 41, 0)];

    /// Builds a [`DateTime<Tz>`] from its calendar components and timezone.
    #[allow(clippy::many_single_char_names)]
    fn build_datetime(tz: Tz, y: i32, m: u32, d: u32, h: u32, min: u32, s: u32) -> DateTime<Tz> {
        tz.with_ymd_and_hms(y, m, d, h, min, s)
            .single()
            .unwrap_or_else(|| {
                panic!("Invalid test data: {y:04}-{m:02}-{d:02} {h:02}:{min:02}:{s:02}");
            })
    }

    /// Verifies that [`calculate_julian_day`] produces the expected Julian Day
    /// for every entry in Table A4.1. The 15 entries jointly exercise both
    /// branches of the Julian/Gregorian split in equation 4, the
    /// January-and-February month renumbering, and a wide range of years
    /// (including negative ones) so that an off-by-one in the year/month
    /// renumbering or a wrong `B` correction would surface here.
    #[test]
    fn julian_day_from_datetime_matches_table_a4_1() {
        for &(y, m, d, h, min, s, expected_jd) in &TABLE_A4_1 {
            let dt = build_datetime(chrono_tz::UTC, y, m, d, h, min, s);
            let jd = calculate_julian_day(&SpaDateTime::new(dt));

            assert!(
                (jd - expected_jd).abs() < f64::EPSILON,
                "Julian day mismatch for {dt}. Expected: {expected_jd}, Got: {jd}"
            );
        }
    }

    /// Verifies that [`calculate_calendar_date_from_julian_day`] recovers the
    /// original calendar date for every Julian Day in Table A4.1. The sweep
    /// covers both the Gregorian branch (equations A15 and A16; e.g.
    /// 2000-01-01) and the Julian branch (`A = Z`; e.g. -1001-08-17), as
    /// well as both year-boundary branches of equation A23 (`m > 2` for
    /// e.g. 837-04-10, `m ≤ 2` for e.g. -122-01-01) and both arms of
    /// equation A22 (`I < 14` for e.g. 1987-06-19, `I ≥ 14` for e.g.
    /// 2000-01-01). Failure of any single entry pins the equation it
    /// happens to cover.
    #[test]
    fn calendar_date_round_trip_matches_table_a4_1() {
        for &(y, m, d, h, min, s, expected_jd) in &TABLE_A4_1 {
            let dt = build_datetime(chrono_tz::UTC, y, m, d, h, min, s);
            let recovered_dt =
                calculate_calendar_date_from_julian_day(expected_jd, chrono_tz::UTC).unwrap();

            assert!(
                (recovered_dt - dt).as_seconds_f64().abs() < f64::EPSILON,
                "Calendar date mismatch for JD {expected_jd}. Expected: {dt}, Got: {recovered_dt}"
            );
        }
    }

    /// Verifies that [`calculate_julian_day`] is invariant under equivalent
    /// UTC and local representations across different effective UTC offsets.
    /// JD lives in UT, so the same instant must yield the same JD whether it
    /// is presented as UTC or wrapped in a non-UTC timezone whose offset
    /// changes between the two probes (Madrid winter UTC+1 vs summer UTC+2).
    #[test]
    fn julian_day_is_timezone_invariant_across_local_offsets() {
        let local_dts: Vec<DateTime<Tz>> = LOCAL_TIME_CASES
            .iter()
            .map(|&(y, m, d, h, min, s)| {
                build_datetime(chrono_tz::Europe::Madrid, y, m, d, h, min, s)
            })
            .collect();

        assert_ne!(
            local_dts[0].offset().fix().local_minus_utc(),
            local_dts[1].offset().fix().local_minus_utc(),
            "Local timezone cases must cover different UTC offsets"
        );

        for local_dt in local_dts {
            let utc_dt = local_dt.with_timezone(&chrono_tz::UTC);

            let local_jd = calculate_julian_day(&SpaDateTime::new(local_dt));
            let utc_jd = calculate_julian_day(&SpaDateTime::new(utc_dt));

            assert!(
                (local_jd - utc_jd).abs() < f64::EPSILON,
                "Julian day must not depend on timezone representation. Local: {local_jd}, UTC: {utc_jd}"
            );
        }
    }

    /// Verifies that [`calculate_calendar_date_from_julian_day`] preserves the
    /// local civil time across different effective UTC offsets, including a
    /// pair that straddles the Madrid DST transition (winter UTC+1 vs summer
    /// UTC+2 in 2026). The round-trip must survive both DST regimes so the
    /// timezone semantics cannot silently drift on a particular offset.
    #[test]
    fn calendar_date_round_trip_preserves_local_time_across_offsets() {
        let local_dts: Vec<DateTime<Tz>> = LOCAL_TIME_CASES
            .iter()
            .map(|&(y, m, d, h, min, s)| {
                build_datetime(chrono_tz::Europe::Madrid, y, m, d, h, min, s)
            })
            .collect();

        assert_ne!(
            local_dts[0].offset().fix().local_minus_utc(),
            local_dts[1].offset().fix().local_minus_utc(),
            "Local timezone cases must cover different UTC offsets"
        );

        for local_dt in local_dts {
            let julian_day = calculate_julian_day(&SpaDateTime::new(local_dt));
            let recovered_dt =
                calculate_calendar_date_from_julian_day(julian_day, chrono_tz::Europe::Madrid)
                    .unwrap();

            assert_eq!(recovered_dt, local_dt);
        }
    }

    /// Pins the Julian/Gregorian threshold at exactly `Z = 2_299_161` (i.e.
    /// `JD = 2_299_160.5`, the start of the Gregorian calendar on
    /// 1582-10-15). Step A.3.2 must take the Julian branch (`A = Z`) one
    /// integer below the threshold and the Gregorian branch (equations A15
    /// and A16) at and above it. The two probes pick up the *consecutive*
    /// civil dates 1582-10-04 (last Julian) and 1582-10-15 (first Gregorian)
    /// — a 10-day jump that historical records confirm. A typo on the
    /// threshold constant (e.g. `2_299_160` or `2_299_162`) would shift the
    /// switch by a full day and break at least one of the two expected
    /// dates here, even when Table A4.1 (whose closest entries are
    /// 1600-01-01 and 1900-01-01) still passes.
    #[test]
    fn calendar_date_picks_julian_or_gregorian_branch_at_2299161() {
        let last_julian =
            calculate_calendar_date_from_julian_day(2_299_159.5, chrono_tz::UTC).unwrap();
        let expected_last_julian = build_datetime(chrono_tz::UTC, 1582, 10, 4, 0, 0, 0);
        assert_eq!(
            last_julian, expected_last_julian,
            "JD 2_299_159.5 (Z = 2_299_160) must take the Julian branch and yield 1582-10-04"
        );

        let first_gregorian =
            calculate_calendar_date_from_julian_day(2_299_160.5, chrono_tz::UTC).unwrap();
        let expected_first_gregorian = build_datetime(chrono_tz::UTC, 1582, 10, 15, 0, 0, 0);
        assert_eq!(
            first_gregorian, expected_first_gregorian,
            "JD 2_299_160.5 (Z = 2_299_161) must take the Gregorian branch and yield 1582-10-15"
        );
    }

    /// Pins the seconds-rollover safety net of step A.3.7. When the
    /// fractional day is so close to one that the integer-second rounding
    /// of `day_fraction · 86_400` lands on `86_400` (reachable for
    /// `day_fraction ≥ 0.999_994_213`), a naive decomposition into
    /// `(hour, minute, second)` would yield the invalid `hour = 24`. The
    /// implementation builds the time of day by adding a [`TimeDelta`] of
    /// `seconds_into_day` seconds to midnight, so the same `86_400`
    /// collapses to the next day at `00:00:00` and the carry cascades
    /// through the month and year for free. The probe lands a JD that the
    /// bare equations A21–A23 read as 1998-12-31 23:59:59.568 (just shy of
    /// the 1999 boundary); rounding lifts the seconds count to a full day,
    /// which must surface as 1999-01-01 00:00:00 after the cascade. A stub
    /// that capped the carry at the day, the month or the year would all
    /// surface here as a wrong date.
    #[test]
    fn calendar_date_handles_full_day_rounding_with_year_boundary_carry() {
        let recovered =
            calculate_calendar_date_from_julian_day(2_451_179.499_995, chrono_tz::UTC).unwrap();
        let expected = build_datetime(chrono_tz::UTC, 1999, 1, 1, 0, 0, 0);
        assert_eq!(
            recovered, expected,
            "A day-fraction that rounds to 86_400 s must carry through day, month and year"
        );
    }

    /// Pins the proleptic-Gregorian validity check at the Julian-leap day
    /// of -1000-02-29. This date exists in the Julian calendar that the
    /// SPA algorithm inhabits for old `Z` values (Table A4.1 lists JD
    /// `1_355_866.5` for it) but not in chrono's proleptic-Gregorian
    /// calendar (-1000 is divisible by 4 but not by 400). The function
    /// must therefore return [`None`] rather than panic or fabricate a
    /// neighbouring date. This is the only path through which the
    /// `Option` short-circuit is reachable in the SPA-validated range,
    /// and exercising it pins both the [`NaiveDate::from_ymd_opt`] guard
    /// and the documented fallthrough behaviour.
    #[test]
    fn calendar_date_returns_none_for_julian_leap_day_invalid_in_proleptic_gregorian() {
        let result = calculate_calendar_date_from_julian_day(1_355_866.5, chrono_tz::UTC);
        assert!(
            result.is_none(),
            "JD 1_355_866.5 (-1000-02-29 Julian) must be rejected by chrono's proleptic Gregorian"
        );
    }

    /// Pins the DUT1 propagation in equation 4. UT = UTC + DUT1, so a
    /// non-zero DUT1 must shift the Julian Day by exactly `DUT1 / 86_400`
    /// days — the same scaling that converts seconds into a fraction of a
    /// day. Both signs are swept so a stray sign flip on the DUT1 addition
    /// (which would leave `|JD|` changing by the right magnitude on one
    /// side only) surfaces here. The 1e-9 absolute tolerance sits a couple
    /// of `ulp(JD) ≈ 4.7·10⁻¹⁰` above the rounding floor for the
    /// 2003-10-17 reference instant, three orders of magnitude below the
    /// expected shift `≈ 5.79·10⁻⁶ d`, so an omitted, miswired or
    /// wrong-scale DUT1 cannot hide here.
    #[test]
    fn julian_day_includes_dut1_correction() {
        let dt = build_datetime(chrono_tz::UTC, 2003, 10, 17, 19, 30, 30);
        let jd_zero = calculate_julian_day(&SpaDateTime::new(dt));

        for &dut1 in &[-0.5_f64, 0.5] {
            let with_dut1 = SpaDateTime::new(dt).try_with_dut1(dut1).unwrap();
            let actual_shift = calculate_julian_day(&with_dut1) - jd_zero;
            let expected_shift = dut1 / 86_400.0;
            assert!(
                (actual_shift - expected_shift).abs() < 1e-9,
                "DUT1 = {dut1} s must shift JD by {expected_shift} d; got {actual_shift}"
            );
        }
    }

    /// Pins sub-second precision in equation 4. The seconds field is
    /// folded into the fraction of a day with the nanoseconds added at
    /// `1e-9 s` resolution; reading it at integer-second precision (a
    /// plausible regression if the nanosecond term were dropped) would
    /// silently zero out 250 ms here. Table A4.1 cannot catch this: every
    /// reference instant in it is second-aligned. The expected shift is
    /// `0.25 / 86_400` d ≈ 2.89·10⁻⁶ d, four orders of magnitude above
    /// `ulp(JD) ≈ 4.7·10⁻¹⁰` at the chosen instant, so a 1e-9 tolerance
    /// rules out a dropped nanosecond term while staying comfortably
    /// above the rounding noise of the `(s + ns) / 60 / 60 / 24` chain.
    #[test]
    fn julian_day_preserves_subsecond_precision() {
        let base = build_datetime(chrono_tz::UTC, 2003, 10, 17, 19, 30, 30);
        let with_nanos = base.with_nanosecond(250_000_000).unwrap();

        let jd_base = calculate_julian_day(&SpaDateTime::new(base));
        let jd_nanos = calculate_julian_day(&SpaDateTime::new(with_nanos));

        let actual_shift = jd_nanos - jd_base;
        let expected_shift = 0.25 / 86_400.0;
        assert!(
            (actual_shift - expected_shift).abs() < 1e-9,
            "250 ms must shift JD by {expected_shift} d; got {actual_shift}"
        );
    }

    /// Verifies that the derived Julian quantities honour the SPA
    /// constants (J2000.0 epoch, century length, seconds per day, JME
    /// ratio). Reference Julian Day taken from Table A5.1.
    #[test]
    fn julian_derived_quantities_pin_spa_constants() {
        // JC vanishes at the J2000.0 epoch.
        assert!(calculate_julian_century(2_451_545.0).abs() < f64::EPSILON);

        // Advancing one Julian century (36 525 days) must shift JC by exactly 1.0.
        assert!(
            (calculate_julian_century(2_451_545.0 + 36_525.0) - 1.0).abs() < f64::EPSILON,
            "Julian century length must be 36 525 days"
        );

        let jd = 2_452_930.312_847; // Table A5.1, 17 October 2003.

        // ΔT of one full day (86 400 s) must shift JDE by exactly one day.
        assert!(
            (calculate_julian_ephemeris_day(jd, 86_400.0) - jd - 1.0).abs() < f64::EPSILON,
            "ΔT of one day must shift JDE by exactly one day"
        );

        // With ΔT equal to zero, JDE collapses to JD, so JCE must coincide with JC.
        let jde = calculate_julian_ephemeris_day(jd, 0.0);
        let jc = calculate_julian_century(jd);
        let jce = calculate_julian_ephemeris_century(jde);
        assert!(
            (jc - jce).abs() < f64::EPSILON,
            "JCE must use the same epoch and century length as JC"
        );

        // JME equals JCE divided by 10 by construction.
        let jme = calculate_julian_ephemeris_millennium(jce);
        assert!(
            (jme * 10.0 - jce).abs() < f64::EPSILON,
            "JME must equal JCE divided by 10"
        );
    }
}
