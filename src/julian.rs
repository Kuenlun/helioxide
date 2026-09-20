// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Julian Day and derived time scales. Section 3.1 plus appendix A.3.

use chrono::{
    DateTime, Datelike, NaiveDate, NaiveDateTime, NaiveTime, Offset, TimeDelta, TimeZone, Timelike,
    Utc,
};
use chrono_tz::Tz;

use crate::SpaDateTime;
use crate::helper::int;

const LAST_JULIAN_DATE: (i32, u32, u32) = (1582, 10, 4);
const FIRST_GREGORIAN_DATE: (i32, u32, u32) = (1582, 10, 15);
const SECONDS_PER_DAY: f64 = 86_400.0;

pub(crate) fn is_gregorian_reform_gap(date: NaiveDate) -> bool {
    let fields = (date.year(), date.month(), date.day());
    fields > LAST_JULIAN_DATE && fields < FIRST_GREGORIAN_DATE
}

/// `Z = INT(JD + 0.5)` at the first Gregorian-calendar day (1582-10-15).
const GREGORIAN_REFORM_Z: f64 = 2_299_161.0;

/// Julian Day from `datetime`, with `UT = UTC + DUT1`. Equation 4.
///
/// Uses Julian calendar labels through 1582-10-04 and Gregorian labels
/// from 1582-10-15. Returns NaN for the skipped dates between them.
#[must_use]
pub fn julian_day<Tz: TimeZone>(datetime: &SpaDateTime<Tz>) -> f64 {
    let dt = datetime.datetime().naive_utc();
    if is_gregorian_reform_gap(dt.date()) {
        return f64::NAN;
    }

    let seconds_of_minute =
        f64::from(dt.second()) + f64::from(dt.nanosecond()) / 1.0e9_f64 + datetime.dut1();
    let day_decimal = f64::from(dt.day())
        + (f64::from(dt.hour())
            + (f64::from(dt.minute()) + seconds_of_minute / 60.0_f64) / 60.0_f64)
            / 24.0_f64;

    // January and February count as months 13 and 14 of the previous year.
    let (year, month) = if dt.month() < 3 {
        (
            f64::from(dt.year()) - 1.0_f64,
            f64::from(dt.month()) + 12.0_f64,
        )
    } else {
        (f64::from(dt.year()), f64::from(dt.month()))
    };

    let julian_day =
        int(365.25 * (year + 4716.0)) + int(30.6001 * (month + 1.0)) + (day_decimal - 1_524.5_f64);
    if (dt.year(), dt.month(), dt.day()) >= FIRST_GREGORIAN_DATE {
        let a = int(year / 100.0);
        julian_day + (2.0 - a + int(a / 4.0))
    } else {
        julian_day
    }
}

/// Calendar date corresponding to `julian_day` (UT), projected onto `tz`.
///
/// Returns [`None`] when the date or its rounded time is outside chrono's
/// representable range, or is valid only in the Julian calendar (for example
/// `-1000-02-29`, Table A4.1).
///
/// Equations A15 to A23.
#[must_use]
#[expect(
    clippy::many_single_char_names,
    clippy::cast_possible_truncation,
    clippy::as_conversions,
    reason = "The SPA equations round or truncate floating-point calendar components before validation."
)]
pub fn calendar_date_from_julian_day(julian_day: f64, tz: Tz) -> Option<DateTime<Tz>> {
    if !julian_day.is_finite() {
        return None;
    }
    let jd_plus_half = julian_day + 0.5_f64;
    let unrounded_day = int(jd_plus_half);
    let rounded_seconds = ((jd_plus_half - unrounded_day) * SECONDS_PER_DAY).round();
    // Advance the astronomical day before decoding its calendar label.
    // Chrono's next Gregorian date can skip a Julian leap day or enter the reform gap.
    let z = unrounded_day + (rounded_seconds / SECONDS_PER_DAY).floor();
    let seconds_into_day = rounded_seconds.rem_euclid(SECONDS_PER_DAY) as i64;

    let a = if z < GREGORIAN_REFORM_Z {
        z
    } else {
        let b = int((z - 1_867_216.25) / 36_524.25);
        z + 1.0_f64 + b - int(b / 4.0)
    };

    let c = a + 1_524.0_f64;
    let d = int((c - 122.1) / 365.25);
    let g = int(365.25 * d);
    let i = int((c - g) / 30.6001);
    let day_decimal = c - g - int(30.6001 * i);

    let i_int = i as i32;
    let month = if i_int < 14_i32 {
        i_int.saturating_sub(1)
    } else {
        i_int.saturating_sub(13)
    };
    let year = if month > 2_i32 {
        (d as i32).saturating_sub(4_716)
    } else {
        (d as i32).saturating_sub(4_715)
    };

    let day_int = day_decimal as i32;
    NaiveDate::from_ymd_opt(year, month.cast_unsigned(), day_int.cast_unsigned()).and_then(|date| {
        // The carry was applied to Z, leaving fewer than 86400 seconds.
        let time = NaiveTime::MIN
            .overflowing_add_signed(TimeDelta::seconds(seconds_into_day))
            .0;
        let local = Utc
            .from_utc_datetime(&NaiveDateTime::new(date, time))
            .with_timezone(&tz);
        local
            .naive_utc()
            .checked_add_offset(local.offset().fix())
            .map(|_| local)
    })
}

/// `JDE = JD + ΔT / 86_400`. Equation 5.
#[must_use]
pub const fn julian_ephemeris_day(julian_day: f64, delta_t: f64) -> f64 {
    julian_day + delta_t / 86_400.0
}

/// `JC = (JD − 2_451_545) / 36_525`. Equation 6.
#[must_use]
pub const fn julian_century(julian_day: f64) -> f64 {
    (julian_day - 2_451_545.0) / 36_525.0
}

/// `JCE = (JDE − 2_451_545) / 36_525`. Equation 7.
#[must_use]
pub const fn julian_ephemeris_century(julian_ephemeris_day: f64) -> f64 {
    (julian_ephemeris_day - 2_451_545.0) / 36_525.0
}

/// `JME = JCE / 10`. Equation 8.
#[must_use]
pub const fn julian_ephemeris_millennium(julian_ephemeris_century: f64) -> f64 {
    julian_ephemeris_century / 10.0
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn calendar_date_rejects_unrepresentable_dates_and_rounded_times() {
        for jd in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MAX,
            f64::MIN,
        ] {
            assert!(calendar_date_from_julian_day(jd, chrono_tz::UTC).is_none());
        }
        let last_instant = SpaDateTime::new(
            DateTime::<Utc>::MAX_UTC
                .checked_sub_signed(TimeDelta::milliseconds(250))
                .unwrap(),
        );
        let jd = julian_day(&last_instant);
        assert!(calendar_date_from_julian_day(jd, chrono_tz::UTC).is_none());
    }

    /// Table A4.1: `(year, month, day, hour, minute, second, expected_jd)`.
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
        // -1000-02-29 (JD 1_355_866.5) is exercised separately: chrono's
        // proleptic Gregorian does not recognise it as a leap day.
        (-1001, 8, 17, 21, 36, 0, 1_355_671.4),
        (-4712, 1, 1, 12, 0, 0, 0.0),
    ];

    const LOCAL_TIME_CASES: [(i32, u32, u32, u32, u32, u32); 2] =
        [(2026, 3, 15, 23, 41, 0), (2026, 3, 30, 23, 41, 0)];

    #[expect(
        clippy::many_single_char_names,
        reason = "Keep the parameter names and grouping used by the SPA equations."
    )]
    fn build_datetime(tz: Tz, y: i32, m: u32, d: u32, h: u32, min: u32, s: u32) -> DateTime<Tz> {
        tz.with_ymd_and_hms(y, m, d, h, min, s).single().unwrap()
    }

    #[test]
    fn julian_day_matches_table_a4_1() {
        for &(y, m, d, h, min, s, expected) in &TABLE_A4_1 {
            let dt = build_datetime(chrono_tz::UTC, y, m, d, h, min, s);
            let jd = julian_day(&SpaDateTime::new(dt));
            assert!(
                (jd - expected).abs() < f64::EPSILON,
                "JD for {dt}: got {jd}"
            );
        }
    }

    #[test]
    fn calendar_date_round_trip_matches_table_a4_1() {
        for &(y, m, d, h, min, s, expected_jd) in &TABLE_A4_1 {
            let dt = build_datetime(chrono_tz::UTC, y, m, d, h, min, s);
            let recovered = calendar_date_from_julian_day(expected_jd, chrono_tz::UTC).unwrap();
            assert!(
                (recovered - dt).as_seconds_f64().abs() < f64::EPSILON,
                "JD {expected_jd}: got {recovered}",
            );
        }
    }

    #[test]
    fn julian_day_is_timezone_invariant() {
        let local_dts = LOCAL_TIME_CASES.map(|(y, m, d, h, min, s)| {
            build_datetime(chrono_tz::Europe::Madrid, y, m, d, h, min, s)
        });
        let [before, after] = &local_dts;
        assert_ne!(
            before.offset().fix().local_minus_utc(),
            after.offset().fix().local_minus_utc(),
        );
        for local in local_dts {
            let utc = local.with_timezone(&chrono_tz::UTC);
            let local_jd = julian_day(&SpaDateTime::new(local));
            let utc_jd = julian_day(&SpaDateTime::new(utc));
            assert!((local_jd - utc_jd).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn calendar_date_preserves_local_time_across_offsets() {
        for &(y, m, d, h, min, s) in &LOCAL_TIME_CASES {
            let local = build_datetime(chrono_tz::Europe::Madrid, y, m, d, h, min, s);
            let jd = julian_day(&SpaDateTime::new(local));
            let recovered = calendar_date_from_julian_day(jd, chrono_tz::Europe::Madrid).unwrap();
            assert_eq!(recovered, local);
        }
    }

    #[test]
    fn calendar_date_picks_julian_or_gregorian_branch_at_2299161() {
        let last_julian = calendar_date_from_julian_day(2_299_159.5, chrono_tz::UTC).unwrap();
        assert_eq!(
            last_julian,
            build_datetime(chrono_tz::UTC, 1582, 10, 4, 0, 0, 0)
        );
        let first_gregorian = calendar_date_from_julian_day(2_299_160.5, chrono_tz::UTC).unwrap();
        assert_eq!(
            first_gregorian,
            build_datetime(chrono_tz::UTC, 1582, 10, 15, 0, 0, 0)
        );
    }

    #[test]
    fn calendar_date_handles_full_day_rounding_with_year_boundary_carry() {
        let recovered = calendar_date_from_julian_day(2_451_179.499_995, chrono_tz::UTC).unwrap();
        assert_eq!(
            recovered,
            build_datetime(chrono_tz::UTC, 1999, 1, 1, 0, 0, 0)
        );
    }

    #[test]
    fn calendar_date_returns_none_for_julian_leap_day() {
        assert!(calendar_date_from_julian_day(1_355_866.5, chrono_tz::UTC).is_none());
    }

    #[test]
    fn julian_day_includes_dut1_correction() {
        let dt = build_datetime(chrono_tz::UTC, 2003, 10, 17, 19, 30, 30);
        let jd_zero = julian_day(&SpaDateTime::new(dt));
        for &dut1 in &[-0.5_f64, 0.5_f64] {
            let with_dut1 = SpaDateTime::new(dt).try_with_dut1(dut1).unwrap();
            let shift = julian_day(&with_dut1) - jd_zero;
            assert!((shift - dut1 / 86_400.0).abs() < 1e-9_f64);
        }
    }

    #[test]
    fn julian_day_preserves_subsecond_precision() {
        let base = build_datetime(chrono_tz::UTC, 2003, 10, 17, 19, 30, 30);
        let with_nanos = base.with_nanosecond(250_000_000).unwrap();
        let shift = julian_day(&SpaDateTime::new(with_nanos)) - julian_day(&SpaDateTime::new(base));
        assert!((shift - 0.25 / 86_400.0).abs() < 1e-9_f64);
    }

    #[test]
    fn julian_derived_quantities_pin_spa_constants() {
        assert!(julian_century(2_451_545.0).abs() < f64::EPSILON);
        assert!((julian_century(2_451_545.0 + 36_525.0) - 1.0).abs() < f64::EPSILON);

        let jd = 2_452_930.312_847_f64;
        assert!((julian_ephemeris_day(jd, 86_400.0) - jd - 1.0).abs() < f64::EPSILON);

        let jde = julian_ephemeris_day(jd, 0.0);
        let jc = julian_century(jd);
        let jce = julian_ephemeris_century(jde);
        assert!((jc - jce).abs() < f64::EPSILON);

        let jme = julian_ephemeris_millennium(jce);
        assert!((jme * 10.0 - jce).abs() < f64::EPSILON);
    }

    #[test]
    fn last_julian_day_does_not_change_calendar_at_noon() {
        for hour in [0, 12, 13, 23] {
            let datetime = Utc.with_ymd_and_hms(1582, 10, 4, hour, 0, 0).unwrap();
            let jd = julian_day(&SpaDateTime::new(datetime));
            assert!((jd - (2_299_159.5 + f64::from(hour) / 24.0)).abs() < 1e-9_f64);
            assert_eq!(
                calendar_date_from_julian_day(jd, chrono_tz::UTC)
                    .unwrap()
                    .naive_utc(),
                datetime.naive_utc()
            );
        }
    }

    #[test]
    fn rounding_carries_in_the_mixed_calendar_before_chrono_conversion() {
        let date = calendar_date_from_julian_day(2_299_160.499_999, chrono_tz::UTC).unwrap();
        assert_eq!(
            date.naive_utc(),
            Utc.with_ymd_and_hms(1582, 10, 15, 0, 0, 0)
                .unwrap()
                .naive_utc()
        );
        assert!(calendar_date_from_julian_day(1_355_866.499_999, chrono_tz::UTC).is_none());
    }

    #[test]
    fn inverse_rejects_local_date_overflow() {
        let datetime = SpaDateTime::new(DateTime::<Utc>::MAX_UTC - TimeDelta::minutes(1));
        let jd = julian_day(&datetime);
        assert!(calendar_date_from_julian_day(jd, chrono_tz::Pacific::Kiritimati).is_none());
    }

    #[test]
    fn skipped_reform_dates_are_rejected_by_both_solar_pipelines() {
        use crate::{Observer, SolarDay, SolarPosition, SpaTimeError};

        let observer = Observer::try_at_sea_level_isa(0.0, 0.0).unwrap();
        for day in 5..=14 {
            let date = SpaDateTime::new(Utc.with_ymd_and_hms(1582, 10, day, 12, 0, 0).unwrap());
            assert!(julian_day(&date).is_nan());
            assert_eq!(
                SolarPosition::compute(&date, observer),
                Err(SpaTimeError::GregorianReformGap)
            );
            assert_eq!(
                SolarDay::compute(&date, observer),
                Err(SpaTimeError::GregorianReformGap)
            );
        }
        for (month, day) in [(10, 4), (10, 15), (11, 1)] {
            let date = SpaDateTime::new(Utc.with_ymd_and_hms(1582, month, day, 12, 0, 0).unwrap());
            assert!(SolarPosition::compute(&date, observer).is_ok());
        }
        assert!(
            SpaTimeError::GregorianReformGap
                .to_string()
                .contains("1582")
        );
    }
}
