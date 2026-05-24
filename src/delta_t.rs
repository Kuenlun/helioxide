// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Polynomial approximation of `ΔT = TT − UT1` (seconds).
//!
//! Implements the piecewise set of Espenak and Meeus from the *Five
//! Millennium Canon of Solar Eclipses: -1999 to +3000* (NASA/TP-2006-214141),
//! together with the long-term parabola of Morrison and Stephenson (2004)
//! used outside the fitted window. Reference: `https://eclipse.gsfc.nasa.gov/SEcat5/deltatpoly.html`.

use chrono::{DateTime, Datelike, TimeZone, Timelike};

const EM_NEG500_TO_500_COEFFS: [f64; 7] = [
    10_583.6,
    -1_014.41,
    33.783_11,
    -5.952_053,
    -0.179_845_2,
    0.022_174_192,
    0.009_031_652_1,
];

const EM_500_TO_1600_COEFFS: [f64; 7] = [
    1_574.2,
    -556.01,
    71.234_72,
    0.319_781,
    -0.850_346_3,
    -0.005_050_998,
    0.008_357_207_3,
];

const EM_1600_TO_1700_COEFFS: [f64; 4] = [120.0, -0.980_8, -0.015_32, 1.0 / 7_129.0];

const EM_1700_TO_1800_COEFFS: [f64; 5] = [
    8.83,
    0.160_3,
    -0.005_928_5,
    0.000_133_36,
    -1.0 / 1_174_000.0,
];

const EM_1800_TO_1860_COEFFS: [f64; 8] = [
    13.72,
    -0.332_447,
    0.006_861_2,
    0.004_111_6,
    -0.000_374_36,
    0.000_012_127_2,
    -0.000_000_169_9,
    0.000_000_000_875,
];

const EM_1860_TO_1900_COEFFS: [f64; 6] = [
    7.62,
    0.573_7,
    -0.251_754,
    0.016_806_68,
    -0.000_447_362_4,
    1.0 / 233_174.0,
];

const EM_1900_TO_1920_COEFFS: [f64; 5] = [-2.79, 1.494_119, -0.059_893_9, 0.006_196_6, -0.000_197];

const EM_1920_TO_1941_COEFFS: [f64; 4] = [21.20, 0.844_93, -0.076_100, 0.002_093_6];

const EM_1941_TO_1961_COEFFS: [f64; 4] = [29.07, 0.407, -1.0 / 233.0, 1.0 / 2_547.0];

const EM_1961_TO_1986_COEFFS: [f64; 4] = [45.45, 1.067, -1.0 / 260.0, -1.0 / 718.0];

const EM_1986_TO_2005_COEFFS: [f64; 6] = [
    63.86,
    0.334_5,
    -0.060_374,
    0.001_727_5,
    0.000_651_814,
    0.000_023_735_99,
];

const EM_2005_TO_2050_COEFFS: [f64; 3] = [62.92, 0.322_17, 0.005_589];

/// End of the linear transition that blends the 2005-2050 polynomial into the
/// long-term parabola, picked so the join at 2050 is continuous.
const TRANSITION_END_YEAR: f64 = 2150.0;

const SECONDS_PER_DAY: f64 = 86_400.0;
const DAYS_IN_COMMON_YEAR: f64 = 365.0;
const DAYS_IN_LEAP_YEAR: f64 = 366.0;

/// Approximate `ΔT` (seconds) at the given Espenak-Meeus decimal year.
#[must_use]
pub fn approximate_delta_t_seconds(decimal_year: f64) -> f64 {
    let y = decimal_year;

    if y > TRANSITION_END_YEAR {
        long_term_parabola(y)
    } else if y > 2050.0 {
        long_term_parabola_transition_to_2150(y)
    } else if y > 2005.0 {
        evaluate_horner(&EM_2005_TO_2050_COEFFS, y - 2000.0)
    } else if y > 1986.0 {
        evaluate_horner(&EM_1986_TO_2005_COEFFS, y - 2000.0)
    } else if y > 1961.0 {
        evaluate_horner(&EM_1961_TO_1986_COEFFS, y - 1975.0)
    } else if y > 1941.0 {
        evaluate_horner(&EM_1941_TO_1961_COEFFS, y - 1950.0)
    } else if y > 1920.0 {
        evaluate_horner(&EM_1920_TO_1941_COEFFS, y - 1920.0)
    } else if y > 1900.0 {
        evaluate_horner(&EM_1900_TO_1920_COEFFS, y - 1900.0)
    } else if y > 1860.0 {
        evaluate_horner(&EM_1860_TO_1900_COEFFS, y - 1860.0)
    } else if y > 1800.0 {
        evaluate_horner(&EM_1800_TO_1860_COEFFS, y - 1800.0)
    } else if y > 1700.0 {
        evaluate_horner(&EM_1700_TO_1800_COEFFS, y - 1700.0)
    } else if y > 1600.0 {
        evaluate_horner(&EM_1600_TO_1700_COEFFS, y - 1600.0)
    } else if y > 500.0 {
        evaluate_horner(&EM_500_TO_1600_COEFFS, (y - 1000.0) / 100.0)
    } else if y >= -500.0 {
        evaluate_horner(&EM_NEG500_TO_500_COEFFS, y / 100.0)
    } else {
        long_term_parabola(y)
    }
}

/// Approximate `ΔT` (seconds) for the UTC instant of `datetime`.
#[must_use]
pub fn approximate_delta_t_seconds_for_datetime<Tz: TimeZone>(datetime: &DateTime<Tz>) -> f64 {
    approximate_delta_t_seconds(decimal_year(datetime))
}

/// Decimal year `year + day_of_year_fraction`, honouring the Gregorian leap rule.
#[must_use]
pub fn decimal_year<Tz: TimeZone>(datetime: &DateTime<Tz>) -> f64 {
    let naive = datetime.naive_utc();
    let year = naive.year();
    let days_in_year = if is_gregorian_leap_year(year) {
        DAYS_IN_LEAP_YEAR
    } else {
        DAYS_IN_COMMON_YEAR
    };

    let seconds_into_day =
        f64::from(naive.nanosecond()).mul_add(1.0e-9, f64::from(naive.num_seconds_from_midnight()));
    let day_fraction = (seconds_into_day / SECONDS_PER_DAY) + f64::from(naive.ordinal0());

    f64::from(year) + day_fraction / days_in_year
}

/// Morrison-Stephenson 2004: `ΔT = -20 + 32·((y - 1820)/100)²` (seconds).
#[inline]
fn long_term_parabola(decimal_year: f64) -> f64 {
    let u = (decimal_year - 1820.0) / 100.0;
    32.0_f64.mul_add(u * u, -20.0)
}

/// Blend the 2005-2050 polynomial into [`long_term_parabola`] by adding the
/// linear correction `-0.5628·(2150 - y)`, which vanishes at `y = 2150`.
#[inline]
fn long_term_parabola_transition_to_2150(decimal_year: f64) -> f64 {
    let secular = long_term_parabola(decimal_year);
    (-0.562_8_f64).mul_add(TRANSITION_END_YEAR - decimal_year, secular)
}

#[inline]
fn evaluate_horner(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0_f64, |acc, &c| acc.mul_add(x, c))
}

#[inline]
const fn is_gregorian_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use chrono::{TimeZone, Timelike, Utc};
    use chrono_tz::Tz;

    /// `(year_at_substitution_zero, constant_term)` for each segment.
    /// Boundary years use a `1e-7 yr` nudge because the dispatch is `prev < y ≤ next`.
    const SEGMENT_CONSTANT_ANCHORS: [(f64, f64); 11] = [
        (0.0, 10_583.6),
        (1_000.0, 1_574.2),
        (1_600.000_000_1, 120.0),
        (1_700.000_000_1, 8.83),
        (1_800.000_000_1, 13.72),
        (1_860.000_000_1, 7.62),
        (1_900.000_000_1, -2.79),
        (1_920.000_000_1, 21.20),
        (1_950.0, 29.07),
        (1_975.0, 45.45),
        (2_000.0, 63.86),
    ];

    const INTERNAL_BOUNDARY_YEARS: [f64; 14] = [
        -500.0, 500.0, 1_600.0, 1_700.0, 1_800.0, 1_860.0, 1_900.0, 1_920.0, 1_941.0, 1_961.0,
        1_986.0, 2_005.0, 2_050.0, 2_150.0,
    ];

    #[test]
    fn every_segment_collapses_to_constant_at_substitution_zero() {
        for (year, expected) in SEGMENT_CONSTANT_ANCHORS {
            let value = approximate_delta_t_seconds(year);
            assert!((value - expected).abs() < 1e-3, "y={year}: got {value}");
        }
    }

    #[test]
    fn short_range_quadratic_matches_hand_computed_values() {
        // 62.92 + 0.32217·t + 0.005589·t² with t = y - 2000.
        assert!((approximate_delta_t_seconds(2_020.0) - 71.599_0).abs() < 1e-4);
        assert!((approximate_delta_t_seconds(2_026.0) - 75.074_584).abs() < 1e-4);
        assert!((approximate_delta_t_seconds(2_050.0) - 93.001_0).abs() < 1e-3);
    }

    #[test]
    fn internal_boundaries_are_continuous_within_two_seconds() {
        let epsilon = 1.0e-6;
        for boundary in INTERNAL_BOUNDARY_YEARS {
            let below = approximate_delta_t_seconds(boundary - epsilon);
            let above = approximate_delta_t_seconds(boundary + epsilon);
            assert!((above - below).abs() < 2.0, "jump at y={boundary}");
        }
    }

    #[test]
    fn long_term_parabola_branch_matches_morrison_stephenson() {
        // -20 + 32·((2500 - 1820)/100)² = 1459.68.
        assert!((approximate_delta_t_seconds(2_500.0) - 1_459.68).abs() < 1e-9);
        // Mirror: -20 + 32·((-1500 - 1820)/100)² = 35251.68.
        assert!((approximate_delta_t_seconds(-1_500.0) - 35_251.68).abs() < 1e-9);
    }

    #[test]
    fn transition_segment_blends_parabola_and_linear_correction() {
        assert!((approximate_delta_t_seconds(2_150.0) - 328.48).abs() < 1e-9);
        assert!((approximate_delta_t_seconds(2_100.0) - 202.74).abs() < 1e-9);

        let just_after_handoff = approximate_delta_t_seconds(2_050.000_001);
        let parabola_at_handoff =
            0.5628_f64.mul_add(-100.0, 32.0_f64.mul_add(2.30_f64.powi(2), -20.0));
        assert!((just_after_handoff - parabola_at_handoff).abs() < 1e-3);
    }

    #[test]
    fn decimal_year_anchors_january_first_to_year_dot_zero() {
        for year in [1_900, 1_999, 2_000, 2_024, 2_026] {
            let dt = Utc.with_ymd_and_hms(year, 1, 1, 0, 0, 0).unwrap();
            assert!((decimal_year(&dt) - f64::from(year)).abs() < f64::EPSILON);
        }
    }

    #[test]
    fn decimal_year_recognises_leap_days() {
        for year in [2_000, 2_024] {
            let dt = Utc.with_ymd_and_hms(year, 2, 29, 0, 0, 0).unwrap();
            let expected = f64::from(year) + 59.0 / 366.0;
            assert!((decimal_year(&dt) - expected).abs() < 1e-9);
        }
    }

    #[test]
    fn decimal_year_last_moment_of_leap_year_stays_inside() {
        let dt = Utc.with_ymd_and_hms(2_024, 12, 31, 23, 59, 59).unwrap();
        let computed = decimal_year(&dt);
        let expected = 2_024.0_f64 + (365.0 + 86_399.0 / 86_400.0) / 366.0;
        assert!((computed - expected).abs() < 1e-12);
        assert!(computed < 2_025.0);
    }

    #[test]
    fn decimal_year_propagates_subsecond_precision() {
        let base = Utc.with_ymd_and_hms(2_026, 5, 16, 12, 0, 0).unwrap();
        let with_nanos = base.with_nanosecond(500_000_000).unwrap();
        let delta = decimal_year(&with_nanos) - decimal_year(&base);
        assert!((delta - 0.5 / 86_400.0 / 365.0).abs() < 1e-12);
    }

    #[test]
    fn decimal_year_is_timezone_invariant() {
        let madrid = Tz::Europe__Madrid
            .with_ymd_and_hms(2_026, 5, 16, 18, 0, 0)
            .unwrap();
        let honolulu = madrid.with_timezone(&Tz::Pacific__Honolulu);
        let utc = madrid.with_timezone(&Utc);
        let m = decimal_year(&madrid);
        assert!((m - decimal_year(&honolulu)).abs() < f64::EPSILON);
        assert!((m - decimal_year(&utc)).abs() < f64::EPSILON);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn for_datetime_is_pure_composition() {
        let dt = Utc.with_ymd_and_hms(2_026, 5, 16, 12, 0, 0).unwrap();
        assert_eq!(
            approximate_delta_t_seconds_for_datetime(&dt),
            approximate_delta_t_seconds(decimal_year(&dt)),
        );
    }

    #[test]
    fn leap_year_predicate_covers_400_year_cycle() {
        assert!(is_gregorian_leap_year(2_000));
        assert!(is_gregorian_leap_year(2_024));
        assert!(!is_gregorian_leap_year(1_900));
        assert!(!is_gregorian_leap_year(2_023));
        assert!(is_gregorian_leap_year(-4));
        assert!(!is_gregorian_leap_year(-1));
    }
}
