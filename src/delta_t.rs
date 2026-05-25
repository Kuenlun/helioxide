// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! `ΔT = TT − UT1` (seconds), observed values with polynomial fallback.
//!
//! [`delta_t_seconds_for_datetime`] returns the published USNO monthly value
//! (linearly interpolated between the first of each month) when the date lies
//! inside the observed window, and otherwise falls back to the piecewise
//! polynomials of Espenak and Meeus from the *Five Millennium Canon of Solar
//! Eclipses: -1999 to +3000* (NASA/TP-2006-214141), with the long-term
//! parabola of Morrison and Stephenson (2004) outside the fitted window.
//!
//! References:
//! - <https://maia.usno.navy.mil/ser7/deltat.data>
//! - <https://eclipse.gsfc.nasa.gov/SEcat5/deltatpoly.html>

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

/// Best available `ΔT` (seconds) for the UTC instant of `datetime`.
///
/// Returns the observed USNO value (linearly interpolated between monthly
/// samples) when inside the published window, otherwise the polynomial
/// approximation.
#[must_use]
pub fn delta_t_seconds_for_datetime<Tz: TimeZone>(datetime: &DateTime<Tz>) -> f64 {
    observed_delta_t_seconds_for_datetime(datetime)
        .unwrap_or_else(|| approximate_delta_t_seconds_for_datetime(datetime))
}

/// Observed `ΔT` (seconds) at the UTC instant of `datetime`, linearly
/// interpolated between adjacent monthly samples of the USNO table. Returns
/// [`None`] when the date lies outside the published window.
#[must_use]
#[allow(
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "month fits in 1..=12 and the post-check `idx < 0` rules out sign loss"
)]
pub fn observed_delta_t_seconds_for_datetime<Tz: TimeZone>(datetime: &DateTime<Tz>) -> Option<f64> {
    let naive = datetime.naive_utc();
    let year = naive.year();
    let month = naive.month();

    let idx = (year - OBSERVED_BASE_YEAR) * 12 + (month as i32 - OBSERVED_BASE_MONTH);
    if idx < 0 || idx as usize + 1 >= OBSERVED_DELTA_T.len() {
        return None;
    }
    let idx = idx as usize;

    let seconds_into_day =
        f64::from(naive.nanosecond()).mul_add(1.0e-9, f64::from(naive.num_seconds_from_midnight()));
    let day_of_month = f64::from(naive.day() - 1);
    let fraction =
        (day_of_month + seconds_into_day / SECONDS_PER_DAY) / f64::from(days_in_month(year, month));

    let a = OBSERVED_DELTA_T[idx];
    let b = OBSERVED_DELTA_T[idx + 1];
    Some((b - a).mul_add(fraction, a))
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

const DAYS_PER_MONTH: [u32; 12] = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

#[inline]
const fn days_in_month(year: i32, month: u32) -> u32 {
    if month == 2 && is_gregorian_leap_year(year) {
        29
    } else {
        DAYS_PER_MONTH[(month - 1) as usize]
    }
}

/// Calendar anchor of the first observed `ΔT` sample.
const OBSERVED_BASE_YEAR: i32 = 1973;
const OBSERVED_BASE_MONTH: i32 = 2;

/// Observed monthly `ΔT` values (seconds), one entry per month starting at
/// (`OBSERVED_BASE_YEAR`, `OBSERVED_BASE_MONTH`), each sampled at 00:00 UTC on
/// the first of the month. Snapshot of <https://maia.usno.navy.mil/ser7/deltat.data>
/// covering Feb 1973 through Apr 2026.
#[rustfmt::skip]
const OBSERVED_DELTA_T: &[f64] = &[
    // 1973
    43.4724, 43.5648, 43.6737, 43.7782, 43.8763, 43.9562, 44.0315, 44.1132, 44.1982, 44.2952,
    44.3936,
    // 1974
    44.4841, 44.5646, 44.6425, 44.7386, 44.8370, 44.9302, 44.9986, 45.0584, 45.1284, 45.2064,
    45.2980, 45.3897,
    // 1975
    45.4761, 45.5632, 45.6450, 45.7375, 45.8284, 45.9133, 45.9820, 46.0407, 46.1067, 46.1825,
    46.2789, 46.3713,
    // 1976
    46.4567, 46.5445, 46.6311, 46.7302, 46.8284, 46.9247, 46.9970, 47.0709, 47.1451, 47.2362,
    47.3413, 47.4319,
    // 1977
    47.5214, 47.6049, 47.6837, 47.7781, 47.8771, 47.9687, 48.0348, 48.0942, 48.1608, 48.2460,
    48.3439, 48.4355,
    // 1978
    48.5344, 48.6325, 48.7294, 48.8365, 48.9353, 49.0319, 49.1013, 49.1591, 49.2286, 49.3070,
    49.4018, 49.4945,
    // 1979
    49.5861, 49.6805, 49.7602, 49.8556, 49.9489, 50.0347, 50.1019, 50.1622, 50.2260, 50.2968,
    50.3831, 50.4599,
    // 1980
    50.5387, 50.6160, 50.6866, 50.7658, 50.8454, 50.9187, 50.9761, 51.0278, 51.0843, 51.1538,
    51.2319, 51.3063,
    // 1981
    51.3808, 51.4526, 51.5160, 51.5985, 51.6809, 51.7573, 51.8133, 51.8532, 51.9014, 51.9603,
    52.0328, 52.0985,
    // 1982
    52.1668, 52.2316, 52.2938, 52.3680, 52.4465, 52.5180, 52.5751, 52.6178, 52.6668, 52.7340,
    52.8056, 52.8792,
    // 1983
    52.9565, 53.0445, 53.1268, 53.2197, 53.3024, 53.3747, 53.4335, 53.4778, 53.5300, 53.5845,
    53.6523, 53.7256,
    // 1984
    53.7882, 53.8367, 53.8830, 53.9443, 54.0042, 54.0536, 54.0856, 54.1084, 54.1463, 54.1914,
    54.2452, 54.2958,
    // 1985
    54.3427, 54.3911, 54.4320, 54.4898, 54.5456, 54.5977, 54.6355, 54.6532, 54.6776, 54.7174,
    54.7741, 54.8253,
    // 1986
    54.8713, 54.9161, 54.9581, 54.9997, 55.0476, 55.0912, 55.1132, 55.1328, 55.1532, 55.1898,
    55.2416, 55.2838,
    // 1987
    55.3222, 55.3613, 55.4063, 55.4629, 55.5111, 55.5524, 55.5812, 55.6004, 55.6262, 55.6656,
    55.7168, 55.7698,
    // 1988
    55.8197, 55.8615, 55.9130, 55.9663, 56.0220, 56.0700, 56.0939, 56.1105, 56.1314, 56.1611,
    56.2068, 56.2583,
    // 1989
    56.3000, 56.3399, 56.3790, 56.4283, 56.4804, 56.5352, 56.5697, 56.5983, 56.6328, 56.6739,
    56.7332, 56.7972,
    // 1990
    56.8553, 56.9111, 56.9755, 57.0471, 57.1136, 57.1738, 57.2226, 57.2597, 57.3073, 57.3643,
    57.4334, 57.5016,
    // 1991
    57.5653, 57.6333, 57.6973, 57.7711, 57.8407, 57.9058, 57.9576, 57.9975, 58.0426, 58.1043,
    58.1679, 58.2389,
    // 1992
    58.3092, 58.3833, 58.4537, 58.5401, 58.6228, 58.6917, 58.7410, 58.7836, 58.8406, 58.8986,
    58.9714, 59.0438,
    // 1993
    59.1218, 59.2003, 59.2747, 59.3574, 59.4434, 59.5242, 59.5850, 59.6343, 59.6928, 59.7588,
    59.8386, 59.9111,
    // 1994
    59.9845, 60.0564, 60.1231, 60.2042, 60.2804, 60.3530, 60.4012, 60.4440, 60.4900, 60.5578,
    60.6324, 60.7059,
    // 1995
    60.7853, 60.8664, 60.9387, 61.0277, 61.1103, 61.1870, 61.2454, 61.2881, 61.3378, 61.4036,
    61.4760, 61.5525,
    // 1996
    61.6287, 61.6846, 61.7433, 61.8132, 61.8823, 61.9497, 61.9969, 62.0343, 62.0714, 62.1202,
    62.1810, 62.2382,
    // 1997
    62.2950, 62.3506, 62.3995, 62.4754, 62.5463, 62.6136, 62.6571, 62.6942, 62.7383, 62.7926,
    62.8567, 62.9146,
    // 1998
    62.9659, 63.0217, 63.0807, 63.1462, 63.2053, 63.2599, 63.2844, 63.2961, 63.3126, 63.3422,
    63.3871, 63.4339,
    // 1999
    63.4673, 63.4979, 63.5319, 63.5679, 63.6104, 63.6444, 63.6642, 63.6739, 63.6926, 63.7147,
    63.7518, 63.7927,
    // 2000
    63.8285, 63.8557, 63.8804, 63.9075, 63.9393, 63.9691, 63.9799, 63.9833, 63.9938, 64.0093,
    64.0400, 64.0670,
    // 2001
    64.0908, 64.1068, 64.1282, 64.1584, 64.1833, 64.2094, 64.2117, 64.2073, 64.2116, 64.2223,
    64.2500, 64.2761,
    // 2002
    64.2998, 64.3192, 64.3450, 64.3735, 64.3943, 64.4151, 64.4132, 64.4118, 64.4097, 64.4168,
    64.4329, 64.4511,
    // 2003
    64.4734, 64.4893, 64.5053, 64.5269, 64.5471, 64.5597, 64.5512, 64.5371, 64.5359, 64.5415,
    64.5544, 64.5654,
    // 2004
    64.5736, 64.5891, 64.6015, 64.6176, 64.6374, 64.6549, 64.6530, 64.6379, 64.6372, 64.6400,
    64.6543, 64.6723,
    // 2005
    64.6876, 64.7052, 64.7313, 64.7575, 64.7811, 64.8001, 64.7995, 64.7876, 64.7831, 64.7921,
    64.8096, 64.8311,
    // 2006
    64.8452, 64.8597, 64.8850, 64.9175, 64.9480, 64.9794, 64.9895, 65.0028, 65.0138, 65.0371,
    65.0773, 65.1122,
    // 2007
    65.1464, 65.1833, 65.2145, 65.2494, 65.2921, 65.3279, 65.3413, 65.3452, 65.3496, 65.3711,
    65.3972, 65.4296,
    // 2008
    65.4573, 65.4868, 65.5152, 65.5450, 65.5781, 65.6127, 65.6288, 65.6370, 65.6493, 65.6760,
    65.7097, 65.7461,
    // 2009
    65.7768, 65.8025, 65.8237, 65.8595, 65.8973, 65.9323, 65.9509, 65.9534, 65.9628, 65.9839,
    66.0147, 66.0420,
    // 2010
    66.0699, 66.0961, 66.1310, 66.1683, 66.2072, 66.2356, 66.2409, 66.2335, 66.2349, 66.2441,
    66.2751, 66.3054,
    // 2011
    66.3246, 66.3406, 66.3624, 66.3957, 66.4289, 66.4619, 66.4749, 66.4751, 66.4829, 66.5056,
    66.5383, 66.5706,
    // 2012
    66.6030, 66.6340, 66.6569, 66.6925, 66.7289, 66.7579, 66.7708, 66.7740, 66.7846, 66.8103,
    66.8400, 66.8779,
    // 2013
    66.9069, 66.9443, 66.9763, 67.0258, 67.0716, 67.1100, 67.1266, 67.1331, 67.1458, 67.1717,
    67.2091, 67.2460,
    // 2014
    67.2810, 67.3136, 67.3457, 67.3890, 67.4318, 67.4666, 67.4858, 67.4989, 67.5111, 67.5353,
    67.5711, 67.6070,
    // 2015
    67.6439, 67.6765, 67.7117, 67.7591, 67.8012, 67.8402, 67.8606, 67.8822, 67.9120, 67.9546,
    68.0055, 68.0514,
    // 2016
    68.1024, 68.1577, 68.2044, 68.2665, 68.3188, 68.3704, 68.3964, 68.4094, 68.4305, 68.4630,
    68.5078, 68.5537,
    // 2017
    68.5927, 68.6298, 68.6671, 68.7135, 68.7623, 68.8033, 68.8245, 68.8373, 68.8477, 68.8689,
    68.9006, 68.9355,
    // 2018
    68.9676, 68.9875, 69.0176, 69.0499, 69.0823, 69.1070, 69.1134, 69.1142, 69.1207, 69.1356,
    69.1646, 69.1964,
    // 2019
    69.2202, 69.2452, 69.2733, 69.3032, 69.3326, 69.3541, 69.3582, 69.3442, 69.3376, 69.3377,
    69.3432, 69.3540,
    // 2020
    69.3612, 69.3752, 69.3890, 69.4092, 69.4265, 69.4386, 69.4241, 69.3921, 69.3693, 69.3575,
    69.3593, 69.3630,
    // 2021
    69.3594, 69.3510, 69.3538, 69.3582, 69.3673, 69.3679, 69.3514, 69.3273, 69.3033, 69.2892,
    69.2881, 69.2908,
    // 2022
    69.2945, 69.2914, 69.2861, 69.2835, 69.2816, 69.2799, 69.2527, 69.2213, 69.1975, 69.1891,
    69.1942, 69.2036,
    // 2023
    69.2039, 69.1986, 69.1993, 69.2084, 69.2183, 69.2300, 69.2201, 69.1988, 69.1814, 69.1723,
    69.1727, 69.1724,
    // 2024
    69.1752, 69.1797, 69.1874, 69.1983, 69.2018, 69.2044, 69.1879, 69.1588, 69.1322, 69.1250,
    69.1304, 69.1345,
    // 2025
    69.1377, 69.1366, 69.1384, 69.1471, 69.1542, 69.1550, 69.1406, 69.1219, 69.0994, 69.0909,
    69.0909, 69.1042,
    // 2026
    69.1099, 69.1133, 69.1168, 69.1330,
];

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
    fn observed_lookup_matches_table_anchor_on_first_of_month() {
        // 2020-01-01 00:00 UTC → first cell of the 2020 row.
        let dt = Utc.with_ymd_and_hms(2_020, 1, 1, 0, 0, 0).unwrap();
        let v = observed_delta_t_seconds_for_datetime(&dt).unwrap();
        assert!((v - 69.3612).abs() < 1e-12);
    }

    #[test]
    fn observed_lookup_interpolates_linearly_between_months() {
        // Halfway through April 2021 (30 days): expect mean of Apr 1 and May 1.
        let dt = Utc.with_ymd_and_hms(2_021, 4, 16, 0, 0, 0).unwrap();
        let expected = f64::midpoint(69.3582, 69.3673);
        let v = observed_delta_t_seconds_for_datetime(&dt).unwrap();
        assert!((v - expected).abs() < 1e-9);
    }

    #[test]
    fn observed_lookup_honours_february_leap_day_count() {
        // 2020 is a leap year: Feb has 29 days, midpoint sits at Feb 15 12:00 UTC.
        let dt = Utc.with_ymd_and_hms(2_020, 2, 15, 12, 0, 0).unwrap();
        let expected = f64::midpoint(69.3752, 69.3890);
        let v = observed_delta_t_seconds_for_datetime(&dt).unwrap();
        assert!((v - expected).abs() < 1e-9);
    }

    #[test]
    fn observed_lookup_returns_none_before_first_sample() {
        let before = Utc.with_ymd_and_hms(1_973, 1, 31, 23, 59, 59).unwrap();
        assert!(observed_delta_t_seconds_for_datetime(&before).is_none());
        let way_before = Utc.with_ymd_and_hms(1_900, 6, 1, 0, 0, 0).unwrap();
        assert!(observed_delta_t_seconds_for_datetime(&way_before).is_none());
    }

    #[test]
    fn observed_lookup_returns_none_at_and_after_last_sample() {
        // The published window ends with the Apr 2026 sample; from there on we
        // need the next month's value to interpolate, so the lookup fails.
        let last = Utc.with_ymd_and_hms(2_026, 4, 1, 0, 0, 0).unwrap();
        assert!(observed_delta_t_seconds_for_datetime(&last).is_none());
        let after = Utc.with_ymd_and_hms(2_030, 1, 1, 0, 0, 0).unwrap();
        assert!(observed_delta_t_seconds_for_datetime(&after).is_none());
    }

    #[test]
    fn observed_lookup_is_timezone_invariant() {
        let madrid = chrono_tz::Tz::Europe__Madrid
            .with_ymd_and_hms(2_024, 7, 1, 2, 0, 0)
            .unwrap();
        let utc = madrid.with_timezone(&Utc);
        let m = observed_delta_t_seconds_for_datetime(&madrid).unwrap();
        let u = observed_delta_t_seconds_for_datetime(&utc).unwrap();
        assert!((m - u).abs() < 1e-12);
    }

    #[test]
    fn unified_delta_t_uses_observed_value_when_available() {
        // 2024-07-01 00:00 UTC: observed table gives 69.1879 exactly. The
        // Espenak-Meeus 2005-2050 quadratic predicts ~69.96 s at that date,
        // so the gap is large enough to verify the lookup is wired in.
        let dt = Utc.with_ymd_and_hms(2_024, 7, 1, 0, 0, 0).unwrap();
        let observed = delta_t_seconds_for_datetime(&dt);
        assert!((observed - 69.1879).abs() < 1e-12);
        assert!((observed - approximate_delta_t_seconds_for_datetime(&dt)).abs() > 0.5);
    }

    #[test]
    fn unified_delta_t_falls_back_to_approximation_outside_window() {
        let pre = Utc.with_ymd_and_hms(1_900, 6, 1, 0, 0, 0).unwrap();
        let post = Utc.with_ymd_and_hms(2_030, 1, 1, 0, 0, 0).unwrap();
        for dt in [pre, post] {
            #[allow(clippy::float_cmp)]
            {
                assert_eq!(
                    delta_t_seconds_for_datetime(&dt),
                    approximate_delta_t_seconds_for_datetime(&dt),
                );
            }
        }
    }

    #[test]
    fn days_in_month_handles_february_leap_rule() {
        assert_eq!(days_in_month(2_020, 2), 29);
        assert_eq!(days_in_month(2_021, 2), 28);
        assert_eq!(days_in_month(1_900, 2), 28);
        assert_eq!(days_in_month(2_000, 2), 29);
        assert_eq!(days_in_month(2_023, 1), 31);
        assert_eq!(days_in_month(2_023, 4), 30);
        assert_eq!(days_in_month(2_023, 12), 31);
    }

    #[test]
    fn observed_table_is_well_formed() {
        // 1973-02 .. 2026-04 inclusive = 11 + 52*12 + 4 = 639 monthly entries.
        assert_eq!(OBSERVED_DELTA_T.len(), 639);
        assert_eq!(OBSERVED_BASE_YEAR, 1_973);
        assert_eq!(OBSERVED_BASE_MONTH, 2);
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
