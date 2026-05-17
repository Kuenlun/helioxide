// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Polynomial approximation of `ΔT = TT − UT1` over the five-millennium
//! window `-1999 ≤ year ≤ +3000`.
//!
//! NREL's Solar Position Algorithm takes ΔT as an input (consumed by
//! [`calculate_julian_ephemeris_day`] via [`SolarPosition::compute`] and
//! [`SolarDay::compute`]), but the quantity itself is *observed*: the
//! authoritative source is the IERS, which publishes the current value
//! month by month from the actual UT1 readings of the Earth rotation
//! service. For applications that cannot reach those bulletins, or that
//! need ΔT at past or future instants for which no observation exists,
//! the canonical fallback is the piecewise polynomial set assembled by
//! Espenak and Meeus for the *Five Millennium Canon of Solar Eclipses:
//! -1999 to +3000* (NASA/TP-2006-214141, 2006).
//!
//! [`approximate_delta_t_seconds`] implements that set: thirteen
//! polynomial segments cover `-500 ≤ y ≤ +2150` and are continued at
//! both tails through the long-term secular parabola of Morrison and
//! Stephenson (2004),
//!
//! ```text
//! ΔT = -20 + 32 · u²,    u = (y - 1820) / 100,
//! ```
//!
//! which encodes the constant deceleration of Earth's rotation imposed
//! by tidal friction. A linear correction `-0.5628 · (2150 − y)` blends
//! the parabola into the 2005-2050 trend across the `2050 < y ≤ 2150`
//! window so the modern era connects to the long-term tail without a
//! visible jump at the 2050 hand-off. The published reference for the
//! complete piecewise expression is the NASA Eclipse page
//! `https://eclipse.gsfc.nasa.gov/SEcat5/deltatpoly.html`.
//!
//! Inside the fitted window the residuals against measured ΔT stay
//! within a few seconds for the modern era and below `~30 s` across the
//! historical record. Outside it the parabola encodes only the secular
//! slowdown and should not be expected to track decadal fluctuations
//! that no proxy record can constrain. At the seconds-resolution that
//! SPA outputs need to land on (`sunrise/sunset` quantises to whole
//! seconds, `topocentric_zenith` to micro-degrees), even a several-second
//! ΔT error is invisible: the convenience entry
//! [`approximate_delta_t_seconds_for_datetime`] is therefore enough for
//! every consumer that does not feed in its own IERS-published value.
//!
//! [`SolarPosition::compute`]: crate::SolarPosition::compute
//! [`SolarDay::compute`]: crate::SolarDay::compute
//! [`calculate_julian_ephemeris_day`]: crate::julian::calculate_julian_ephemeris_day

use chrono::{DateTime, Datelike, TimeZone, Timelike};

/// Espenak-Meeus polynomial for `-500 ≤ y ≤ +500`. Coefficients are
/// ordered constant-first so Horner's method evaluates them in a single
/// reverse fold. Substitution variable: `u = y / 100`.
const EM_NEG500_TO_500_COEFFS: [f64; 7] = [
    10_583.6,
    -1_014.41,
    33.783_11,
    -5.952_053,
    -0.179_845_2,
    0.022_174_192,
    0.009_031_652_1,
];

/// Espenak-Meeus polynomial for `+500 < y ≤ +1600`. Substitution
/// variable: `u = (y − 1000) / 100`.
const EM_500_TO_1600_COEFFS: [f64; 7] = [
    1_574.2,
    -556.01,
    71.234_72,
    0.319_781,
    -0.850_346_3,
    -0.005_050_998,
    0.008_357_207_3,
];

/// Espenak-Meeus polynomial for `+1600 < y ≤ +1700`. Substitution
/// variable: `t = y − 1600`.
const EM_1600_TO_1700_COEFFS: [f64; 4] = [120.0, -0.980_8, -0.015_32, 1.0 / 7_129.0];

/// Espenak-Meeus polynomial for `+1700 < y ≤ +1800`. Substitution
/// variable: `t = y − 1700`.
const EM_1700_TO_1800_COEFFS: [f64; 5] = [
    8.83,
    0.160_3,
    -0.005_928_5,
    0.000_133_36,
    -1.0 / 1_174_000.0,
];

/// Espenak-Meeus polynomial for `+1800 < y ≤ +1860`. Substitution
/// variable: `t = y − 1800`.
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

/// Espenak-Meeus polynomial for `+1860 < y ≤ +1900`. Substitution
/// variable: `t = y − 1860`.
const EM_1860_TO_1900_COEFFS: [f64; 6] = [
    7.62,
    0.573_7,
    -0.251_754,
    0.016_806_68,
    -0.000_447_362_4,
    1.0 / 233_174.0,
];

/// Espenak-Meeus polynomial for `+1900 < y ≤ +1920`. Substitution
/// variable: `t = y − 1900`.
const EM_1900_TO_1920_COEFFS: [f64; 5] = [-2.79, 1.494_119, -0.059_893_9, 0.006_196_6, -0.000_197];

/// Espenak-Meeus polynomial for `+1920 < y ≤ +1941`. Substitution
/// variable: `t = y − 1920`.
const EM_1920_TO_1941_COEFFS: [f64; 4] = [21.20, 0.844_93, -0.076_100, 0.002_093_6];

/// Espenak-Meeus polynomial for `+1941 < y ≤ +1961`. Substitution
/// variable: `t = y − 1950`.
const EM_1941_TO_1961_COEFFS: [f64; 4] = [29.07, 0.407, -1.0 / 233.0, 1.0 / 2_547.0];

/// Espenak-Meeus polynomial for `+1961 < y ≤ +1986`. Substitution
/// variable: `t = y − 1975`.
const EM_1961_TO_1986_COEFFS: [f64; 4] = [45.45, 1.067, -1.0 / 260.0, -1.0 / 718.0];

/// Espenak-Meeus polynomial for `+1986 < y ≤ +2005`. Substitution
/// variable: `t = y − 2000`.
const EM_1986_TO_2005_COEFFS: [f64; 6] = [
    63.86,
    0.334_5,
    -0.060_374,
    0.001_727_5,
    0.000_651_814,
    0.000_023_735_99,
];

/// Espenak-Meeus polynomial for `+2005 < y ≤ +2050`. Substitution
/// variable: `t = y − 2000`. This three-term quadratic is the segment
/// that covers the present civil era and is the only polynomial most
/// modern consumers actually evaluate.
const EM_2005_TO_2050_COEFFS: [f64; 3] = [62.92, 0.322_17, 0.005_589];

/// Anchor year `1820` of the Morrison-Stephenson long-term parabola.
/// The choice of anchor is empirical: the parabola is the least-squares
/// fit to the pre-telescopic eclipse record under the assumption that
/// the secular trend of `ΔT` is dominated by tidal braking, and `1820`
/// is the epoch at which Stephenson and Morrison (2004) report the
/// residual ΔT against that fit goes through zero.
const PARABOLA_ANCHOR_YEAR: f64 = 1820.0;

/// Constant term `−20 s` of the Morrison-Stephenson parabola. Encodes
/// the offset between Stephenson and Morrison's recalibrated lunar
/// secular acceleration and the previous IAU value, evaluated at the
/// 1820 anchor.
const PARABOLA_CONSTANT_OFFSET_SECONDS: f64 = -20.0;

/// Curvature `32 s/century²` of the Morrison-Stephenson parabola. The
/// natural variable of the fit is `u = (y − 1820) / 100`, so `32 · u²`
/// recovers `ΔT` in seconds without further unit conversion.
const PARABOLA_QUADRATIC_COEFFICIENT_SECONDS_PER_CENTURY_SQUARED: f64 = 32.0;

/// Boundary year `2150` where the `2050-2150` transition reverts to the
/// bare Morrison-Stephenson parabola.
const TRANSITION_END_YEAR: f64 = 2150.0;

/// Linear slope of the transition correction across `2050 < y ≤ 2150`
/// (seconds of `ΔT` per year of approach to 2150). Reproduced verbatim
/// from the Five Millennium Canon: chosen so the transition joins the
/// 1986-2005 polynomial trend at the 2050 hand-off without a visible
/// jump.
const TRANSITION_CORRECTION_SLOPE: f64 = -0.562_8;

/// Seconds per UT day, used by [`decimal_year`] to add the sub-day
/// residue of the input instant to the day-of-year index.
const SECONDS_PER_DAY: f64 = 86_400.0;

/// Day-count denominators used by [`decimal_year`] for common and leap
/// Gregorian years.
const DAYS_IN_COMMON_YEAR: f64 = 365.0;
const DAYS_IN_LEAP_YEAR: f64 = 366.0;

/// Approximate `ΔT = TT − UT1` (seconds) for a given decimal year.
///
/// `decimal_year` is `y` in the Espenak-Meeus convention: the calendar
/// year plus the fraction of that year that has elapsed by the instant
/// of interest. The polynomial set is smooth on a year scale, so any
/// reasonable sub-year resolution (mid-month per the original paper,
/// day-of-year for finer resolution) maps to the same `ΔT` within a
/// fraction of a second.
///
/// Inside `-500 ≤ y ≤ +2150` the function dispatches to the Espenak-Meeus
/// piecewise polynomial published in the *Five Millennium Canon of Solar
/// Eclipses* (NASA/TP-2006-214141, 2006). Outside that window it returns
/// the long-term parabola `ΔT = -20 + 32 · ((y - 1820) / 100)²` of
/// Morrison and Stephenson (2004). The `2050 < y ≤ 2150` interval applies
/// the canon's `-0.5628 · (2150 - y)` linear blend onto the parabola so
/// the secular tail joins the 2005-2050 trend without a jump.
///
/// The dispatch chain is ordered modern-first, so the contemporary year
/// branches resolve in two or three comparisons.
///
/// # Examples
///
/// ```
/// use helioxide::delta_t::approximate_delta_t_seconds;
///
/// // Centre of the 1986-2005 polynomial: at t = 0 the result is the
/// // constant term, 63.86 s, matching the IERS value for 2000.0 to a
/// // few hundredths of a second.
/// let dt_2000 = approximate_delta_t_seconds(2000.0);
/// assert!((dt_2000 - 63.86).abs() < 1e-9);
///
/// // Lower endpoint of the 2005-2050 short-range quadratic: at
/// // `t = y - 2000 ≈ 5` the polynomial returns
/// // `62.92 + 0.32217·5 + 0.005589·25 = 64.670_575 s`.
/// let dt_2005_plus = approximate_delta_t_seconds(2_005.000_001);
/// assert!((dt_2005_plus - 64.670_575).abs() < 1e-4);
/// ```
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

/// Approximate `ΔT` (seconds) for the calendar instant `datetime`.
///
/// The wall-clock instant is collapsed to UTC and then mapped to a
/// decimal year via [`decimal_year`] before being fed to
/// [`approximate_delta_t_seconds`]. The timezone of the input only
/// affects the calendar-year boundary it crosses near January 1, which
/// is invisible to `ΔT` at the seconds-resolution the polynomial set
/// resolves.
///
/// `ΔT` is the offset between two *time scales* (TT and UT1), so the
/// signature deliberately takes a bare [`DateTime`] rather than a
/// [`SpaDateTime`]: the DUT1 correction carried by [`SpaDateTime`] is a
/// sub-second adjustment to UT1 alone and has no bearing on the
/// year-resolution polynomial input.
///
/// # Examples
///
/// ```
/// use chrono::{TimeZone, Utc};
/// use helioxide::delta_t::approximate_delta_t_seconds_for_datetime;
///
/// let noon = Utc.with_ymd_and_hms(2026, 5, 16, 12, 0, 0).single().unwrap();
/// let delta_t = approximate_delta_t_seconds_for_datetime(&noon);
/// // 2026 sits well inside the 2005-2050 quadratic. The polynomial
/// // overestimates the present IERS value by a few seconds because
/// // Earth's rotation has been steadier than projected.
/// assert!((60.0..80.0).contains(&delta_t));
/// ```
///
/// [`SpaDateTime`]: crate::SpaDateTime
#[must_use]
pub fn approximate_delta_t_seconds_for_datetime<Tz: TimeZone>(datetime: &DateTime<Tz>) -> f64 {
    approximate_delta_t_seconds(decimal_year(datetime))
}

/// Decimal year of `datetime` in the Espenak-Meeus convention.
///
/// Computed as `year + (day_of_year_zero_indexed + seconds_into_day /
/// 86_400) / days_in_year`, so January 1 at 00:00 UT maps exactly to
/// `year.0` and December 31 at 23:59:59 UT maps to just below
/// `year + 1`. The denominator honours the proleptic-Gregorian leap
/// rule (`year % 4 == 0 ∧ (year % 100 ≠ 0 ∨ year % 400 == 0)`), which
/// is the rule chrono itself applies across the entire datetime axis.
///
/// The wall-clock instant is collapsed to UTC first so that two
/// observers in different timezones at the same physical instant
/// produce the same decimal year, which keeps the `ΔT` estimate
/// timezone-invariant.
///
/// # Examples
///
/// ```
/// use chrono::{TimeZone, Utc};
/// use helioxide::delta_t::decimal_year;
///
/// let jan_1_2026 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).single().unwrap();
/// assert!((decimal_year(&jan_1_2026) - 2026.0).abs() < f64::EPSILON);
/// ```
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

/// Morrison-Stephenson long-term parabola.
#[inline]
fn long_term_parabola(decimal_year: f64) -> f64 {
    let u = (decimal_year - PARABOLA_ANCHOR_YEAR) / 100.0;
    PARABOLA_QUADRATIC_COEFFICIENT_SECONDS_PER_CENTURY_SQUARED
        .mul_add(u * u, PARABOLA_CONSTANT_OFFSET_SECONDS)
}

/// `2050 < y ≤ 2150` transition: the secular parabola plus a linear
/// correction tapering to zero at `y = 2150`. The correction lifts the
/// parabola down by `~56 s` at `y = 2050` so it lands flush with the
/// 2005-2050 polynomial trend, and decays to zero at `y = 2150` so the
/// hand-off to the bare parabola is continuous.
#[inline]
fn long_term_parabola_transition_to_2150(decimal_year: f64) -> f64 {
    let secular = long_term_parabola(decimal_year);
    TRANSITION_CORRECTION_SLOPE.mul_add(TRANSITION_END_YEAR - decimal_year, secular)
}

/// Evaluate a polynomial by Horner's method.
///
/// `coefficients` are ordered constant-first
/// (`[c₀, c₁, …, cₙ]` for `Σ cᵢ · xⁱ`), so the reverse fold collapses
/// to `((cₙ · x + cₙ₋₁) · x + … + c₁) · x + c₀`. Each step is fused
/// through `mul_add`, sharing one rounding between the multiplication
/// and the accumulation and keeping the dominant terms precise against
/// the much smaller higher-order corrections.
#[inline]
fn evaluate_horner(coefficients: &[f64], x: f64) -> f64 {
    coefficients
        .iter()
        .rev()
        .fold(0.0_f64, |acc, &c| acc.mul_add(x, c))
}

/// Proleptic-Gregorian leap-year predicate.
#[inline]
const fn is_gregorian_leap_year(year: i32) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{
        approximate_delta_t_seconds, approximate_delta_t_seconds_for_datetime, decimal_year,
        is_gregorian_leap_year,
    };
    use chrono::{TimeZone, Timelike, Utc};
    use chrono_tz::Tz;

    /// Anchor year of each Espenak-Meeus segment that makes the
    /// substitution variable vanish, paired with the segment's
    /// constant term. The dispatch convention `prev < y ≤ next` means a
    /// year on a segment's lower boundary resolves to the *previous*
    /// branch, so anchors at internal boundaries (`y = 1600`, `1700`,
    /// `1800`, `1860`, `1900`, `1920`) are nudged just past the
    /// boundary by `1e-7 yr`. The linear coefficient of every nudged
    /// segment is below `2 s/yr`, so the displaced anchor still
    /// collapses to the constant term within the `1e-3` test
    /// tolerance.
    ///
    /// The 2005-2050 quadratic cannot be probed at its substitution
    /// zero (`y = 2000` resolves to the 1986-2005 branch and `y > 2005`
    /// puts `t ≥ 5`), so its coefficients are pinned by
    /// [`short_range_quadratic_matches_hand_computed_values`] instead.
    /// The 2050-2150 transition, the `y > 2150` tail and the
    /// `y < -500` tail are not polynomial segments in the usual sense
    /// and are pinned by their own dedicated tests below.
    const SEGMENT_CONSTANT_ANCHORS: [(f64, f64); 11] = [
        (0.0, 10_583.6),          // -500 to +500, u = 0 ⇔ y = 0
        (1_000.0, 1_574.2),       // +500 to +1600, u = 0 ⇔ y = 1000
        (1_600.000_000_1, 120.0), // +1600 to +1700, t ≈ 0 just past lower boundary
        (1_700.000_000_1, 8.83),  // +1700 to +1800, t ≈ 0
        (1_800.000_000_1, 13.72), // +1800 to +1860, t ≈ 0
        (1_860.000_000_1, 7.62),  // +1860 to +1900, t ≈ 0
        (1_900.000_000_1, -2.79), // +1900 to +1920, t ≈ 0
        (1_920.000_000_1, 21.20), // +1920 to +1941, t ≈ 0
        (1_950.0, 29.07),         // +1941 to +1961, t = 0 ⇔ y = 1950
        (1_975.0, 45.45),         // +1961 to +1986, t = 0 ⇔ y = 1975
        (2_000.0, 63.86),         // +1986 to +2005, t = 0 ⇔ y = 2000
    ];

    /// Interior boundary years of the dispatch chain, used by the
    /// continuity test. The list covers every place where the chain
    /// switches branches: the twelve Espenak-Meeus segment boundaries
    /// advertised on the NASA page, plus the two parabola-to-polynomial
    /// hand-offs at `y = -500` and `y = 2150`. The test evaluates `ΔT`
    /// infinitesimally on both sides of every boundary and confirms the
    /// discontinuity stays below a few seconds.
    const INTERNAL_BOUNDARY_YEARS: [f64; 14] = [
        -500.0, 500.0, 1_600.0, 1_700.0, 1_800.0, 1_860.0, 1_900.0, 1_920.0, 1_941.0, 1_961.0,
        1_986.0, 2_005.0, 2_050.0, 2_150.0,
    ];

    /// Each Espenak-Meeus segment, evaluated at the year that maps to
    /// the zero of its substitution variable, must collapse to the
    /// published constant term of that segment. A typo on any single
    /// constant coefficient surfaces exactly in the corresponding entry
    /// while the rest of the table still passes. This is the strongest
    /// single sanity check on the per-segment coefficient table.
    ///
    /// The eleven anchors jointly exercise every polynomial branch of
    /// the modern-first dispatch chain from `y > 1986` down to
    /// `y >= -500` except the 2005-2050 quadratic (whose constant zero
    /// `y = 2000` resolves to the 1986-2005 branch, and is therefore
    /// pinned separately by
    /// [`short_range_quadratic_matches_hand_computed_values`]).
    #[test]
    fn every_segment_collapses_to_its_constant_at_the_substitution_zero() {
        for (year, expected) in SEGMENT_CONSTANT_ANCHORS {
            let value = approximate_delta_t_seconds(year);
            assert!(
                (value - expected).abs() < 1e-3,
                "ΔT mismatch at y = {year}: got {value}, expected {expected}",
            );
        }
    }

    /// Hand-computed reference values from the Espenak-Meeus 2005-2050
    /// short-range quadratic (the segment the solar-clock crate already
    /// relied on). The quadratic is the only segment most modern
    /// consumers ever evaluate, so the three probes pin it tightly
    /// against the documented coefficients independently of the segment
    /// table sweep.
    #[test]
    fn short_range_quadratic_matches_hand_computed_values() {
        // 2020: 62.92 + 0.32217·20 + 0.005589·400 = 71.5990
        assert!((approximate_delta_t_seconds(2_020.0) - 71.599_0).abs() < 1e-4);
        // 2026: 62.92 + 0.32217·26 + 0.005589·676 = 75.074_584
        assert!((approximate_delta_t_seconds(2_026.0) - 75.074_584).abs() < 1e-4);
        // 2050: 62.92 + 0.32217·50 + 0.005589·2500 = 93.0010
        assert!((approximate_delta_t_seconds(2_050.0) - 93.001_0).abs() < 1e-3);
    }

    /// The piecewise polynomial set is designed (by least-squares fit
    /// of each segment to the IERS-published ΔT series) so its segment
    /// boundaries are nearly continuous. Espenak and Meeus' published
    /// criterion is that the discontinuity at each interior boundary
    /// stays below a few seconds. This test pins that property at
    /// every boundary by evaluating ΔT `ε` away on each side and
    /// asserting the jump is below 2.0 s. A regression in any single
    /// segment polynomial that broke its endpoint match against the
    /// neighbouring segment would surface here.
    #[test]
    fn every_internal_boundary_is_continuous_within_two_seconds() {
        // `ε` of `1e-6` years sits well above `f64` ulp for the typical
        // year magnitude (~2000), so the two probes stay numerically
        // distinct after the dispatch chain selects different branches
        // for each.
        let epsilon = 1.0e-6;
        for boundary in INTERNAL_BOUNDARY_YEARS {
            let below = approximate_delta_t_seconds(boundary - epsilon);
            let above = approximate_delta_t_seconds(boundary + epsilon);
            let jump = (above - below).abs();
            assert!(
                jump < 2.0,
                "ΔT jump at y = {boundary} exceeds 2 s: below = {below}, above = {above}, \
                 jump = {jump}",
            );
        }
    }

    /// The `y > 2150` branch is the bare Morrison-Stephenson parabola.
    /// Hand-compute `ΔT(2500) = -20 + 32 · ((2500 - 1820) / 100)²
    /// = -20 + 32 · 46.24 = 1459.68 s` and pin the branch.
    #[test]
    fn long_term_parabola_branch_matches_morrison_stephenson() {
        let value = approximate_delta_t_seconds(2_500.0);
        assert!(
            (value - 1_459.68).abs() < 1e-9,
            "ΔT mismatch at y = 2500: got {value}, expected 1459.68",
        );
    }

    /// Symmetric pin for the `y < -500` branch, also the long-term
    /// parabola. Hand-compute `ΔT(-1500) = -20 + 32 · ((-1500 - 1820)
    /// / 100)² = -20 + 32 · 1102.24 = 35251.68 s`.
    #[test]
    fn long_term_parabola_branch_also_serves_the_far_past() {
        let value = approximate_delta_t_seconds(-1_500.0);
        assert!(
            (value - 35_251.68).abs() < 1e-9,
            "ΔT mismatch at y = -1500: got {value}, expected 35251.68",
        );
    }

    /// `2050 < y ≤ 2150` is the only segment that combines the secular
    /// parabola with a linear correction. Pin both endpoints and the
    /// midpoint to lock the transition formula in place.
    ///
    /// At `y = 2150`: `ΔT = -20 + 32·(3.30)² - 0.5628·0   = 328.48 s`.
    /// At `y = 2100`: `ΔT = -20 + 32·(2.80)² - 0.5628·50  = 202.74 s`.
    /// At `y = 2050`: `ΔT = -20 + 32·(2.30)² - 0.5628·100 =  93.00 s`
    ///   (the 2005-2050 polynomial gives `93.001` at the same year,
    ///   so the canon's chosen slope locks the two segments together
    ///   within `~0.01 s` at the hand-off).
    #[test]
    fn transition_segment_blends_parabola_and_linear_correction() {
        let at_end = approximate_delta_t_seconds(2_150.0);
        assert!(
            (at_end - 328.48).abs() < 1e-9,
            "transition endpoint at y = 2150: got {at_end}, expected 328.48",
        );

        let at_midpoint = approximate_delta_t_seconds(2_100.0);
        assert!(
            (at_midpoint - 202.74).abs() < 1e-9,
            "transition midpoint at y = 2100: got {at_midpoint}, expected 202.74",
        );

        // At the 2050 hand-off the transition formula and the
        // 2005-2050 polynomial agree to within ~0.01 s; the dispatch
        // picks the 2005-2050 polynomial at exactly y = 2050, so probe
        // just past it to land on the transition branch instead.
        let just_after_handoff = approximate_delta_t_seconds(2_050.000_001);
        let parabola_at_handoff =
            0.5628_f64.mul_add(-100.0, 32.0_f64.mul_add(2.30_f64.powi(2), -20.0));
        assert!(
            (just_after_handoff - parabola_at_handoff).abs() < 1e-3,
            "transition lower endpoint at y → 2050⁺: got {just_after_handoff}, \
             expected ≈ {parabola_at_handoff}",
        );
    }

    /// January 1 at 00:00 UT of any year must map to exactly `year.0`
    /// in decimal-year form (the leap-rule denominator cancels at the
    /// year boundary). A bug that started day-of-year from `1` instead
    /// of `0` would shift the result by `1 / days_in_year` and fail
    /// here. A bug that wrapped the day count modulo `365` instead of
    /// honouring the leap rule would fail symmetrically on the last
    /// instant of a leap year (see the `leap_year_last_moment` test).
    #[test]
    fn decimal_year_anchors_january_first_midnight_to_year_dot_zero() {
        for year in [1_900, 1_999, 2_000, 2_024, 2_026] {
            let dt = Utc.with_ymd_and_hms(year, 1, 1, 0, 0, 0).single().unwrap();
            let computed = decimal_year(&dt);
            assert!(
                (computed - f64::from(year)).abs() < f64::EPSILON,
                "decimal year at {year}-01-01 00:00 UT: got {computed}, expected {year}.0",
            );
        }
    }

    /// February 29 of a leap year exists (`day_of_year = 60`,
    /// 0-indexed) and the denominator `days_in_year` is `366`. A bug
    /// that picked the wrong leap rule (e.g. treating 2000 as
    /// non-leap, which the bare `year % 4 == 0` rule does correctly
    /// only by accident) would fail here on at least one of the
    /// century-year probes. Both `2000` (leap by the 400-year rule)
    /// and `2024` (leap by the 4-year rule) are exercised.
    #[test]
    fn decimal_year_recognises_leap_days() {
        for year in [2_000, 2_024] {
            let dt = Utc.with_ymd_and_hms(year, 2, 29, 0, 0, 0).single().unwrap();
            let computed = decimal_year(&dt);
            // Day-of-year (1-indexed) for Feb 29 is 60, so 0-indexed
            // is 59. With days_in_year = 366: 59 / 366 ≈ 0.161202.
            let expected = f64::from(year) + 59.0 / 366.0;
            assert!(
                (computed - expected).abs() < 1.0e-9,
                "decimal year at {year}-02-29: got {computed}, expected ≈ {expected}",
            );
        }
    }

    /// The last possible instant of a leap year must map to just below
    /// `year + 1`. This pins the denominator at `366` for leap years
    /// (a regression that hard-coded `365` would shift the result by
    /// `~1 / 365 ≈ 0.0027` here) and confirms that the seconds-into-day
    /// term reaches the day boundary properly.
    #[test]
    fn decimal_year_last_moment_of_leap_year_stays_within_the_year() {
        let dt = Utc
            .with_ymd_and_hms(2_024, 12, 31, 23, 59, 59)
            .single()
            .unwrap();
        let computed = decimal_year(&dt);
        // 0-indexed day-of-year for Dec 31 of a leap year is 365;
        // seconds_into_day = 86_399; days_in_year = 366.
        let expected = 2_024.0_f64 + (365.0 + 86_399.0 / 86_400.0) / 366.0;
        assert!(
            (computed - expected).abs() < 1.0e-12,
            "decimal year at 2024-12-31 23:59:59: got {computed}, expected ≈ {expected}",
        );
        assert!(
            computed < 2_025.0,
            "decimal year must stay strictly below the next year boundary",
        );
    }

    /// Sub-second precision must propagate from `nanosecond()` through
    /// the `seconds_into_day` accumulator. A `500_000_000 ns` shift
    /// (half a second) must show up as `0.5 / 86_400 / days_in_year`
    /// in the decimal-year result. Dropping the nanosecond term (a
    /// plausible regression if the formula were simplified to whole
    /// seconds) would zero this delta out and fail here. The
    /// `1e-12` tolerance sits a couple of orders of magnitude above
    /// the f64 ulp at the `~2026` year magnitude (`~2.3·10⁻¹³`),
    /// which is the noise floor on the subtraction of two same-year
    /// decimal-year values.
    #[test]
    fn decimal_year_propagates_subsecond_precision() {
        let base = Utc
            .with_ymd_and_hms(2_026, 5, 16, 12, 0, 0)
            .single()
            .unwrap();
        let with_nanos = base.with_nanosecond(500_000_000).unwrap();
        let delta = decimal_year(&with_nanos) - decimal_year(&base);
        let expected = 0.5 / 86_400.0 / 365.0;
        assert!(
            (delta - expected).abs() < 1.0e-12,
            "sub-second shift mismatch: got {delta}, expected {expected}",
        );
    }

    /// Two observers at the same physical instant must produce the
    /// same decimal year (and the same `ΔT`) regardless of their wall
    /// clocks. The pair below picks a Madrid afternoon that lands the
    /// day after in Honolulu, so a regression that read the calendar
    /// year off the local instant instead of the UT instant would
    /// produce different decimal years here.
    #[test]
    fn decimal_year_is_timezone_invariant_across_offsets() {
        let madrid = Tz::Europe__Madrid
            .with_ymd_and_hms(2_026, 5, 16, 18, 0, 0)
            .single()
            .unwrap();
        let honolulu = madrid.with_timezone(&Tz::Pacific__Honolulu);
        let utc = madrid.with_timezone(&Utc);
        let madrid_y = decimal_year(&madrid);
        let honolulu_y = decimal_year(&honolulu);
        let utc_y = decimal_year(&utc);
        assert!(
            (madrid_y - honolulu_y).abs() < f64::EPSILON && (madrid_y - utc_y).abs() < f64::EPSILON,
            "decimal year must not depend on timezone representation: \
             madrid = {madrid_y}, honolulu = {honolulu_y}, utc = {utc_y}",
        );
    }

    /// The convenience [`approximate_delta_t_seconds_for_datetime`]
    /// must compose [`decimal_year`] with [`approximate_delta_t_seconds`]
    /// without introducing any other transformation. Pin it bit-for-bit
    /// against the explicit composition.
    #[test]
    #[allow(clippy::float_cmp)]
    fn for_datetime_is_a_pure_composition_of_the_two_primitives() {
        let dt = Utc
            .with_ymd_and_hms(2_026, 5, 16, 12, 0, 0)
            .single()
            .unwrap();
        let via_for_datetime = approximate_delta_t_seconds_for_datetime(&dt);
        let via_composition = approximate_delta_t_seconds(decimal_year(&dt));
        assert_eq!(via_for_datetime, via_composition);
    }

    /// The Gregorian leap-rule predicate must distinguish the four
    /// corner cases of the 400-year cycle: `2000` (leap by `% 400`),
    /// `2024` (leap by `% 4`), `1900` (non-leap by `% 100`),
    /// `2023` (non-leap by `% 4`). Negative years exercise the
    /// proleptic extension chrono itself applies.
    #[test]
    fn leap_year_predicate_handles_all_four_corners_of_the_400_year_cycle() {
        assert!(is_gregorian_leap_year(2_000));
        assert!(is_gregorian_leap_year(2_024));
        assert!(!is_gregorian_leap_year(1_900));
        assert!(!is_gregorian_leap_year(2_023));
        // Proleptic extension: -4 is leap by the % 4 rule, -1 is not.
        assert!(is_gregorian_leap_year(-4));
        assert!(!is_gregorian_leap_year(-1));
    }
}
