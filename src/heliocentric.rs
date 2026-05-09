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

//! Earth heliocentric longitude (`L`), latitude (`B`) and radius vector (`R`).
//!
//! Implements section 3.2 of NREL/TP-560-34302. Each quantity is the value of
//! a truncated polynomial in the Julian Ephemeris Millennium (JME) whose
//! coefficients are themselves Σ A·cos(B + C·JME) over the rows published in
//! Table A4.2 of the same report.

use crate::helper::limit_degrees;
use tables::{B_TERMS, L_TERMS, R_TERMS};

/// Earth heliocentric longitude `L` (degrees), wrapped into `[0°, 360°)`.
///
/// Refer to section 3.2, steps 3.2.1 through 3.2.6.
///
/// # Examples
///
/// ```
/// use helioxide::heliocentric::earth_heliocentric_longitude;
/// use helioxide::julian::{
///     calculate_julian_ephemeris_century, calculate_julian_ephemeris_day,
///     calculate_julian_ephemeris_millennium,
/// };
///
/// // Table A5.1 reference: 17 October 2003, 12:30:30 LST, ΔT = 67 s.
/// let jde = calculate_julian_ephemeris_day(2_452_930.312_847, 67.0);
/// let jce = calculate_julian_ephemeris_century(jde);
/// let jme = calculate_julian_ephemeris_millennium(jce);
///
/// let l = earth_heliocentric_longitude(jme);
/// assert!((l - 24.018_261_691_7).abs() < 1e-6);
/// ```
#[must_use]
pub fn earth_heliocentric_longitude(jme: f64) -> f64 {
    limit_degrees(evaluate_periodic_series(&L_TERMS, jme).to_degrees())
}

/// Earth heliocentric latitude `B` (degrees, signed, not range-limited).
///
/// Refer to section 3.2, step 3.2.7.
///
/// # Examples
///
/// ```
/// use helioxide::heliocentric::earth_heliocentric_latitude;
/// use helioxide::julian::{
///     calculate_julian_ephemeris_century, calculate_julian_ephemeris_day,
///     calculate_julian_ephemeris_millennium,
/// };
///
/// let jde = calculate_julian_ephemeris_day(2_452_930.312_847, 67.0);
/// let jce = calculate_julian_ephemeris_century(jde);
/// let jme = calculate_julian_ephemeris_millennium(jce);
///
/// let b = earth_heliocentric_latitude(jme);
/// assert!((b - -0.000_101_121_9).abs() < 1e-9);
/// ```
#[must_use]
pub fn earth_heliocentric_latitude(jme: f64) -> f64 {
    evaluate_periodic_series(&B_TERMS, jme).to_degrees()
}

/// Earth radius vector `R` (astronomical units).
///
/// Refer to section 3.2, step 3.2.8.
///
/// # Examples
///
/// ```
/// use helioxide::heliocentric::earth_radius_vector;
/// use helioxide::julian::{
///     calculate_julian_ephemeris_century, calculate_julian_ephemeris_day,
///     calculate_julian_ephemeris_millennium,
/// };
///
/// let jde = calculate_julian_ephemeris_day(2_452_930.312_847, 67.0);
/// let jce = calculate_julian_ephemeris_century(jde);
/// let jme = calculate_julian_ephemeris_millennium(jce);
///
/// let r = earth_radius_vector(jme);
/// assert!((r - 0.996_542_297_4).abs() < 1e-9);
/// ```
#[must_use]
pub fn earth_radius_vector(jme: f64) -> f64 {
    evaluate_periodic_series(&R_TERMS, jme)
}

/// Σ Aᵢ·cos(Bᵢ + Cᵢ·JME) over the rows of one A4.2 sub-block (equation 9).
///
/// Returns radians for `L_TERMS`/`B_TERMS` and astronomical units for
/// `R_TERMS`, before the `1 / 10⁸` scaling applied in [`evaluate_periodic_series`].
#[inline]
#[must_use]
fn periodic_subseries_sum(rows: &[[f64; 3]], jme: f64) -> f64 {
    rows.iter()
        .map(|&[a, b, c]| a * c.mul_add(jme, b).cos())
        .sum()
}

/// Evaluates `(S₀ + S₁·JME + … + Sₙ·JMEⁿ) / 10⁸` (equations 10 and 11)
/// using Horner's method, with each `Sₖ` produced by [`periodic_subseries_sum`].
#[inline]
#[must_use]
fn evaluate_periodic_series(subseries: &[&[[f64; 3]]], jme: f64) -> f64 {
    subseries.iter().rev().fold(0.0_f64, |acc, rows| {
        acc.mul_add(jme, periodic_subseries_sum(rows, jme))
    }) / 1.0e8
}

// The constants below mirror Table A4.2 verbatim. Clippy's stylistic checks
// on numeric literals are silenced here because rounding or substituting
// would alter the agreement with the published reference values.
#[allow(
    clippy::unreadable_literal,
    clippy::approx_constant,
    clippy::excessive_precision
)]
mod tables {
    const L0: &[[f64; 3]] = &[
        [175347046.0, 0.0, 0.0],
        [3341656.0, 4.6692568, 6283.07585],
        [34894.0, 4.6261, 12566.1517],
        [3497.0, 2.7441, 5753.3849],
        [3418.0, 2.8289, 3.5231],
        [3136.0, 3.6277, 77713.7715],
        [2676.0, 4.4181, 7860.4194],
        [2343.0, 6.1352, 3930.2097],
        [1324.0, 0.7425, 11506.7698],
        [1273.0, 2.0371, 529.691],
        [1199.0, 1.1096, 1577.3435],
        [990.0, 5.233, 5884.927],
        [902.0, 2.045, 26.298],
        [857.0, 3.508, 398.149],
        [780.0, 1.179, 5223.694],
        [753.0, 2.533, 5507.553],
        [505.0, 4.583, 18849.228],
        [492.0, 4.205, 775.523],
        [357.0, 2.92, 0.067],
        [317.0, 5.849, 11790.629],
        [284.0, 1.899, 796.298],
        [271.0, 0.315, 10977.079],
        [243.0, 0.345, 5486.778],
        [206.0, 4.806, 2544.314],
        [205.0, 1.869, 5573.143],
        [202.0, 2.458, 6069.777],
        [156.0, 0.833, 213.299],
        [132.0, 3.411, 2942.463],
        [126.0, 1.083, 20.775],
        [115.0, 0.645, 0.98],
        [103.0, 0.636, 4694.003],
        [102.0, 0.976, 15720.839],
        [102.0, 4.267, 7.114],
        [99.0, 6.21, 2146.17],
        [98.0, 0.68, 155.42],
        [86.0, 5.98, 161000.69],
        [85.0, 1.3, 6275.96],
        [85.0, 3.67, 71430.7],
        [80.0, 1.81, 17260.15],
        [79.0, 3.04, 12036.46],
        [75.0, 1.76, 5088.63],
        [74.0, 3.5, 3154.69],
        [74.0, 4.68, 801.82],
        [70.0, 0.83, 9437.76],
        [62.0, 3.98, 8827.39],
        [61.0, 1.82, 7084.9],
        [57.0, 2.78, 6286.6],
        [56.0, 4.39, 14143.5],
        [56.0, 3.47, 6279.55],
        [52.0, 0.19, 12139.55],
        [52.0, 1.33, 1748.02],
        [51.0, 0.28, 5856.48],
        [49.0, 0.49, 1194.45],
        [41.0, 5.37, 8429.24],
        [41.0, 2.4, 19651.05],
        [39.0, 6.17, 10447.39],
        [37.0, 6.04, 10213.29],
        [37.0, 2.57, 1059.38],
        [36.0, 1.71, 2352.87],
        [36.0, 1.78, 6812.77],
        [33.0, 0.59, 17789.85],
        [30.0, 0.44, 83996.85],
        [30.0, 2.74, 1349.87],
        [25.0, 3.16, 4690.48],
    ];

    const L1: &[[f64; 3]] = &[
        [628331966747.0, 0.0, 0.0],
        [206059.0, 2.678235, 6283.07585],
        [4303.0, 2.6351, 12566.1517],
        [425.0, 1.59, 3.523],
        [119.0, 5.796, 26.298],
        [109.0, 2.966, 1577.344],
        [93.0, 2.59, 18849.23],
        [72.0, 1.14, 529.69],
        [68.0, 1.87, 398.15],
        [67.0, 4.41, 5507.55],
        [59.0, 2.89, 5223.69],
        [56.0, 2.17, 155.42],
        [45.0, 0.4, 796.3],
        [36.0, 0.47, 775.52],
        [29.0, 2.65, 7.11],
        [21.0, 5.34, 0.98],
        [19.0, 1.85, 5486.78],
        [19.0, 4.97, 213.3],
        [17.0, 2.99, 6275.96],
        [16.0, 0.03, 2544.31],
        [16.0, 1.43, 2146.17],
        [15.0, 1.21, 10977.08],
        [12.0, 2.83, 1748.02],
        [12.0, 3.26, 5088.63],
        [12.0, 5.27, 1194.45],
        [12.0, 2.08, 4694.0],
        [11.0, 0.77, 553.57],
        [10.0, 1.3, 6286.6],
        [10.0, 4.24, 1349.87],
        [9.0, 2.7, 242.73],
        [9.0, 5.64, 951.72],
        [8.0, 5.3, 2352.87],
        [6.0, 2.65, 9437.76],
        [6.0, 4.67, 4690.48],
    ];

    const L2: &[[f64; 3]] = &[
        [52919.0, 0.0, 0.0],
        [8720.0, 1.0721, 6283.0758],
        [309.0, 0.867, 12566.152],
        [27.0, 0.05, 3.52],
        [16.0, 5.19, 26.3],
        [16.0, 3.68, 155.42],
        [10.0, 0.76, 18849.23],
        [9.0, 2.06, 77713.77],
        [7.0, 0.83, 775.52],
        [5.0, 4.66, 1577.34],
        [4.0, 1.03, 7.11],
        [4.0, 3.44, 5573.14],
        [3.0, 5.14, 796.3],
        [3.0, 6.05, 5507.55],
        [3.0, 1.19, 242.73],
        [3.0, 6.12, 529.69],
        [3.0, 0.31, 398.15],
        [3.0, 2.28, 553.57],
        [2.0, 4.38, 5223.69],
        [2.0, 3.75, 0.98],
    ];

    const L3: &[[f64; 3]] = &[
        [289.0, 5.844, 6283.076],
        [35.0, 0.0, 0.0],
        [17.0, 5.49, 12566.15],
        [3.0, 5.2, 155.42],
        [1.0, 4.72, 3.52],
        [1.0, 5.3, 18849.23],
        [1.0, 5.97, 242.73],
    ];

    const L4: &[[f64; 3]] = &[
        [114.0, 3.142, 0.0],
        [8.0, 4.13, 6283.08],
        [1.0, 3.84, 12566.15],
    ];

    const L5: &[[f64; 3]] = &[[1.0, 3.14, 0.0]];

    pub(super) const L_TERMS: [&[[f64; 3]]; 6] = [L0, L1, L2, L3, L4, L5];

    const B0: &[[f64; 3]] = &[
        [280.0, 3.199, 84334.662],
        [102.0, 5.422, 5507.553],
        [80.0, 3.88, 5223.69],
        [44.0, 3.7, 2352.87],
        [32.0, 4.0, 1577.34],
    ];

    const B1: &[[f64; 3]] = &[[9.0, 3.9, 5507.55], [6.0, 1.73, 5223.69]];

    pub(super) const B_TERMS: [&[[f64; 3]]; 2] = [B0, B1];

    const R0: &[[f64; 3]] = &[
        [100013989.0, 0.0, 0.0],
        [1670700.0, 3.0984635, 6283.07585],
        [13956.0, 3.05525, 12566.1517],
        [3084.0, 5.1985, 77713.7715],
        [1628.0, 1.1739, 5753.3849],
        [1576.0, 2.8469, 7860.4194],
        [925.0, 5.453, 11506.77],
        [542.0, 4.564, 3930.21],
        [472.0, 3.661, 5884.927],
        [346.0, 0.964, 5507.553],
        [329.0, 5.9, 5223.694],
        [307.0, 0.299, 5573.143],
        [243.0, 4.273, 11790.629],
        [212.0, 5.847, 1577.344],
        [186.0, 5.022, 10977.079],
        [175.0, 3.012, 18849.228],
        [110.0, 5.055, 5486.778],
        [98.0, 0.89, 6069.78],
        [86.0, 5.69, 15720.84],
        [86.0, 1.27, 161000.69],
        [65.0, 0.27, 17260.15],
        [63.0, 0.92, 529.69],
        [57.0, 2.01, 83996.85],
        [56.0, 5.24, 71430.7],
        [49.0, 3.25, 2544.31],
        [47.0, 2.58, 775.52],
        [45.0, 5.54, 9437.76],
        [43.0, 6.01, 6275.96],
        [39.0, 5.36, 4694.0],
        [38.0, 2.39, 8827.39],
        [37.0, 0.83, 19651.05],
        [37.0, 4.9, 12139.55],
        [36.0, 1.67, 12036.46],
        [35.0, 1.84, 2942.46],
        [33.0, 0.24, 7084.9],
        [32.0, 0.18, 5088.63],
        [32.0, 1.78, 398.15],
        [28.0, 1.21, 6286.6],
        [28.0, 1.9, 6279.55],
        [26.0, 4.59, 10447.39],
    ];

    const R1: &[[f64; 3]] = &[
        [103019.0, 1.10749, 6283.07585],
        [1721.0, 1.0644, 12566.1517],
        [702.0, 3.142, 0.0],
        [32.0, 1.02, 18849.23],
        [31.0, 2.84, 5507.55],
        [25.0, 1.32, 5223.69],
        [18.0, 1.42, 1577.34],
        [10.0, 5.91, 10977.08],
        [9.0, 1.42, 6275.96],
        [9.0, 0.27, 5486.78],
    ];

    const R2: &[[f64; 3]] = &[
        [4359.0, 5.7846, 6283.0758],
        [124.0, 5.579, 12566.152],
        [12.0, 3.14, 0.0],
        [9.0, 3.63, 77713.77],
        [6.0, 1.87, 5573.14],
        [3.0, 5.47, 18849.23],
    ];

    const R3: &[[f64; 3]] = &[[145.0, 4.273, 6283.076], [7.0, 3.92, 12566.15]];

    const R4: &[[f64; 3]] = &[[4.0, 2.56, 6283.08]];

    pub(super) const R_TERMS: [&[[f64; 3]]; 5] = [R0, R1, R2, R3, R4];
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::tables::{B_TERMS, L_TERMS, R_TERMS};
    use super::{
        earth_heliocentric_latitude, earth_heliocentric_longitude, earth_radius_vector,
        evaluate_periodic_series, periodic_subseries_sum,
    };
    use crate::test_fixtures::reference_jme;

    /// Reference subseries totals (the raw `Σ A·cos(B + C·JME)` values, before
    /// the JME-polynomial combination and the `1 / 10⁸` scaling), as published
    /// in Table A5.1. Splitting them out lets a regression flag the broken
    /// sub-block instead of pointing at an aggregate `L`/`B`/`R` mismatch.
    #[allow(clippy::excessive_precision)] // Published trailing digits are pinned for traceability.
    const A5_1_L_SUBSERIES: [f64; 6] = [
        172_067_561.526_586,
        628_332_010_650.051_147,
        61_368.682_493,
        -26.902_819,
        -121.279_536,
        -0.999_999,
    ];
    const A5_1_B_SUBSERIES: [f64; 2] = [-176.502_688, 3.067_582];
    const A5_1_R_SUBSERIES: [f64; 5] = [
        99_653_849.037_796,
        100_378.567_146,
        -1_140.953_507,
        -141.115_419,
        1.232_361,
    ];

    /// Hybrid tolerance: tracks the trailing-digit precision of the table
    /// (≈10⁻⁶ absolute) for small terms while loosening to 1 ulp at the
    /// magnitude of the largest term (`L1 ≈ 6.28·10¹¹`, where 1 ulp ≈ 10⁻⁴).
    fn assert_close(actual: f64, expected: f64, label: &str) {
        let tolerance = expected.abs().mul_add(1e-12, 1e-6);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}: expected {expected}, got {actual} (|Δ| > {tolerance:e})",
        );
    }

    /// Each `L_k` subseries (radians) must match the value Table A5.1
    /// reports for that block at the reference JME.
    #[test]
    fn periodic_subseries_sums_for_l_match_table_a5_1() {
        let jme = reference_jme();
        for (k, expected) in A5_1_L_SUBSERIES.iter().enumerate() {
            let actual = periodic_subseries_sum(L_TERMS[k], jme);
            assert_close(actual, *expected, &format!("L{k}"));
        }
    }

    /// Each `B_k` subseries must reproduce the Table A5.1 value, including
    /// the negative `B0`, which exercises the signed accumulation path.
    #[test]
    fn periodic_subseries_sums_for_b_match_table_a5_1() {
        let jme = reference_jme();
        for (k, expected) in A5_1_B_SUBSERIES.iter().enumerate() {
            let actual = periodic_subseries_sum(B_TERMS[k], jme);
            assert_close(actual, *expected, &format!("B{k}"));
        }
    }

    /// Each `R_k` subseries must reproduce the Table A5.1 value. `R2` and
    /// `R3` are negative, again exercising the signed accumulation path.
    #[test]
    fn periodic_subseries_sums_for_r_match_table_a5_1() {
        let jme = reference_jme();
        for (k, expected) in A5_1_R_SUBSERIES.iter().enumerate() {
            let actual = periodic_subseries_sum(R_TERMS[k], jme);
            assert_close(actual, *expected, &format!("R{k}"));
        }
    }

    /// Aggregated `L` (after JME polynomial, radian→degree conversion and
    /// wrapping) must match the published Table A5.1 figure within the
    /// SPA-stated ±0.0003° envelope by several orders of magnitude.
    #[test]
    fn earth_heliocentric_longitude_matches_table_a5_1() {
        let l = earth_heliocentric_longitude(reference_jme());
        assert!(
            (l - 24.018_261_691_7).abs() < 1e-6,
            "L mismatch at A5.1 reference JME: got {l}",
        );
    }

    /// Aggregated `B` is signed and tiny (≈10⁻⁴°). The tolerance is set well
    /// below the magnitude of the value itself so a sign flip would fail.
    #[test]
    fn earth_heliocentric_latitude_matches_table_a5_1() {
        let b = earth_heliocentric_latitude(reference_jme());
        assert!(
            (b - -0.000_101_121_9).abs() < 1e-9,
            "B mismatch at A5.1 reference JME: got {b}",
        );
    }

    /// Aggregated `R` (in AU) must match Table A5.1. Tolerance below 1e-9
    /// is well within the ulp of the published value.
    #[test]
    fn earth_radius_vector_matches_table_a5_1() {
        let r = earth_radius_vector(reference_jme());
        assert!(
            (r - 0.996_542_297_4).abs() < 1e-9,
            "R mismatch at A5.1 reference JME: got {r}",
        );
    }

    /// Step 3.2.6 mandates the half-open interval `[0°, 360°)` for `L`. The
    /// `L1` term grows by ≈360 000°/JME, so even a one-millennium offset is
    /// far past one revolution and stresses the wrapping.
    #[test]
    fn earth_heliocentric_longitude_is_wrapped_into_zero_360() {
        for &jme in &[-2.0, -0.5, 0.0, 0.5, 1.5, reference_jme()] {
            let l = earth_heliocentric_longitude(jme);
            assert!(
                (0.0..360.0).contains(&l),
                "L escaped [0, 360) at JME = {jme}: got {l}",
            );
        }
    }

    /// `B` carries no range constraint and must remain signed.
    /// At the Table A5.1 reference JME `B` is published as a negative value,
    /// which would never occur if the function silently wrapped to a
    /// non-negative range.
    #[test]
    fn earth_heliocentric_latitude_preserves_sign() {
        let b = earth_heliocentric_latitude(reference_jme());
        assert!(b < 0.0, "B must be signed; got {b} at the A5.1 JME");
    }

    /// `R` is a physical distance and must stay finite and positive across
    /// the SPA validity window (-2000 to 6000), which spans roughly
    /// JME ∈ `[-4, 4]`.
    #[test]
    fn earth_radius_vector_is_strictly_positive_across_validity_window() {
        for &jme in &[-4.0, -1.0, 0.0, 1.0, 4.0] {
            let r = earth_radius_vector(jme);
            assert!(
                r.is_finite() && r > 0.0,
                "R must be finite and positive at JME = {jme}: got {r}",
            );
        }
    }

    /// `evaluate_periodic_series` reduces to a single subseries when the
    /// outer slice has length 1, and that single value is returned divided
    /// by 10⁸. This pins the Horner reduction at the polynomial's base case
    /// and the `1 / 10⁸` scaling demanded by equation 11.
    #[test]
    fn evaluate_periodic_series_reduces_to_scaled_subseries_when_polynomial_is_constant() {
        // A single row whose B and C are both 0 makes the sum equal to A,
        // independent of JME; the only thing the function should add is the
        // 1/10⁸ scaling.
        let single: &[[f64; 3]] = &[[1.0e8, 0.0, 0.0]];
        let outer: &[&[[f64; 3]]; 1] = &[single];
        let value = evaluate_periodic_series(outer, 0.123_456);
        assert!(
            (value - 1.0).abs() < f64::EPSILON,
            "constant polynomial must collapse to the scaled subseries: got {value}",
        );
    }
}
