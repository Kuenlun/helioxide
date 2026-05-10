// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Nutation in longitude (`Δψ`) and obliquity (`Δε`).
//!
//! Implements section 3.4 of NREL/TP-560-34302. The two quantities share
//! the five fundamental angles `X₀..X₄` evaluated by Horner's method on
//! the Julian Ephemeris Century (equations 15 to 19) and the same row-by-
//! row argument `∑ⱼ Xⱼ·Yᵢⱼ` used as the input to `sin` (for `Δψ`) and
//! `cos` (for `Δε`) in the Table A4.3 series. Producing both in a single
//! sweep — and a single `sin_cos` call per row — yields the same result
//! as evaluating equations 22 and 23 independently, at half the
//! trigonometric cost.

use tables::{TABLE_A4_3, X_POLYNOMIALS};

/// One row of Table A4.3, packed as `(Y, [a, b, c, d])`.
///
/// `Y` lists the integer multipliers `(Y₀..Y₄)` applied to the fundamental
/// angles inside `sin` and `cos`. `[a, b, c, d]` carries the linear-in-JCE
/// coefficient pairs published next to each row: `(a, b)` is the pair for
/// the `Δψ` series, `(c, d)` the pair for the `Δε` series. All four
/// floating-point coefficients are expressed in `0.0001`-arc-second units,
/// matching the unscaled output of equations 20 and 21 prior to the
/// `1 / 36 000 000` conversion to degrees.
type NutationTerm = ([i8; 5], [f64; 4]);

/// Nutation in longitude (`Δψ`) and obliquity (`Δε`), both in degrees,
/// returned together as `(Δψ, Δε)`.
///
/// Implements section 3.4, equations 15 to 23. The two values are emitted
/// together because they share the fundamental-arguments evaluation and
/// the per-row dot product `∑ⱼ Xⱼ·Yᵢⱼ`; only the choice of `sin` versus
/// `cos` and the coefficient pair (`a, b` for `Δψ`; `c, d` for `Δε`)
/// differ, so a single sweep over Table A4.3 fills both accumulators.
///
/// `jce` is the Julian Ephemeris Century, as produced by
/// [`calculate_julian_ephemeris_century`].
///
/// # Examples
///
/// ```
/// use helioxide::julian::{
///     calculate_julian_ephemeris_century, calculate_julian_ephemeris_day,
/// };
/// use helioxide::nutation::nutation_in_longitude_and_obliquity;
///
/// // Table A5.1 reference: 17 October 2003, 12:30:30 LST, ΔT = 67 s.
/// let jde = calculate_julian_ephemeris_day(2_452_930.312_847, 67.0);
/// let jce = calculate_julian_ephemeris_century(jde);
///
/// let (delta_psi, delta_epsilon) = nutation_in_longitude_and_obliquity(jce);
/// assert!((delta_psi - -0.003_998_40).abs() < 1e-8);
/// assert!((delta_epsilon - 0.001_666_57).abs() < 1e-8);
/// ```
///
/// [`calculate_julian_ephemeris_century`]: crate::julian::calculate_julian_ephemeris_century
#[must_use]
pub fn nutation_in_longitude_and_obliquity(jce: f64) -> (f64, f64) {
    let x = fundamental_arguments(jce);

    let (psi, epsilon) =
        TABLE_A4_3
            .iter()
            .fold((0.0_f64, 0.0_f64), |(psi, epsilon), &(y, [a, b, c, d])| {
                let argument = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&xj, &yj)| xj * f64::from(yj))
                    .sum::<f64>()
                    .to_radians();
                let (sin_argument, cos_argument) = argument.sin_cos();
                (
                    b.mul_add(jce, a).mul_add(sin_argument, psi),
                    d.mul_add(jce, c).mul_add(cos_argument, epsilon),
                )
            });

    // Equations 22 and 23: 1° = 36 000 000 × 0.0001 arc seconds.
    (psi / 36_000_000.0, epsilon / 36_000_000.0)
}

/// Fundamental angles `X₀..X₄` (degrees) at the supplied Julian Ephemeris
/// Century, evaluated by Horner's method.
///
/// Refer to section 3.4, equations 15 to 19. Limiting the result to
/// `[0°, 360°)` would be an arithmetic no-op for the only downstream
/// consumer (`sin` and `cos` of *integer* linear combinations of these
/// angles), so the wrap is intentionally skipped.
#[inline]
fn fundamental_arguments(jce: f64) -> [f64; 5] {
    X_POLYNOMIALS.map(|[c0, c1, c2, c3]| c3.mul_add(jce, c2).mul_add(jce, c1).mul_add(jce, c0))
}

// The constants below mirror equations 15 to 19 and Table A4.3 verbatim.
// Clippy's stylistic checks on numeric literals are silenced here because
// rounding or substituting digits would alter the agreement with the
// published reference values.
#[allow(
    clippy::unreadable_literal,
    clippy::approx_constant,
    clippy::excessive_precision
)]
mod tables {
    use super::NutationTerm;

    /// Polynomial coefficients of the fundamental angles `X₀..X₄` in
    /// `[c₀, c₁, c₂, c₃]` order so that `Xₖ = c₀ + c₁·JCE + c₂·JCE² + c₃·JCE³`.
    /// Reproduced verbatim from equations 15 to 19.
    pub(super) const X_POLYNOMIALS: [[f64; 4]; 5] = [
        // X₀: mean elongation of the moon from the sun (equation 15).
        [297.85036, 445267.111480, -0.0019142, 1.0 / 189474.0],
        // X₁: mean anomaly of the sun, that is, of the Earth (equation 16).
        [357.52772, 35999.050340, -0.0001603, -1.0 / 300000.0],
        // X₂: mean anomaly of the moon (equation 17).
        [134.96298, 477198.867398, 0.0086972, 1.0 / 56250.0],
        // X₃: moon's argument of latitude (equation 18).
        [93.27191, 483202.017538, -0.0036825, 1.0 / 327270.0],
        // X₄: longitude of the ascending node of the moon's mean orbit on
        // the ecliptic, measured from the mean equinox of the date
        // (equation 19).
        [125.04452, -1934.136261, 0.0020708, 1.0 / 450000.0],
    ];

    /// Periodic terms for the nutation in longitude and obliquity, in the
    /// order published in Table A4.3 (63 rows). Each entry is
    /// `(Y, [a, b, c, d])`; see [`NutationTerm`] for the field semantics.
    pub(super) const TABLE_A4_3: &[NutationTerm] = &[
        ([0, 0, 0, 0, 1], [-171996.0, -174.2, 92025.0, 8.9]),
        ([-2, 0, 0, 2, 2], [-13187.0, -1.6, 5736.0, -3.1]),
        ([0, 0, 0, 2, 2], [-2274.0, -0.2, 977.0, -0.5]),
        ([0, 0, 0, 0, 2], [2062.0, 0.2, -895.0, 0.5]),
        ([0, 1, 0, 0, 0], [1426.0, -3.4, 54.0, -0.1]),
        ([0, 0, 1, 0, 0], [712.0, 0.1, -7.0, 0.0]),
        ([-2, 1, 0, 2, 2], [-517.0, 1.2, 224.0, -0.6]),
        ([0, 0, 0, 2, 1], [-386.0, -0.4, 200.0, 0.0]),
        ([0, 0, 1, 2, 2], [-301.0, 0.0, 129.0, -0.1]),
        ([-2, -1, 0, 2, 2], [217.0, -0.5, -95.0, 0.3]),
        ([-2, 0, 1, 0, 0], [-158.0, 0.0, 0.0, 0.0]),
        ([-2, 0, 0, 2, 1], [129.0, 0.1, -70.0, 0.0]),
        ([0, 0, -1, 2, 2], [123.0, 0.0, -53.0, 0.0]),
        ([2, 0, 0, 0, 0], [63.0, 0.0, 0.0, 0.0]),
        ([0, 0, 1, 0, 1], [63.0, 0.1, -33.0, 0.0]),
        ([2, 0, -1, 2, 2], [-59.0, 0.0, 26.0, 0.0]),
        ([0, 0, -1, 0, 1], [-58.0, -0.1, 32.0, 0.0]),
        ([0, 0, 1, 2, 1], [-51.0, 0.0, 27.0, 0.0]),
        ([-2, 0, 2, 0, 0], [48.0, 0.0, 0.0, 0.0]),
        ([0, 0, -2, 2, 1], [46.0, 0.0, -24.0, 0.0]),
        ([2, 0, 0, 2, 2], [-38.0, 0.0, 16.0, 0.0]),
        ([0, 0, 2, 2, 2], [-31.0, 0.0, 13.0, 0.0]),
        ([0, 0, 2, 0, 0], [29.0, 0.0, 0.0, 0.0]),
        ([-2, 0, 1, 2, 2], [29.0, 0.0, -12.0, 0.0]),
        ([0, 0, 0, 2, 0], [26.0, 0.0, 0.0, 0.0]),
        ([-2, 0, 0, 2, 0], [-22.0, 0.0, 0.0, 0.0]),
        ([0, 0, -1, 2, 1], [21.0, 0.0, -10.0, 0.0]),
        ([0, 2, 0, 0, 0], [17.0, -0.1, 0.0, 0.0]),
        ([2, 0, -1, 0, 1], [16.0, 0.0, -8.0, 0.0]),
        ([-2, 2, 0, 2, 2], [-16.0, 0.1, 7.0, 0.0]),
        ([0, 1, 0, 0, 1], [-15.0, 0.0, 9.0, 0.0]),
        ([-2, 0, 1, 0, 1], [-13.0, 0.0, 7.0, 0.0]),
        ([0, -1, 0, 0, 1], [-12.0, 0.0, 6.0, 0.0]),
        ([0, 0, 2, -2, 0], [11.0, 0.0, 0.0, 0.0]),
        ([2, 0, -1, 2, 1], [-10.0, 0.0, 5.0, 0.0]),
        ([2, 0, 1, 2, 2], [-8.0, 0.0, 3.0, 0.0]),
        ([0, 1, 0, 2, 2], [7.0, 0.0, -3.0, 0.0]),
        ([-2, 1, 1, 0, 0], [-7.0, 0.0, 0.0, 0.0]),
        ([0, -1, 0, 2, 2], [-7.0, 0.0, 3.0, 0.0]),
        ([2, 0, 0, 2, 1], [-7.0, 0.0, 3.0, 0.0]),
        ([2, 0, 1, 0, 0], [6.0, 0.0, 0.0, 0.0]),
        ([-2, 0, 2, 2, 2], [6.0, 0.0, -3.0, 0.0]),
        ([-2, 0, 1, 2, 1], [6.0, 0.0, -3.0, 0.0]),
        ([2, 0, -2, 0, 1], [-6.0, 0.0, 3.0, 0.0]),
        ([2, 0, 0, 0, 1], [-6.0, 0.0, 3.0, 0.0]),
        ([0, -1, 1, 0, 0], [5.0, 0.0, 0.0, 0.0]),
        ([-2, -1, 0, 2, 1], [-5.0, 0.0, 3.0, 0.0]),
        ([-2, 0, 0, 0, 1], [-5.0, 0.0, 3.0, 0.0]),
        ([0, 0, 2, 2, 1], [-5.0, 0.0, 3.0, 0.0]),
        ([-2, 0, 2, 0, 1], [4.0, 0.0, 0.0, 0.0]),
        ([-2, 1, 0, 2, 1], [4.0, 0.0, 0.0, 0.0]),
        ([0, 0, 1, -2, 0], [4.0, 0.0, 0.0, 0.0]),
        ([-1, 0, 1, 0, 0], [-4.0, 0.0, 0.0, 0.0]),
        ([-2, 1, 0, 0, 0], [-4.0, 0.0, 0.0, 0.0]),
        ([1, 0, 0, 0, 0], [-4.0, 0.0, 0.0, 0.0]),
        ([0, 0, 1, 2, 0], [3.0, 0.0, 0.0, 0.0]),
        ([0, 0, -2, 2, 2], [-3.0, 0.0, 0.0, 0.0]),
        ([-1, -1, 1, 0, 0], [-3.0, 0.0, 0.0, 0.0]),
        ([0, 1, 1, 0, 0], [-3.0, 0.0, 0.0, 0.0]),
        ([0, -1, 1, 2, 2], [-3.0, 0.0, 0.0, 0.0]),
        ([2, -1, -1, 2, 2], [-3.0, 0.0, 0.0, 0.0]),
        ([0, 0, 3, 2, 2], [-3.0, 0.0, 0.0, 0.0]),
        ([2, -1, 0, 2, 2], [-3.0, 0.0, 0.0, 0.0]),
    ];
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::nutation_in_longitude_and_obliquity;
    use crate::test_fixtures::reference_jce;

    /// `Δψ` at the Table A5.1 reference instant must reproduce the
    /// published `-0.00399840°`. Failure here is a global red flag: an
    /// end-to-end mismatch can stem from a wrong polynomial coefficient
    /// (equations 15 to 19), a Table A4.3 transcription error, a
    /// degree-versus-radian confusion in the trigonometric argument, or
    /// a wrong divisor in equation 22. Tolerance is set just below the
    /// trailing-digit precision of the published value so a real bug
    /// cannot hide behind rounding.
    #[test]
    fn delta_psi_matches_table_a5_1() {
        let (delta_psi, _) = nutation_in_longitude_and_obliquity(reference_jce());
        assert!(
            (delta_psi - -0.003_998_40).abs() < 1e-8,
            "Δψ mismatch at A5.1 reference JCE: got {delta_psi}",
        );
    }

    /// `Δε` at the Table A5.1 reference instant must reproduce the
    /// published `+0.00166657°`. The opposite sign and the smaller
    /// magnitude relative to `Δψ` ensure that a tuple-order swap between
    /// the two outputs cannot be hidden by both tests passing on the
    /// wrong component.
    #[test]
    fn delta_epsilon_matches_table_a5_1() {
        let (_, delta_epsilon) = nutation_in_longitude_and_obliquity(reference_jce());
        assert!(
            (delta_epsilon - 0.001_666_57).abs() < 1e-8,
            "Δε mismatch at A5.1 reference JCE: got {delta_epsilon}",
        );
    }
}
