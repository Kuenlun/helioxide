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

//! True obliquity of the ecliptic (`ε`).
//!
//! Implements section 3.5 of NREL/TP-560-34302. The mean obliquity `ε₀`
//! (equation 24) is a tenth-degree polynomial in `U = JME / 10` that yields
//! arc seconds; equation 25 then converts to degrees and adds the nutation
//! in obliquity `Δε`. The two steps are fused in a single call because the
//! intermediate `ε₀` has no other downstream consumer, so a separate public
//! function would only widen the API surface.

/// Polynomial coefficients of `ε₀` (arc seconds) in `U = JME / 10`, ordered
/// from the constant term up to `U¹⁰`. Reproduced verbatim from equation 24.
const MEAN_OBLIQUITY_COEFFICIENTS: [f64; 11] = [
    84_381.448, -4_680.93, -1.55, 1_999.25, -51.38, -249.67, -39.05, 7.12, 27.87, 5.79, 2.45,
];

/// True obliquity of the ecliptic `ε` (degrees).
///
/// `jme` is the Julian Ephemeris Millennium, as produced by
/// [`calculate_julian_ephemeris_millennium`]. `delta_epsilon` is the
/// nutation in obliquity `Δε` (degrees), as produced by
/// [`nutation_in_longitude_and_obliquity`].
///
/// Refer to section 3.5, equations 24 and 25.
///
/// # Examples
///
/// ```
/// use helioxide::julian::{
///     calculate_julian_ephemeris_century, calculate_julian_ephemeris_day,
///     calculate_julian_ephemeris_millennium,
/// };
/// use helioxide::nutation::nutation_in_longitude_and_obliquity;
/// use helioxide::obliquity::true_obliquity_of_ecliptic;
///
/// // Table A5.1 reference: 17 October 2003, 12:30:30 LST, ΔT = 67 s.
/// let jde = calculate_julian_ephemeris_day(2_452_930.312_847, 67.0);
/// let jce = calculate_julian_ephemeris_century(jde);
/// let jme = calculate_julian_ephemeris_millennium(jce);
/// let (_, delta_epsilon) = nutation_in_longitude_and_obliquity(jce);
///
/// let epsilon = true_obliquity_of_ecliptic(jme, delta_epsilon);
/// assert!((epsilon - 23.440_465).abs() < 1e-6);
/// ```
///
/// [`calculate_julian_ephemeris_millennium`]: crate::julian::calculate_julian_ephemeris_millennium
/// [`nutation_in_longitude_and_obliquity`]: crate::nutation::nutation_in_longitude_and_obliquity
#[must_use]
pub fn true_obliquity_of_ecliptic(jme: f64, delta_epsilon: f64) -> f64 {
    // Equation 24: ε₀ in arc seconds, evaluated by Horner's method on U.
    let u = jme / 10.0;
    let mean_obliquity_arcseconds = MEAN_OBLIQUITY_COEFFICIENTS
        .iter()
        .rev()
        .fold(0.0_f64, |acc, &coefficient| acc.mul_add(u, coefficient));

    // Equation 25: ε = ε₀ / 3600 + Δε  (arc seconds → degrees, then add Δε).
    mean_obliquity_arcseconds / 3600.0 + delta_epsilon
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::true_obliquity_of_ecliptic;
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::test_fixtures::{reference_jce, reference_jme};

    /// `ε` at the Table A5.1 reference instant must reproduce the published
    /// `23.440465°`. This pins the integration with the upstream JCE/JME and
    /// `Δε` chain: a wrong polynomial coefficient in equation 24, a missing
    /// arc-second-to-degree conversion in equation 25, or a sign flip on
    /// `Δε` would all surface here as a mismatch larger than the trailing
    /// digit of the published reference.
    #[test]
    fn true_obliquity_matches_table_a5_1() {
        let (_, delta_epsilon) = nutation_in_longitude_and_obliquity(reference_jce());
        let epsilon = true_obliquity_of_ecliptic(reference_jme(), delta_epsilon);
        assert!(
            (epsilon - 23.440_465).abs() < 1e-6,
            "ε mismatch at A5.1 reference: got {epsilon}",
        );
    }

    /// At `JME = 0` (J2000.0 epoch) `U = 0`, so equation 24 collapses to its
    /// constant term `84_381.448"`. With `Δε = 0`, equation 25 must therefore
    /// return exactly `84_381.448 / 3600°`. This isolates the constant term
    /// and the arc-second-to-degree factor from every other coefficient: a
    /// stray `_` typo in `84_381.448` or a wrong `3600` divisor would fail
    /// here even when the A5.1 reference test still passes.
    #[test]
    fn true_obliquity_at_j2000_collapses_to_constant_term() {
        let epsilon = true_obliquity_of_ecliptic(0.0, 0.0);
        let expected = 84_381.448 / 3600.0;
        assert!(
            (epsilon - expected).abs() < f64::EPSILON,
            "ε at JME=0 must equal 84_381.448 / 3600°: got {epsilon}",
        );
    }

    /// Equation 25 adds `Δε` unchanged. Calling the function with the same
    /// `JME` but two different `Δε` values must shift the output by exactly
    /// the difference: this isolates the addition step from the polynomial
    /// evaluation, so a sign flip on `Δε` (which would silently survive the
    /// A5.1 reference if the polynomial happened to compensate) fails here.
    /// The sweep includes both signs and a value much larger than the IAU
    /// envelope so a typo in the addition cannot hide behind the `1e-6`
    /// reference tolerance.
    #[test]
    fn true_obliquity_offsets_by_delta_epsilon() {
        let jme = reference_jme();
        let baseline = true_obliquity_of_ecliptic(jme, 0.0);
        for &delta in &[-0.5_f64, -1e-3, 1e-6, 0.5] {
            let shifted = true_obliquity_of_ecliptic(jme, delta);
            assert!(
                (shifted - baseline - delta).abs() < 1e-13,
                "ε must shift by Δε; got {shifted} - {baseline} ≠ {delta}",
            );
        }
    }
}
