// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Mean (`ε₀`) and true (`ε`) obliquity of the ecliptic.
//!
//! Implements section 3.5 of NREL/TP-560-34302. The mean obliquity `ε₀`
//! (equation 24) is a tenth-degree polynomial in `U = JME / 10` that
//! yields arc seconds. Equation 25 then converts `ε₀` to degrees and adds
//! the nutation in obliquity `Δε` to obtain the true obliquity `ε`. Each
//! quantity is surfaced through its own public function so callers can
//! consume `ε₀` directly in the unit equation 24 publishes it (arc
//! seconds) without having to back it out of `ε`.

/// Polynomial coefficients of `ε₀` (arc seconds) in `U = JME / 10`, ordered
/// from the constant term up to `U¹⁰`. Reproduced verbatim from equation 24.
const MEAN_OBLIQUITY_COEFFICIENTS: [f64; 11] = [
    84_381.448, -4_680.93, -1.55, 1_999.25, -51.38, -249.67, -39.05, 7.12, 27.87, 5.79, 2.45,
];

/// Mean obliquity of the ecliptic `ε₀` (arc seconds).
///
/// `jme` is the Julian Ephemeris Millennium, as produced by
/// [`calculate_julian_ephemeris_millennium`].
///
/// Refer to section 3.5, equation 24:
/// `ε₀ = 84381.448 + U·(-4680.93 + U·(-1.55 + U·(1999.25 + U·(-51.38 + U·(-249.67 + U·(-39.05 + U·(7.12 + U·(27.87 + U·(5.79 + U·2.45)))))))))`
/// where `U = JME / 10`. The output is in the same unit equation 24
/// publishes it (arc seconds); [`true_obliquity_of_ecliptic`] performs
/// the arc-second-to-degree conversion mandated by equation 25.
///
/// # Examples
///
/// ```
/// use helioxide::julian::{
///     calculate_julian_ephemeris_century, calculate_julian_ephemeris_day,
///     calculate_julian_ephemeris_millennium,
/// };
/// use helioxide::obliquity::mean_obliquity_of_ecliptic_arcseconds;
///
/// // Table A5.1 reference: 17 October 2003, 12:30:30 LST, ΔT = 67 s.
/// let jde = calculate_julian_ephemeris_day(2_452_930.312_847, 67.0);
/// let jce = calculate_julian_ephemeris_century(jde);
/// let jme = calculate_julian_ephemeris_millennium(jce);
///
/// let epsilon0 = mean_obliquity_of_ecliptic_arcseconds(jme);
/// // ε₀ ≈ 84_379.67" at this instant (`ε ≈ 23.440465°` minus the
/// // nutation `Δε ≈ 0.001666°`, scaled back to arc seconds).
/// assert!((epsilon0 - 84_379.67).abs() < 0.1);
/// ```
///
/// [`calculate_julian_ephemeris_millennium`]: crate::julian::calculate_julian_ephemeris_millennium
#[must_use]
pub fn mean_obliquity_of_ecliptic_arcseconds(jme: f64) -> f64 {
    // Equation 24: ε₀ in arc seconds, evaluated by Horner's method on U.
    let u = jme / 10.0;
    MEAN_OBLIQUITY_COEFFICIENTS
        .iter()
        .rev()
        .fold(0.0_f64, |acc, &coefficient| acc.mul_add(u, coefficient))
}

/// True obliquity of the ecliptic `ε` (degrees).
///
/// `jme` is the Julian Ephemeris Millennium, as produced by
/// [`calculate_julian_ephemeris_millennium`]. `delta_epsilon` is the
/// nutation in obliquity `Δε` (degrees), as produced by
/// [`nutation_in_longitude_and_obliquity`].
///
/// Refer to section 3.5, equation 25: `ε = ε₀ / 3600 + Δε`. `ε₀` is the
/// mean obliquity in arc seconds, as produced by
/// [`mean_obliquity_of_ecliptic_arcseconds`].
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
/// [`mean_obliquity_of_ecliptic_arcseconds`]: self::mean_obliquity_of_ecliptic_arcseconds
#[must_use]
pub fn true_obliquity_of_ecliptic(jme: f64, delta_epsilon: f64) -> f64 {
    // Equation 25: ε = ε₀ / 3600 + Δε  (arc seconds → degrees, then add Δε).
    mean_obliquity_of_ecliptic_arcseconds(jme) / 3600.0 + delta_epsilon
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{mean_obliquity_of_ecliptic_arcseconds, true_obliquity_of_ecliptic};
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

    /// `ε₀` at the Table A5.1 reference instant must match the value
    /// derived from the published true obliquity by subtracting the
    /// published nutation (`Δε`) and scaling back to arc seconds:
    /// `(ε − Δε) · 3600`. Pinning it directly here ensures the
    /// arc-seconds output (the unit equation 24 publishes it in) is
    /// not silently rescaled by [`true_obliquity_of_ecliptic`]'s
    /// `1/3600` divisor.
    #[test]
    fn mean_obliquity_arcseconds_matches_table_a5_1_via_round_trip() {
        let (_, delta_epsilon) = nutation_in_longitude_and_obliquity(reference_jce());
        let epsilon0_via_function = mean_obliquity_of_ecliptic_arcseconds(reference_jme());
        let epsilon0_via_round_trip = (23.440_465_f64 - delta_epsilon) * 3600.0;
        assert!(
            (epsilon0_via_function - epsilon0_via_round_trip).abs() < 0.1,
            "ε₀ at A5.1 reference must equal (ε − Δε)·3600 within the published \
             precision: got {epsilon0_via_function}\" vs round-trip \
             {epsilon0_via_round_trip}\"",
        );
    }

    /// At `JME = 0` (J2000.0 epoch) `U = 0`, so equation 24 collapses
    /// to its constant term `84_381.448"`. This isolates the constant
    /// from every other coefficient: a stray `_` typo in `84_381.448`
    /// would fail here even when the A5.1 reference test still passes
    /// (because the latter exercises a single non-zero `JME` where
    /// compensating typos in the higher-order coefficients could mask
    /// the bug).
    #[test]
    #[allow(clippy::float_cmp)]
    fn mean_obliquity_arcseconds_at_j2000_collapses_to_constant_term() {
        let epsilon0 = mean_obliquity_of_ecliptic_arcseconds(0.0);
        assert_eq!(
            epsilon0, 84_381.448,
            "ε₀ at JME = 0 must equal 84_381.448\"",
        );
    }
}
