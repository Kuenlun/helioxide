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

//! Mean (`ν₀`) and apparent (`ν`) sidereal time at Greenwich.
//!
//! Implements section 3.8 of NREL/TP-560-34302. The mean sidereal time
//! `ν₀` (equation 28) is a third-order polynomial whose dominant term is
//! linear in `(JD - JD_J2000)` (capturing Earth's diurnal rotation),
//! plus quadratic and cubic Julian-century corrections. The result is
//! wrapped into `[0°, 360°)` per step 3.8.2. The apparent sidereal time
//! `ν` (equation 29) corrects `ν₀` for the equation of the equinoxes by
//! adding `Δψ · cos(ε)`.

use crate::helper::limit_degrees;

/// Mean sidereal time at Greenwich `ν₀` (degrees), wrapped into `[0°, 360°)`.
///
/// `julian_day` is the Julian Day in UT, as produced by
/// [`calculate_julian_day`]. The Julian Century used by the higher-order
/// terms is derived internally so that callers cannot accidentally pair a
/// `JD` with a `JC` evaluated against a different epoch.
///
/// Refer to section 3.8, equation 28 and step 3.8.2:
/// `ν₀ = 280.46061837 + 360.98564736629 · (JD - 2_451_545)`
/// `   + 0.000387933 · JC² - JC³ / 38_710_000`
/// where `JC = (JD - 2_451_545) / 36_525`. The two `JC`-driven terms are
/// fused into one `JC² · (C - D · JC)` product so the coupled subtraction
/// shares a single rounded step, and the dominant linear-in-`JD` term
/// (reaching `~5·10⁵°` on the Table A5.1 reference date) is folded
/// against the small offset with `mul_add` to retain precision.
///
/// # Examples
///
/// ```
/// use helioxide::sidereal::mean_sidereal_time;
///
/// // Table A5.1 reference: JD ≈ 2_452_930.312_847 → ν₀ ≈ 318.515578°.
/// let nu0 = mean_sidereal_time(2_452_930.312_847);
/// assert!((nu0 - 318.515_578).abs() < 1e-4);
/// ```
///
/// [`calculate_julian_day`]: crate::julian::calculate_julian_day
#[must_use]
pub fn mean_sidereal_time(julian_day: f64) -> f64 {
    let days_since_j2000 = julian_day - 2_451_545.0;
    let jc = days_since_j2000 / 36_525.0;

    // Equation 28. `JC² · (C - D · JC)` keeps the two coupled JC
    // corrections in a single rounded product. The dominant
    // linear-in-`JD` term is then FMA-folded against the small constant
    // offset so it shares one rounded step with the constant.
    let jc_polynomial = (jc * jc).mul_add(
        (-1.0_f64 / 38_710_000.0).mul_add(jc, 0.000_387_933),
        280.460_618_37,
    );
    let nu0 = 360.985_647_366_29_f64.mul_add(days_since_j2000, jc_polynomial);

    limit_degrees(nu0)
}

/// Apparent sidereal time at Greenwich `ν` (degrees, signed, not wrapped).
///
/// `mean_sidereal_time` is the mean sidereal time `ν₀` (in degrees), as
/// produced by [`mean_sidereal_time`]. `delta_psi` is the nutation in
/// longitude `Δψ` (in degrees), as produced by
/// [`nutation_in_longitude_and_obliquity`]. `epsilon` is the true
/// obliquity of the ecliptic `ε` (in degrees), as produced by
/// [`true_obliquity_of_ecliptic`].
///
/// Refer to section 3.8, equation 29: `ν = ν₀ + Δψ · cos(ε)`. The
/// equation-of-the-equinoxes correction is bounded by `|Δψ| ≈ 17"`, so
/// `ν` lies within `~5·10⁻³°` of `ν₀` and is therefore returned without
/// an explicit wrap. The next consumer (equation 32, `H = ν + σ - α`)
/// reduces modulo 360° anyway.
///
/// # Examples
///
/// ```
/// use helioxide::sidereal::{apparent_sidereal_time, mean_sidereal_time};
///
/// // Table A5.1: ν₀ ≈ 318.515578°, Δψ = -0.00399840°, ε = 23.440465°
/// // → ν ≈ 318.511910°.
/// let nu0 = mean_sidereal_time(2_452_930.312_847);
/// let nu = apparent_sidereal_time(nu0, -0.003_998_40, 23.440_465);
/// assert!((nu - 318.511_910).abs() < 1e-4);
/// ```
///
/// [`mean_sidereal_time`]: self::mean_sidereal_time
/// [`nutation_in_longitude_and_obliquity`]: crate::nutation::nutation_in_longitude_and_obliquity
/// [`true_obliquity_of_ecliptic`]: crate::obliquity::true_obliquity_of_ecliptic
#[inline]
#[must_use]
pub fn apparent_sidereal_time(mean_sidereal_time: f64, delta_psi: f64, epsilon: f64) -> f64 {
    delta_psi.mul_add(epsilon.to_radians().cos(), mean_sidereal_time)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{apparent_sidereal_time, mean_sidereal_time};
    use crate::helper::limit_degrees;
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::obliquity::true_obliquity_of_ecliptic;
    use crate::test_fixtures::{reference_jce, reference_jd, reference_jme};

    /// `ν₀` at the Table A5.1 reference instant must reproduce the
    /// `318.515578°` value back-derived from the published `H`, `α` and
    /// observer longitude `σ` via equation 32 (then folded by
    /// `Δψ · cos(ε)`). Failure here is a global red flag: a wrong
    /// polynomial coefficient (equation 28), a wrong `36 525`-day
    /// Julian-century divisor, the J2000.0 epoch constant or the wrap
    /// to `[0°, 360°)` would all surface here. The 1e-4° tolerance sits
    /// just below the trailing-digit precision of the binding input
    /// `α = 202.22741°` (rounded to five decimals in the PDF), so a
    /// real bug cannot hide behind rounding.
    #[test]
    fn mean_sidereal_time_matches_table_a5_1() {
        let nu0 = mean_sidereal_time(reference_jd());
        assert!(
            (nu0 - 318.515_578).abs() < 1e-4,
            "ν₀ mismatch at A5.1 reference: got {nu0}",
        );
    }

    /// `ν` at the Table A5.1 reference instant must reproduce the
    /// `318.511910°` value back-derived from `H = 11.105900°`,
    /// `α = 202.22741°` and `σ = -105.1786°` via equation 32. Failure
    /// pins equation 29's `Δψ · cos(ε)` correction in addition to the
    /// upstream `ν₀`: a sign flip, a missing degrees-to-radians
    /// conversion on `ε`, or a `cos`/`sin` swap (the two differ
    /// significantly at `ε ≈ 23°`) would surface as a mismatch larger
    /// than the 1e-4° tolerance.
    #[test]
    fn apparent_sidereal_time_matches_table_a5_1() {
        let (delta_psi, delta_epsilon) = nutation_in_longitude_and_obliquity(reference_jce());
        let epsilon = true_obliquity_of_ecliptic(reference_jme(), delta_epsilon);
        let nu0 = mean_sidereal_time(reference_jd());

        let nu = apparent_sidereal_time(nu0, delta_psi, epsilon);
        assert!(
            (nu - 318.511_910).abs() < 1e-4,
            "ν mismatch at A5.1 reference: got {nu}",
        );
    }

    /// At `JD = 2_451_545.0` (the J2000.0 epoch) `JD - JD_J2000 = 0` and
    /// `JC = 0`, so equation 28 collapses to its constant term
    /// `280.460_618_37°` (already inside `[0°, 360°)`, so the wrap is a
    /// no-op). This isolates the constant from every other coefficient:
    /// a typo in `280.460_618_37` or in the J2000.0 epoch reference
    /// `2_451_545.0` would fail here even when the A5.1 reference test
    /// still passes, because the latter exercises a single instant
    /// where compensating typos can mask the bug.
    #[test]
    fn mean_sidereal_time_at_j2000_collapses_to_constant_term() {
        let nu0 = mean_sidereal_time(2_451_545.0);
        assert!(
            (nu0 - 280.460_618_37).abs() < 1e-12,
            "ν₀ at JD = J2000 must equal 280.460_618_37°: got {nu0}",
        );
    }

    /// Advancing `JD` by exactly one day at the J2000.0 epoch must shift
    /// `ν₀` by `360.985_647_366_29° mod 360° = 0.985_647_366_29°` (the
    /// quadratic and cubic corrections at `JC ≈ 1/36 525` contribute
    /// at most `~3·10⁻¹³°`, well below the assertion tolerance). This
    /// isolates the dominant linear coefficient from every other term:
    /// a typo in `360.985_647_366_29` would fail here even when the
    /// A5.1 reference test still passes (because at the reference date
    /// that term exceeds `5·10⁵°` and a low-digit typo could partly
    /// survive the wrap).
    #[test]
    fn mean_sidereal_time_advances_by_published_diurnal_rate_per_day() {
        let nu0_at_j2000 = mean_sidereal_time(2_451_545.0);
        let nu0_one_day_later = mean_sidereal_time(2_451_546.0);
        let advance = nu0_one_day_later - nu0_at_j2000;
        assert!(
            (advance - 0.985_647_366_29).abs() < 1e-9,
            "ν₀ daily advance must equal 0.985_647_366_29°: got {advance}",
        );
    }

    /// At `JD = J2000 + 100·36 525` (so `JC = 100` exactly) the quadratic
    /// coefficient `0.000_387_933` contributes `+3.879_33°` and the cubic
    /// divisor `38_710_000` contributes `-0.025_833_118°` to `ν₀` pre-wrap,
    /// summing to `~3.853_496_882°`. At the A5.1 reference (`JCE ≈ 0.038`)
    /// the same coefficients contribute only `~5.6·10⁻⁷°` and `~1.4·10⁻¹²°`,
    /// so a typo in either coefficient would survive the A5.1 test (whose
    /// 1e-4 floor sits well above those magnitudes) but fail here. The
    /// wrapped difference between the full polynomial and the linear-only
    /// baseline equals `(0.000_387_933·JC² - JC³/38_710_000) mod 360°` by
    /// the modular identity `limit_degrees(a + b) mod 360 = limit_degrees(a) + b`
    /// for bounded `b`, isolating the higher-order contribution.
    #[test]
    fn mean_sidereal_time_pins_higher_order_jc_terms_at_jc_100() {
        let jc = 100.0_f64;
        let jd = jc.mul_add(36_525.0, 2_451_545.0);
        let full = mean_sidereal_time(jd);
        let linear_only =
            limit_degrees(360.985_647_366_29_f64.mul_add(jd - 2_451_545.0, 280.460_618_37));
        let higher_order = limit_degrees(full - linear_only);
        let expected = 3.853_496_881_94_f64;
        assert!(
            (higher_order - expected).abs() < 5e-6,
            "Higher-order JC contribution at JC={jc} must equal \
             0.000_387_933·JC² - JC³/38_710_000 ≈ {expected}°: got {higher_order}",
        );
    }

    /// Step 3.8.2 mandates the half-open interval `[0°, 360°)` for `ν₀`.
    /// The sweep covers values inside and well outside the SPA-validated
    /// civil-date range so that `limit_degrees` is exercised across many
    /// full revolutions of the dominant linear term, including a
    /// negative-`JD` epoch where the wrap must lift a value far below
    /// zero up into `[0°, 360°)`.
    #[test]
    fn mean_sidereal_time_is_wrapped_into_zero_360() {
        for &jd in &[
            -1_000_000.0_f64,  // ~7400 BC, far before the SPA-validated range
            0.0,               // 4713 BC noon (the Julian Day epoch)
            2_451_545.0,       // J2000.0 (no linear contribution)
            2_452_930.312_847, // A5.1 reference, dominant linear ≈ 5·10⁵°
            5_000_000.0,       // ~AD 8975, far after the SPA-validated range
        ] {
            let nu0 = mean_sidereal_time(jd);
            assert!(
                (0.0..360.0).contains(&nu0),
                "ν₀ escaped [0°, 360°) for JD = {jd}: got {nu0}",
            );
        }
    }

    /// Equation 29 adds `Δψ · cos(ε)` linearly. Calling the function
    /// with the same `ν₀` and `ε` but two different `Δψ` values must
    /// shift the output by exactly `(Δψ_b - Δψ_a) · cos(ε)`. This
    /// isolates the equation-of-the-equinoxes correction: a sign flip
    /// on `Δψ`, a missing `cos(ε)` factor, or a `cos`/`sin` swap (which
    /// differ at `ε ≈ 23°`) would fail here even when the A5.1
    /// reference test still passes (because at the reference instant a
    /// compensating typo could hide behind one specific input). Both
    /// signs and several magnitudes of `Δψ` are swept so a typo cannot
    /// hide behind the comparison tolerance.
    #[test]
    fn apparent_sidereal_time_shifts_linearly_with_delta_psi() {
        let nu0 = 100.0_f64;
        let epsilon = 23.440_465_f64;
        let cos_epsilon = epsilon.to_radians().cos();
        let baseline = apparent_sidereal_time(nu0, 0.0, epsilon);
        for &delta_psi in &[-1e-2_f64, -1e-4, 1e-6, 5e-3] {
            let shifted = apparent_sidereal_time(nu0, delta_psi, epsilon);
            let expected_shift = delta_psi * cos_epsilon;
            assert!(
                (shifted - baseline - expected_shift).abs() < 1e-13,
                "ν must shift by Δψ · cos(ε) for Δψ = {delta_psi}: \
                 got {shifted} - {baseline} ≠ {expected_shift}",
            );
        }
    }

    /// At `ε = 90°` `cos(ε) = 0` (modulo `f64` rounding of `π/2`), so
    /// equation 29 collapses to `ν = ν₀` regardless of `Δψ`. This pins
    /// the degrees-to-radians conversion on `ε`: if the cosine were
    /// applied to the raw degree value, `cos(90 rad) ≈ -0.448` and the
    /// deliberately large `Δψ` swept here would surface as a clear
    /// mismatch. Both signs of `Δψ` are swept so a stray sign flip
    /// cannot hide behind a single input.
    #[test]
    fn apparent_sidereal_time_correction_vanishes_at_quarter_turn_obliquity() {
        let nu0 = 217.5_f64;
        for &delta_psi in &[-1.0_f64, 0.0, 1.0] {
            let nu = apparent_sidereal_time(nu0, delta_psi, 90.0);
            assert!(
                (nu - nu0).abs() < 1e-13,
                "ν must collapse to ν₀ at ε = 90° for Δψ = {delta_psi}: got {nu}",
            );
        }
    }
}
