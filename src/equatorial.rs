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

//! Sun geocentric right ascension (`α`) and declination (`δ`).
//!
//! Implements sections 3.9 and 3.10 of NREL/TP-560-34302. Both quantities
//! are coordinates of the same geocentric equatorial frame and consume the
//! same input triplet `(λ, β, ε)` (apparent sun longitude, geocentric
//! latitude, true obliquity), so they are presented together. Equation 30
//! uses `atan2` to keep `α` in the correct quadrant before the wrap into
//! `[0°, 360°)` mandated by step 3.9.2; equation 31 uses `arcsin`, whose
//! output is naturally in `[-90°, 90°]` and is therefore returned signed,
//! with positive values placing the sun north of the celestial equator.

use crate::helper::limit_degrees;

/// Sun geocentric right ascension `α` (degrees), wrapped into `[0°, 360°)`.
///
/// `apparent_sun_longitude` is the apparent sun longitude `λ` (in degrees),
/// as produced by [`apparent_sun_longitude`]. `geocentric_latitude` is the
/// sun geocentric latitude `β` (in degrees), as produced by
/// [`geocentric_latitude`]. `true_obliquity` is the true obliquity of the
/// ecliptic `ε` (in degrees), as produced by [`true_obliquity_of_ecliptic`].
///
/// Refer to section 3.9, equation 30:
/// `α = atan2(sin λ · cos ε − tan β · sin ε, cos λ)`. The two-argument
/// arctangent preserves the quadrant of `α` even when `cos λ < 0` (the
/// half-year `λ ∈ (90°, 270°)` between the summer and winter solstices,
/// spanning the autumnal equinox), where a single-argument
/// `atan(num/den)` would lose 180°. The numerator is folded with
/// `mul_add` so the dominant `sin λ · cos ε` product retains full
/// precision against the much smaller `tan β · sin ε` correction
/// (≈ 10⁻⁶ at the sun's geocentric latitude, which never exceeds a few
/// arc seconds).
///
/// # Examples
///
/// ```
/// use helioxide::equatorial::geocentric_right_ascension;
///
/// // Table A5.1: λ = 204.0085519281°, β = 0.0001011219°, ε = 23.440465°
/// // → α ≈ 202.22741°.
/// let alpha = geocentric_right_ascension(204.008_551_928_1, 0.000_101_121_9, 23.440_465);
/// assert!((alpha - 202.227_41).abs() < 1e-4);
/// ```
///
/// [`apparent_sun_longitude`]: crate::apparent::apparent_sun_longitude
/// [`geocentric_latitude`]: crate::geocentric::geocentric_latitude
/// [`true_obliquity_of_ecliptic`]: crate::obliquity::true_obliquity_of_ecliptic
#[inline]
#[must_use]
pub fn geocentric_right_ascension(
    apparent_sun_longitude: f64,
    geocentric_latitude: f64,
    true_obliquity: f64,
) -> f64 {
    let (sin_lambda, cos_lambda) = apparent_sun_longitude.to_radians().sin_cos();
    let (sin_epsilon, cos_epsilon) = true_obliquity.to_radians().sin_cos();
    let tan_beta = geocentric_latitude.to_radians().tan();

    let numerator = sin_lambda.mul_add(cos_epsilon, -(tan_beta * sin_epsilon));
    limit_degrees(numerator.atan2(cos_lambda).to_degrees())
}

/// Sun geocentric declination `δ` (degrees, signed, in `[-90°, 90°]`).
///
/// `apparent_sun_longitude` is the apparent sun longitude `λ` (in degrees),
/// as produced by [`apparent_sun_longitude`]. `geocentric_latitude` is the
/// sun geocentric latitude `β` (in degrees), as produced by
/// [`geocentric_latitude`]. `true_obliquity` is the true obliquity of the
/// ecliptic `ε` (in degrees), as produced by [`true_obliquity_of_ecliptic`].
///
/// Refer to section 3.10, equation 31:
/// `δ = arcsin(sin β · cos ε + cos β · sin ε · sin λ)`. `arcsin` returns
/// a value in `[-π/2, π/2]` by construction, so the output is already in
/// `[-90°, 90°]` after the radians-to-degrees conversion and no wrap is
/// applied: `δ > 0` places the sun north of the celestial equator,
/// `δ < 0` south. The dominant `cos β · sin ε · sin λ` triple product is
/// folded inside the `mul_add` so that its last multiplication is
/// rounded only once against the small `sin β · cos ε` correction.
///
/// # Examples
///
/// ```
/// use helioxide::equatorial::geocentric_declination;
///
/// // Table A5.1: λ = 204.0085519281°, β = 0.0001011219°, ε = 23.440465°
/// // → δ ≈ -9.31434°.
/// let delta = geocentric_declination(204.008_551_928_1, 0.000_101_121_9, 23.440_465);
/// assert!((delta - -9.314_34).abs() < 1e-4);
/// ```
///
/// [`apparent_sun_longitude`]: crate::apparent::apparent_sun_longitude
/// [`geocentric_latitude`]: crate::geocentric::geocentric_latitude
/// [`true_obliquity_of_ecliptic`]: crate::obliquity::true_obliquity_of_ecliptic
#[inline]
#[must_use]
pub fn geocentric_declination(
    apparent_sun_longitude: f64,
    geocentric_latitude: f64,
    true_obliquity: f64,
) -> f64 {
    let (sin_beta, cos_beta) = geocentric_latitude.to_radians().sin_cos();
    let (sin_epsilon, cos_epsilon) = true_obliquity.to_radians().sin_cos();
    let sin_lambda = apparent_sun_longitude.to_radians().sin();

    (cos_beta * sin_epsilon)
        .mul_add(sin_lambda, sin_beta * cos_epsilon)
        .asin()
        .to_degrees()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{geocentric_declination, geocentric_right_ascension};
    use crate::apparent::{aberration_correction, apparent_sun_longitude};
    use crate::geocentric::{geocentric_latitude, geocentric_longitude};
    use crate::heliocentric::{
        earth_heliocentric_latitude, earth_heliocentric_longitude, earth_radius_vector,
    };
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::obliquity::true_obliquity_of_ecliptic;
    use crate::test_fixtures::{reference_jce, reference_jme};

    /// Drives the full upstream chain that produces `(λ, β, ε)` at the
    /// Table A5.1 reference instant. Reused by every reference-instant
    /// test in this module so a single integration regression upstream
    /// surfaces in exactly one place.
    fn reference_lambda_beta_epsilon() -> (f64, f64, f64) {
        let jme = reference_jme();
        let theta = geocentric_longitude(earth_heliocentric_longitude(jme));
        let beta = geocentric_latitude(earth_heliocentric_latitude(jme));
        let (delta_psi, delta_epsilon) = nutation_in_longitude_and_obliquity(reference_jce());
        let epsilon = true_obliquity_of_ecliptic(jme, delta_epsilon);
        let delta_tau = aberration_correction(earth_radius_vector(jme));
        let lambda = apparent_sun_longitude(theta, delta_psi, delta_tau);
        (lambda, beta, epsilon)
    }

    /// `α` at the Table A5.1 reference instant must reproduce the published
    /// `202.22741°`. Failure here is a global integration red flag: a wrong
    /// sign in equation 30's numerator, a `sin`/`cos` swap, an `atan` in
    /// place of `atan2`, a missing wrap, or any upstream regression on
    /// `λ`, `β` or `ε` would all surface here. The 1e-4° tolerance sits
    /// just below the trailing-digit precision of the published value so
    /// a real bug cannot hide behind rounding.
    #[test]
    fn geocentric_right_ascension_matches_table_a5_1() {
        let (lambda, beta, epsilon) = reference_lambda_beta_epsilon();
        let alpha = geocentric_right_ascension(lambda, beta, epsilon);
        assert!(
            (alpha - 202.227_41).abs() < 1e-4,
            "α mismatch at A5.1 reference: got {alpha}",
        );
    }

    /// `δ` at the Table A5.1 reference instant must reproduce the published
    /// `-9.31434°`. The opposite sign and the order of magnitude difference
    /// from `α` ensure that an accidental swap of the two outputs cannot be
    /// hidden by both reference tests passing on the wrong component.
    #[test]
    fn geocentric_declination_matches_table_a5_1() {
        let (lambda, beta, epsilon) = reference_lambda_beta_epsilon();
        let delta = geocentric_declination(lambda, beta, epsilon);
        assert!(
            (delta - -9.314_34).abs() < 1e-4,
            "δ mismatch at A5.1 reference: got {delta}",
        );
    }

    /// At `β = 0` and `ε = 0`, equation 30 collapses to
    /// `α = atan2(sin λ, cos λ) = λ (mod 360°)`. Sweeping `λ` across all
    /// four quadrants and well outside `[0°, 360°)` pins three independent
    /// requirements at once: the radians-to-degrees conversion (a missing
    /// `to_radians` would explode `sin`/`cos` arguments), the use of
    /// `atan2` over `atan` (the two agree here because the wrap absorbs
    /// the discrepancy on this axis, but step 3.9.2's wrap into
    /// `[0°, 360°)` is exercised), and the absence of any spurious
    /// β- or ε-dependent term left over when both are zero.
    #[test]
    fn geocentric_right_ascension_collapses_to_apparent_longitude_when_beta_and_epsilon_zero() {
        for &lambda in &[15.0_f64, 75.0, 165.0, 200.0, 269.5, 350.0, -45.0, 720.5] {
            let alpha = geocentric_right_ascension(lambda, 0.0, 0.0);
            let expected = lambda.rem_euclid(360.0);
            assert!(
                (alpha - expected).abs() < 1e-10,
                "α must equal λ (mod 360°) at β = ε = 0: got {alpha} for λ = {lambda}",
            );
        }
    }

    /// At `λ = 180°` (and `β = 0`) equation 30's numerator is `≈ 0` while
    /// `cos λ = -1`, so `atan2` must return `π` and `α` must wrap to
    /// exactly `180°`. A single-argument `atan(num/den)` would return
    /// `atan(0 / -1) = 0`, off by 180°. At `λ = 270°` the numerator is
    /// `-cos ε`, `cos λ ≈ 0` from the negative side, and `atan2` resolves
    /// the third quadrant to `-π/2` (then wraps to `270°`), where a naive
    /// `atan(-cos ε / 0⁻)` would either NaN or collapse to `+π/2`. Pinning
    /// both rotations isolates the `atan2` quadrant resolution from the
    /// rest of the formula.
    #[test]
    fn geocentric_right_ascension_resolves_atan2_quadrant_at_cos_lambda_singularities() {
        for &(lambda, expected, label) in &[
            (180.0_f64, 180.0_f64, "λ = 180° (cos λ = -1, num ≈ 0)"),
            (270.0, 270.0, "λ = 270° (cos λ ≈ 0, num < 0)"),
        ] {
            let alpha = geocentric_right_ascension(lambda, 0.0, 23.44);
            assert!(
                (alpha - expected).abs() < 1e-10,
                "α at {label} must equal {expected}°: got {alpha}",
            );
        }
    }

    /// At `λ = 0`, equation 30's numerator collapses to `-tan β · sin ε`
    /// and the denominator to `cos λ = 1`, so `α = atan(-tan β · sin ε)`
    /// (mod 360°). Sweeping non-zero `β` and `ε` of both signs isolates
    /// the `tan β · sin ε` correction from the dominant `sin λ · cos ε`
    /// term: a missing `tan β` factor, a sign flip on it, a swap with
    /// `sin ε`, or a missing radians conversion on `β` would all fail
    /// here even when the A5.1 reference test passes (because at the
    /// published instant `β ≈ 10⁻⁴°` and the term contributes only
    /// `~10⁻⁶°` to the trailing decimals of `α`).
    #[test]
    fn geocentric_right_ascension_pins_tan_beta_correction_term() {
        for &(beta_deg, epsilon_deg) in
            &[(0.001_f64, 23.0), (-0.001, 23.0), (0.5, 45.0), (-0.5, 45.0)]
        {
            let alpha = geocentric_right_ascension(0.0, beta_deg, epsilon_deg);
            let numerator = -beta_deg.to_radians().tan() * epsilon_deg.to_radians().sin();
            let expected = numerator.atan().to_degrees().rem_euclid(360.0);
            assert!(
                (alpha - expected).abs() < 1e-12,
                "α at λ = 0 must equal atan(-tan β · sin ε) (mod 360°): \
                 got {alpha} vs expected {expected} for β = {beta_deg}, ε = {epsilon_deg}",
            );
        }
    }

    /// Step 3.9.2 mandates the half-open interval `[0°, 360°)` for `α`.
    /// The sweep covers `λ` values that drive the raw `atan2(...).to_degrees()`
    /// well into `(-180°, 0°)` (negative `cos λ` with negative numerator),
    /// so `limit_degrees` must do the lifting on every iteration. Without
    /// the wrap, `α` would carry the negative branch from `atan2`.
    #[test]
    fn geocentric_right_ascension_is_wrapped_into_zero_360() {
        for &lambda in &[-720.0_f64, -10.0, 250.0, 1000.0] {
            let alpha = geocentric_right_ascension(lambda, 0.000_1, 23.44);
            assert!(
                (0.0..360.0).contains(&alpha),
                "α escaped [0°, 360°) for λ = {lambda}: got {alpha}",
            );
        }
    }

    /// At `ε = 0`, equation 31 collapses to `δ = arcsin(sin β) = β` (for
    /// `|β| ≤ 90°`). Sweeping both signs of `β` and several `λ` values
    /// pins three things at once: that the `cos ε` factor on the `sin β`
    /// term is honored (a missing `cos ε` would still pass at `ε = 0`,
    /// so we ALSO check below that the `cos β · sin ε · sin λ` term
    /// vanishes when `ε = 0`), that no spurious `λ`-dependent term
    /// survives `ε = 0`, and that `δ` is signed (not wrapped). A wrap to
    /// `[0°, 360°)` would push negative `β` to ~360° and fail loudly.
    #[test]
    fn geocentric_declination_collapses_to_geocentric_latitude_when_epsilon_zero() {
        for &beta in &[-30.0_f64, -1e-3, 1e-3, 30.0, 89.999] {
            for &lambda in &[0.0_f64, 90.0, 180.0, 270.0] {
                let delta = geocentric_declination(lambda, beta, 0.0);
                assert!(
                    (delta - beta).abs() < 1e-10,
                    "δ must equal β when ε = 0: got {delta} for β = {beta}, λ = {lambda}",
                );
            }
        }
    }

    /// At `β = 0`, equation 31 collapses to
    /// `δ = arcsin(sin ε · sin λ)`. Sweeping `λ` across all four quadrants
    /// and `ε` across the full IAU envelope `[0°, 90°]` pins the
    /// `cos β · sin ε · sin λ` term in isolation: a missing factor, a
    /// `sin`/`cos` swap on `λ` or `ε`, or a sign flip on `sin λ` would
    /// all fail here. The negative-`λ`-quadrant entries (`λ > 180°`) flip
    /// the sign of `sin λ` and therefore of the expected `δ`, which a
    /// stray `|sin λ|` would not capture.
    #[test]
    fn geocentric_declination_pins_arcsin_sin_eps_sin_lambda_when_beta_zero() {
        for &lambda in &[15.0_f64, 90.0, 150.0, 210.0, 300.0] {
            for &epsilon in &[5.0_f64, 23.44, 45.0, 89.0] {
                let delta = geocentric_declination(lambda, 0.0, epsilon);
                let expected = (epsilon.to_radians().sin() * lambda.to_radians().sin())
                    .asin()
                    .to_degrees();
                assert!(
                    (delta - expected).abs() < 1e-12,
                    "δ at β = 0 must equal arcsin(sin ε · sin λ): \
                     got {delta} vs expected {expected} for λ = {lambda}, ε = {epsilon}",
                );
            }
        }
    }
}
