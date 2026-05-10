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

//! Topocentric elevation (`e₀`), atmospheric refraction (`Δe`),
//! topocentric zenith angle (`θ`), astronomers' azimuth (`Γ`) and
//! topocentric azimuth (`Φ`).
//!
//! Implements sections 3.14 and 3.15 of NREL/TP-560-34302. Section 3.14
//! lifts the topocentric equatorial pair `(δ', H')` and the observer
//! latitude `φ` into the local horizon frame: equation 41 produces the
//! geometric elevation `e₀` (the sun above the unrefracted horizon),
//! equation 42 corrects for atmospheric refraction `Δe`, and equation 44
//! flips the corrected elevation `e = e₀ + Δe` into the zenith angle
//! `θ = 90° − e`. Section 3.15 closes the chain: equation 45 produces the
//! astronomers' azimuth `Γ`, measured westward from south and consumed by
//! section 3.16's incidence angle; equation 46 rotates `Γ` into the
//! navigators' convention `Φ = Γ + 180°`, measured eastward from north.

use crate::helper::limit_degrees;

/// Standard atmospheric pressure used as the denominator of the pressure
/// ratio in equation 42 (millibars). Reproduced verbatim from the paper.
const STANDARD_PRESSURE_MILLIBARS: f64 = 1010.0;

/// Reference temperature `283 K` (i.e. `10 °C`) appearing as the numerator
/// of the temperature ratio in equation 42.
const REFERENCE_TEMPERATURE_KELVIN: f64 = 283.0;

/// Offset that converts Celsius to Kelvin inside equation 42's denominator
/// `273 + T`. Reproduced verbatim from the paper (the `0.15` of the strict
/// IAU conversion is dropped: at the magnitudes the formula targets, its
/// influence on `Δe` is below the trailing digit of the published example).
const KELVIN_OFFSET_FROM_CELSIUS: f64 = 273.0;

/// Saemundsson refraction-formula numerator `1.02` (arc minutes), the
/// dimensional constant of equation 42.
const SAEMUNDSSON_NUMERATOR_ARCMINUTES: f64 = 1.02;

/// Arc minutes per degree, the divisor that brings the equation 42
/// numerator from arc minutes to degrees.
const ARC_MINUTES_PER_DEGREE: f64 = 60.0;

/// Numerator `10.3` (degrees) of the auxiliary correction `10.3/(e₀+5.11)`
/// added to `e₀` inside the `tan` of equation 42.
const REFRACTION_AUXILIARY_NUMERATOR_DEGREES: f64 = 10.3;

/// Offset `5.11` (degrees) added to `e₀` in the denominator of the same
/// auxiliary correction.
const REFRACTION_AUXILIARY_OFFSET_DEGREES: f64 = 5.11;

/// Sun apparent angular radius (degrees). Pairs with the standard
/// horizon-level refraction below to define the cutoff at which equation
/// 42 collapses to zero per its "below the horizon" note. Both constants
/// are the canonical values used in section A.2 to fix the rising/setting
/// elevation `h'₀ = −0.8333°`.
const SUN_DISK_RADIUS_DEGREES: f64 = 0.26667;

/// Standard atmospheric refraction at the visible horizon (degrees).
/// See [`SUN_DISK_RADIUS_DEGREES`] for context.
const STANDARD_REFRACTION_AT_HORIZON_DEGREES: f64 = 0.5667;

/// Geometric elevation below which the sun is fully below the visible
/// horizon and atmospheric refraction is no longer modelled (degrees).
/// Derived from [`SUN_DISK_RADIUS_DEGREES`] and
/// [`STANDARD_REFRACTION_AT_HORIZON_DEGREES`] so that a sun whose centre
/// sits this far below the geometric horizon would not have its upper
/// limb lifted into view by horizon-level refraction.
const HORIZON_CUTOFF_ELEVATION_DEGREES: f64 =
    -(SUN_DISK_RADIUS_DEGREES + STANDARD_REFRACTION_AT_HORIZON_DEGREES);

/// Topocentric elevation angle without atmospheric refraction `e₀` (degrees,
/// signed, in `[-90°, 90°]`).
///
/// `observer_latitude` is the geographical latitude `φ` (degrees, positive
/// north of the equator per section 3.12.2). `topocentric_declination` is
/// `δ'` (degrees), as produced by [`topocentric_equatorial_coordinates`].
/// `topocentric_local_hour_angle` is `H'` (degrees), as produced by
/// [`topocentric_local_hour_angle`].
///
/// Refer to section 3.14.1, equation 41:
/// `e₀ = arcsin(sin φ · sin δ' + cos φ · cos δ' · cos H')`. The dominant
/// `cos φ · cos δ' · cos H'` triple product is folded inside the `mul_add`
/// so that its last multiplication is rounded only once against the small
/// `sin φ · sin δ'` correction. `arcsin` returns a value in `[-π/2, π/2]`
/// by construction, so the output lies in `[-90°, 90°]` after the
/// radians-to-degrees conversion: positive values place the sun above the
/// geometric horizon, negative below.
///
/// # Examples
///
/// ```
/// use helioxide::horizontal::topocentric_elevation_without_refraction;
///
/// // Table A5.1: φ = 39.742476°, δ' = -9.316179°, H' = 11.10629°
/// // → e₀ ≈ 39.872°.
/// let e0 = topocentric_elevation_without_refraction(39.742_476, -9.316_179, 11.106_29);
/// assert!((e0 - 39.872).abs() < 1e-3);
/// ```
///
/// [`topocentric_equatorial_coordinates`]: crate::parallax::topocentric_equatorial_coordinates
/// [`topocentric_local_hour_angle`]: crate::hour_angle::topocentric_local_hour_angle
#[inline]
#[must_use]
pub fn topocentric_elevation_without_refraction(
    observer_latitude: f64,
    topocentric_declination: f64,
    topocentric_local_hour_angle: f64,
) -> f64 {
    let (sin_phi, cos_phi) = observer_latitude.to_radians().sin_cos();
    let (sin_delta_prime, cos_delta_prime) = topocentric_declination.to_radians().sin_cos();
    let cos_h_prime = topocentric_local_hour_angle.to_radians().cos();

    (cos_phi * cos_delta_prime)
        .mul_add(cos_h_prime, sin_phi * sin_delta_prime)
        .asin()
        .to_degrees()
}

/// Atmospheric refraction correction `Δe` (degrees, non-negative away
/// from the zenith).
///
/// `elevation_without_refraction` is `e₀` (degrees), as produced by
/// [`topocentric_elevation_without_refraction`]. `pressure_millibars` is
/// the annual average local pressure `P` (millibars). `temperature_celsius`
/// is the annual average local temperature `T` (degrees Celsius).
///
/// Refer to section 3.14.2, equation 42:
/// `Δe = (P/1010) · (283/(273+T)) · 1.02 / (60 · tan(e₀ + 10.3/(e₀+5.11)))`.
/// The auxiliary tangent argument lives in degrees and is converted to
/// radians for the trig call. Section 3.14.2 stipulates `Δe = 0` once the
/// sun drops below the visible horizon; the cutoff is set at
/// `e₀ < −(0.26667° + 0.5667°)` so that even the upper limb of the disk
/// (`+0.26667°`) lifted by horizon-level refraction (`+0.5667°` per
/// section A.2) would still sit at or below the geometric horizon. Above
/// `e₀ ≈ 89.89°` the auxiliary `e₀ + 10.3/(e₀+5.11)` crosses `90°` and
/// equation 42 returns a vanishingly small negative artefact (a few
/// `10⁻⁵°`); the paper is reproduced verbatim because refraction is
/// itself physically negligible that close to the zenith.
///
/// # Examples
///
/// ```
/// use helioxide::horizontal::atmospheric_refraction;
///
/// // Table A5.1: e₀ ≈ 39.872°, P = 820 mbar, T = 11 °C → Δe ≈ 0.01633°.
/// let delta_e = atmospheric_refraction(39.872, 820.0, 11.0);
/// assert!((delta_e - 0.016_33).abs() < 1e-4);
///
/// // Below the horizon refraction is not modelled.
/// assert_eq!(atmospheric_refraction(-1.0, 820.0, 11.0), 0.0);
/// ```
///
/// [`topocentric_elevation_without_refraction`]: self::topocentric_elevation_without_refraction
#[inline]
#[must_use]
pub fn atmospheric_refraction(
    elevation_without_refraction: f64,
    pressure_millibars: f64,
    temperature_celsius: f64,
) -> f64 {
    if elevation_without_refraction < HORIZON_CUTOFF_ELEVATION_DEGREES {
        return 0.0;
    }

    // Equation 42's auxiliary `e₀ + 10.3/(e₀+5.11)` (degrees), lifting
    // the tangent argument so that it stays comfortably away from the
    // `90°` singularity even as `e₀` itself approaches it from below.
    let auxiliary_degrees = REFRACTION_AUXILIARY_NUMERATOR_DEGREES
        / (elevation_without_refraction + REFRACTION_AUXILIARY_OFFSET_DEGREES)
        + elevation_without_refraction;

    let pressure_ratio = pressure_millibars / STANDARD_PRESSURE_MILLIBARS;
    let temperature_ratio =
        REFERENCE_TEMPERATURE_KELVIN / (KELVIN_OFFSET_FROM_CELSIUS + temperature_celsius);

    pressure_ratio * temperature_ratio * SAEMUNDSSON_NUMERATOR_ARCMINUTES
        / (ARC_MINUTES_PER_DEGREE * auxiliary_degrees.to_radians().tan())
}

/// Topocentric zenith angle `θ` (degrees, signed, not range-limited).
///
/// `elevation_without_refraction` is `e₀` (degrees), as produced by
/// [`topocentric_elevation_without_refraction`]. `refraction` is `Δe`
/// (degrees), as produced by [`atmospheric_refraction`].
///
/// Refer to sections 3.14.3 and 3.14.4, equations 43 and 44:
/// `e = e₀ + Δe`, `θ = 90° − e`. The two steps fuse into one expression
/// because the corrected elevation `e` has no other downstream consumer
/// (section 3.16's incidence equation 47 takes `θ`, not `e`); a separate
/// public helper would only widen the API surface for an algebraic
/// rearrangement of `θ`.
///
/// # Examples
///
/// ```
/// use helioxide::horizontal::topocentric_zenith_angle;
///
/// // Table A5.1: e₀ ≈ 39.872°, Δe ≈ 0.01633° → θ ≈ 50.11162°.
/// let theta = topocentric_zenith_angle(39.872, 0.016_33);
/// assert!((theta - 50.11167).abs() < 1e-3);
/// ```
///
/// [`topocentric_elevation_without_refraction`]: self::topocentric_elevation_without_refraction
/// [`atmospheric_refraction`]: self::atmospheric_refraction
#[inline]
#[must_use]
pub const fn topocentric_zenith_angle(elevation_without_refraction: f64, refraction: f64) -> f64 {
    90.0 - (elevation_without_refraction + refraction)
}

/// Topocentric astronomers' azimuth `Γ` (degrees), wrapped into `[0°, 360°)`.
///
/// `topocentric_local_hour_angle` is `H'` (degrees), as produced by
/// [`topocentric_local_hour_angle`]. `observer_latitude` is `φ` (degrees,
/// positive north of the equator per section 3.12.2).
/// `topocentric_declination` is `δ'` (degrees), as produced by
/// [`topocentric_equatorial_coordinates`].
///
/// Refer to section 3.15.1, equation 45:
/// `Γ = atan2(sin H', cos H' · sin φ − tan δ' · cos φ)`. The two-argument
/// arctangent preserves the quadrant of `Γ` even when the denominator
/// turns negative (the half of the sky where the sun has crossed the
/// observer's prime vertical), where a single-argument `atan(num/den)`
/// would lose 180°. The output is wrapped into `[0°, 360°)` per step
/// 3.15.1 and is measured westward from south, the convention consumed
/// by section 3.16's surface incidence angle.
///
/// # Examples
///
/// ```
/// use helioxide::horizontal::astronomers_azimuth;
///
/// // Table A5.1: H' = 11.10629°, φ = 39.742476°, δ' = -9.316179°
/// // → Γ ≈ 14.34024°.
/// let gamma = astronomers_azimuth(11.106_29, 39.742_476, -9.316_179);
/// assert!((gamma - 14.340_24).abs() < 1e-4);
/// ```
///
/// [`topocentric_local_hour_angle`]: crate::hour_angle::topocentric_local_hour_angle
/// [`topocentric_equatorial_coordinates`]: crate::parallax::topocentric_equatorial_coordinates
#[inline]
#[must_use]
pub fn astronomers_azimuth(
    topocentric_local_hour_angle: f64,
    observer_latitude: f64,
    topocentric_declination: f64,
) -> f64 {
    let (sin_h_prime, cos_h_prime) = topocentric_local_hour_angle.to_radians().sin_cos();
    let (sin_phi, cos_phi) = observer_latitude.to_radians().sin_cos();
    let tan_delta_prime = topocentric_declination.to_radians().tan();

    let denominator = cos_h_prime.mul_add(sin_phi, -(tan_delta_prime * cos_phi));
    limit_degrees(sin_h_prime.atan2(denominator).to_degrees())
}

/// Topocentric azimuth angle `Φ` (degrees), wrapped into `[0°, 360°)`.
///
/// `astronomers_azimuth` is `Γ` (degrees), as produced by
/// [`astronomers_azimuth`].
///
/// Refer to section 3.15.2, equation 46: `Φ = Γ + 180°`, then wrapped
/// into `[0°, 360°)` per step 3.2.6. The `+180°` rotation flips the
/// astronomers' "westward from south" convention into the navigators'
/// "eastward from north" convention; the wrap takes care of the upper
/// half of the input range, where the raw sum exceeds `360°`.
///
/// # Examples
///
/// ```
/// use helioxide::horizontal::topocentric_azimuth_angle;
///
/// // Table A5.1: Γ ≈ 14.34024° → Φ ≈ 194.34024°.
/// let phi = topocentric_azimuth_angle(14.340_24);
/// assert!((phi - 194.340_24).abs() < 1e-12);
///
/// // The wrap closes the second half of the range.
/// let phi = topocentric_azimuth_angle(200.0);
/// assert!((phi - 20.0).abs() < 1e-12);
/// ```
///
/// [`astronomers_azimuth`]: self::astronomers_azimuth
#[inline]
#[must_use]
pub const fn topocentric_azimuth_angle(astronomers_azimuth: f64) -> f64 {
    limit_degrees(astronomers_azimuth + 180.0)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{
        HORIZON_CUTOFF_ELEVATION_DEGREES, REFERENCE_TEMPERATURE_KELVIN,
        STANDARD_PRESSURE_MILLIBARS, astronomers_azimuth, atmospheric_refraction,
        topocentric_azimuth_angle, topocentric_elevation_without_refraction,
        topocentric_zenith_angle,
    };
    use crate::apparent::{aberration_correction, apparent_sun_longitude};
    use crate::equatorial::{geocentric_declination, geocentric_right_ascension};
    use crate::geocentric::{geocentric_latitude, geocentric_longitude};
    use crate::heliocentric::{
        earth_heliocentric_latitude, earth_heliocentric_longitude, earth_radius_vector,
    };
    use crate::hour_angle::{observer_local_hour_angle, topocentric_local_hour_angle};
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::obliquity::true_obliquity_of_ecliptic;
    use crate::parallax::{equatorial_horizontal_parallax, topocentric_equatorial_coordinates};
    use crate::sidereal::{apparent_sidereal_time, mean_sidereal_time};
    use crate::test_fixtures::{
        REFERENCE_ELEVATION_METRES, REFERENCE_LATITUDE_DEGREES, REFERENCE_LONGITUDE_DEGREES,
        reference_jce, reference_jd, reference_jme,
    };

    /// Reference observer pressure for the Table A5.1 worked example
    /// (millibars), per section A.5.
    const REFERENCE_PRESSURE_MILLIBARS: f64 = 820.0;
    /// Reference observer temperature for the Table A5.1 worked example
    /// (degrees Celsius), per section A.5.
    const REFERENCE_TEMPERATURE_CELSIUS: f64 = 11.0;

    /// Drives the full upstream chain to produce `(δ', H')` at the
    /// Table A5.1 reference instant. Pinning the pair together lets every
    /// downstream reference test surface a regression on either input in
    /// exactly one place.
    fn reference_delta_prime_and_h_prime() -> (f64, f64) {
        let jd = reference_jd();
        let jce = reference_jce();
        let jme = reference_jme();

        let (delta_psi, delta_epsilon) = nutation_in_longitude_and_obliquity(jce);
        let epsilon = true_obliquity_of_ecliptic(jme, delta_epsilon);
        let nu = apparent_sidereal_time(mean_sidereal_time(jd), delta_psi, epsilon);

        let theta = geocentric_longitude(earth_heliocentric_longitude(jme));
        let beta = geocentric_latitude(earth_heliocentric_latitude(jme));
        let r = earth_radius_vector(jme);
        let delta_tau = aberration_correction(r);
        let lambda = apparent_sun_longitude(theta, delta_psi, delta_tau);

        let alpha = geocentric_right_ascension(lambda, beta, epsilon);
        let delta = geocentric_declination(lambda, beta, epsilon);
        let h = observer_local_hour_angle(nu, REFERENCE_LONGITUDE_DEGREES, alpha);
        let xi = equatorial_horizontal_parallax(r);
        let topocentric = topocentric_equatorial_coordinates(
            alpha,
            delta,
            h,
            xi,
            REFERENCE_LATITUDE_DEGREES,
            REFERENCE_ELEVATION_METRES,
        );
        let h_prime = topocentric_local_hour_angle(h, topocentric.parallax_in_right_ascension);

        (topocentric.declination, h_prime)
    }

    /// Drives the full upstream chain plus equation 41 to produce `e₀` at
    /// the Table A5.1 reference instant. Reused by the refraction, zenith,
    /// and reference-θ tests so that a single integration regression
    /// upstream surfaces in exactly one place.
    fn reference_elevation_without_refraction() -> f64 {
        let (delta_prime, h_prime) = reference_delta_prime_and_h_prime();
        topocentric_elevation_without_refraction(REFERENCE_LATITUDE_DEGREES, delta_prime, h_prime)
    }

    /// `θ` at the Table A5.1 reference instant must reproduce the
    /// published `50.11162°`. Failure is a global integration red flag:
    /// a wrong sign on equation 41's `sin φ · sin δ'` term, a `sin`/`cos`
    /// swap, a missing radians conversion on any input, a sign flip on
    /// equation 42's correction, a wrong `90°` complement in equation 44,
    /// or any upstream regression on `(φ, δ', H', P, T)` would all surface
    /// here. The 1e-4° tolerance sits just below the trailing-digit
    /// precision of the published value, so a real bug cannot hide
    /// behind rounding.
    #[test]
    fn topocentric_zenith_angle_matches_table_a5_1() {
        let e0 = reference_elevation_without_refraction();
        let delta_e = atmospheric_refraction(
            e0,
            REFERENCE_PRESSURE_MILLIBARS,
            REFERENCE_TEMPERATURE_CELSIUS,
        );
        let theta = topocentric_zenith_angle(e0, delta_e);
        assert!(
            (theta - 50.111_62).abs() < 1e-4,
            "θ mismatch at A5.1 reference: got {theta}",
        );
    }

    /// `Γ` at the Table A5.1 reference instant must reproduce the value
    /// `14.34024°` produced by equation 45 fed with the published
    /// `(H' = 11.10629°, φ = 39.742476°, δ' = -9.316179°)`. The opposite
    /// sign and the order-of-magnitude difference from `Φ` (the next
    /// test) ensure that an accidental swap of the two outputs cannot be
    /// hidden by both reference tests passing on the wrong field. Failure
    /// pins equation 45's `cos H' · sin φ − tan δ' · cos φ` denominator,
    /// the use of `atan2` over `atan`, and the wrap into `[0°, 360°)`.
    #[test]
    fn astronomers_azimuth_matches_published_intermediate_values() {
        let gamma = astronomers_azimuth(11.106_29, 39.742_476, -9.316_179);
        assert!(
            (gamma - 14.340_24).abs() < 1e-4,
            "Γ mismatch at A5.1 published intermediates: got {gamma}",
        );
    }

    /// `Φ` at the Table A5.1 reference instant must reproduce the
    /// published `194.34024°`. Failure pins equation 46's `+180°` rotation
    /// in addition to the upstream `Γ` chain. A wrong sign on the rotation
    /// (`Γ − 180°`) would shift `Φ` by `360°` and the wrap would mask the
    /// bug; only a magnitude error on `180` would surface here.
    #[test]
    fn topocentric_azimuth_angle_matches_table_a5_1() {
        let (delta_prime, h_prime) = reference_delta_prime_and_h_prime();
        let gamma = astronomers_azimuth(h_prime, REFERENCE_LATITUDE_DEGREES, delta_prime);
        let phi = topocentric_azimuth_angle(gamma);
        assert!(
            (phi - 194.340_24).abs() < 1e-4,
            "Φ mismatch at A5.1 reference: got {phi}",
        );
    }

    /// At the celestial equator (`δ' = 0`) and with the sun on the local
    /// meridian (`H' = 0`, so `cos H' = 1`), equation 41 collapses to
    /// `e₀ = arcsin(cos φ) = 90° − |φ|` for `|φ| ≤ 90°`. Sweeping `φ`
    /// across both hemispheres and the equatorial pole/equator extremes
    /// pins three things at once: that `δ' = 0` removes the `sin φ · sin δ'`
    /// addend (a sign flip there would still pass at `δ' = 0` if the
    /// `cos φ · cos δ' · cos H'` term happened to compensate, but the
    /// pole-vs-equator sweep widens the window), that the `cos H' = 1`
    /// substitution leaves the `cos φ · cos δ'` factor intact (a missing
    /// `cos H'` would still pass here, so a separate test below pins the
    /// `cos H'` factor in isolation), and that the radians conversion on
    /// `φ` is in place. A typo dropping `cos φ` would yield `arcsin(1) = 90°`
    /// regardless of `φ` and fail loudly across the sweep.
    #[test]
    fn topocentric_elevation_collapses_to_complement_of_latitude_at_celestial_meridian() {
        for &phi in &[-89.0_f64, -45.0, 0.0, 30.0, 60.0, 89.0] {
            let e0 = topocentric_elevation_without_refraction(phi, 0.0, 0.0);
            let expected = 90.0 - phi.abs();
            assert!(
                (e0 - expected).abs() < 1e-12,
                "e₀ at δ'=H'=0 must equal 90° − |φ| for φ = {phi}: \
                 got {e0} vs expected {expected}",
            );
        }
    }

    /// At the celestial equator (`δ' = 0`) and the geographical equator
    /// (`φ = 0`), equation 41 collapses to `e₀ = arcsin(cos H')`. Sweeping
    /// `H'` across both branches isolates the `cos H'` factor: a missing
    /// `cos H'` would yield `arcsin(0) = 0°` regardless of `H'`, and a
    /// swap with `sin H'` would shift the curve by `90°`. The chosen
    /// inputs cover both signs and exercise the `arcsin` mapping over its
    /// full positive lobe.
    #[test]
    fn topocentric_elevation_collapses_to_arcsin_cos_h_prime_on_equator() {
        for &h_prime in &[-90.0_f64, -45.0, 0.0, 30.0, 60.0, 90.0] {
            let e0 = topocentric_elevation_without_refraction(0.0, 0.0, h_prime);
            let expected = h_prime.to_radians().cos().asin().to_degrees();
            assert!(
                (e0 - expected).abs() < 1e-12,
                "e₀ at φ=δ'=0 must equal arcsin(cos H') for H' = {h_prime}: \
                 got {e0} vs expected {expected}",
            );
        }
    }

    /// At the same latitude as the topocentric declination (`φ = δ'`) and
    /// on the celestial meridian (`H' = 0`), equation 41 collapses to
    /// `e₀ = arcsin(sin²φ + cos²φ) = arcsin(1) = 90°` (the sun overhead).
    /// This pins the `sin φ · sin δ'` addend in isolation: dropping it
    /// would leave `arcsin(cos²φ)` which only equals `90°` at `φ = 0`,
    /// failing for every other `φ` in the sweep. A sign flip on the
    /// addend would yield `arcsin(cos 2φ)` and fail equally loudly.
    #[test]
    fn topocentric_elevation_reaches_zenith_when_latitude_equals_declination_on_meridian() {
        for &phi in &[-60.0_f64, -23.5, 0.0, 23.5, 60.0] {
            let e0 = topocentric_elevation_without_refraction(phi, phi, 0.0);
            assert!(
                (e0 - 90.0).abs() < 1e-12,
                "e₀ at φ = δ' on meridian must equal 90° for φ = {phi}: got {e0}",
            );
        }
    }

    /// Section 3.14.2 mandates `Δe = 0` once the sun drops below the
    /// visible horizon. The cutoff is fixed at
    /// `−(0.26667° + 0.5667°) = −0.83337°`. The sweep covers values
    /// strictly below it (for which the function must return zero) and
    /// values strictly at or above it (for which the function must return
    /// the equation-42 value); no intermediate fudge is allowed. Together
    /// with the next test, this exercises both branches of the cutoff.
    #[test]
    #[allow(clippy::float_cmp)] // The function returns the literal `0.0`, so direct comparison is exact.
    fn atmospheric_refraction_vanishes_strictly_below_horizon_cutoff() {
        for &e0 in &[
            -90.0_f64,
            -10.0,
            -1.0,
            HORIZON_CUTOFF_ELEVATION_DEGREES - f64::EPSILON,
        ] {
            let delta_e = atmospheric_refraction(e0, 1010.0, 10.0);
            assert_eq!(
                delta_e, 0.0,
                "Δe must be exactly zero strictly below the horizon cutoff for e₀ = {e0}",
            );
        }
    }

    /// At the standard reference atmosphere (`P = 1010 mbar`, `T = 10 °C`)
    /// the pressure and temperature ratios in equation 42 both collapse
    /// to one, so `Δe = 1.02 / (60 · tan(e₀ + 10.3/(e₀+5.11)))`. Sweeping
    /// `e₀` above the horizon pins the auxiliary `e₀ + 10.3/(e₀+5.11)`
    /// degree argument, the radians conversion before `tan`, and the
    /// `1.02 / 60` numerator. A typo on either constant would surface as
    /// a non-trivial mismatch even when the A5.1 reference test still
    /// passes (because at the published `(P, T)` a compensating typo on
    /// the ratios could mask the bug).
    #[test]
    fn atmospheric_refraction_collapses_at_standard_pressure_and_temperature() {
        let pressure = STANDARD_PRESSURE_MILLIBARS;
        let temperature = REFERENCE_TEMPERATURE_KELVIN - 273.0;
        for &e0 in &[HORIZON_CUTOFF_ELEVATION_DEGREES, 0.0, 10.0, 45.0, 89.0] {
            let delta_e = atmospheric_refraction(e0, pressure, temperature);
            let aux = e0 + 10.3 / (e0 + 5.11);
            let expected = 1.02 / (60.0 * aux.to_radians().tan());
            assert!(
                (delta_e - expected).abs() < 1e-15,
                "Δe at standard atmosphere must equal 1.02/(60·tan(e₀+10.3/(e₀+5.11))) \
                 for e₀ = {e0}: got {delta_e} vs expected {expected}",
            );
        }
    }

    /// Equation 42 is linear in the pressure ratio `P / 1010`. Holding the
    /// other inputs fixed, doubling `P` must double `Δe`. This pins the
    /// `1010 mbar` divisor in isolation: a typo there (e.g. `1013`, the
    /// other commonly tabulated standard) would still produce a finite
    /// `Δe` at the A5.1 reference, but the proportionality test would
    /// catch the resulting deviation.
    #[test]
    fn atmospheric_refraction_is_linear_in_pressure_ratio() {
        let baseline = atmospheric_refraction(20.0, 500.0, 10.0);
        let doubled = atmospheric_refraction(20.0, 1000.0, 10.0);
        let residual = 2.0_f64.mul_add(-baseline, doubled);
        assert!(
            residual.abs() < 1e-15,
            "Δe must scale linearly with P: got {doubled} vs expected {expected}",
            expected = 2.0 * baseline,
        );
    }

    /// Equation 42 is linear in the temperature ratio `283 / (273 + T)`.
    /// Holding the other inputs fixed, the ratio of `Δe` between two
    /// temperatures `T_a` and `T_b` must equal `(273 + T_b) / (273 + T_a)`.
    /// This pins the `283` numerator and the `273` Celsius-to-Kelvin
    /// offset together: a typo on either would shift the ratio away from
    /// the expected one. The `283` in the numerator further cancels in
    /// this ratio, so it is pinned by the
    /// `atmospheric_refraction_collapses_at_standard_pressure_and_temperature`
    /// test above (where `T = 10 °C` makes the temperature ratio one).
    #[test]
    fn atmospheric_refraction_inverse_temperature_ratio_holds() {
        let cold = atmospheric_refraction(20.0, 1010.0, -30.0);
        let hot = atmospheric_refraction(20.0, 1010.0, 40.0);
        let expected_ratio = (273.0 + -30.0_f64) / (273.0 + 40.0);
        let actual_ratio = hot / cold;
        assert!(
            (actual_ratio - expected_ratio).abs() < 1e-12,
            "Δe(hot)/Δe(cold) must equal (273+T_cold)/(273+T_hot): \
             got {actual_ratio} vs expected {expected_ratio}",
        );
    }

    /// Equations 43 and 44 together give `θ = 90° − (e₀ + Δe)` linearly
    /// in both inputs. Shifting one input by `δ` while holding the other
    /// fixed must shift `θ` by `−δ`. This isolates the two coefficients
    /// and signs from each other: a sign flip on either, a swap of the
    /// two inputs, or a stray multiplicative factor would all fail here
    /// even when the A5.1 reference test still passes (because at one
    /// specific instant compensating typos can mask the bug). The
    /// baseline `(e₀ = 45°, Δe = 0.01°)` keeps the raw difference well
    /// inside `[−90°, 90°]` so the small `δ` values do not interact with
    /// the `arcsin` boundary upstream.
    #[test]
    fn topocentric_zenith_angle_is_linear_in_each_input() {
        let baseline_e0 = 45.0_f64;
        let baseline_delta_e = 0.01_f64;
        let baseline = topocentric_zenith_angle(baseline_e0, baseline_delta_e);
        for &delta in &[-1.0_f64, -1e-4, 1e-6, 0.5] {
            let from_e0 = topocentric_zenith_angle(baseline_e0 + delta, baseline_delta_e);
            let from_delta_e = topocentric_zenith_angle(baseline_e0, baseline_delta_e + delta);
            assert!(
                (from_e0 - baseline + delta).abs() < 1e-13,
                "θ must shift by -δ when e₀ shifts by δ = {delta}: \
                 got {from_e0} - {baseline} ≠ {expected}",
                expected = -delta,
            );
            assert!(
                (from_delta_e - baseline + delta).abs() < 1e-13,
                "θ must shift by -δ when Δe shifts by δ = {delta}: \
                 got {from_delta_e} - {baseline} ≠ {expected}",
                expected = -delta,
            );
        }
    }

    /// Step 3.15.1 mandates the half-open interval `[0°, 360°)` for `Γ`.
    /// At `H' = 0` and any `φ`, `δ'` with `cos H' · sin φ − tan δ' · cos φ < 0`
    /// (i.e. the sun crossed the prime vertical to the north of the
    /// observer), `atan2(0, negative)` returns `±π`, which the wrap must
    /// lift to exactly `180°`. The sweep also includes `H' < 0` cases
    /// where the raw `atan2` output is negative, so `limit_degrees` must
    /// do the lifting on every iteration. Without the wrap, `Γ` would
    /// carry the negative branch from `atan2`.
    #[test]
    fn astronomers_azimuth_is_wrapped_into_zero_360() {
        for &(h_prime, phi, delta_prime) in &[
            (0.0_f64, 0.0, 30.0),            // raw atan2 returns +π → must wrap to 180°
            (-30.0, 39.742_476, -9.316_179), // raw atan2 negative
            (-90.0, 39.742_476, -9.316_179),
            (-179.999, 0.0, 0.0),
        ] {
            let gamma = astronomers_azimuth(h_prime, phi, delta_prime);
            assert!(
                (0.0..360.0).contains(&gamma),
                "Γ escaped [0°, 360°) for (H', φ, δ') = ({h_prime}, {phi}, {delta_prime}): got {gamma}",
            );
        }
    }

    /// Equation 45's numerator `sin H'` is odd in `H'` while the
    /// denominator `cos H' · sin φ − tan δ' · cos φ` is even, so on the
    /// equator (`φ = 0`, where the denominator collapses to
    /// `−tan δ' · cos φ` and is `H'`-independent) `Γ(−H')` must equal
    /// `−Γ(H')` modulo `360°`. A typo flipping a sign on `sin H'` or
    /// using `|sin H'|` would break this anti-symmetry. The sweep covers
    /// diverse `H'` magnitudes so a coincidence at any single point
    /// cannot mask the bug; the equator removes the `cos H' · sin φ` term
    /// so the test isolates the numerator's `sin H'` factor.
    #[test]
    fn astronomers_azimuth_is_odd_in_h_prime_on_equator() {
        let phi = 0.0_f64;
        let delta_prime = 10.0_f64;
        for &h_prime in &[1.0_f64, 30.0, 89.0, 175.0] {
            let plus = astronomers_azimuth(h_prime, phi, delta_prime);
            let minus = astronomers_azimuth(-h_prime, phi, delta_prime);
            let sum = (plus + minus).rem_euclid(360.0);
            assert!(
                sum < 1e-12 || (sum - 360.0).abs() < 1e-12,
                "Γ(H') + Γ(-H') must vanish mod 360° on the equator for H' = {h_prime}: \
                 got plus = {plus}, minus = {minus}, sum mod 360 = {sum}",
            );
        }
    }

    /// Step 3.15.2 mandates the wrap into `[0°, 360°)` for `Φ`. Equation
    /// 46 is `Φ = Γ + 180°`, so for `Γ ∈ [0°, 360°)` the raw sum lies in
    /// `[180°, 540°)`: the lower half (`Γ < 180°`) passes through; the
    /// upper half (`Γ ≥ 180°`) needs the wrap to lift it back into range.
    /// The sweep covers both halves and the boundaries to exercise both
    /// branches of `limit_degrees`.
    #[test]
    fn topocentric_azimuth_angle_wraps_both_halves_of_the_input_range() {
        for &(gamma, expected) in &[
            (0.0_f64, 180.0_f64),
            (90.0, 270.0),
            (179.999, 359.999),
            (180.0, 0.0),
            (270.0, 90.0),
            (359.999, 179.999),
        ] {
            let phi = topocentric_azimuth_angle(gamma);
            assert!(
                (phi - expected).abs() < 1e-12,
                "Φ at Γ = {gamma}° must equal {expected}°: got {phi}",
            );
            assert!(
                (0.0..360.0).contains(&phi),
                "Φ escaped [0°, 360°) for Γ = {gamma}°: got {phi}",
            );
        }
    }
}
