// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Observer local hour angle (`H`) and topocentric local hour angle (`H'`).
//!
//! Implements sections 3.11 and 3.13 of NREL/TP-560-34302. Equation 32
//! pairs the upstream apparent sidereal time `ν` (Earth's instantaneous
//! orientation at Greenwich) with the observer's geographical longitude
//! `σ` and subtracts the sun's geocentric right ascension `α`. The result
//! is reduced into `[0°, 360°)` per step 3.11 and measured westward from
//! south. Equation 40 then refines `H` for the observer's local position
//! by subtracting the parallax in right ascension `Δα` produced by section
//! 3.12; the topocentric `H'` shares the same westward-from-south
//! convention.

use crate::helper::limit_degrees;

/// Observer local hour angle `H` (degrees), wrapped into `[0°, 360°)`.
///
/// `apparent_sidereal_time` is `ν` (in degrees), as produced by
/// [`apparent_sidereal_time`]. `observer_longitude` is `σ` (in degrees),
/// **measured positive east of Greenwich** (see "Sign convention" below).
/// `geocentric_right_ascension` is `α` (in degrees), as produced by
/// [`geocentric_right_ascension`].
///
/// Refer to section 3.11, equation 32: `H = ν + σ - α`. The output is
/// reduced into `[0°, 360°)` per step 3.11, which itself defers to
/// step 3.2.6's modular wrap, and is measured westward from south.
///
/// # Sign convention
///
/// Section 3.11 specifies `σ` as "positive or negative for east or west
/// of Greenwich, respectively". Passing a sign-flipped `σ` would silently
/// shift `H` by `2·σ` modulo `360°`.
///
/// # Examples
///
/// ```
/// use helioxide::hour_angle::observer_local_hour_angle;
///
/// // Table A5.1: ν ≈ 318.511910°, σ = -105.1786°, α ≈ 202.22741°
/// // → H ≈ 11.105900°.
/// let h = observer_local_hour_angle(318.511_910, -105.1786, 202.227_41);
/// assert!((h - 11.105_900).abs() < 1e-4);
/// ```
///
/// [`apparent_sidereal_time`]: crate::sidereal::apparent_sidereal_time
/// [`geocentric_right_ascension`]: crate::equatorial::geocentric_right_ascension
#[inline]
#[must_use]
pub const fn observer_local_hour_angle(
    apparent_sidereal_time: f64,
    observer_longitude: f64,
    geocentric_right_ascension: f64,
) -> f64 {
    limit_degrees(apparent_sidereal_time + observer_longitude - geocentric_right_ascension)
}

/// Topocentric local hour angle `H'` (degrees, signed, not wrapped).
///
/// `observer_local_hour_angle` is `H` (in degrees), as produced by
/// [`observer_local_hour_angle`]. `parallax_in_right_ascension` is `Δα`
/// (in degrees), as produced by [`topocentric_equatorial_coordinates`].
///
/// Refer to section 3.13, equation 40: `H' = H − Δα`. Section 3.13 does
/// not mandate a wrap into `[0°, 360°)`. In any physical solar
/// configuration `|Δα| ≤ ξ` (arc-second order, ≤ 0.0025° at one AU), so
/// subtracting `Δα` from the upstream `H ∈ [0°, 360°)` yields a value at
/// most a few arc seconds outside that interval. The downstream consumers
/// (equations 41 and 45) take only `sin H'` and `cos H'`, where any signed
/// excursion across the boundary is absorbed by the periodicity of the
/// trig functions.
///
/// # Examples
///
/// ```
/// use helioxide::hour_angle::topocentric_local_hour_angle;
///
/// // Table A5.1: H = 11.105900°, H' = 11.10629°, so Δα = H − H' = -0.000390°.
/// let h_prime = topocentric_local_hour_angle(11.105_900, -0.000_390);
/// assert!((h_prime - 11.10629).abs() < 1e-5);
/// ```
///
/// [`observer_local_hour_angle`]: self::observer_local_hour_angle
/// [`topocentric_equatorial_coordinates`]: crate::parallax::topocentric_equatorial_coordinates
#[inline]
#[must_use]
pub const fn topocentric_local_hour_angle(
    observer_local_hour_angle: f64,
    parallax_in_right_ascension: f64,
) -> f64 {
    observer_local_hour_angle - parallax_in_right_ascension
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{observer_local_hour_angle, topocentric_local_hour_angle};
    use crate::apparent::{aberration_correction, apparent_sun_longitude};
    use crate::equatorial::{geocentric_declination, geocentric_right_ascension};
    use crate::geocentric::{geocentric_latitude, geocentric_longitude};
    use crate::heliocentric::{
        earth_heliocentric_latitude, earth_heliocentric_longitude, earth_radius_vector,
    };
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::obliquity::true_obliquity_of_ecliptic;
    use crate::parallax::{equatorial_horizontal_parallax, topocentric_equatorial_coordinates};
    use crate::sidereal::{apparent_sidereal_time, mean_sidereal_time};
    use crate::test_fixtures::{
        REFERENCE_ELEVATION_METRES, REFERENCE_LATITUDE_DEGREES, REFERENCE_LONGITUDE_DEGREES,
        reference_jce, reference_jd, reference_jme,
    };

    /// Drives the upstream chain to produce `(ν, α)` at the Table A5.1
    /// reference instant. Two integration paths feed `H`: the sidereal
    /// branch (`JD → ν₀ → ν` via `(Δψ, ε)`) and the equatorial branch
    /// (`JME → (L, B, R) → (Θ, β, R) → (λ, ε) → α` via `(Δψ, Δτ)`).
    /// Both upstream regressions surface in exactly one place.
    fn reference_nu_and_alpha() -> (f64, f64) {
        let jd = reference_jd();
        let jce = reference_jce();
        let jme = reference_jme();

        let (delta_psi, delta_epsilon) = nutation_in_longitude_and_obliquity(jce);
        let epsilon = true_obliquity_of_ecliptic(jme, delta_epsilon);
        let nu = apparent_sidereal_time(mean_sidereal_time(jd), delta_psi, epsilon);

        let theta = geocentric_longitude(earth_heliocentric_longitude(jme));
        let beta = geocentric_latitude(earth_heliocentric_latitude(jme));
        let delta_tau = aberration_correction(earth_radius_vector(jme));
        let lambda = apparent_sun_longitude(theta, delta_psi, delta_tau);
        let alpha = geocentric_right_ascension(lambda, beta, epsilon);

        (nu, alpha)
    }

    /// Drives the full upstream chain (sections 3.2 through 3.12) to
    /// produce `(H, Δα)` at the Table A5.1 reference instant. Pinning the
    /// pair together lets the `H'` reference test surface a regression on
    /// either input in exactly one place.
    fn reference_h_and_delta_alpha() -> (f64, f64) {
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

        (h, topocentric.parallax_in_right_ascension)
    }

    /// `H` at the Table A5.1 reference instant must reproduce the
    /// published `11.105900°` (with `σ = -105.1786°`). Failure is a
    /// global integration red flag: a wrong sign on `σ` (positive-west
    /// instead of positive-east), an `α + σ` instead of `α - σ`, the
    /// missing `[0°, 360°)` wrap, or any upstream regression on `ν` or
    /// `α` would all surface here. The 1e-4 tolerance sits just below
    /// the trailing-digit precision of the published value, so a real
    /// bug cannot hide behind rounding.
    #[test]
    fn observer_local_hour_angle_matches_table_a5_1() {
        let (nu, alpha) = reference_nu_and_alpha();
        let h = observer_local_hour_angle(nu, -105.1786, alpha);
        assert!(
            (h - 11.105_900).abs() < 1e-4,
            "H mismatch at A5.1 reference: got {h}",
        );
    }

    /// Section 3.11 pins `σ` as positive east of Greenwich. Therefore
    /// for any `(ν, α)` and any magnitude `L` the difference
    /// `H(σ = +L) - H(σ = -L)` must equal `2L` modulo `360°`. A bug that
    /// flipped the sign on `σ` would invert this difference, making the
    /// test fail for any non-zero `L`. The sweep covers small
    /// displacements from the prime meridian, the Table A5.1 magnitude
    /// `105.1786°`, and a near-antipodal `179°` (`180°` itself would be
    /// degenerate: `+σ ≡ −σ (mod 360°)` and the property would hold
    /// trivially).
    #[test]
    fn observer_local_hour_angle_treats_observer_longitude_as_positive_east() {
        let nu = 100.0_f64;
        let alpha = 50.0_f64;
        for &sigma in &[10.0_f64, 45.0, 105.1786, 179.0] {
            let h_east = observer_local_hour_angle(nu, sigma, alpha);
            let h_west = observer_local_hour_angle(nu, -sigma, alpha);
            let actual = (h_east - h_west).rem_euclid(360.0);
            let expected = (2.0 * sigma).rem_euclid(360.0);
            assert!(
                (actual - expected).abs() < 1e-12,
                "H(σ=+{sigma}) - H(σ=-{sigma}) must equal 2σ mod 360°: \
                 got {actual} vs expected {expected}",
            );
        }
    }

    /// Step 3.11 mandates the half-open interval `[0°, 360°)` for `H`.
    /// The sweep drives the raw signed sum `ν + σ - α` to both sides of
    /// zero and beyond a full revolution, exercising both branches of
    /// the modular wrap (the negative-remainder lift and the
    /// already-non-negative pass-through). Without the wrap the function
    /// would return the raw signed sum.
    #[test]
    fn observer_local_hour_angle_is_wrapped_into_zero_360() {
        for &(nu, sigma, alpha) in &[
            (0.0_f64, -180.0, 0.0), // raw -180°: negative remainder
            (10.0, -180.0, 50.0),   // raw -220°: negative remainder, > one turn below
            (350.0, 180.0, 0.0),    // raw  530°: positive remainder, > one turn above
            (720.0, 0.0, -100.0),   // raw  820°: positive remainder, > two turns above
            (-100.0, 0.0, 0.0),     // raw -100°: negative remainder, sub-turn below
        ] {
            let h = observer_local_hour_angle(nu, sigma, alpha);
            assert!(
                (0.0..360.0).contains(&h),
                "H escaped [0°, 360°) for (ν, σ, α) = ({nu}, {sigma}, {alpha}): got {h}",
            );
        }
    }

    /// Equation 32 sums `ν + σ - α` linearly. Shifting one input by `δ`
    /// while holding the others fixed must shift `H` by `+δ`, `+δ`, `-δ`
    /// respectively. This isolates each input's coefficient and sign
    /// from the others: a missing addend, a wrong sign on `σ` or `α`,
    /// or a stray multiplicative factor would fail here even if the
    /// A5.1 reference test still passed (because at one specific
    /// instant compensating typos can mask the bug). The baseline
    /// `(200°, 50°, 100°)` keeps the raw sum well inside `[0°, 360°)`
    /// so the small `δ` values do not trip the wrap, isolating the
    /// linear sum from `limit_degrees`.
    #[test]
    fn observer_local_hour_angle_is_linear_in_each_input_inside_range() {
        let baseline_nu = 200.0_f64;
        let baseline_sigma = 50.0_f64;
        let baseline_alpha = 100.0_f64;
        let baseline = observer_local_hour_angle(baseline_nu, baseline_sigma, baseline_alpha);
        for &delta in &[-1.0_f64, -1e-3, 1e-6, 0.5] {
            let from_nu =
                observer_local_hour_angle(baseline_nu + delta, baseline_sigma, baseline_alpha);
            let from_sigma =
                observer_local_hour_angle(baseline_nu, baseline_sigma + delta, baseline_alpha);
            let from_alpha =
                observer_local_hour_angle(baseline_nu, baseline_sigma, baseline_alpha + delta);
            assert!(
                (from_nu - baseline - delta).abs() < 1e-13,
                "H must shift by +δ when ν shifts by δ = {delta}: \
                 got {from_nu} - {baseline} ≠ {delta}",
            );
            assert!(
                (from_sigma - baseline - delta).abs() < 1e-13,
                "H must shift by +δ when σ shifts by δ = {delta}: \
                 got {from_sigma} - {baseline} ≠ {delta}",
            );
            assert!(
                (from_alpha - baseline + delta).abs() < 1e-13,
                "H must shift by -δ when α shifts by δ = {delta}: \
                 got {from_alpha} - {baseline} ≠ {expected}",
                expected = -delta,
            );
        }
    }

    /// `H'` at the Table A5.1 reference instant must reproduce the
    /// published `11.10629°`. Section 3.13's one-line refinement is the
    /// terminal consumer of `Δα`, so failure pins both the equation 40
    /// subtraction and the upstream `(H, Δα)` chain. A wrong sign on
    /// `Δα` (i.e. `H + Δα`) shifts `H'` by `~8·10⁻⁴°` at this instant —
    /// well above the 1e-4° tolerance — so it cannot survive even though
    /// `|Δα|` itself is `~4·10⁻⁴°`.
    #[test]
    fn topocentric_local_hour_angle_matches_table_a5_1() {
        let (h, delta_alpha) = reference_h_and_delta_alpha();
        let h_prime = topocentric_local_hour_angle(h, delta_alpha);
        assert!(
            (h_prime - 11.106_29).abs() < 1e-4,
            "H' mismatch at A5.1 reference: got {h_prime}",
        );
    }

    /// Equation 40 subtracts `Δα` from `H` linearly. Shifting one input
    /// by `δ` while holding the other fixed must shift `H'` by `+δ` for
    /// `H` and `-δ` for `Δα`. This isolates each input's coefficient and
    /// sign from the other: a wrong sign on `Δα` (`H + Δα`), a wrong sign
    /// on `H`, a swap of the two inputs, or a stray multiplicative
    /// factor would all fail here even when the A5.1 reference test
    /// still passes (because at one specific instant compensating typos
    /// can mask the bug). The baseline `(H = 100°, Δα = 10⁻³°)` keeps the
    /// raw difference well inside `[0°, 360°)` so the small `δ` values
    /// do not interact with any boundary behaviour.
    #[test]
    fn topocentric_local_hour_angle_is_linear_in_each_input() {
        let baseline_h = 100.0_f64;
        let baseline_delta_alpha = 1e-3_f64;
        let baseline = topocentric_local_hour_angle(baseline_h, baseline_delta_alpha);
        for &delta in &[-1.0_f64, -1e-4, 1e-6, 0.5] {
            let from_h = topocentric_local_hour_angle(baseline_h + delta, baseline_delta_alpha);
            let from_delta_alpha =
                topocentric_local_hour_angle(baseline_h, baseline_delta_alpha + delta);
            assert!(
                (from_h - baseline - delta).abs() < 1e-13,
                "H' must shift by +δ when H shifts by δ = {delta}: \
                 got {from_h} - {baseline} ≠ {delta}",
            );
            assert!(
                (from_delta_alpha - baseline + delta).abs() < 1e-13,
                "H' must shift by -δ when Δα shifts by δ = {delta}: \
                 got {from_delta_alpha} - {baseline} ≠ {expected}",
                expected = -delta,
            );
        }
    }

    /// Section 3.13 does not mandate a wrap into `[0°, 360°)` (contrast
    /// section 3.11, which explicitly invokes step 3.2.6). Driving `H − Δα`
    /// below zero (small positive `H`, larger positive `Δα`) and above
    /// `360°` (`H` near the wrap, negative `Δα`) must therefore produce
    /// the raw signed value, not a `limit_degrees`-lifted one. A future
    /// contributor adding a wrap "to be safe" would silently shift `H'` by
    /// `360°` at these inputs; the downstream `sin`/`cos` consumers would
    /// still agree, masking the contract change everywhere except this
    /// test.
    #[test]
    fn topocentric_local_hour_angle_does_not_wrap_into_zero_360() {
        for &(h, delta_alpha, expected) in &[
            (1e-4_f64, 1.0, 1e-4 - 1.0), // raw < 0
            (359.9, -1.0, 360.9),        // raw ≥ 360
        ] {
            let h_prime = topocentric_local_hour_angle(h, delta_alpha);
            assert!(
                (h_prime - expected).abs() < 1e-13,
                "H' must remain signed (no wrap into [0°, 360°)) for \
                 (H, Δα) = ({h}, {delta_alpha}): got {h_prime} vs expected {expected}",
            );
        }
    }
}
