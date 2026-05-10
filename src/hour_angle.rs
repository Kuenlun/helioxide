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

//! Observer local hour angle (`H`).
//!
//! Implements section 3.11 of NREL/TP-560-34302. Equation 32 pairs the
//! upstream apparent sidereal time `ν` (Earth's instantaneous orientation
//! at Greenwich) with the observer's geographical longitude `σ` and
//! subtracts the sun's geocentric right ascension `α`. The result is
//! reduced into `[0°, 360°)` per step 3.11 and measured westward from
//! south.

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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::observer_local_hour_angle;
    use crate::apparent::{aberration_correction, apparent_sun_longitude};
    use crate::equatorial::geocentric_right_ascension;
    use crate::geocentric::{geocentric_latitude, geocentric_longitude};
    use crate::heliocentric::{
        earth_heliocentric_latitude, earth_heliocentric_longitude, earth_radius_vector,
    };
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::obliquity::true_obliquity_of_ecliptic;
    use crate::sidereal::{apparent_sidereal_time, mean_sidereal_time};
    use crate::test_fixtures::{reference_jce, reference_jd, reference_jme};

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
}
