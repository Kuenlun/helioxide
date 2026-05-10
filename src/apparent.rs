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

//! Apparent sun longitude (`λ`) and aberration correction (`Δτ`).
//!
//! Implements sections 3.6 and 3.7 of NREL/TP-560-34302. The aberration
//! correction `Δτ` (equation 26) accounts for the apparent westward
//! displacement of the Sun caused by the finite speed of light over the
//! Earth-Sun distance, and the apparent longitude `λ` (equation 27) is the
//! geocentric longitude `Θ` corrected for both the nutation in longitude
//! `Δψ` and the aberration `Δτ`.

/// Constant of aberration in arc seconds — the apparent westward displacement
/// of the Sun at one astronomical unit, caused by the finite speed of light.
/// Reproduced verbatim from equation 26.
const ABERRATION_CONSTANT_ARCSECONDS: f64 = 20.4898;

/// Aberration correction `Δτ` (degrees).
///
/// `earth_radius_vector` is the Earth radius vector `R` (in astronomical
/// units), as produced by [`earth_radius_vector`]; it is the Earth-Sun
/// distance and is therefore strictly positive in any physical situation.
///
/// Refer to section 3.6, equation 26: `Δτ = -20.4898" / (3600 · R)`. The
/// published constant lives in arc seconds, hence the `1 / 3600` factor
/// that brings it to degrees. The negative sign reflects the apparent
/// westward displacement of the Sun.
///
/// # Examples
///
/// ```
/// use helioxide::apparent::aberration_correction;
///
/// // Table A5.1: R = 0.9965422974 AU → Δτ ≈ -0.005711359°.
/// let delta_tau = aberration_correction(0.996_542_297_4);
/// assert!((delta_tau - -0.005_711_359).abs() < 1e-9);
/// ```
///
/// [`earth_radius_vector`]: crate::heliocentric::earth_radius_vector
#[inline]
#[must_use]
pub const fn aberration_correction(earth_radius_vector: f64) -> f64 {
    -ABERRATION_CONSTANT_ARCSECONDS / (3600.0 * earth_radius_vector)
}

/// Apparent sun longitude `λ` (degrees, signed, not range-limited).
///
/// `geocentric_longitude` is the sun geocentric longitude `Θ` (in degrees),
/// as produced by [`geocentric_longitude`]. `delta_psi` is the nutation in
/// longitude `Δψ` (in degrees), as produced by
/// [`nutation_in_longitude_and_obliquity`]. `delta_tau` is the aberration
/// correction `Δτ` (in degrees), as produced by [`aberration_correction`].
///
/// Refer to section 3.7, equation 27: `λ = Θ + Δψ + Δτ`. The two corrections
/// are summed first, so that `Θ` (≈ 200°) absorbs them in a single
/// `ulp(Θ) ≈ 3·10⁻¹⁴` rounding step instead of two: `Δψ` and `Δτ` share
/// magnitude (≈ 10⁻³°) and combine cleanly, whereas a left-to-right
/// `(Θ + Δψ) + Δτ` would round each tiny addend against `Θ` independently.
///
/// # Examples
///
/// ```
/// use helioxide::apparent::{aberration_correction, apparent_sun_longitude};
///
/// // Table A5.1: Θ = 204.0182616917°, Δψ = -0.00399840°, R = 0.9965422974 AU
/// // → λ = 204.0085519281°.
/// let delta_tau = aberration_correction(0.996_542_297_4);
/// let lambda = apparent_sun_longitude(204.018_261_691_7, -0.003_998_40, delta_tau);
/// assert!((lambda - 204.008_551_928_1).abs() < 1e-6);
/// ```
///
/// [`geocentric_longitude`]: crate::geocentric::geocentric_longitude
/// [`nutation_in_longitude_and_obliquity`]: crate::nutation::nutation_in_longitude_and_obliquity
/// [`aberration_correction`]: self::aberration_correction
#[inline]
#[must_use]
pub const fn apparent_sun_longitude(
    geocentric_longitude: f64,
    delta_psi: f64,
    delta_tau: f64,
) -> f64 {
    geocentric_longitude + (delta_psi + delta_tau)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{aberration_correction, apparent_sun_longitude};
    use crate::geocentric::geocentric_longitude;
    use crate::heliocentric::{earth_heliocentric_longitude, earth_radius_vector};
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::test_fixtures::{reference_jce, reference_jme};

    /// `λ` at the Table A5.1 reference instant must reproduce the published
    /// `204.0085519281°`. Failure is an integration-level red flag: a wrong
    /// sign on `Δτ`, a missing arc-second-to-degree conversion in equation
    /// 26, a swapped addend in equation 27, or any upstream regression on
    /// `Θ`, `Δψ` or `R` would all surface here. The 1e-6 tolerance sits just
    /// below the trailing-digit precision of the published value, so a real
    /// bug cannot hide behind rounding.
    #[test]
    fn apparent_sun_longitude_matches_table_a5_1() {
        let jme = reference_jme();
        let theta = geocentric_longitude(earth_heliocentric_longitude(jme));
        let (delta_psi, _) = nutation_in_longitude_and_obliquity(reference_jce());
        let delta_tau = aberration_correction(earth_radius_vector(jme));

        let lambda = apparent_sun_longitude(theta, delta_psi, delta_tau);
        assert!(
            (lambda - 204.008_551_928_1).abs() < 1e-6,
            "λ mismatch at A5.1 reference: got {lambda}",
        );
    }

    /// At `R = 1 AU` equation 26 collapses to `Δτ = -20.4898 / 3600°`. This
    /// isolates the literal aberration constant and the `3600`
    /// arc-second-to-degree divisor from the `R` dependence: a stray digit
    /// in `20.4898` or a wrong divisor would fail here even when the A5.1
    /// reference test still passes (since the latter exercises only one
    /// specific R, where a compensating typo upstream could mask the bug).
    #[test]
    fn aberration_correction_at_unit_distance_equals_published_constant_in_degrees() {
        let dt = aberration_correction(1.0);
        let expected = -20.4898 / 3600.0;
        assert!(
            (dt - expected).abs() < f64::EPSILON,
            "Δτ at R = 1 AU must equal -20.4898 / 3600°: got {dt}",
        );
    }

    /// Equation 26 mandates `Δτ ∝ 1/R`; therefore `Δτ(R) · R` must be a
    /// constant equal to `Δτ(1)`. A typo such as `R²`, `R + 1`, or any other
    /// non-linear-in-R denominator would break this invariant. The sweep
    /// covers values inside Earth's annual range (≈ [0.983, 1.017] AU) and
    /// well outside it so the proportionality is pinned across orders of
    /// magnitude rather than at a single operating point.
    #[test]
    fn aberration_correction_is_inversely_proportional_to_earth_radius_vector() {
        let dt_unit = aberration_correction(1.0);
        for &r in &[0.5_f64, 0.95, 1.05, 2.0, 10.0] {
            let dt = aberration_correction(r);
            assert!(
                dt.mul_add(r, -dt_unit).abs() < 1e-15,
                "Δτ · R must equal Δτ(1) for R = {r}: got {dt}",
            );
        }
    }

    /// Equation 27 sums `Θ`, `Δψ` and `Δτ` unchanged. Calling the function
    /// with the same `Δτ` but a shifted `Θ` (resp. `Δψ`) must shift the
    /// output by exactly the same delta: this isolates the addition step
    /// from the aberration term, so a sign flip or a wrong coefficient on
    /// either addend fails here even when the A5.1 reference test still
    /// passes (because at one specific input the summands could happen to
    /// compensate). The sweep includes both signs and several magnitudes so
    /// a typo cannot hide behind the comparison tolerance.
    #[test]
    fn apparent_sun_longitude_shifts_linearly_with_theta_and_delta_psi() {
        let delta_tau = aberration_correction(1.0);
        let baseline = apparent_sun_longitude(0.0, 0.0, delta_tau);
        for &delta in &[-1.0_f64, -1e-3, 1e-6, 0.5] {
            let shifted_theta = apparent_sun_longitude(delta, 0.0, delta_tau);
            let shifted_psi = apparent_sun_longitude(0.0, delta, delta_tau);
            assert!(
                (shifted_theta - baseline - delta).abs() < 1e-13,
                "λ must shift by Δ in Θ; got {shifted_theta} - {baseline} ≠ {delta}",
            );
            assert!(
                (shifted_psi - baseline - delta).abs() < 1e-13,
                "λ must shift by Δ in Δψ; got {shifted_psi} - {baseline} ≠ {delta}",
            );
        }
    }
}
