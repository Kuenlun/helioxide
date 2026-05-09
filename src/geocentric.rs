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

//! Sun geocentric longitude (`Θ`) and latitude (`β`).
//!
//! Implements section 3.3 of NREL/TP-560-34302. The geocentric coordinates
//! place the Sun in a frame centred on the Earth: rotating the Earth
//! heliocentric longitude `L` by 180° (equation 13) and inverting the Earth
//! heliocentric latitude `B` (equation 14) yields the position of the Sun
//! as seen from the Earth's centre.

use crate::helper::limit_degrees;

/// Sun geocentric longitude `Θ` (degrees), wrapped into `[0°, 360°)`.
///
/// `heliocentric_longitude` is the Earth heliocentric longitude `L` (in
/// degrees), as produced by [`earth_heliocentric_longitude`].
///
/// Refer to section 3.3, steps 3.3.1 and 3.3.2 (equation 13).
///
/// # Examples
///
/// ```
/// use helioxide::geocentric::geocentric_longitude;
///
/// // Table A5.1: L = 24.0182616917° → Θ = 204.0182616917°.
/// let theta = geocentric_longitude(24.018_261_691_7);
/// assert!((theta - 204.018_261_691_7).abs() < 1e-9);
/// ```
///
/// [`earth_heliocentric_longitude`]: crate::heliocentric::earth_heliocentric_longitude
#[inline]
#[must_use]
pub const fn geocentric_longitude(heliocentric_longitude: f64) -> f64 {
    limit_degrees(heliocentric_longitude + 180.0)
}

/// Sun geocentric latitude `β` (degrees, signed, not range-limited).
///
/// `heliocentric_latitude` is the Earth heliocentric latitude `B` (in
/// degrees), as produced by [`earth_heliocentric_latitude`].
///
/// Refer to section 3.3, step 3.3.3 (equation 14).
///
/// # Examples
///
/// ```
/// # #![allow(clippy::float_cmp)] // f64 negation only flips the sign bit, so equality is bit-exact.
/// use helioxide::geocentric::geocentric_latitude;
///
/// // Table A5.1: B = -0.0001011219° → β = 0.0001011219°.
/// let beta = geocentric_latitude(-0.000_101_121_9);
/// assert_eq!(beta, 0.000_101_121_9);
/// ```
///
/// [`earth_heliocentric_latitude`]: crate::heliocentric::earth_heliocentric_latitude
#[inline]
#[must_use]
pub const fn geocentric_latitude(heliocentric_latitude: f64) -> f64 {
    -heliocentric_latitude
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{geocentric_latitude, geocentric_longitude};
    use crate::heliocentric::{earth_heliocentric_latitude, earth_heliocentric_longitude};
    use crate::test_fixtures::reference_jme;

    /// `Θ` at the Table A5.1 reference instant must reproduce the published
    /// `204.0182616917°`. Feeding the actual upstream `L` from
    /// [`earth_heliocentric_longitude`] also exercises the integration with
    /// section 3.2, ensuring that nothing is lost between the two stages.
    #[test]
    fn geocentric_longitude_matches_table_a5_1() {
        let l = earth_heliocentric_longitude(reference_jme());
        let theta = geocentric_longitude(l);
        assert!(
            (theta - 204.018_261_691_7).abs() < 1e-6,
            "Θ mismatch at A5.1 reference: got {theta}",
        );
    }

    /// `β` at the Table A5.1 reference instant must reproduce the published
    /// `+0.0001011219°` against an upstream `B = -0.0001011219°`. The
    /// magnitude must survive the negation and, crucially, the sign must
    /// flip — so we assert the strict-positivity in addition to the value.
    #[test]
    fn geocentric_latitude_matches_table_a5_1() {
        let b = earth_heliocentric_latitude(reference_jme());
        let beta = geocentric_latitude(b);
        assert!(
            (beta - 0.000_101_121_9).abs() < 1e-9,
            "β mismatch at A5.1 reference: got {beta}",
        );
        assert!(beta > 0.0, "β must flip the sign of B; got {beta}");
    }

    /// Step 3.3.2 mandates the half-open interval `[0°, 360°)` for `Θ`. The
    /// sweep brackets the wrap point at `L = 180°` and includes negative
    /// and large inputs so that `limit_degrees` is exercised even when
    /// `L + 180` lands well outside `[0°, 360°)`.
    #[test]
    fn geocentric_longitude_is_wrapped_into_zero_360() {
        for &l in &[0.0_f64, 90.0, 179.999, 180.0, 181.0, 359.999, -45.0, 720.5] {
            let theta = geocentric_longitude(l);
            assert!(
                (0.0..360.0).contains(&theta),
                "Θ escaped [0, 360) for L = {l}: got {theta}",
            );
        }
    }

    /// `Θ ≡ L + 180 (mod 360)`: the geometric meaning of equation 13 (the
    /// Sun and the Earth sit on opposite sides of the line of sight).
    /// Pinning this identity catches sign flips and constant typos that an
    /// ulp-level reference comparison at one JME might tolerate.
    #[test]
    fn geocentric_longitude_is_input_offset_by_180_modulo_360() {
        for &l in &[0.0_f64, 30.0, 179.999, 180.0, 200.0, 350.0] {
            let theta = geocentric_longitude(l);
            // Map the residue into `[0°, 360°)` and collapse the
            // wrap-around case so the tolerance is symmetric around zero.
            let residue = (theta - l - 180.0).rem_euclid(360.0);
            let distance_to_zero = residue.min(360.0 - residue);
            assert!(
                distance_to_zero < 1e-12,
                "Θ - L ≢ 180 (mod 360) for L = {l}: got Θ = {theta}",
            );
        }
    }

    /// `β = -B` is unconditional. Sweeping over both signs and both signed
    /// zeros ensures no spurious branch (e.g. an `if B > 0` shortcut) was
    /// introduced. Direct equality is meaningful here: f64 negation only
    /// flips the sign bit, so the result is bit-exact.
    #[test]
    #[allow(clippy::float_cmp)]
    fn geocentric_latitude_negates_input_unconditionally() {
        for &b in &[-1.0_f64, -1e-6, -0.0, 0.0, 1e-6, 1.0] {
            assert_eq!(geocentric_latitude(b), -b, "β must equal -B for B = {b}");
        }
    }
}
