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

//! Sun mean longitude (`M`) and equation of time (`E`).
//!
//! Implements appendix A.1 of NREL/TP-560-34302. The sun mean longitude
//! `M` (equation A2) is a fifth-degree polynomial in the Julian Ephemeris
//! Millennium that locates the sun's mean position along the ecliptic;
//! step 3.2.6 wraps the result into `[0°, 360°)`. The equation of time
//! `E` (equation A1) is the difference between solar apparent and mean
//! time. It is first expressed in degrees and then scaled to minutes of
//! time by the `1° ↔ 4 min` factor mandated by the same section. A final
//! `±1440` shift collapses the discrete `±360°` ambiguity left over by
//! `M` and `α` having been independently wrapped into `[0°, 360°)`.

use crate::helper::limit_degrees;

/// Empirical correction `0.0057183°` of equation A1: the small offset
/// between the sun's mean and apparent longitudes that is not already
/// absorbed by the explicit `α` and `Δψ · cos ε` terms. Reproduced
/// verbatim from the appendix.
const APPARENT_MEAN_LONGITUDE_CORRECTION_DEGREES: f64 = 0.005_718_3;

/// `1° ↔ 4 minutes of time`: Earth rotates `360°` in `24 · 60 = 1440`
/// minutes, hence `1°` corresponds to `4` minutes of time.
const MINUTES_OF_TIME_PER_DEGREE: f64 = 4.0;

/// Minutes of time per full revolution (`360°`). `M` and `α` are each
/// independently wrapped into `[0°, 360°)` upstream, so the raw
/// `M − α` can land off by a full revolution; in minutes of time that
/// off-by-one is exactly `±1440`. Used to fold the result back into
/// the physical `[−20 min, +20 min]` band of the equation of time.
const MINUTES_OF_TIME_PER_REVOLUTION: f64 = 1440.0;

/// Empirical bound `|E| ≤ 20 min` of the appendix; the equation of
/// time stays within `~±16 min` over the algorithm's validity window,
/// so any computed value above this band is the residual `±1440`
/// ambiguity and must be brought back inside.
const EQUATION_OF_TIME_BOUND_MINUTES: f64 = 20.0;

/// Polynomial coefficients of `M` (degrees) in `JME`, ordered from the
/// constant term up to `JME⁵`. Reproduced verbatim from equation A2.
const SUN_MEAN_LONGITUDE_COEFFICIENTS: [f64; 6] = [
    280.466_456_7,
    360_007.698_277_9,
    0.030_320_28,
    1.0 / 49_931.0,
    -1.0 / 15_300.0,
    -1.0 / 2_000_000.0,
];

/// Sun mean longitude `M` (degrees), wrapped into `[0°, 360°)`.
///
/// `jme` is the Julian Ephemeris Millennium, as produced by
/// [`calculate_julian_ephemeris_millennium`].
///
/// Refer to appendix A.1, equation A2:
/// `M = 280.4664567 + 360007.6982779 · JME + 0.03032028 · JME²`
/// `  + JME³ / 49931 − JME⁴ / 15300 − JME⁵ / 2000000`,
/// limited to `[0°, 360°)` per step 3.2.6. The polynomial is evaluated
/// by Horner's method, so each coefficient is folded against the
/// running product in a single `mul_add`, keeping the dominant
/// `360_007.6982779 · JME` linear term (which reaches `~10⁶°` at the
/// edges of the SPA validity window) precise against the much smaller
/// higher-order corrections.
///
/// # Examples
///
/// ```
/// use helioxide::equation_of_time::sun_mean_longitude;
/// use helioxide::julian::{
///     calculate_julian_ephemeris_century, calculate_julian_ephemeris_day,
///     calculate_julian_ephemeris_millennium,
/// };
///
/// // Table A5.1 reference: 17 October 2003, 12:30:30 LST, ΔT = 67 s.
/// let jde = calculate_julian_ephemeris_day(2_452_930.312_847, 67.0);
/// let jce = calculate_julian_ephemeris_century(jde);
/// let jme = calculate_julian_ephemeris_millennium(jce);
///
/// let m = sun_mean_longitude(jme);
/// assert!((m - 205.897_172_2).abs() < 1e-6);
/// ```
///
/// [`calculate_julian_ephemeris_millennium`]: crate::julian::calculate_julian_ephemeris_millennium
#[must_use]
pub fn sun_mean_longitude(jme: f64) -> f64 {
    let raw = SUN_MEAN_LONGITUDE_COEFFICIENTS
        .iter()
        .rev()
        .fold(0.0_f64, |acc, &coefficient| acc.mul_add(jme, coefficient));
    limit_degrees(raw)
}

/// Equation of time `E` (minutes of time).
///
/// `sun_mean_longitude` is `M` (in degrees, wrapped into `[0°, 360°)`),
/// as produced by [`sun_mean_longitude`]. `geocentric_right_ascension`
/// is `α` (in degrees, wrapped into `[0°, 360°)`), as produced by
/// [`geocentric_right_ascension`]. `delta_psi` is the nutation in
/// longitude `Δψ` (in degrees), as produced by
/// [`nutation_in_longitude_and_obliquity`]. `true_obliquity` is `ε`
/// (in degrees), as produced by [`true_obliquity_of_ecliptic`].
///
/// Refer to appendix A.1, equation A1:
/// `E = M − 0.0057183 − α + Δψ · cos ε`. The result is then multiplied
/// by `4` to convert from degrees to minutes of time, after which the
/// appendix mandates folding the value back into the physical
/// `[−20, +20]` minute band by adding or subtracting `1440` once. The
/// fold is required because `M` and `α` were each independently wrapped
/// into `[0°, 360°)` upstream: when one sits just above the boundary
/// and the other just below, the raw difference is off by one
/// revolution, which is exactly `±1440` minutes of time. The actual
/// equation of time stays within `~±16 min` over the algorithm's
/// validity window, so a single `±1440` shift is enough.
///
/// The two corrections `Δψ · cos ε` and `−0.0057183` share magnitude
/// (`~10⁻³°`), so they are combined first inside a single `mul_add`
/// (rounding the multiplication and the constant subtraction together)
/// before being added to the dominant `M − α` difference.
///
/// # Examples
///
/// ```
/// use helioxide::equation_of_time::equation_of_time;
///
/// // Table A5.1: M = 205.8971722516°, α = 202.22741°,
/// // Δψ = -0.00399840°, ε = 23.440465° → E ≈ 14.641503 min.
/// let e = equation_of_time(205.897_172_251_6, 202.227_41, -0.003_998_40, 23.440_465);
/// assert!((e - 14.641_503).abs() < 1e-4);
/// ```
///
/// [`sun_mean_longitude`]: self::sun_mean_longitude
/// [`geocentric_right_ascension`]: crate::equatorial::geocentric_right_ascension
/// [`nutation_in_longitude_and_obliquity`]: crate::nutation::nutation_in_longitude_and_obliquity
/// [`true_obliquity_of_ecliptic`]: crate::obliquity::true_obliquity_of_ecliptic
#[inline]
#[must_use]
pub fn equation_of_time(
    sun_mean_longitude: f64,
    geocentric_right_ascension: f64,
    delta_psi: f64,
    true_obliquity: f64,
) -> f64 {
    let cos_epsilon = true_obliquity.to_radians().cos();
    let nutation_correction =
        delta_psi.mul_add(cos_epsilon, -APPARENT_MEAN_LONGITUDE_CORRECTION_DEGREES);
    let degrees = sun_mean_longitude - geocentric_right_ascension + nutation_correction;
    let minutes = degrees * MINUTES_OF_TIME_PER_DEGREE;

    if minutes.abs() > EQUATION_OF_TIME_BOUND_MINUTES {
        minutes - MINUTES_OF_TIME_PER_REVOLUTION.copysign(minutes)
    } else {
        minutes
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{APPARENT_MEAN_LONGITUDE_CORRECTION_DEGREES, equation_of_time, sun_mean_longitude};
    use crate::apparent::{aberration_correction, apparent_sun_longitude};
    use crate::equatorial::geocentric_right_ascension;
    use crate::geocentric::{geocentric_latitude, geocentric_longitude};
    use crate::heliocentric::{
        earth_heliocentric_latitude, earth_heliocentric_longitude, earth_radius_vector,
    };
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::obliquity::true_obliquity_of_ecliptic;
    use crate::test_fixtures::{reference_jce, reference_jme};

    /// Drives the upstream chain (sections 3.2 through 3.10 plus equation A2)
    /// to produce `(M, α, Δψ, ε)` at the Table A5.1 reference instant.
    /// Any upstream regression on `M`, `α`, `Δψ`, or `ε` surfaces in
    /// exactly one place via the `equation_of_time` reference test.
    fn reference_inputs() -> (f64, f64, f64, f64) {
        let jce = reference_jce();
        let jme = reference_jme();

        let m = sun_mean_longitude(jme);
        let (delta_psi, delta_epsilon) = nutation_in_longitude_and_obliquity(jce);
        let epsilon = true_obliquity_of_ecliptic(jme, delta_epsilon);

        let theta = geocentric_longitude(earth_heliocentric_longitude(jme));
        let beta = geocentric_latitude(earth_heliocentric_latitude(jme));
        let delta_tau = aberration_correction(earth_radius_vector(jme));
        let lambda = apparent_sun_longitude(theta, delta_psi, delta_tau);
        let alpha = geocentric_right_ascension(lambda, beta, epsilon);

        (m, alpha, delta_psi, epsilon)
    }

    /// `M` at the Table A5.1 reference instant must reproduce the
    /// published `205.8971722516°`. Failure pins a typo in the equation
    /// A2 polynomial coefficients, a missing `[0°, 360°)` wrap (the raw
    /// polynomial value is `~1645.9°` at this JME, so it actually does
    /// cross one full revolution), or a regression in the upstream JME
    /// chain. The `1e-6` tolerance sits well above the trailing-digit
    /// precision of the published value, absorbing roundoff without
    /// masking a real bug.
    #[test]
    fn sun_mean_longitude_matches_table_a5_1() {
        let m = sun_mean_longitude(reference_jme());
        assert!(
            (m - 205.897_172_251_6).abs() < 1e-6,
            "M mismatch at A5.1 reference: got {m}",
        );
    }

    /// At `JME = 0` (J2000.0 epoch) every monomial above the constant
    /// term vanishes, so the polynomial collapses to its constant
    /// coefficient `280.4664567°`, already inside `[0°, 360°)`, so the
    /// wrap is the identity. This isolates the constant term from every
    /// other coefficient and from the wrap: a stray digit in
    /// `280.4664567` would fail here even when the A5.1 reference test
    /// still passes by happy accident on a single non-zero JME.
    #[test]
    fn sun_mean_longitude_at_j2000_collapses_to_constant_term() {
        let m = sun_mean_longitude(0.0);
        assert!(
            (m - 280.466_456_7).abs() < f64::EPSILON,
            "M at JME = 0 must equal 280.4664567°: got {m}",
        );
    }

    /// Step 3.2.6 mandates the half-open interval `[0°, 360°)` for `M`.
    /// The sweep covers JME values across the SPA validity window
    /// (≈ -2000 to 6000, i.e. JME ∈ [-4, 4]) including both signs and
    /// magnitudes large enough to drive the dominant `360_007.7 · JME`
    /// linear term well past a single revolution, stressing both the
    /// negative-remainder and the positive-remainder branches of
    /// `limit_degrees` from this call site.
    #[test]
    fn sun_mean_longitude_is_wrapped_into_zero_360() {
        for &jme in &[-4.0, -1.0, -1e-3, 0.0, 1e-3, 1.0, 4.0] {
            let m = sun_mean_longitude(jme);
            assert!(
                (0.0..360.0).contains(&m),
                "M escaped [0°, 360°) at JME = {jme}: got {m}",
            );
        }
    }

    /// `E` at the Table A5.1 reference instant must reproduce the
    /// published `14.641503` minutes of time. Failure is a global
    /// integration red flag: a wrong sign on `α`, a wrong coupling on
    /// `Δψ` (e.g. `sin ε` instead of `cos ε`), a stray digit in the
    /// `0.0057183` correction, a missing `×4` minute conversion, or any
    /// upstream regression on `M`, `α`, `Δψ`, or `ε` would all surface
    /// here. The `1e-4` tolerance sits well above the trailing-digit
    /// precision of the published value, absorbing drift accumulated
    /// through the upstream chain without masking a real bug. The
    /// reference value sits comfortably inside the `±20 min` band, so
    /// this test also pins the no-clamp branch of the appendix's
    /// residual fold.
    #[test]
    fn equation_of_time_matches_table_a5_1() {
        let (m, alpha, delta_psi, epsilon) = reference_inputs();
        let e = equation_of_time(m, alpha, delta_psi, epsilon);
        assert!(
            (e - 14.641_503).abs() < 1e-4,
            "E mismatch at A5.1 reference: got {e}",
        );
    }

    /// With `M = α` and `Δψ = 0`, equation A1 collapses to its empirical
    /// correction: `E = −0.0057183° → −0.0228732 min`. This isolates
    /// the literal `0.0057183` and the `×4` unit conversion from every
    /// other input: a stray digit in the constant or a missing `×4`
    /// would fail here even when the A5.1 reference test still passes
    /// because of compensating typos elsewhere. Sweeping a few values
    /// of `M = α` shows the result is independent of the common value
    /// (the `M − α` term vanishes for any choice).
    #[test]
    fn equation_of_time_collapses_to_correction_constant_when_inputs_neutralize() {
        let expected = -APPARENT_MEAN_LONGITUDE_CORRECTION_DEGREES * 4.0;
        for &m_alpha in &[0.0_f64, 90.0, 200.0] {
            let e = equation_of_time(m_alpha, m_alpha, 0.0, 23.44);
            assert!(
                (e - expected).abs() < 1e-13,
                "E at M = α = {m_alpha}, Δψ = 0 must equal -0.0057183 · 4 min: \
                 got {e} vs expected {expected}",
            );
        }
    }

    /// Equation A1 is linear in `M` (slope `+1°` per degree of input)
    /// and in `α` (slope `−1°` per degree of input). After the `×4`
    /// minute conversion the slopes become `±4 min/°`. Shifting `M` by
    /// `δ` while holding the other inputs fixed must therefore shift
    /// `E` by `+4 · δ`, and shifting `α` by `δ` must shift `E` by
    /// `−4 · δ`. This isolates the `+M` and `−α` coefficients and the
    /// `×4` unit conversion simultaneously: a wrong sign on `α`, a
    /// stray multiplicative factor, or a wrong unit conversion would
    /// all fail here even when the A5.1 reference test still passes
    /// (since at one specific instant compensating typos can mask the
    /// bug). The baseline keeps the result well inside the `±20 min`
    /// band so no `δ` in the sweep crosses the clamp boundary.
    #[test]
    fn equation_of_time_is_linear_in_sun_mean_longitude_and_geocentric_right_ascension() {
        let baseline_m = 100.0_f64;
        let baseline_alpha = 99.0_f64;
        let delta_psi = 0.0_f64;
        let epsilon = 23.44_f64;
        let baseline = equation_of_time(baseline_m, baseline_alpha, delta_psi, epsilon);
        for &delta in &[-1.0_f64, -1e-3, 1e-6, 0.5] {
            let from_m = equation_of_time(baseline_m + delta, baseline_alpha, delta_psi, epsilon);
            let from_alpha =
                equation_of_time(baseline_m, baseline_alpha + delta, delta_psi, epsilon);
            let expected = 4.0 * delta;
            assert!(
                4.0_f64.mul_add(-delta, from_m - baseline).abs() < 1e-12,
                "E must shift by +4δ when M shifts by δ = {delta}: \
                 got {from_m} - {baseline} ≠ {expected}",
            );
            assert!(
                4.0_f64.mul_add(delta, from_alpha - baseline).abs() < 1e-12,
                "E must shift by -4δ when α shifts by δ = {delta}: \
                 got {from_alpha} - {baseline} ≠ {opposite}",
                opposite = -expected,
            );
        }
    }

    /// Equation A1 couples `Δψ` to `cos ε`, so shifting `Δψ` by `δ` at
    /// fixed `ε` must shift `E` by `4 · δ · cos(ε)` minutes. A typo
    /// such as `sin ε`, `cos² ε`, or omitting the trig factor altogether
    /// would fail here while the A5.1 reference test could still pass
    /// (since at that single instant `Δψ · cos ε ≈ −3.7·10⁻³°` is small
    /// enough that several wrong formulas remain numerically close). The
    /// sweep covers `ε` from `0°` (slope `+4`) through the canonical
    /// `23.44°` (slope `~3.67`) up to `60°` (slope `+2`), so any wrong
    /// trig identity diverges at at least one operating point.
    #[test]
    fn equation_of_time_scales_delta_psi_by_cos_epsilon() {
        let baseline_m = 100.0_f64;
        let baseline_alpha = 99.0_f64;
        for &epsilon in &[0.0_f64, 23.44, 60.0] {
            let baseline = equation_of_time(baseline_m, baseline_alpha, 0.0, epsilon);
            let cos_epsilon = epsilon.to_radians().cos();
            for &delta in &[-1e-3_f64, 1e-6, 0.5] {
                let shifted = equation_of_time(baseline_m, baseline_alpha, delta, epsilon);
                let expected = 4.0 * delta * cos_epsilon;
                assert!(
                    (shifted - baseline - expected).abs() < 1e-12,
                    "E must shift by 4·δ·cos(ε) for (ε, δ) = ({epsilon}, {delta}): \
                     got {shifted} - {baseline} ≠ {expected}",
                );
            }
        }
    }

    /// The appendix mandates folding `E` by `±1440` minutes whenever
    /// `|E| > 20`. Picking `M = 358°`, `α = 1°`, `Δψ = 0` yields
    /// `E_raw ≈ 1427.977 min` (well outside the band), which the fold
    /// must bring down to `E_raw − 1440 ≈ −12.023 min`. The mirror
    /// configuration `M = 1°`, `α = 358°` produces
    /// `E_raw ≈ −1428.023 min` and tests the fold's negative-overshoot
    /// branch (`+1440` shift). A typo replacing `copysign` with a
    /// constant `+1440` or `−1440`, a swap of the inequality direction,
    /// or a missing fold would fail in at least one branch.
    #[test]
    fn equation_of_time_clamps_overshoot_by_one_revolution() {
        let positive = equation_of_time(358.0, 1.0, 0.0, 23.44);
        let positive_raw = 4.0 * (358.0 - 1.0 - APPARENT_MEAN_LONGITUDE_CORRECTION_DEGREES);
        assert!(
            positive_raw.abs() > 20.0,
            "test setup must produce |E_raw| > 20 (got {positive_raw})",
        );
        let positive_expected = positive_raw - 1440.0;
        assert!(
            (positive - positive_expected).abs() < 1e-12,
            "positive-overshoot E must be folded by -1440 min: \
             got {positive} vs expected {positive_expected}",
        );

        let negative = equation_of_time(1.0, 358.0, 0.0, 23.44);
        let negative_raw = 4.0 * (1.0 - 358.0 - APPARENT_MEAN_LONGITUDE_CORRECTION_DEGREES);
        assert!(
            negative_raw.abs() > 20.0,
            "test setup must produce |E_raw| > 20 (got {negative_raw})",
        );
        let negative_expected = negative_raw + 1440.0;
        assert!(
            (negative - negative_expected).abs() < 1e-12,
            "negative-overshoot E must be folded by +1440 min: \
             got {negative} vs expected {negative_expected}",
        );
    }
}
