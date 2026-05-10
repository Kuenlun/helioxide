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

//! Equatorial horizontal parallax (`ξ`) and topocentric equatorial
//! coordinates (`α'`, `δ'`).
//!
//! Implements section 3.12 of NREL/TP-560-34302. The equatorial horizontal
//! parallax `ξ` (equation 33) shrinks linearly with the Earth-Sun distance
//! `R` and sets the angular footprint of the Earth as seen from the Sun.
//! The topocentric equatorial coordinates `α'` (equation 38) and `δ'`
//! (equation 39) refine the geocentric pair `(α, δ)` for the observer's
//! actual position on the Earth's spheroid: latitude `φ`, elevation `E`,
//! and the local hour angle `H`. Both refinements run through the same
//! `(u, x, y)` reduction (equations 34 to 36) and share the denominator
//! `cos δ − x · sin ξ · cos H` of equations 37 and 39, so they are emitted
//! together to evaluate the trig scaffolding once.

/// Earth's polar-radius factor `1 − f`, where `f` is the flattening.
/// Weights `tan φ` inside the auxiliary angle `u` (equation 34) and
/// `sin u` inside `y` (equation 36). Reproduced verbatim from the paper's
/// note next to equation 34.
const EARTH_POLAR_RADIUS_FACTOR: f64 = 0.996_647_19;

/// Earth's equatorial radius (metres). Acts as the divisor of the observer
/// elevation `E` in equations 35 and 36, normalising the small `E·cos φ`
/// and `E·sin φ` corrections against the dimensionless `cos u` and
/// `(1 − f)·sin u` terms.
const EARTH_EQUATORIAL_RADIUS_METRES: f64 = 6_378_140.0;

/// Equatorial horizontal parallax of the sun at one astronomical unit
/// (arc seconds). Reproduced verbatim from equation 33.
const SOLAR_HORIZONTAL_PARALLAX_AT_UNIT_DISTANCE_ARCSECONDS: f64 = 8.794;

/// Equatorial horizontal parallax of the sun `ξ` (degrees).
///
/// `earth_radius_vector` is the Earth-Sun distance `R` (in astronomical
/// units), as produced by [`earth_radius_vector`]; physically positive in
/// any solar configuration.
///
/// Refer to section 3.12.1, equation 33: `ξ = 8.794" / (3600 · R)`. The
/// constant lives in arc seconds, hence the `1 / 3600` factor that brings
/// it to degrees. `ξ` shrinks as `1 / R`, so it is at its largest near
/// perihelion and at its smallest near aphelion.
///
/// # Examples
///
/// ```
/// use helioxide::parallax::equatorial_horizontal_parallax;
///
/// // Table A5.1: R = 0.9965422974 AU → ξ = 8.794 / (3600 · R) ≈ 0.002451°.
/// let xi = equatorial_horizontal_parallax(0.996_542_297_4);
/// assert!((xi - 0.002_451_253).abs() < 1e-8);
/// ```
///
/// [`earth_radius_vector`]: crate::heliocentric::earth_radius_vector
#[inline]
#[must_use]
pub const fn equatorial_horizontal_parallax(earth_radius_vector: f64) -> f64 {
    SOLAR_HORIZONTAL_PARALLAX_AT_UNIT_DISTANCE_ARCSECONDS / (3600.0 * earth_radius_vector)
}

/// Topocentric sun right ascension `α'`, declination `δ'`, and the
/// parallax in right ascension `Δα` they were derived from.
///
/// All three values are returned together because equations 37, 38 and 39
/// share the geodetic reduction `(u, x, y)` (equations 34 to 36) and the
/// denominator `cos δ − x · sin ξ · cos H`. Splitting them across separate
/// public functions would either recompute the scaffolding twice or force
/// callers to plumb intermediate `f64`s.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopocentricEquatorialCoordinates {
    /// Topocentric sun right ascension `α'` (degrees), as produced by
    /// equation 38.
    pub right_ascension: f64,
    /// Topocentric sun declination `δ'` (degrees, signed, in `[-90°, 90°]`),
    /// as produced by equation 39.
    pub declination: f64,
    /// Parallax in the sun right ascension `Δα` (degrees, signed), as
    /// produced by equation 37. Required by step 3.13 to compute the
    /// topocentric local hour angle `H' = H − Δα`.
    pub parallax_in_right_ascension: f64,
}

/// Topocentric equatorial coordinates of the sun.
///
/// `geocentric_right_ascension` is the sun geocentric right ascension `α`
/// (degrees), as produced by [`geocentric_right_ascension`].
/// `geocentric_declination` is the sun geocentric declination `δ`
/// (degrees), as produced by [`geocentric_declination`].
/// `observer_local_hour_angle` is `H` (degrees), as produced by
/// [`observer_local_hour_angle`].
/// `equatorial_horizontal_parallax` is `ξ` (degrees), as produced by
/// [`equatorial_horizontal_parallax`].
/// `observer_latitude` is the observer geographical latitude `φ`
/// (degrees, positive north of the equator).
/// `observer_elevation` is the observer elevation `E` above sea level
/// (metres).
///
/// Refer to section 3.12.2 through 3.12.7, equations 34 to 39.
/// `u = arctan((1 − f)·tan φ)` (eq 34) maps the geodetic latitude `φ` to
/// the corresponding point on the auxiliary sphere. `x = cos u + (E/r_eq)·cos φ`
/// (eq 35) and `y = (1 − f)·sin u + (E/r_eq)·sin φ` (eq 36) are then the
/// geocentric projections `D·cos φ'` and `D·sin φ'` of the observer's
/// distance `D` from Earth's centre. Equation 37's denominator
/// `cos δ − x·sin ξ·cos H` is shared verbatim with equation 39, so it is
/// computed once and reused; each `cos δ − …` and `sin δ − …` subtraction
/// is folded into a single `mul_add` rounding step so the small `x·sin ξ`
/// and `y·sin ξ` corrections (≈ 10⁻⁵) stay precise against the dominant
/// `cos δ` and `sin δ` (≈ 1).
///
/// # Examples
///
/// ```
/// use helioxide::parallax::{
///     equatorial_horizontal_parallax, topocentric_equatorial_coordinates,
/// };
///
/// // Table A5.1: α = 202.22741°, δ = -9.31434°, H = 11.105900°,
/// // R = 0.9965422974 AU, φ = 39.742476°, E = 1830.14 m
/// // → α' ≈ 202.22704°, δ' ≈ -9.316179°.
/// let xi = equatorial_horizontal_parallax(0.996_542_297_4);
/// let coords = topocentric_equatorial_coordinates(
///     202.227_41, -9.314_34, 11.105_900, xi, 39.742_476, 1830.14,
/// );
/// assert!((coords.right_ascension - 202.227_04).abs() < 1e-4);
/// assert!((coords.declination - -9.316_179).abs() < 1e-4);
/// ```
///
/// [`geocentric_right_ascension`]: crate::equatorial::geocentric_right_ascension
/// [`geocentric_declination`]: crate::equatorial::geocentric_declination
/// [`observer_local_hour_angle`]: crate::hour_angle::observer_local_hour_angle
/// [`equatorial_horizontal_parallax`]: self::equatorial_horizontal_parallax
#[inline]
#[must_use]
pub fn topocentric_equatorial_coordinates(
    geocentric_right_ascension: f64,
    geocentric_declination: f64,
    observer_local_hour_angle: f64,
    equatorial_horizontal_parallax: f64,
    observer_latitude: f64,
    observer_elevation: f64,
) -> TopocentricEquatorialCoordinates {
    // Equations 34 to 36: geodetic-to-geocentric reduction. The
    // observer's geodetic latitude φ collapses to the auxiliary angle u
    // on the unit ellipsoid; (x, y) then carry the geocentric projections
    // D·cos φ' and D·sin φ' of the observer's distance from Earth's
    // centre, with E folded against the equatorial radius in a single
    // `mul_add` so the small E/r_eq term stays precise.
    let phi_rad = observer_latitude.to_radians();
    let (sin_phi, cos_phi) = phi_rad.sin_cos();
    let (sin_u, cos_u) = (EARTH_POLAR_RADIUS_FACTOR * phi_rad.tan()).atan().sin_cos();
    let elevation_normalised = observer_elevation / EARTH_EQUATORIAL_RADIUS_METRES;
    let x = elevation_normalised.mul_add(cos_phi, cos_u);
    let y = elevation_normalised.mul_add(sin_phi, EARTH_POLAR_RADIUS_FACTOR * sin_u);

    // Equation 37: parallax in the sun right ascension Δα. The
    // denominator `cos δ − x·sin ξ·cos H` is the same as equation 39's,
    // so it is evaluated once and reused below.
    let (sin_h, cos_h) = observer_local_hour_angle.to_radians().sin_cos();
    let (sin_delta, cos_delta) = geocentric_declination.to_radians().sin_cos();
    let sin_xi = equatorial_horizontal_parallax.to_radians().sin();
    let x_sin_xi = x * sin_xi;
    let denominator = (-x_sin_xi).mul_add(cos_h, cos_delta);
    let delta_alpha_rad = (-x_sin_xi * sin_h).atan2(denominator);
    let delta_alpha = delta_alpha_rad.to_degrees();

    // Equation 39: topocentric declination δ'. The numerator carries the
    // small `y·sin ξ` correction folded into a single `mul_add` so that
    // it stays precise against the dominant `sin δ`. The `cos Δα` factor
    // couples the declination correction to the right-ascension
    // correction, accounting for the observer's sideways displacement on
    // the celestial sphere as `Δα` accumulates.
    let topocentric_declination_numerator = (-y).mul_add(sin_xi, sin_delta) * delta_alpha_rad.cos();
    let delta_prime = topocentric_declination_numerator
        .atan2(denominator)
        .to_degrees();

    TopocentricEquatorialCoordinates {
        // Equation 38: α' = α + Δα. The PDF does not mandate a wrap into
        // [0°, 360°) here; in any physical solar configuration |Δα| ≤ ξ
        // (arc-second order), so α + Δα stays inside the interval that
        // upstream `α` already lives in.
        right_ascension: geocentric_right_ascension + delta_alpha,
        declination: delta_prime,
        parallax_in_right_ascension: delta_alpha,
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{
        EARTH_EQUATORIAL_RADIUS_METRES, SOLAR_HORIZONTAL_PARALLAX_AT_UNIT_DISTANCE_ARCSECONDS,
        equatorial_horizontal_parallax, topocentric_equatorial_coordinates,
    };
    use crate::apparent::{aberration_correction, apparent_sun_longitude};
    use crate::equatorial::{geocentric_declination, geocentric_right_ascension};
    use crate::geocentric::{geocentric_latitude, geocentric_longitude};
    use crate::heliocentric::{
        earth_heliocentric_latitude, earth_heliocentric_longitude, earth_radius_vector,
    };
    use crate::hour_angle::observer_local_hour_angle;
    use crate::nutation::nutation_in_longitude_and_obliquity;
    use crate::obliquity::true_obliquity_of_ecliptic;
    use crate::sidereal::{apparent_sidereal_time, mean_sidereal_time};
    use crate::test_fixtures::{reference_jce, reference_jd, reference_jme};

    /// Reference site parameters from Appendix A.5: latitude `φ`,
    /// longitude `σ`, elevation `E`. Used by every reference-instant
    /// test in this module so the published civil instant feeds a single,
    /// canonical observer.
    const REFERENCE_LATITUDE_DEGREES: f64 = 39.742_476;
    const REFERENCE_LONGITUDE_DEGREES: f64 = -105.178_6;
    const REFERENCE_ELEVATION_METRES: f64 = 1830.14;

    /// Drives the full upstream chain to produce `(α, δ, H, ξ)` at the
    /// Table A5.1 reference instant. Reused by every reference-instant
    /// test so a single integration regression upstream surfaces in
    /// exactly one place.
    fn reference_alpha_delta_h_xi() -> (f64, f64, f64, f64) {
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

        (alpha, delta, h, xi)
    }

    /// `α'` at the Table A5.1 reference instant must reproduce the
    /// published `202.22704°`. Failure is a global integration red flag:
    /// a wrong sign on equation 37's numerator or denominator, a swapped
    /// `sin H`/`cos H`, a missing radians conversion on `ξ`, `H` or `δ`,
    /// a wrong `(1 − f)` factor, or any upstream regression on `α`, `δ`,
    /// `H` or `ξ` would all surface here. The 1e-4° tolerance sits just
    /// below the trailing-digit precision of the published value, so a
    /// real bug cannot hide behind rounding.
    #[test]
    fn topocentric_right_ascension_matches_table_a5_1() {
        let (alpha, delta, h, xi) = reference_alpha_delta_h_xi();
        let coords = topocentric_equatorial_coordinates(
            alpha,
            delta,
            h,
            xi,
            REFERENCE_LATITUDE_DEGREES,
            REFERENCE_ELEVATION_METRES,
        );
        let alpha_prime = coords.right_ascension;
        assert!(
            (alpha_prime - 202.227_04).abs() < 1e-4,
            "α' mismatch at A5.1 reference: got {alpha_prime}",
        );
    }

    /// `δ'` at the Table A5.1 reference instant must reproduce the
    /// published `-9.316179°`. The opposite sign and the order-of-
    /// magnitude difference from `α'` ensure that an accidental swap of
    /// the two outputs cannot be hidden by both reference tests passing
    /// on the wrong field. Failure also pins the `(1 − f)` weight on
    /// `sin u` inside `y` (equation 36) in aggregate, since `y · sin ξ`
    /// is the only place `(1 − f)` enters the published `δ'`.
    #[test]
    fn topocentric_declination_matches_table_a5_1() {
        let (alpha, delta, h, xi) = reference_alpha_delta_h_xi();
        let coords = topocentric_equatorial_coordinates(
            alpha,
            delta,
            h,
            xi,
            REFERENCE_LATITUDE_DEGREES,
            REFERENCE_ELEVATION_METRES,
        );
        let delta_prime = coords.declination;
        assert!(
            (delta_prime - -9.316_179).abs() < 1e-4,
            "δ' mismatch at A5.1 reference: got {delta_prime}",
        );
    }

    /// `H − Δα` at the Table A5.1 reference instant must reproduce the
    /// published `H' = 11.10629°`. Section 3.13's one-line refinement
    /// `H' = H − Δα` is the only published downstream consumer of `Δα`,
    /// so this test pins the exposed scaffolding output that no other
    /// reference test exercises directly. A wrong sign on `Δα` (which
    /// would still keep `α' = α + Δα` close to its published value, since
    /// `Δα` itself is `~10⁻⁴°`) would surface here as a `~2·10⁻⁴°`
    /// mismatch, well above the 1e-4° tolerance.
    #[test]
    fn parallax_in_right_ascension_drives_published_topocentric_hour_angle() {
        let (alpha, delta, h, xi) = reference_alpha_delta_h_xi();
        let coords = topocentric_equatorial_coordinates(
            alpha,
            delta,
            h,
            xi,
            REFERENCE_LATITUDE_DEGREES,
            REFERENCE_ELEVATION_METRES,
        );
        let h_prime = h - coords.parallax_in_right_ascension;
        assert!(
            (h_prime - 11.106_29).abs() < 1e-4,
            "H − Δα mismatch at A5.1 reference: got {h_prime}",
        );
    }

    /// At `R = 1 AU` equation 33 collapses to `ξ = 8.794 / 3600°`. This
    /// isolates the literal aberration-of-distance constant and the
    /// `3600` arc-second-to-degree divisor from the `R` dependence: a
    /// stray digit in `8.794` or a wrong divisor would fail here even
    /// when the reference test still passes (since the latter exercises
    /// only one specific `R`, where a compensating typo upstream could
    /// mask the bug).
    #[test]
    fn equatorial_horizontal_parallax_at_unit_distance_equals_published_constant_in_degrees() {
        let xi = equatorial_horizontal_parallax(1.0);
        let expected = SOLAR_HORIZONTAL_PARALLAX_AT_UNIT_DISTANCE_ARCSECONDS / 3600.0;
        assert!(
            (xi - expected).abs() < f64::EPSILON,
            "ξ at R = 1 AU must equal 8.794 / 3600°: got {xi}",
        );
    }

    /// Equation 33 mandates `ξ ∝ 1/R`; therefore `ξ(R) · R` must be
    /// constant equal to `ξ(1)`. A typo such as `R²`, `R + 1`, or any
    /// other non-linear-in-`R` denominator would break this invariant.
    /// The sweep covers `R` values inside Earth's annual range
    /// (≈ [0.983, 1.017] AU) and well outside it so the proportionality
    /// is pinned across orders of magnitude rather than at a single
    /// operating point.
    #[test]
    fn equatorial_horizontal_parallax_is_inversely_proportional_to_radius_vector() {
        let xi_unit = equatorial_horizontal_parallax(1.0);
        for &r in &[0.5_f64, 0.95, 1.05, 2.0, 10.0] {
            let xi = equatorial_horizontal_parallax(r);
            assert!(
                xi.mul_add(r, -xi_unit).abs() < 1e-15,
                "ξ · R must equal ξ(1) for R = {r}: got {xi}",
            );
        }
    }

    /// At `ξ = 0` (the Sun is infinitely far away, so its parallax
    /// vanishes) equations 37 and 39 collapse: the numerator of equation
    /// 37 vanishes and its denominator becomes `cos δ`, so `Δα = 0`
    /// regardless of `(α, δ, H, φ, E)`. Equation 38 then gives `α' = α`,
    /// and equation 39 reduces to `δ' = atan2(sin δ, cos δ) = δ`. This
    /// pins five things in one test: the `sin ξ` factor on the `x` term
    /// in equation 37's numerator and denominator, the `sin ξ` factor on
    /// the `y` term in equation 39's numerator, the `cos δ` denominator
    /// structure, and the `α + Δα` form of equation 38. A typo dropping
    /// any `sin ξ` factor would survive the reference test (where `ξ`
    /// is small but non-zero) but fail here.
    #[test]
    fn topocentric_corrections_vanish_at_zero_equatorial_horizontal_parallax() {
        for &(alpha, delta, h, phi, elevation) in &[
            (10.0_f64, -30.0, 0.0, 0.0, 0.0),
            (200.0, 5.5, 90.0, 39.742_476, 1830.14),
            (350.0, -85.0, -45.0, -60.0, 5_000.0),
        ] {
            let coords = topocentric_equatorial_coordinates(alpha, delta, h, 0.0, phi, elevation);
            assert!(
                coords.parallax_in_right_ascension.abs() < 1e-15,
                "Δα must vanish at ξ = 0 for (α, δ, H, φ, E) = \
                 ({alpha}, {delta}, {h}, {phi}, {elevation}): got {}",
                coords.parallax_in_right_ascension,
            );
            assert!(
                (coords.right_ascension - alpha).abs() < 1e-13,
                "α' must equal α at ξ = 0: got {} for α = {alpha}",
                coords.right_ascension,
            );
            assert!(
                (coords.declination - delta).abs() < 1e-13,
                "δ' must equal δ at ξ = 0: got {} for δ = {delta}",
                coords.declination,
            );
        }
    }

    /// At `H = 0` (the sun crosses the local meridian) `sin H = 0`, so
    /// equation 37's numerator `−x · sin ξ · sin H` vanishes and `Δα`
    /// must be exactly zero — independently of `(α, δ, ξ, φ, E)`. This
    /// isolates the `sin H` factor in equation 37's numerator: a typo
    /// substituting `cos H` or `1` would still keep `Δα` small at the
    /// reference instant (because `x · sin ξ` is itself small), but it
    /// would not vanish at `H = 0` and the test would fail. The sweep
    /// covers diverse `(δ, φ, E)` so that no specific configuration can
    /// hide the bug.
    #[test]
    fn parallax_in_right_ascension_vanishes_at_meridian_transit() {
        let xi = equatorial_horizontal_parallax(1.0);
        for &(delta, phi, elevation) in &[
            (0.0_f64, 0.0, 0.0),
            (-9.314_34, 39.742_476, 1830.14),
            (45.0, -45.0, 0.0),
            (-89.0, 89.0, 5_000.0),
        ] {
            let coords =
                topocentric_equatorial_coordinates(123.456, delta, 0.0, xi, phi, elevation);
            assert!(
                coords.parallax_in_right_ascension.abs() < 1e-15,
                "Δα must vanish at H = 0 for (δ, φ, E) = \
                 ({delta}, {phi}, {elevation}): got {}",
                coords.parallax_in_right_ascension,
            );
        }
    }

    /// Equation 37's numerator `−x · sin ξ · sin H` is odd in `H` while
    /// the denominator `cos δ − x · sin ξ · cos H` is even, so
    /// `Δα(H) = −Δα(−H)` for any common `(α, δ, ξ, φ, E)`. A typo that
    /// flipped a sign on the numerator, used `|sin H|`, or replaced one
    /// of the `H`-trig calls with the wrong one would break this
    /// symmetry. The sweep covers diverse `(δ, H, φ, E)` configurations,
    /// including the published reference instant, so a coincidence at
    /// any single point cannot mask the bug.
    #[test]
    fn parallax_in_right_ascension_is_odd_in_observer_local_hour_angle() {
        let xi = equatorial_horizontal_parallax(0.996_542_297_4);
        for &(delta, h, phi, elevation) in &[
            (0.0_f64, 30.0, 0.0, 0.0),
            (-9.314_34, 11.105_900, 39.742_476, 1830.14),
            (45.0, 60.0, -45.0, 500.0),
            (-30.0, 89.999, 60.0, 1_500.0),
        ] {
            let plus = topocentric_equatorial_coordinates(100.0, delta, h, xi, phi, elevation);
            let minus = topocentric_equatorial_coordinates(100.0, delta, -h, xi, phi, elevation);
            let sum = plus.parallax_in_right_ascension + minus.parallax_in_right_ascension;
            assert!(
                sum.abs() < 1e-13,
                "Δα(H) + Δα(-H) must vanish for (δ, H, φ, E) = \
                 ({delta}, {h}, {phi}, {elevation}): got {sum}",
            );
        }
    }

    /// Equation 38 sums `α` and `Δα` linearly; `α` appears nowhere else
    /// in the topocentric chain. Shifting `α` by `κ` while holding
    /// `(δ, H, ξ, φ, E)` fixed must therefore shift `α'` by exactly `κ`
    /// while leaving `Δα` and `δ'` unchanged. This isolates equation 38
    /// from the rest of the formula: a typo that fed `α` into equation
    /// 37 or 39 would surface here even when the reference test still
    /// passes (because at one specific `α` a compensating typo could
    /// mask the bug). The sweep covers both signs and several
    /// magnitudes of `κ` so a wrong coefficient cannot hide behind the
    /// comparison tolerance.
    #[test]
    fn topocentric_right_ascension_shifts_linearly_with_geocentric_right_ascension() {
        let xi = equatorial_horizontal_parallax(1.0);
        let baseline =
            topocentric_equatorial_coordinates(100.0, 10.0, 30.0, xi, 39.742_476, 1830.14);
        for &kappa in &[-30.0_f64, -1e-3, 1e-6, 0.5, 50.0] {
            let shifted = topocentric_equatorial_coordinates(
                100.0 + kappa,
                10.0,
                30.0,
                xi,
                39.742_476,
                1830.14,
            );
            let alpha_prime_shift = shifted.right_ascension - baseline.right_ascension;
            assert!(
                (alpha_prime_shift - kappa).abs() < 1e-13,
                "α' must shift by κ when α shifts by κ = {kappa}: got {alpha_prime_shift}",
            );
            let delta_alpha_shift =
                shifted.parallax_in_right_ascension - baseline.parallax_in_right_ascension;
            assert!(
                delta_alpha_shift.abs() < 1e-15,
                "Δα must not depend on α: shift = {delta_alpha_shift}",
            );
            let delta_prime_shift = shifted.declination - baseline.declination;
            assert!(
                delta_prime_shift.abs() < 1e-15,
                "δ' must not depend on α: shift = {delta_prime_shift}",
            );
        }
    }

    /// Equation 35 contributes `(E / 6_378_140) · cos φ` to `x`. At the
    /// equator (`φ = 0`) `cos φ = 1`, so `x = 1 + E / r_eq`, and an
    /// elevation equal to one Earth equatorial radius doubles `x` from
    /// `1` to `2`. Equation 37 is approximately linear in `x` for small
    /// `ξ`, so `|Δα(E = r_eq)| / |Δα(E = 0)|` must equal `2` to leading
    /// order at the equator with `sin H = 1`. This pins three things at
    /// once: that `E` enters `x` (not `y` or somewhere else), that the
    /// divisor is `6_378_140` (a typo in the divisor would break the
    /// ratio), and that the addition is `cos u + (E/r_eq)·cos φ` rather
    /// than a multiplicative or other combination. The next-order
    /// correction is `O((sin ξ)²) ≈ 2·10⁻⁹`, well below the tolerance.
    #[test]
    fn parallax_in_right_ascension_doubles_when_elevation_equals_one_earth_equatorial_radius_at_equator()
     {
        let xi = equatorial_horizontal_parallax(1.0);
        let sea_level = topocentric_equatorial_coordinates(100.0, 0.0, 90.0, xi, 0.0, 0.0);
        let one_radius_up = topocentric_equatorial_coordinates(
            100.0,
            0.0,
            90.0,
            xi,
            0.0,
            EARTH_EQUATORIAL_RADIUS_METRES,
        );
        let ratio =
            one_radius_up.parallax_in_right_ascension / sea_level.parallax_in_right_ascension;
        assert!(
            (ratio - 2.0).abs() < 1e-7,
            "|Δα| must double when E equals one equatorial radius at the equator: got {ratio}",
        );
    }
}
