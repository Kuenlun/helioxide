// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Surface incidence angle (`I`).
//!
//! Implements section 3.16 of NREL/TP-560-34302. Equation 47 lifts the
//! topocentric solar position `(θ, Γ)` produced by section 3.15 onto a
//! planar surface of arbitrary orientation, parameterised by its slope
//! `ω` (degrees from horizontal) and its azimuth rotation `γ` (degrees
//! from south, positive west and negative east). The output `I` is the
//! angle between the surface normal and the sun direction; in vector
//! form it is `arccos` of the dot product of the two unit vectors,
//! which the paper writes out via the spherical-cosine identity.

/// Surface incidence angle `I` (degrees, in `[0°, 180°]`).
///
/// `topocentric_zenith_angle` is `θ` (in degrees), as produced by
/// [`topocentric_zenith_angle`]. `astronomers_azimuth` is `Γ` (in
/// degrees), **the astronomers' azimuth measured westward from south**
/// as produced by [`astronomers_azimuth`], not the navigators' azimuth
/// `Φ`: passing `Φ` would shift `Γ − γ` by `180°` and flip the sign of
/// `cos(Γ − γ)`. `surface_slope` is `ω` (in degrees), the slope of the
/// surface measured from the horizontal plane (`ω = 0°` is horizontal,
/// `ω = 90°` is vertical). `surface_azimuth_rotation` is `γ` (in
/// degrees), the rotation of the surface azimuth from south to the
/// projection of the surface normal on the horizontal plane, **positive
/// westward and negative eastward** to match `Γ`'s convention.
///
/// Refer to section 3.16, equation 47:
/// `I = arccos(cos θ · cos ω + sin ω · sin θ · cos(Γ − γ))`. The two
/// terms are folded inside a single `mul_add` so the cross-term's last
/// multiplication and the accumulation share one rounding instead of
/// two. `Γ` and `γ` enter only via their difference, so the `cos`
/// absorbs any `±360°` wrap on either input. `arccos` returns a value
/// in `[0, π]` by construction, so the output sits in `[0°, 180°]`
/// after the radians-to-degrees conversion.
///
/// # Examples
///
/// ```
/// use helioxide::incidence::surface_incidence_angle;
///
/// // Table A5.1: θ ≈ 50.11162°, Γ ≈ 14.34024°, ω = 30°, γ = -10°
/// // → I ≈ 25.18700°.
/// let i = surface_incidence_angle(50.111_62, 14.340_24, 30.0, -10.0);
/// assert!((i - 25.187_00).abs() < 1e-4);
/// ```
///
/// [`topocentric_zenith_angle`]: crate::horizontal::topocentric_zenith_angle
/// [`astronomers_azimuth`]: crate::horizontal::astronomers_azimuth
#[inline]
#[must_use]
pub fn surface_incidence_angle(
    topocentric_zenith_angle: f64,
    astronomers_azimuth: f64,
    surface_slope: f64,
    surface_azimuth_rotation: f64,
) -> f64 {
    let (sin_theta, cos_theta) = topocentric_zenith_angle.to_radians().sin_cos();
    let (sin_omega, cos_omega) = surface_slope.to_radians().sin_cos();
    let cos_azimuth_difference = (astronomers_azimuth - surface_azimuth_rotation)
        .to_radians()
        .cos();

    (sin_omega * sin_theta)
        .mul_add(cos_azimuth_difference, cos_theta * cos_omega)
        .acos()
        .to_degrees()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::surface_incidence_angle;
    use crate::test_fixtures::reference_theta_and_gamma;

    /// Reference surface slope `ω` for the Table A5.1 worked example
    /// (degrees), per section A.5.
    const REFERENCE_SURFACE_SLOPE_DEGREES: f64 = 30.0;
    /// Reference surface azimuth rotation `γ` for the Table A5.1 worked
    /// example (degrees, positive west / negative east of south), per
    /// section A.5.
    const REFERENCE_SURFACE_AZIMUTH_ROTATION_DEGREES: f64 = -10.0;

    /// `I` at the Table A5.1 reference instant must reproduce the
    /// published `25.18700°`. Failure is a global integration red flag:
    /// a wrong sign on equation 47's `sin ω · sin θ · cos(Γ − γ)` term,
    /// a swap of `Γ` and `Φ` (which would shift the cosine argument by
    /// `180°`), a swap of `γ`'s sign convention (the paper measures it
    /// westward from south, sign-mirrored from `Γ`), a missing radians
    /// conversion on any input, or any upstream regression on `(θ, Γ)`
    /// would all surface here. The 1e-4° tolerance sits just below the
    /// trailing-digit precision of the published value, so a real bug
    /// cannot hide behind rounding.
    #[test]
    fn surface_incidence_angle_matches_table_a5_1() {
        let (theta, gamma) = reference_theta_and_gamma();
        let i = surface_incidence_angle(
            theta,
            gamma,
            REFERENCE_SURFACE_SLOPE_DEGREES,
            REFERENCE_SURFACE_AZIMUTH_ROTATION_DEGREES,
        );
        assert!(
            (i - 25.187_00).abs() < 1e-4,
            "I mismatch at A5.1 reference: got {i}",
        );
    }

    /// On a horizontal surface (`ω = 0°`) the surface normal points to
    /// the zenith, so the incidence angle collapses to the topocentric
    /// zenith angle: `I = θ` for every `(Γ, γ)`. Equation 47 reduces to
    /// `arccos(cos θ · 1 + 0 · sin θ · cos(Γ − γ)) = θ`. This pins three
    /// things at once: the `cos ω` factor on `cos θ` (a typo using
    /// `sin ω` would yield `arccos(0) = 90°` regardless of `θ`), the
    /// `sin ω` factor that kills the second term (a typo using `cos ω`
    /// would leak `cos(Γ − γ)` into the result), and that `Γ` and `γ`
    /// have no other coupling than via the second term. The sweep over
    /// `(θ, Γ, γ)` rules out a coincidence at any single point.
    #[test]
    fn surface_incidence_angle_collapses_to_zenith_angle_on_horizontal_surface() {
        for &(theta, gamma, gamma_s) in &[
            (10.0_f64, 0.0, 0.0),
            (45.0, 90.0, -45.0),
            (89.999, 200.0, 170.0),
            (135.0, -30.0, 30.0),
        ] {
            let i = surface_incidence_angle(theta, gamma, 0.0, gamma_s);
            assert!(
                (i - theta).abs() < 1e-13,
                "I must equal θ on a horizontal surface for (θ, Γ, γ) = \
                 ({theta}, {gamma}, {gamma_s}): got {i}",
            );
        }
    }

    /// With the sun at the local zenith (`θ = 0°`) the sun direction
    /// coincides with the surface's vertical axis, so the incidence
    /// angle equals the surface slope: `I = ω` for every `(Γ, γ)`.
    /// Equation 47 reduces to `arccos(1 · cos ω + sin ω · 0 · …) = ω`.
    /// This pins the dual structure of the previous test: the `cos θ`
    /// factor on `cos ω` (a typo using `sin θ` would yield
    /// `arccos(sin ω · cos(Γ − γ))`, generally non-zero in `ω`), and
    /// that `sin θ = 0` collapses the cross-term independently of `(Γ, γ)`.
    /// The sweep covers slopes from horizontal to past vertical
    /// (`ω = 135°`, the surface tipped past upside-down) so an error
    /// growing in `ω` cannot hide behind a single operating point.
    #[test]
    fn surface_incidence_angle_collapses_to_surface_slope_when_sun_at_zenith() {
        for &(omega, gamma, gamma_s) in &[
            (0.0_f64, 0.0, 0.0),
            (30.0, 14.340_24, -10.0),
            (90.0, 200.0, 170.0),
            (135.0, -90.0, 90.0),
        ] {
            let i = surface_incidence_angle(0.0, gamma, omega, gamma_s);
            assert!(
                (i - omega).abs() < 1e-13,
                "I must equal ω when θ = 0 for (ω, Γ, γ) = \
                 ({omega}, {gamma}, {gamma_s}): got {i}",
            );
        }
    }

    /// When the sun's azimuth is aligned with the surface azimuth
    /// (`Γ = γ`, so `cos(Γ − γ) = 1`), equation 47 collapses to
    /// `arccos(cos θ · cos ω + sin θ · sin ω) = arccos(cos(θ − ω)) = |θ − ω|`
    /// for `|θ − ω| ≤ 180°`. This pins the addend coefficient on the
    /// cross-term: a sign flip on the second term would yield the
    /// opposite-azimuth identity `θ + ω` instead, and the difference
    /// from the next test (`Γ − γ = 180°`) catches the swap. The sweep
    /// keeps `θ` and `ω` distinct so the argument of `arccos` stays
    /// safely below `1` and the f64 result is well-conditioned.
    #[test]
    fn surface_incidence_angle_equals_absolute_difference_when_azimuths_aligned() {
        for &(theta, omega) in &[
            (50.0_f64, 30.0),
            (30.0, 50.0),
            (10.0, 80.0),
            (89.0, 1.0),
            (60.0, 60.5), // close but unequal: stresses the |θ − ω| sign without hitting acos(1)
        ] {
            let gamma = 14.340_24_f64;
            let i = surface_incidence_angle(theta, gamma, omega, gamma);
            let expected = (theta - omega).abs();
            assert!(
                (i - expected).abs() < 1e-12,
                "I must equal |θ − ω| at Γ = γ for (θ, ω) = ({theta}, {omega}): \
                 got {i} vs expected {expected}",
            );
        }
    }

    /// When the sun's azimuth is opposite the surface azimuth
    /// (`Γ − γ = ±180°`, so `cos(Γ − γ) = −1`), equation 47 collapses
    /// to `arccos(cos θ · cos ω − sin θ · sin ω) = arccos(cos(θ + ω)) = θ + ω`
    /// for `θ + ω ≤ 180°`. This pins the `−1` value of `cos(Γ − γ)` on
    /// the cross-term: a typo dropping the `cos(Γ − γ)` factor or
    /// taking its absolute value would still pass the aligned-azimuth
    /// test above (since `|cos 0°| = cos 0° = 1`), but it would fail
    /// here because `|−1| ≠ −1`. The sweep includes `Γ − γ = +180°` and
    /// `Γ − γ = −180°` to confirm the cosine's even symmetry, and keeps
    /// `θ + ω < 180°` so the argument of `arccos` stays above `−1`.
    #[test]
    fn surface_incidence_angle_equals_sum_when_azimuths_opposite() {
        for &(theta, omega, signed_offset) in &[
            (30.0_f64, 30.0, 180.0),
            (30.0, 30.0, -180.0),
            (10.0, 80.0, 180.0),
            (45.0, 60.0, -180.0),
            (89.0, 89.0, 180.0),
        ] {
            let gamma_s = 14.0_f64;
            let gamma = gamma_s + signed_offset;
            let i = surface_incidence_angle(theta, gamma, omega, gamma_s);
            let expected = theta + omega;
            assert!(
                (i - expected).abs() < 1e-12,
                "I must equal θ + ω at Γ − γ = ±180° for (θ, ω, offset) = \
                 ({theta}, {omega}, {signed_offset}): got {i} vs expected {expected}",
            );
        }
    }

    /// Equation 47 takes `Γ` and `γ` only via their difference inside
    /// `cos(Γ − γ)`. Therefore shifting both by the same `κ` (including
    /// a full revolution `κ = 360°`, which exercises the cosine's
    /// `2π`-periodicity) must leave `I` unchanged. A typo that fed
    /// either `Γ` or `γ` into the formula independently of the other —
    /// for example via `cos Γ − cos γ`, `cos Γ + γ`, or any non-difference
    /// combination — would break this invariance even when both the
    /// reference and the previous structural tests still pass (since
    /// those fix `γ` or `Γ` to specific values that a wrong formulation
    /// could compensate for). The sweep covers `θ` and `ω` away from
    /// the boundaries to keep the `arccos` well-conditioned.
    #[test]
    fn surface_incidence_angle_depends_on_azimuths_only_via_their_difference() {
        let theta = 50.0_f64;
        let omega = 30.0_f64;
        let gamma = 14.340_24_f64;
        let gamma_s = -10.0_f64;
        let baseline = surface_incidence_angle(theta, gamma, omega, gamma_s);
        for &kappa in &[-180.0_f64, -1.0, 1e-6, 47.5, 360.0] {
            let shifted = surface_incidence_angle(theta, gamma + kappa, omega, gamma_s + kappa);
            assert!(
                (shifted - baseline).abs() < 1e-12,
                "I must be invariant under Γ ↦ Γ + κ, γ ↦ γ + κ for κ = {kappa}: \
                 got {shifted} vs baseline {baseline}",
            );
        }
    }
}
