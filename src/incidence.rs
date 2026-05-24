// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Surface incidence angle (`I`). Section 3.16, equation 47.

/// `I = arccos(cos θ · cos ω + sin ω · sin θ · cos(Γ − γ))` (degrees,
/// in `[0°, 180°]`). Equation 47.
///
/// `Γ` is the astronomers' azimuth (westward from south), not the
/// navigators' `Φ`. `γ` shares the same convention.
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

    const REFERENCE_SURFACE_SLOPE_DEGREES: f64 = 30.0;
    const REFERENCE_SURFACE_AZIMUTH_ROTATION_DEGREES: f64 = -10.0;

    #[test]
    fn surface_incidence_angle_matches_table_a5_1() {
        let (theta, gamma) = reference_theta_and_gamma();
        let i = surface_incidence_angle(
            theta,
            gamma,
            REFERENCE_SURFACE_SLOPE_DEGREES,
            REFERENCE_SURFACE_AZIMUTH_ROTATION_DEGREES,
        );
        assert!((i - 25.187_00).abs() < 1e-4);
    }

    #[test]
    fn horizontal_surface_yields_zenith_angle() {
        for &(theta, gamma, gamma_s) in &[
            (10.0_f64, 0.0, 0.0),
            (45.0, 90.0, -45.0),
            (89.999, 200.0, 170.0),
            (135.0, -30.0, 30.0),
        ] {
            let i = surface_incidence_angle(theta, gamma, 0.0, gamma_s);
            assert!((i - theta).abs() < 1e-13);
        }
    }

    #[test]
    fn sun_at_zenith_yields_surface_slope() {
        for &(omega, gamma, gamma_s) in &[
            (0.0_f64, 0.0, 0.0),
            (30.0, 14.340_24, -10.0),
            (90.0, 200.0, 170.0),
            (135.0, -90.0, 90.0),
        ] {
            let i = surface_incidence_angle(0.0, gamma, omega, gamma_s);
            assert!((i - omega).abs() < 1e-13);
        }
    }

    #[test]
    fn aligned_azimuths_yield_abs_difference() {
        for &(theta, omega) in &[
            (50.0_f64, 30.0),
            (30.0, 50.0),
            (10.0, 80.0),
            (89.0, 1.0),
            (60.0, 60.5),
        ] {
            let gamma = 14.340_24;
            let i = surface_incidence_angle(theta, gamma, omega, gamma);
            assert!((i - (theta - omega).abs()).abs() < 1e-12);
        }
    }

    #[test]
    fn opposite_azimuths_yield_sum() {
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
            assert!((i - (theta + omega)).abs() < 1e-12);
        }
    }

    #[test]
    fn azimuths_enter_only_via_difference() {
        let baseline = surface_incidence_angle(50.0, 14.340_24, 30.0, -10.0);
        for &kappa in &[-180.0_f64, -1.0, 1e-6, 47.5, 360.0] {
            let shifted = surface_incidence_angle(50.0, 14.340_24 + kappa, 30.0, -10.0 + kappa);
            assert!((shifted - baseline).abs() < 1e-12);
        }
    }
}
