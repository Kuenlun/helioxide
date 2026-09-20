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
        .clamp(-1.0, 1.0)
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
        assert!((i - 25.187_00).abs() < 1e-4_f64);
    }

    #[test]
    fn horizontal_surface_yields_zenith_angle() {
        for &(theta, gamma, gamma_s) in &[
            (10.0_f64, 0.0_f64, 0.0_f64),
            (45.0_f64, 90.0_f64, -45.0_f64),
            (89.999_f64, 200.0_f64, 170.0_f64),
            (135.0_f64, -30.0_f64, 30.0_f64),
        ] {
            let i = surface_incidence_angle(theta, gamma, 0.0, gamma_s);
            assert!((i - theta).abs() < 1e-13_f64);
        }
    }

    #[test]
    fn sun_at_zenith_yields_surface_slope() {
        for &(omega, gamma, gamma_s) in &[
            (0.0_f64, 0.0_f64, 0.0_f64),
            (30.0_f64, 14.340_24_f64, -10.0_f64),
            (90.0_f64, 200.0_f64, 170.0_f64),
            (135.0_f64, -90.0_f64, 90.0_f64),
        ] {
            let i = surface_incidence_angle(0.0, gamma, omega, gamma_s);
            assert!((i - omega).abs() < 1e-13_f64);
        }
    }

    #[test]
    fn aligned_azimuths_yield_abs_difference() {
        for &(theta, omega) in &[
            (50.0_f64, 30.0_f64),
            (30.0_f64, 50.0_f64),
            (10.0_f64, 80.0_f64),
            (89.0_f64, 1.0_f64),
            (60.0_f64, 60.5_f64),
        ] {
            let gamma = 14.340_24_f64;
            let i = surface_incidence_angle(theta, gamma, omega, gamma);
            assert!((i - (theta - omega).abs()).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn opposite_azimuths_yield_sum() {
        for &(theta, omega, signed_offset) in &[
            (30.0_f64, 30.0_f64, 180.0_f64),
            (30.0_f64, 30.0_f64, -180.0_f64),
            (10.0_f64, 80.0_f64, 180.0_f64),
            (45.0_f64, 60.0_f64, -180.0_f64),
            (89.0_f64, 89.0_f64, 180.0_f64),
        ] {
            let gamma_s = 14.0_f64;
            let gamma = gamma_s + signed_offset;
            let i = surface_incidence_angle(theta, gamma, omega, gamma_s);
            assert!((i - (theta + omega)).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn azimuths_enter_only_via_difference() {
        let baseline = surface_incidence_angle(50.0, 14.340_24, 30.0, -10.0);
        for &kappa in &[-180.0_f64, -1.0_f64, 1e-6_f64, 47.5_f64, 360.0_f64] {
            let shifted = surface_incidence_angle(50.0, 14.340_24 + kappa, 30.0, -10.0 + kappa);
            assert!((shifted - baseline).abs() < 1e-12_f64);
        }
    }

    #[test]
    fn aligned_and_antipodal_vectors_remain_finite() {
        for theta in [0.015_f64, 26.477_169_330_695_055_f64] {
            let aligned = surface_incidence_angle(theta, 0.0, theta, 0.0);
            let opposite = surface_incidence_angle(theta, 0.0, 180.0 - theta, 180.0);
            assert!(aligned.abs() < 2e-6_f64);
            assert!((opposite - 180.0).abs() < 2e-6_f64);
        }
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(surface_incidence_angle(invalid, 0.0, 0.0, 0.0).is_nan());
        }
    }

    #[test]
    fn surfaces_aligned_to_hourly_solar_positions_remain_finite() {
        use crate::{Observer, SolarPosition, SpaDateTime, Surface};
        use chrono::{TimeDelta, TimeZone, Utc};

        let observer = Observer::try_at_sea_level_isa(0.0, 0.0).unwrap();
        let start = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        for hour in 0..8760 {
            let datetime = SpaDateTime::new(start + TimeDelta::hours(hour));
            let position = SolarPosition::compute_with_delta_t(&datetime, 69.1, observer).unwrap();
            let surface = Surface::try_new(
                position.topocentric_zenith,
                position.astronomers_azimuth_signed,
            )
            .unwrap();
            assert!(position.surface_incidence(surface).abs() < 2e-6_f64);
        }
    }
}
