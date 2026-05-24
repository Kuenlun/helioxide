// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Nutation in longitude (`Δψ`) and obliquity (`Δε`). Section 3.4.
//!
//! Both quantities share the five fundamental angles `X₀..X₄` (equations 15
//! to 19) and the per-row argument `Σⱼ Xⱼ · Yᵢⱼ` of Table A4.3, so emitting
//! them together folds equations 22 and 23 into a single sweep with one
//! `sin_cos` per row.

use tables::{TABLE_A4_3, X_POLYNOMIALS};

/// One Table A4.3 row: `(Y, [a, b, c, d])` with the coefficients in
/// `0.0001`-arc-second units. `(a, b)` apply to `Δψ` (equation 20),
/// `(c, d)` to `Δε` (equation 21).
type NutationTerm = ([i8; 5], [f64; 4]);

/// `(Δψ, Δε)` in degrees. Equations 15 to 23.
#[must_use]
pub fn nutation_in_longitude_and_obliquity(jce: f64) -> (f64, f64) {
    let x = fundamental_arguments(jce);

    let (psi, epsilon) =
        TABLE_A4_3
            .iter()
            .fold((0.0_f64, 0.0_f64), |(psi, epsilon), &(y, [a, b, c, d])| {
                let argument = x
                    .iter()
                    .zip(y.iter())
                    .map(|(&xj, &yj)| xj * f64::from(yj))
                    .sum::<f64>()
                    .to_radians();
                let (sin_argument, cos_argument) = argument.sin_cos();
                (
                    b.mul_add(jce, a).mul_add(sin_argument, psi),
                    d.mul_add(jce, c).mul_add(cos_argument, epsilon),
                )
            });

    // `1° = 36_000_000 × 0.0001"`.
    (psi / 36_000_000.0, epsilon / 36_000_000.0)
}

/// Fundamental angles `[X₀, X₁, X₂, X₃, X₄]` (degrees, not wrapped to
/// `[0°, 360°)`). Equations 15 to 19. Downstream uses take `sin`/`cos` of
/// integer linear combinations, so the wrap is unnecessary.
#[inline]
#[must_use]
pub fn fundamental_arguments(jce: f64) -> [f64; 5] {
    X_POLYNOMIALS.map(|[c0, c1, c2, c3]| c3.mul_add(jce, c2).mul_add(jce, c1).mul_add(jce, c0))
}

#[allow(
    clippy::unreadable_literal,
    clippy::approx_constant,
    clippy::excessive_precision
)]
mod tables {
    use super::NutationTerm;

    /// `Xₖ = c₀ + c₁·JCE + c₂·JCE² + c₃·JCE³`. Equations 15 to 19.
    pub(super) const X_POLYNOMIALS: [[f64; 4]; 5] = [
        [297.85036, 445267.111480, -0.0019142, 1.0 / 189474.0],
        [357.52772, 35999.050340, -0.0001603, -1.0 / 300000.0],
        [134.96298, 477198.867398, 0.0086972, 1.0 / 56250.0],
        [93.27191, 483202.017538, -0.0036825, 1.0 / 327270.0],
        [125.04452, -1934.136261, 0.0020708, 1.0 / 450000.0],
    ];

    pub(super) const TABLE_A4_3: &[NutationTerm] = &[
        ([0, 0, 0, 0, 1], [-171996.0, -174.2, 92025.0, 8.9]),
        ([-2, 0, 0, 2, 2], [-13187.0, -1.6, 5736.0, -3.1]),
        ([0, 0, 0, 2, 2], [-2274.0, -0.2, 977.0, -0.5]),
        ([0, 0, 0, 0, 2], [2062.0, 0.2, -895.0, 0.5]),
        ([0, 1, 0, 0, 0], [1426.0, -3.4, 54.0, -0.1]),
        ([0, 0, 1, 0, 0], [712.0, 0.1, -7.0, 0.0]),
        ([-2, 1, 0, 2, 2], [-517.0, 1.2, 224.0, -0.6]),
        ([0, 0, 0, 2, 1], [-386.0, -0.4, 200.0, 0.0]),
        ([0, 0, 1, 2, 2], [-301.0, 0.0, 129.0, -0.1]),
        ([-2, -1, 0, 2, 2], [217.0, -0.5, -95.0, 0.3]),
        ([-2, 0, 1, 0, 0], [-158.0, 0.0, 0.0, 0.0]),
        ([-2, 0, 0, 2, 1], [129.0, 0.1, -70.0, 0.0]),
        ([0, 0, -1, 2, 2], [123.0, 0.0, -53.0, 0.0]),
        ([2, 0, 0, 0, 0], [63.0, 0.0, 0.0, 0.0]),
        ([0, 0, 1, 0, 1], [63.0, 0.1, -33.0, 0.0]),
        ([2, 0, -1, 2, 2], [-59.0, 0.0, 26.0, 0.0]),
        ([0, 0, -1, 0, 1], [-58.0, -0.1, 32.0, 0.0]),
        ([0, 0, 1, 2, 1], [-51.0, 0.0, 27.0, 0.0]),
        ([-2, 0, 2, 0, 0], [48.0, 0.0, 0.0, 0.0]),
        ([0, 0, -2, 2, 1], [46.0, 0.0, -24.0, 0.0]),
        ([2, 0, 0, 2, 2], [-38.0, 0.0, 16.0, 0.0]),
        ([0, 0, 2, 2, 2], [-31.0, 0.0, 13.0, 0.0]),
        ([0, 0, 2, 0, 0], [29.0, 0.0, 0.0, 0.0]),
        ([-2, 0, 1, 2, 2], [29.0, 0.0, -12.0, 0.0]),
        ([0, 0, 0, 2, 0], [26.0, 0.0, 0.0, 0.0]),
        ([-2, 0, 0, 2, 0], [-22.0, 0.0, 0.0, 0.0]),
        ([0, 0, -1, 2, 1], [21.0, 0.0, -10.0, 0.0]),
        ([0, 2, 0, 0, 0], [17.0, -0.1, 0.0, 0.0]),
        ([2, 0, -1, 0, 1], [16.0, 0.0, -8.0, 0.0]),
        ([-2, 2, 0, 2, 2], [-16.0, 0.1, 7.0, 0.0]),
        ([0, 1, 0, 0, 1], [-15.0, 0.0, 9.0, 0.0]),
        ([-2, 0, 1, 0, 1], [-13.0, 0.0, 7.0, 0.0]),
        ([0, -1, 0, 0, 1], [-12.0, 0.0, 6.0, 0.0]),
        ([0, 0, 2, -2, 0], [11.0, 0.0, 0.0, 0.0]),
        ([2, 0, -1, 2, 1], [-10.0, 0.0, 5.0, 0.0]),
        ([2, 0, 1, 2, 2], [-8.0, 0.0, 3.0, 0.0]),
        ([0, 1, 0, 2, 2], [7.0, 0.0, -3.0, 0.0]),
        ([-2, 1, 1, 0, 0], [-7.0, 0.0, 0.0, 0.0]),
        ([0, -1, 0, 2, 2], [-7.0, 0.0, 3.0, 0.0]),
        ([2, 0, 0, 2, 1], [-7.0, 0.0, 3.0, 0.0]),
        ([2, 0, 1, 0, 0], [6.0, 0.0, 0.0, 0.0]),
        ([-2, 0, 2, 2, 2], [6.0, 0.0, -3.0, 0.0]),
        ([-2, 0, 1, 2, 1], [6.0, 0.0, -3.0, 0.0]),
        ([2, 0, -2, 0, 1], [-6.0, 0.0, 3.0, 0.0]),
        ([2, 0, 0, 0, 1], [-6.0, 0.0, 3.0, 0.0]),
        ([0, -1, 1, 0, 0], [5.0, 0.0, 0.0, 0.0]),
        ([-2, -1, 0, 2, 1], [-5.0, 0.0, 3.0, 0.0]),
        ([-2, 0, 0, 0, 1], [-5.0, 0.0, 3.0, 0.0]),
        ([0, 0, 2, 2, 1], [-5.0, 0.0, 3.0, 0.0]),
        ([-2, 0, 2, 0, 1], [4.0, 0.0, 0.0, 0.0]),
        ([-2, 1, 0, 2, 1], [4.0, 0.0, 0.0, 0.0]),
        ([0, 0, 1, -2, 0], [4.0, 0.0, 0.0, 0.0]),
        ([-1, 0, 1, 0, 0], [-4.0, 0.0, 0.0, 0.0]),
        ([-2, 1, 0, 0, 0], [-4.0, 0.0, 0.0, 0.0]),
        ([1, 0, 0, 0, 0], [-4.0, 0.0, 0.0, 0.0]),
        ([0, 0, 1, 2, 0], [3.0, 0.0, 0.0, 0.0]),
        ([0, 0, -2, 2, 2], [-3.0, 0.0, 0.0, 0.0]),
        ([-1, -1, 1, 0, 0], [-3.0, 0.0, 0.0, 0.0]),
        ([0, 1, 1, 0, 0], [-3.0, 0.0, 0.0, 0.0]),
        ([0, -1, 1, 2, 2], [-3.0, 0.0, 0.0, 0.0]),
        ([2, -1, -1, 2, 2], [-3.0, 0.0, 0.0, 0.0]),
        ([0, 0, 3, 2, 2], [-3.0, 0.0, 0.0, 0.0]),
        ([2, -1, 0, 2, 2], [-3.0, 0.0, 0.0, 0.0]),
    ];
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{fundamental_arguments, nutation_in_longitude_and_obliquity};
    use crate::test_fixtures::reference_jce;

    #[test]
    fn delta_psi_matches_table_a5_1() {
        let (delta_psi, _) = nutation_in_longitude_and_obliquity(reference_jce());
        assert!((delta_psi - -0.003_998_40).abs() < 1e-8);
    }

    #[test]
    fn delta_epsilon_matches_table_a5_1() {
        let (_, delta_epsilon) = nutation_in_longitude_and_obliquity(reference_jce());
        assert!((delta_epsilon - 0.001_666_57).abs() < 1e-8);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn fundamental_arguments_at_j2000_collapse_to_constant_terms() {
        let [x0, x1, x2, x3, x4] = fundamental_arguments(0.0);
        assert_eq!(x0, 297.85036);
        assert_eq!(x1, 357.52772);
        assert_eq!(x2, 134.96298);
        assert_eq!(x3, 93.27191);
        assert_eq!(x4, 125.04452);
    }

    #[test]
    fn fundamental_arguments_evaluate_horner_consistently() {
        let jce = reference_jce();
        let [x0, x1, x2, x3, x4] = fundamental_arguments(jce);
        let expected = |c: [f64; 4]| -> f64 {
            c[3].mul_add(jce, c[2])
                .mul_add(jce, c[1])
                .mul_add(jce, c[0])
        };
        assert!(
            (x0 - expected([297.85036, 445_267.111_480, -0.001_914_2, 1.0 / 189_474.0])).abs()
                < 1e-9
        );
        assert!(
            (x1 - expected([357.52772, 35_999.050_340, -0.000_160_3, -1.0 / 300_000.0])).abs()
                < 1e-9
        );
        assert!(
            (x2 - expected([134.96298, 477_198.867_398, 0.008_697_2, 1.0 / 56_250.0])).abs() < 1e-9
        );
        assert!(
            (x3 - expected([93.27191, 483_202.017_538, -0.003_682_5, 1.0 / 327_270.0])).abs()
                < 1e-9
        );
        assert!(
            (x4 - expected([125.04452, -1_934.136_261, 0.002_070_8, 1.0 / 450_000.0])).abs() < 1e-9
        );
    }
}
