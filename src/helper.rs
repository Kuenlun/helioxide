// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Shared scalar helpers used across the SPA pipeline.

const FULL_REVOLUTION_DEGREES: f64 = 360.0;

/// `INT` of the NREL SPA paper: truncate towards zero.
///
/// ```
/// use helioxide::helper::int;
/// assert_eq!(int(8.7), 8.0);
/// assert_eq!(int(-8.7), -8.0);
/// ```
#[inline]
#[must_use]
pub const fn int(x: f64) -> f64 {
    x.trunc()
}

/// Wrap a degree value into `[0°, 360°)`, per step 3.2.6.
#[inline]
#[must_use]
pub const fn limit_degrees(degrees: f64) -> f64 {
    wrap_positive_period(degrees, FULL_REVOLUTION_DEGREES)
}

/// Wrap into `[0, period)` for a positive finite period, preserving non-finite inputs as NaN.
#[inline]
pub(crate) const fn wrap_positive_period(value: f64, period: f64) -> f64 {
    // Hand-rolled `rem_euclid` because the stdlib one is not yet `const`.
    let remainder = value % period;
    if remainder < 0.0 {
        let wrapped = remainder + period;
        // A tiny negative remainder can round up to the excluded endpoint.
        if wrapped >= period { 0.0 } else { wrapped }
    } else {
        remainder
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "These cases require exact preservation of stored values or exact boundary results."
    )]
    fn int_truncates_towards_zero() {
        assert_eq!(int(8.7), 8.0_f64);
        assert_eq!(int(8.2), 8.0_f64);
        assert_eq!(int(-8.7), -8.0_f64);
        assert_eq!(int(-8.2), -8.0_f64);
        assert_eq!(int(0.0), 0.0_f64);
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "These cases require exact preservation of stored values or exact boundary results."
    )]
    fn limit_degrees_inside_range_is_identity() {
        assert_eq!(limit_degrees(0.0), 0.0_f64);
        assert_eq!(limit_degrees(45.0), 45.0_f64);
        assert_eq!(limit_degrees(360.0), 0.0_f64);
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "These cases require exact preservation of stored values or exact boundary results."
    )]
    fn limit_degrees_wraps() {
        assert_eq!(limit_degrees(370.0), 10.0_f64);
        assert_eq!(limit_degrees(720.0), 0.0_f64);
        assert_eq!(limit_degrees(-30.0), 330.0_f64);
        assert_eq!(limit_degrees(-720.0), 0.0_f64);
        assert_eq!(limit_degrees(-721.0), 359.0_f64);
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "Rounding at the excluded upper endpoint must produce exact zero."
    )]
    fn negative_roundoff_cannot_reach_the_excluded_full_revolution() {
        for angle in [-1e-16_f64, -f64::MIN_POSITIVE, -f64::from_bits(1)] {
            assert_eq!(limit_degrees(angle), 0.0_f64);
        }
        let below_full_revolution = 360.0_f64.next_down();
        assert_eq!(limit_degrees(below_full_revolution), below_full_revolution);
        for invalid in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(limit_degrees(invalid).is_nan());
        }
    }
}
