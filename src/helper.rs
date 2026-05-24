// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Shared scalar helpers used across the SPA pipeline.

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
    // Hand-rolled `rem_euclid` because the stdlib one is not yet `const`.
    let r = degrees % 360.0;
    if r < 0.0 { r + 360.0 } else { r }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::float_cmp)]
    fn int_truncates_towards_zero() {
        assert_eq!(int(8.7), 8.0);
        assert_eq!(int(8.2), 8.0);
        assert_eq!(int(-8.7), -8.0);
        assert_eq!(int(-8.2), -8.0);
        assert_eq!(int(0.0), 0.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn limit_degrees_inside_range_is_identity() {
        assert_eq!(limit_degrees(0.0), 0.0);
        assert_eq!(limit_degrees(45.0), 45.0);
        assert_eq!(limit_degrees(360.0), 0.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn limit_degrees_wraps() {
        assert_eq!(limit_degrees(370.0), 10.0);
        assert_eq!(limit_degrees(720.0), 0.0);
        assert_eq!(limit_degrees(-30.0), 330.0);
        assert_eq!(limit_degrees(-720.0), 0.0);
        assert_eq!(limit_degrees(-721.0), 359.0);
    }
}
