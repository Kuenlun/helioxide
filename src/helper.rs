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

/// Enables conversion from an `f64` by extracting its integer component.
pub const trait FromTruncatedF64 {
    /// Constructs `Self` by truncating the fractional part towards zero.
    fn from_truncated(val: f64) -> Self;
}

impl const FromTruncatedF64 for f64 {
    /// Retains the integer component as an `f64`, discarding the fraction.
    #[inline]
    fn from_truncated(val: f64) -> Self {
        val.trunc()
    }
}

impl const FromTruncatedF64 for i32 {
    /// Converts to an `i32` by truncating the value towards zero.
    #[allow(clippy::cast_possible_truncation)] // Truncation is intentional
    #[inline]
    fn from_truncated(val: f64) -> Self {
        val as Self
    }
}

/// Implements the `INT` function from the NREL SPA article, truncating towards zero.
///
/// # Examples
///
/// ```
/// use helioxide::helper::int;
///
/// let a: i32 = int(8.7);
/// let b: i32 = int(8.2);
/// let c: i32 = int(-8.7);
///
/// assert_eq!(a, 8);
/// assert_eq!(b, 8);
/// assert_eq!(c, -8);
/// ```
#[inline]
#[must_use]
pub const fn int<T: const FromTruncatedF64>(x: f64) -> T {
    T::from_truncated(x)
}

/// Wraps a degree value into the half-open interval `[0°, 360°)`.
///
/// Refer to step 3.2.6.
///
/// # Examples
///
/// ```
/// use helioxide::helper::limit_degrees;
///
/// let wrapped = limit_degrees(370.0);
/// assert!((0.0..360.0).contains(&wrapped));
/// ```
#[inline]
#[must_use]
pub fn limit_degrees(degrees: f64) -> f64 {
    degrees.rem_euclid(360.0)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    /// Verifies that `int::<i32>` truncates positive and negative values toward zero.
    #[test]
    fn test_int_truncates_towards_zero_i32() {
        assert_eq!(int::<i32>(8.7), 8_i32);
        assert_eq!(int::<i32>(8.2), 8_i32);
        assert_eq!(int::<i32>(-8.7), -8_i32);
        assert_eq!(int::<i32>(-8.2), -8_i32);
        assert_eq!(int::<i32>(0.0), 0_i32);
    }

    /// Verifies that `int::<f64>` preserves only the integer part by truncating toward zero.
    #[test]
    #[allow(clippy::float_cmp)] // Truncation is exact, so direct comparison is valid
    fn test_int_truncates_towards_zero_f64() {
        assert_eq!(int::<f64>(8.7), 8.0_f64);
        assert_eq!(int::<f64>(8.2), 8.0_f64);
        assert_eq!(int::<f64>(-8.7), -8.0_f64);
        assert_eq!(int::<f64>(-8.2), -8.0_f64);
        assert_eq!(int::<f64>(0.0), 0.0_f64);
    }

    /// Pins the in-range identity: any value already inside `[0, 360)` must
    /// pass through unchanged. The chosen inputs (0, 360, and a generic value)
    /// are exactly representable in `f64`, so direct equality is meaningful.
    #[test]
    #[allow(clippy::float_cmp)]
    fn limit_degrees_is_identity_inside_open_zero_360_interval() {
        assert_eq!(limit_degrees(0.0), 0.0);
        assert_eq!(limit_degrees(45.0), 45.0);
        // Step 3.2.6 specifies a half-open interval, so 360° collapses to 0°.
        assert_eq!(limit_degrees(360.0), 0.0);
    }

    /// Pins the wrap behaviour for over- and under-shoots, including values
    /// that span multiple full revolutions. All operands are integers exactly
    /// representable in `f64`, so the modular arithmetic is exact.
    #[test]
    #[allow(clippy::float_cmp)]
    fn limit_degrees_wraps_into_zero_360_for_over_and_undershoots() {
        assert_eq!(limit_degrees(370.0), 10.0);
        assert_eq!(limit_degrees(720.0), 0.0);
        assert_eq!(limit_degrees(-30.0), 330.0);
        assert_eq!(limit_degrees(-720.0), 0.0);
        assert_eq!(limit_degrees(-721.0), 359.0);
    }
}
