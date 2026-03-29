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
pub trait FromTruncatedF64 {
    /// Constructs `Self` by truncating the fractional part towards zero.
    fn from_truncated(val: f64) -> Self;
}

impl FromTruncatedF64 for f64 {
    /// Retains the integer component as an `f64`, discarding the fraction.
    #[inline]
    fn from_truncated(val: f64) -> Self {
        val.trunc()
    }
}

impl FromTruncatedF64 for i32 {
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
pub fn int<T: FromTruncatedF64>(x: f64) -> T {
    T::from_truncated(x)
}

#[cfg(test)]
#[coverage(off)]
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
}
