// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Time inputs for the SPA pipeline.
//!
//! [`SpaDateTime`] pairs a [`chrono::DateTime`] with the DUT1 (= UT1 − UTC)
//! correction required by NREL's Solar Position Algorithm. DUT1 is validated
//! against the IERS bound of ±1 s on construction.

use chrono::{DateTime, TimeZone};
use thiserror::Error;

/// DUT1 (= UT1 − UTC) is bounded by ±1 s; leap seconds are inserted to keep it inside that range.
const DUT1_LIMIT_SECONDS: f64 = 1.0;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum SpaTimeError {
    #[error("DUT1 must lie in the open interval (-1, 1) s, got {0}")]
    Dut1OutOfRange(f64),
}

/// Instant tagged with the DUT1 (= UT1 − UTC) correction required by the
/// NREL Solar Position Algorithm.
///
/// Use [`SpaDateTime::new`] when DUT1 is unknown (defaults to 0). Use
/// [`SpaDateTime::try_new`] when an IERS-published value is available.
///
/// # Examples
///
/// ```
/// use chrono::Utc;
/// use helioxide::SpaDateTime;
///
/// let now = Utc::now();
/// let default_dut1 = SpaDateTime::new(now);
/// let with_iers = SpaDateTime::try_new(now, -0.123).expect("DUT1 inside (-1, 1) s");
/// let via_into: SpaDateTime<_> = now.into();
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SpaDateTime<Tz: TimeZone> {
    datetime: DateTime<Tz>,
    dut1: f64,
}

impl<Tz: TimeZone> SpaDateTime<Tz> {
    /// Builds a [`SpaDateTime`] with `dut1 = 0`.
    #[must_use]
    pub const fn new(datetime: DateTime<Tz>) -> Self {
        Self {
            datetime,
            dut1: 0.0,
        }
    }

    /// Builds a [`SpaDateTime`] with the given DUT1 in seconds.
    ///
    /// # Errors
    /// Returns [`SpaTimeError::Dut1OutOfRange`] if `dut1` is non-finite or `|dut1| >= 1`.
    pub fn try_new(datetime: DateTime<Tz>, dut1: f64) -> Result<Self, SpaTimeError> {
        Self::new(datetime).try_with_dut1(dut1)
    }

    /// Returns a copy of `self` with the given DUT1 in seconds.
    ///
    /// # Errors
    /// Returns [`SpaTimeError::Dut1OutOfRange`] if `dut1` is non-finite or `|dut1| >= 1`.
    pub fn try_with_dut1(mut self, dut1: f64) -> Result<Self, SpaTimeError> {
        if !dut1.is_finite() || dut1.abs() >= DUT1_LIMIT_SECONDS {
            return Err(SpaTimeError::Dut1OutOfRange(dut1));
        }
        self.dut1 = dut1;
        Ok(self)
    }

    #[must_use]
    pub const fn datetime(&self) -> &DateTime<Tz> {
        &self.datetime
    }

    #[must_use]
    pub const fn dut1(&self) -> f64 {
        self.dut1
    }
}

impl<Tz: TimeZone> From<DateTime<Tz>> for SpaDateTime<Tz> {
    /// Equivalent to [`SpaDateTime::new`]: wraps the instant with `dut1 = 0`.
    fn from(datetime: DateTime<Tz>) -> Self {
        Self::new(datetime)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use chrono::Utc;

    /// Throwaway timestamp; its value is irrelevant to DUT1 validation.
    fn dt() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 9, 12, 9, 30).unwrap()
    }

    /// Pins the documented default of [`SpaDateTime::new`].
    // Direct float comparison is safe here: 0.0 is a documented constant
    // stored verbatim, with no arithmetic that could introduce rounding.
    #[test]
    #[allow(clippy::float_cmp)]
    fn new_initialises_dut1_to_zero() {
        assert_eq!(SpaDateTime::new(dt()).dut1(), 0.0);
    }

    /// Every finite DUT1 strictly inside (-1, 1) s must round-trip through
    /// the validating constructor and the getter.
    // Direct float comparison is safe here: the same bit pattern flows
    // through the constructor and getter unchanged, with no arithmetic.
    #[test]
    #[allow(clippy::float_cmp)]
    fn try_with_dut1_accepts_finite_values_inside_open_unit_interval() {
        for dut1 in [-0.999, -0.5, 0.0, 0.5, 0.999] {
            let stored = SpaDateTime::new(dt())
                .try_with_dut1(dut1)
                .expect("|dut1| < 1 must be accepted")
                .dut1();
            assert_eq!(stored, dut1);
        }
    }

    /// Pins the *open* interval: |dut1| = 1 must be rejected, matching the
    /// IERS bound advertised by the `Display` message.
    // Direct float comparison is safe here: the error variant echoes the
    // input verbatim, so we are comparing identical bit patterns.
    #[test]
    #[allow(clippy::float_cmp)]
    fn try_with_dut1_rejects_values_at_or_beyond_unit_boundary() {
        for dut1 in [-1.5, -1.0, 1.0, 1.5] {
            let err = SpaDateTime::new(dt())
                .try_with_dut1(dut1)
                .expect_err("|dut1| >= 1 must be rejected");
            match err {
                SpaTimeError::Dut1OutOfRange(reported) => {
                    assert_eq!(reported, dut1, "error must echo the offending DUT1");
                }
            }
        }
    }

    /// `NaN.abs() >= 1` is `false`, so the magnitude check alone would let
    /// non-finite values through. This pins `is_finite()` as the guard.
    #[test]
    fn try_with_dut1_rejects_non_finite_values() {
        for dut1 in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let result = SpaDateTime::new(dt()).try_with_dut1(dut1);
            assert!(
                matches!(result, Err(SpaTimeError::Dut1OutOfRange(_))),
                "non-finite DUT1 ({dut1}) must be rejected"
            );
        }
    }

    /// Pins the [`From`] impl as a sugar-only alias for [`SpaDateTime::new`],
    /// which also enables `impl Into<SpaDateTime<_>>` bounds at call sites.
    #[test]
    fn from_datetime_is_equivalent_to_new() {
        let via_new = SpaDateTime::new(dt());
        let via_into: SpaDateTime<Utc> = dt().into();
        assert_eq!(via_new, via_into);
    }

    /// Pins the public surface of [`SpaDateTime::try_new`]: it must validate
    /// DUT1 with the same rules as [`SpaDateTime::try_with_dut1`].
    #[test]
    fn try_new_validates_dut1_like_try_with_dut1() {
        assert!(SpaDateTime::try_new(dt(), 0.5).is_ok());
        assert!(matches!(
            SpaDateTime::try_new(dt(), 1.0),
            Err(SpaTimeError::Dut1OutOfRange(_))
        ));
        assert!(matches!(
            SpaDateTime::try_new(dt(), f64::NAN),
            Err(SpaTimeError::Dut1OutOfRange(_))
        ));
    }
}
