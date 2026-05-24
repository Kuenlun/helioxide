// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Time inputs for the SPA pipeline.

use chrono::{DateTime, TimeZone};
use thiserror::Error;

/// IERS bound on `|DUT1| < 1 s` (leap seconds keep UT1 within this band).
const DUT1_LIMIT_SECONDS: f64 = 1.0;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum SpaTimeError {
    #[error("DUT1 must lie in the open interval (-1, 1) s, got {0}")]
    Dut1OutOfRange(f64),
}

/// Instant tagged with `DUT1 = UT1 − UTC` (seconds).
///
/// Default DUT1 is `0`. Use [`Self::try_new`] to attach an IERS value.
#[derive(Debug, Clone, PartialEq)]
pub struct SpaDateTime<Tz: TimeZone> {
    datetime: DateTime<Tz>,
    dut1: f64,
}

impl<Tz: TimeZone> SpaDateTime<Tz> {
    /// Build with `dut1 = 0`.
    #[must_use]
    pub const fn new(datetime: DateTime<Tz>) -> Self {
        Self {
            datetime,
            dut1: 0.0,
        }
    }

    /// # Errors
    /// [`SpaTimeError::Dut1OutOfRange`] if `dut1` is non-finite or `|dut1| >= 1`.
    pub fn try_new(datetime: DateTime<Tz>, dut1: f64) -> Result<Self, SpaTimeError> {
        Self::new(datetime).try_with_dut1(dut1)
    }

    /// # Errors
    /// [`SpaTimeError::Dut1OutOfRange`] if `dut1` is non-finite or `|dut1| >= 1`.
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

    /// Swap the wall-clock instant, preserving DUT1 (already validated on `self`).
    #[must_use]
    pub const fn with_datetime<NewTz: TimeZone>(
        &self,
        datetime: DateTime<NewTz>,
    ) -> SpaDateTime<NewTz> {
        SpaDateTime {
            datetime,
            dut1: self.dut1,
        }
    }
}

impl<Tz: TimeZone> From<DateTime<Tz>> for SpaDateTime<Tz> {
    fn from(datetime: DateTime<Tz>) -> Self {
        Self::new(datetime)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use chrono::Utc;

    fn dt() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 5, 9, 12, 9, 30).unwrap()
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn new_defaults_dut1_to_zero() {
        assert_eq!(SpaDateTime::new(dt()).dut1(), 0.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn try_with_dut1_accepts_open_unit_interval() {
        for dut1 in [-0.999, -0.5, 0.0, 0.5, 0.999] {
            let stored = SpaDateTime::new(dt()).try_with_dut1(dut1).unwrap().dut1();
            assert_eq!(stored, dut1);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn try_with_dut1_rejects_boundary_and_beyond() {
        for dut1 in [-1.5, -1.0, 1.0, 1.5] {
            let SpaTimeError::Dut1OutOfRange(reported) =
                SpaDateTime::new(dt()).try_with_dut1(dut1).unwrap_err();
            assert_eq!(reported, dut1);
        }
    }

    #[test]
    fn try_with_dut1_rejects_non_finite() {
        for dut1 in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                SpaDateTime::new(dt()).try_with_dut1(dut1),
                Err(SpaTimeError::Dut1OutOfRange(_)),
            ));
        }
    }

    #[test]
    fn from_datetime_equals_new() {
        let via_into: SpaDateTime<Utc> = dt().into();
        assert_eq!(SpaDateTime::new(dt()), via_into);
    }

    #[test]
    fn try_new_validates_like_try_with_dut1() {
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

    #[test]
    #[allow(clippy::float_cmp)]
    fn with_datetime_preserves_dut1_and_swaps_instant() {
        let original = SpaDateTime::try_new(dt(), 0.25).unwrap();
        let new_instant = Utc.with_ymd_and_hms(2030, 1, 2, 3, 4, 5).unwrap();
        let retargeted = original.with_datetime(new_instant);
        assert_eq!(retargeted.dut1(), 0.25);
        assert_eq!(retargeted.datetime(), &new_instant);
    }
}
