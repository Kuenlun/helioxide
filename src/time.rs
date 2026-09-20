// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Time inputs for the SPA pipeline.

use chrono::{DateTime, Datelike, TimeZone};
use thiserror::Error;

/// IERS bound on `|DUT1| < 1 s` (leap seconds keep UT1 within this band).
const DUT1_LIMIT_SECONDS: f64 = 1.0;

/// Earliest UTC year covered by the SPA model.
pub const MIN_SPA_YEAR: i32 = -2000;
/// Latest UTC year covered by the SPA model.
pub const MAX_SPA_YEAR: i32 = 6000;
/// One day bounds the TT shift while retaining the polynomial estimates
/// throughout the SPA year range, including ancient dates.
pub const MAX_ABS_DELTA_T_SECONDS: f64 = 86_400.0;

/// Invalid time input or unrepresentable solar event.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SpaTimeError {
    /// DUT1 is non-finite or outside the open interval (-1, 1) seconds.
    #[error("DUT1 must lie in the open interval (-1, 1) s, got {0}")]
    Dut1OutOfRange(f64),
    /// UTC year lies outside the SPA model's supported interval.
    #[error("UTC year {0} must lie in [-2000, 6000]")]
    YearOutOfRange(i32),
    /// Calendar label belongs to the skipped Gregorian reform dates.
    #[error("dates 1582-10-05 through 1582-10-14 do not exist in the SPA mixed calendar")]
    GregorianReformGap,
    /// Delta T is non-finite or exceeds one day in magnitude.
    #[error("delta T {0} s must be finite and lie in [-86400, 86400]")]
    DeltaTOutOfRange(f64),
    /// An event lies outside the bounded search interval or date range.
    #[error("solar event is outside the representable date or search interval")]
    EventOutOfRange,
}

pub(crate) fn validate_spa_time<Tz: TimeZone>(
    datetime: &SpaDateTime<Tz>,
    delta_t: f64,
) -> Result<(), SpaTimeError> {
    let year = datetime.datetime().naive_utc().year();
    if !(MIN_SPA_YEAR..=MAX_SPA_YEAR).contains(&year) {
        return Err(SpaTimeError::YearOutOfRange(year));
    }
    if crate::julian::is_gregorian_reform_gap(datetime.datetime().naive_utc().date()) {
        return Err(SpaTimeError::GregorianReformGap);
    }
    if !(-MAX_ABS_DELTA_T_SECONDS..=MAX_ABS_DELTA_T_SECONDS).contains(&delta_t) {
        return Err(SpaTimeError::DeltaTOutOfRange(delta_t));
    }
    Ok(())
}

/// Instant tagged with `DUT1 = UT1 − UTC` (seconds).
///
/// Default DUT1 is `0`. Use [`Self::try_new`] to attach an IERS value.
/// This wrapper accepts Chrono's date range for calendar conversions.
/// The solar orchestrators separately enforce the SPA year and delta-T limits.
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

    /// Civil instant and its timezone.
    #[must_use]
    pub const fn datetime(&self) -> &DateTime<Tz> {
        &self.datetime
    }

    /// UT1 minus UTC, in seconds.
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
    #[expect(
        clippy::float_cmp,
        reason = "These cases require exact preservation of stored values or exact boundary results."
    )]
    fn new_defaults_dut1_to_zero() {
        assert_eq!(SpaDateTime::new(dt()).dut1(), 0.0_f64);
    }

    #[test]
    #[expect(
        clippy::float_cmp,
        reason = "These cases require exact preservation of stored values or exact boundary results."
    )]
    fn try_with_dut1_accepts_open_unit_interval() {
        for dut1 in [-0.999_f64, -0.5_f64, 0.0_f64, 0.5_f64, 0.999_f64] {
            let stored = SpaDateTime::new(dt()).try_with_dut1(dut1).unwrap().dut1();
            assert_eq!(stored, dut1);
        }
    }

    #[test]
    fn try_with_dut1_rejects_boundary_and_beyond() {
        for dut1 in [-1.5_f64, -1.0_f64, 1.0_f64, 1.5_f64] {
            let error = SpaDateTime::new(dt()).try_with_dut1(dut1).unwrap_err();
            assert_eq!(error, SpaTimeError::Dut1OutOfRange(dut1));
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
    #[expect(
        clippy::float_cmp,
        reason = "These cases require exact preservation of stored values or exact boundary results."
    )]
    fn with_datetime_preserves_dut1_and_swaps_instant() {
        let original = SpaDateTime::try_new(dt(), 0.25).unwrap();
        let new_instant = Utc.with_ymd_and_hms(2030, 1, 2, 3, 4, 5).unwrap();
        let retargeted = original.with_datetime(new_instant);
        assert_eq!(retargeted.dut1(), 0.25_f64);
        assert_eq!(retargeted.datetime(), &new_instant);
    }

    #[test]
    fn solar_orchestrators_reject_unsupported_dates_and_non_finite_delta_t() {
        use crate::{Observer, SolarDay, SolarPosition};

        let observer = Observer::try_at_sea_level_isa(0.0, -179.0).unwrap();
        for invalid in [DateTime::<Utc>::MIN_UTC, DateTime::<Utc>::MAX_UTC] {
            let datetime = SpaDateTime::new(invalid);
            assert!(matches!(
                SolarPosition::compute(&datetime, observer),
                Err(SpaTimeError::YearOutOfRange(_))
            ));
            assert!(matches!(
                SolarDay::compute(&datetime, observer),
                Err(SpaTimeError::YearOutOfRange(_))
            ));
        }
        let datetime = SpaDateTime::new(dt());
        for delta_t in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -86_400.001_f64,
            86_400.001_f64,
        ] {
            assert!(matches!(
                SolarPosition::compute_with_delta_t(&datetime, delta_t, observer),
                Err(SpaTimeError::DeltaTOutOfRange(_))
            ));
            assert!(matches!(
                SolarDay::compute_with_delta_t(&datetime, delta_t, observer),
                Err(SpaTimeError::DeltaTOutOfRange(_))
            ));
        }
    }

    #[test]
    fn solar_position_supports_the_model_endpoints_and_polynomial_delta_t() {
        use crate::{Observer, SolarPosition};

        let observer = Observer::try_at_sea_level_isa(0.0, 0.0).unwrap();
        for year in [MIN_SPA_YEAR, MAX_SPA_YEAR] {
            let datetime = SpaDateTime::new(Utc.with_ymd_and_hms(year, 1, 1, 12, 0, 0).unwrap());
            assert!(
                SolarPosition::compute(&datetime, observer)
                    .unwrap()
                    .topocentric_zenith
                    .is_finite()
            );
            for delta_t in [-MAX_ABS_DELTA_T_SECONDS, MAX_ABS_DELTA_T_SECONDS] {
                assert!(
                    SolarPosition::compute_with_delta_t(&datetime, delta_t, observer)
                        .unwrap()
                        .topocentric_azimuth
                        .is_finite()
                );
            }
        }
    }

    #[test]
    fn time_errors_describe_the_invalid_quantity() {
        for (error, word) in [
            (SpaTimeError::Dut1OutOfRange(1.0), "DUT1"),
            (SpaTimeError::YearOutOfRange(6001), "year"),
            (SpaTimeError::DeltaTOutOfRange(f64::NAN), "delta T"),
            (SpaTimeError::EventOutOfRange, "event"),
        ] {
            assert!(error.to_string().contains(word));
        }
    }
}
