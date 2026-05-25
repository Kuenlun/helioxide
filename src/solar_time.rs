// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Sun transit, sunrise and sunset for a given civil day. Appendix A.2.
//!
//! [`SolarDay::compute`] anchors `D₀` on the local civil date of the input
//! (section 3.1.1 allows driving JD from local time and section A.3 confirms
//! the same shift on read-out), so the three returned events fall on the
//! input's local civil date when the timezone tracks the observer.

use core::fmt;

use chrono::{DateTime, MappedLocalTime, TimeDelta, TimeZone, Utc};

use crate::SpaDateTime;
use crate::apparent::{aberration_correction, apparent_sun_longitude};
use crate::equatorial::{geocentric_declination, geocentric_right_ascension};
use crate::geocentric::{geocentric_latitude, geocentric_longitude};
use crate::heliocentric::{
    earth_heliocentric_latitude, earth_heliocentric_longitude, earth_radius_vector,
};
use crate::julian::{
    calculate_julian_day, calculate_julian_ephemeris_century, calculate_julian_ephemeris_day,
    calculate_julian_ephemeris_millennium,
};
use crate::nutation::nutation_in_longitude_and_obliquity;
use crate::obliquity::true_obliquity_of_ecliptic;
use crate::sidereal::{apparent_sidereal_time, mean_sidereal_time};
use crate::spa::Observer;

/// `h'₀ = -0.8333°` (solar disk radius `0.26667°` plus horizon-level
/// refraction `0.5667°`), per appendix A.2.
pub const SUN_ELEVATION_AT_HORIZON_DEGREES: f64 = -0.8333;

/// Earth's sidereal rotation rate `360.985647°/day` of equation A7.
const EARTH_SIDEREAL_DAILY_ROTATION_DEGREES: f64 = 360.985_647;

const SECONDS_PER_DAY: f64 = 86_400.0;
const HALF_REVOLUTION_DEGREES: f64 = 180.0;
const FULL_REVOLUTION_DEGREES: f64 = 360.0;

/// Threshold above which interpolation differences are treated as a `360°`
/// wrap (step A.2.10). `2°` comfortably exceeds the diurnal motion of `α`
/// (`≤ ~1°/day`) and `δ` (`≤ ~0.4°/day`).
const INTERPOLATION_WRAP_THRESHOLD_DEGREES: f64 = 2.0;

const J2000_EPOCH_JD: f64 = 2_451_545.0;

/// `m₀ = (α₀ − σ − ν) / 360`, wrapped into `[0, 1)`. Equation A3.
#[inline]
#[must_use]
pub fn approximate_sun_transit_time(
    geocentric_right_ascension_at_0tt: f64,
    observer_longitude: f64,
    apparent_sidereal_time_at_0ut: f64,
) -> f64 {
    let raw =
        (geocentric_right_ascension_at_0tt - observer_longitude - apparent_sidereal_time_at_0ut)
            / FULL_REVOLUTION_DEGREES;
    raw.rem_euclid(1.0)
}

/// Local hour angle `H₀` at sun centre `h'₀`, or `None` for polar day/night.
/// Equation A4.
#[inline]
#[must_use]
pub fn sunrise_sunset_local_hour_angle(
    observer_latitude: f64,
    geocentric_declination_at_0tt: f64,
    sun_horizon_elevation: f64,
) -> Option<f64> {
    let (sin_phi, cos_phi) = observer_latitude.to_radians().sin_cos();
    let (sin_delta, cos_delta) = geocentric_declination_at_0tt.to_radians().sin_cos();
    let sin_h0 = sun_horizon_elevation.to_radians().sin();

    let argument = (-sin_phi).mul_add(sin_delta, sin_h0) / (cos_phi * cos_delta);
    if !(-1.0..=1.0).contains(&argument) {
        return None;
    }
    Some(argument.acos().to_degrees())
}

/// `m₁ = m₀ − H₀ / 360`, wrapped into `[0, 1)`. Equation A5.
#[inline]
#[must_use]
pub fn approximate_sunrise_time(
    approximate_sun_transit_time: f64,
    sunrise_sunset_local_hour_angle: f64,
) -> f64 {
    (approximate_sun_transit_time - sunrise_sunset_local_hour_angle / FULL_REVOLUTION_DEGREES)
        .rem_euclid(1.0)
}

/// `m₂ = m₀ + H₀ / 360`, wrapped into `[0, 1)`. Equation A6.
#[inline]
#[must_use]
pub fn approximate_sunset_time(
    approximate_sun_transit_time: f64,
    sunrise_sunset_local_hour_angle: f64,
) -> f64 {
    (approximate_sun_transit_time + sunrise_sunset_local_hour_angle / FULL_REVOLUTION_DEGREES)
        .rem_euclid(1.0)
}

/// `νᵢ = ν + 360.985647 · mᵢ` (degrees, signed, not wrapped). Equation A7.
#[inline]
#[must_use]
pub const fn sidereal_time_at_event(
    apparent_sidereal_time_at_0ut: f64,
    approximate_event_time: f64,
) -> f64 {
    apparent_sidereal_time_at_0ut + EARTH_SIDEREAL_DAILY_ROTATION_DEGREES * approximate_event_time
}

/// `nᵢ = mᵢ + ΔT / 86_400`. Equation A8.
#[inline]
#[must_use]
pub const fn delta_t_corrected_event_time(
    approximate_event_time: f64,
    delta_t_seconds: f64,
) -> f64 {
    approximate_event_time + delta_t_seconds / SECONDS_PER_DAY
}

/// Three-point Stirling interpolation of an angular quantity at `nᵢ`,
/// folding `360°` wraps in the first differences (step A.2.10). Equations A9
/// and A10.
#[inline]
#[must_use]
pub fn interpolate_three_day_value(
    value_minus_one_day: f64,
    value_zero_day: f64,
    value_plus_one_day: f64,
    delta_t_corrected_event_time: f64,
) -> f64 {
    let a = wrap_interpolation_difference(value_zero_day - value_minus_one_day);
    let b = wrap_interpolation_difference(value_plus_one_day - value_zero_day);
    let c = b - a;
    let n = delta_t_corrected_event_time;
    c.mul_add(n, a + b).mul_add(n / 2.0, value_zero_day)
}

fn wrap_interpolation_difference(diff: f64) -> f64 {
    if diff.abs() > INTERPOLATION_WRAP_THRESHOLD_DEGREES {
        FULL_REVOLUTION_DEGREES.mul_add(-(diff / FULL_REVOLUTION_DEGREES).round(), diff)
    } else {
        diff
    }
}

/// `H'ᵢ = νᵢ + σ − α'ᵢ`, wrapped into `(-180°, 180°]`. Equation A11.
#[inline]
#[must_use]
pub fn event_local_hour_angle(
    sidereal_time_at_event: f64,
    observer_longitude: f64,
    interpolated_right_ascension: f64,
) -> f64 {
    let raw = sidereal_time_at_event + observer_longitude - interpolated_right_ascension;
    let wrapped = raw.rem_euclid(FULL_REVOLUTION_DEGREES);
    if wrapped > HALF_REVOLUTION_DEGREES {
        wrapped - FULL_REVOLUTION_DEGREES
    } else {
        wrapped
    }
}

/// `hᵢ = arcsin(sin φ · sin δ'ᵢ + cos φ · cos δ'ᵢ · cos H'ᵢ)` (degrees).
/// Equation A12.
#[inline]
#[must_use]
pub fn sun_altitude_at_event(
    observer_latitude: f64,
    interpolated_declination: f64,
    event_local_hour_angle: f64,
) -> f64 {
    let (sin_phi, cos_phi) = observer_latitude.to_radians().sin_cos();
    let (sin_delta, cos_delta) = interpolated_declination.to_radians().sin_cos();
    let cos_h = event_local_hour_angle.to_radians().cos();

    (cos_phi * cos_delta)
        .mul_add(cos_h, sin_phi * sin_delta)
        .asin()
        .to_degrees()
}

/// `T = m₀ − H'₀ / 360`. Equation A13.
#[inline]
#[must_use]
pub fn sun_transit_time(
    approximate_sun_transit_time: f64,
    event_local_hour_angle_at_transit: f64,
) -> f64 {
    approximate_sun_transit_time - event_local_hour_angle_at_transit / FULL_REVOLUTION_DEGREES
}

/// `R or S = mᵢ + (hᵢ − h'₀) / (360 · cos δ'ᵢ · cos φ · sin H'ᵢ)`.
/// Equation A14 (Newton step of `h(t) = h'₀`), applied with `i = 1` for
/// sunrise and `i = 2` for sunset per step A.2.15.
#[inline]
#[must_use]
#[allow(clippy::too_many_arguments)]
pub fn sunrise_or_sunset_time(
    approximate_event_time: f64,
    sun_altitude_at_event: f64,
    sun_horizon_elevation: f64,
    interpolated_declination: f64,
    observer_latitude: f64,
    event_local_hour_angle: f64,
) -> f64 {
    let cos_delta = interpolated_declination.to_radians().cos();
    let cos_phi = observer_latitude.to_radians().cos();
    let sin_h = event_local_hour_angle.to_radians().sin();
    let denominator = FULL_REVOLUTION_DEGREES * cos_delta * cos_phi * sin_h;
    approximate_event_time + (sun_altitude_at_event - sun_horizon_elevation) / denominator
}

/// Sun transit (solar noon), sunrise and sunset for a single civil day.
///
/// `sunrise` and `sunset` are `None` for polar day or polar night. `transit`
/// is always populated.
#[derive(Debug, Clone, PartialEq)]
pub struct SolarDay<Tz: TimeZone> {
    pub transit: DateTime<Tz>,
    pub sunrise: Option<DateTime<Tz>>,
    pub sunset: Option<DateTime<Tz>>,
    /// Sun altitude at transit (degrees, in `[-90°, 90°]`). Equation A12.
    pub sun_transit_altitude: f64,
    /// `H'₁` at sunrise (degrees). Equation A11. `None` for polar day/night.
    pub sunrise_hour_angle: Option<f64>,
    /// `H'₂` at sunset (degrees). Equation A11. `None` for polar day/night.
    pub sunset_hour_angle: Option<f64>,
}

impl<Tz: TimeZone> SolarDay<Tz> {
    /// Run appendix A.2 end to end.
    ///
    /// `D₀` is the local civil date of `datetime` in its own timezone, and
    /// the result is rendered on the same timezone. Pass a [`DateTime<Utc>`]
    /// to anchor on the UTC civil day.
    ///
    /// `ΔT` is resolved via [`crate::delta_t::delta_t_seconds_for_datetime`]
    /// (observed USNO value when available, polynomial approximation
    /// otherwise). Reach for [`Self::compute_with_delta_t`] when a specific
    /// `ΔT` is required, for example to reproduce NREL reference values.
    ///
    /// `observer.elevation`, `pressure` and `temperature` are unused:
    /// appendix A.2 absorbs the horizon-level refraction into
    /// [`SUN_ELEVATION_AT_HORIZON_DEGREES`].
    ///
    /// [`DateTime<Utc>`]: chrono::DateTime
    #[must_use]
    pub fn compute(datetime: &SpaDateTime<Tz>, observer: Observer) -> Self {
        let delta_t = crate::delta_t::delta_t_seconds_for_datetime(datetime.datetime());
        Self::compute_with_delta_t(datetime, delta_t, observer)
    }

    /// Compute the same sunrise, transit and sunset readout as [`Self::compute`]
    /// with an explicit `ΔT = TT − UT1` (seconds).
    #[must_use]
    #[allow(clippy::many_single_char_names, clippy::similar_names)]
    pub fn compute_with_delta_t(
        datetime: &SpaDateTime<Tz>,
        delta_t_seconds: f64,
        observer: Observer,
    ) -> Self {
        let tz = datetime.datetime().timezone();
        let utc_anchor = local_civil_midnight_in_utc(datetime.datetime());
        let datetime_at_anchor = datetime.with_datetime(utc_anchor);

        // Step A.2.1: ν at the anchor.
        let jd_0 = calculate_julian_day(&datetime_at_anchor);
        let jde_0 = calculate_julian_ephemeris_day(jd_0, delta_t_seconds);
        let jce_0 = calculate_julian_ephemeris_century(jde_0);
        let jme_0 = calculate_julian_ephemeris_millennium(jce_0);
        let (delta_psi_0, delta_epsilon_0) = nutation_in_longitude_and_obliquity(jce_0);
        let epsilon_0 = true_obliquity_of_ecliptic(jme_0, delta_epsilon_0);
        let nu = apparent_sidereal_time(mean_sidereal_time(jd_0), delta_psi_0, epsilon_0);

        // Step A.2.2: (α, δ) on D₋₁, D₀, D₊₁ (TT).
        let (alpha_minus, delta_minus) = right_ascension_and_declination(jde_0 - 1.0);
        let (alpha_zero, delta_zero) = right_ascension_and_declination(jde_0);
        let (alpha_plus, delta_plus) = right_ascension_and_declination(jde_0 + 1.0);

        // Step A.2.3: m₀.
        let m_0 = approximate_sun_transit_time(alpha_zero, observer.longitude(), nu);

        // Step A.2.4: H₀ (None for polar day/night).
        let h_0 = sunrise_sunset_local_hour_angle(
            observer.latitude(),
            delta_zero,
            SUN_ELEVATION_AT_HORIZON_DEGREES,
        );

        let transit_event = refined_event_fraction_of_day(
            EventKind::Transit,
            m_0,
            nu,
            observer,
            delta_t_seconds,
            (alpha_minus, alpha_zero, alpha_plus),
            (delta_minus, delta_zero, delta_plus),
        );

        let (sunrise_event, sunset_event) = h_0.map_or((None, None), |h0| {
            let m_1 = approximate_sunrise_time(m_0, h0);
            let m_2 = approximate_sunset_time(m_0, h0);
            let r = refined_event_fraction_of_day(
                EventKind::Sunrise,
                m_1,
                nu,
                observer,
                delta_t_seconds,
                (alpha_minus, alpha_zero, alpha_plus),
                (delta_minus, delta_zero, delta_plus),
            );
            let s = refined_event_fraction_of_day(
                EventKind::Sunset,
                m_2,
                nu,
                observer,
                delta_t_seconds,
                (alpha_minus, alpha_zero, alpha_plus),
                (delta_minus, delta_zero, delta_plus),
            );
            (Some(r), Some(s))
        });

        // Wrap T into [0, 1) and unwrap R, S to the closest representative
        // around T, preserving sunrise < transit < sunset across day boundaries.
        let transit_wrapped = transit_event.fraction_of_day.rem_euclid(1.0);
        let to_datetime = |fraction_of_day: f64| -> DateTime<Tz> {
            // Round to whole milliseconds: appendix A.2 publishes to 0.01 s.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let milliseconds = (fraction_of_day * (SECONDS_PER_DAY * 1000.0)).round() as i64;
            (utc_anchor + TimeDelta::milliseconds(milliseconds)).with_timezone(&tz)
        };
        let unwrap_to_transit = |fraction: f64| -> f64 {
            let raw = fraction - transit_wrapped;
            if raw > 0.5 {
                fraction - 1.0
            } else if raw < -0.5 {
                fraction + 1.0
            } else {
                fraction
            }
        };

        let sun_transit_altitude = sun_altitude_at_event(
            observer.latitude(),
            transit_event.interpolated_declination,
            transit_event.local_hour_angle,
        );

        Self {
            transit: to_datetime(transit_wrapped),
            sunrise: sunrise_event.map(|r| to_datetime(unwrap_to_transit(r.fraction_of_day))),
            sunset: sunset_event.map(|s| to_datetime(unwrap_to_transit(s.fraction_of_day))),
            sun_transit_altitude,
            sunrise_hour_angle: sunrise_event.map(|r| r.local_hour_angle),
            sunset_hour_angle: sunset_event.map(|s| s.local_hour_angle),
        }
    }
}

impl<Tz: TimeZone> fmt::Display for SolarDay<Tz>
where
    DateTime<Tz>: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match &self.sunrise {
            Some(sunrise) => writeln!(f, "Sunrise:                {sunrise}")?,
            None => writeln!(f, "Sunrise:                none (polar day or polar night)")?,
        }
        writeln!(f, "Sun transit:            {}", self.transit)?;
        match &self.sunset {
            Some(sunset) => writeln!(f, "Sunset:                 {sunset}")?,
            None => writeln!(f, "Sunset:                 none (polar day or polar night)")?,
        }
        match self.sunrise_hour_angle {
            Some(hour_angle) => writeln!(f, "Sunrise hour angle:     {hour_angle}°")?,
            None => writeln!(f, "Sunrise hour angle:     none (polar day or polar night)")?,
        }
        match self.sunset_hour_angle {
            Some(hour_angle) => writeln!(f, "Sunset hour angle:      {hour_angle}°")?,
            None => writeln!(f, "Sunset hour angle:      none (polar day or polar night)")?,
        }
        write!(f, "Sun transit altitude:   {}°", self.sun_transit_altitude)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventKind {
    Transit,
    Sunrise,
    Sunset,
}

#[derive(Debug, Clone, Copy)]
struct RefinedEvent {
    fraction_of_day: f64,
    local_hour_angle: f64,
    interpolated_declination: f64,
}

#[allow(clippy::similar_names, clippy::too_many_arguments)]
fn refined_event_fraction_of_day(
    kind: EventKind,
    approximate_event_time: f64,
    apparent_sidereal_time_at_0ut: f64,
    observer: Observer,
    delta_t_seconds: f64,
    alpha_three_day: (f64, f64, f64),
    delta_three_day: (f64, f64, f64),
) -> RefinedEvent {
    let m = approximate_event_time;
    let nu_i = sidereal_time_at_event(apparent_sidereal_time_at_0ut, m);
    let n_i = delta_t_corrected_event_time(m, delta_t_seconds);
    let (alpha_minus, alpha_zero, alpha_plus) = alpha_three_day;
    let (delta_minus, delta_zero, delta_plus) = delta_three_day;
    let alpha_prime = interpolate_three_day_value(alpha_minus, alpha_zero, alpha_plus, n_i);
    let delta_prime = interpolate_three_day_value(delta_minus, delta_zero, delta_plus, n_i);
    let h_prime = event_local_hour_angle(nu_i, observer.longitude(), alpha_prime);

    let fraction_of_day = match kind {
        EventKind::Transit => sun_transit_time(m, h_prime),
        EventKind::Sunrise | EventKind::Sunset => {
            let h_at_event = sun_altitude_at_event(observer.latitude(), delta_prime, h_prime);
            sunrise_or_sunset_time(
                m,
                h_at_event,
                SUN_ELEVATION_AT_HORIZON_DEGREES,
                delta_prime,
                observer.latitude(),
                h_prime,
            )
        }
    };

    RefinedEvent {
        fraction_of_day,
        local_hour_angle: h_prime,
        interpolated_declination: delta_prime,
    }
}

/// UT instant of local civil midnight at the start of `datetime`'s local date.
///
/// DST corner cases:
/// * `Ambiguous` (fall back over midnight): take the earliest representation.
/// * `None` (spring forward over midnight): reinterpret naive local midnight
///   as UT, so events may be off by the DST gap on the input's wall clock.
fn local_civil_midnight_in_utc<Tz: TimeZone>(datetime: &DateTime<Tz>) -> DateTime<Utc> {
    let local_midnight = datetime.date_naive().and_time(chrono::NaiveTime::MIN);
    match datetime.timezone().from_local_datetime(&local_midnight) {
        MappedLocalTime::Single(t) | MappedLocalTime::Ambiguous(t, _) => t.with_timezone(&Utc),
        MappedLocalTime::None => local_midnight.and_utc(),
    }
}

/// `(α, δ)` (degrees) at the given Julian Ephemeris Day.
fn right_ascension_and_declination(julian_ephemeris_day: f64) -> (f64, f64) {
    let jce = (julian_ephemeris_day - J2000_EPOCH_JD) / 36_525.0;
    let jme = jce / 10.0;

    let l = earth_heliocentric_longitude(jme);
    let b = earth_heliocentric_latitude(jme);
    let r = earth_radius_vector(jme);

    let theta = geocentric_longitude(l);
    let beta = geocentric_latitude(b);

    let (delta_psi, delta_epsilon) = nutation_in_longitude_and_obliquity(jce);
    let epsilon = true_obliquity_of_ecliptic(jme, delta_epsilon);
    let delta_tau = aberration_correction(r);
    let lambda = apparent_sun_longitude(theta, delta_psi, delta_tau);

    let alpha = geocentric_right_ascension(lambda, beta, epsilon);
    let delta = geocentric_declination(lambda, beta, epsilon);
    (alpha, delta)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{
        EARTH_SIDEREAL_DAILY_ROTATION_DEGREES, EventKind, SUN_ELEVATION_AT_HORIZON_DEGREES,
        SolarDay, approximate_sun_transit_time, approximate_sunrise_time, approximate_sunset_time,
        delta_t_corrected_event_time, event_local_hour_angle, interpolate_three_day_value,
        refined_event_fraction_of_day, right_ascension_and_declination, sidereal_time_at_event,
        sun_altitude_at_event, sun_transit_time, sunrise_or_sunset_time,
        sunrise_sunset_local_hour_angle, wrap_interpolation_difference,
    };
    use crate::spa::Observer;
    use crate::test_fixtures::{
        REFERENCE_ELEVATION_METRES, REFERENCE_LATITUDE_DEGREES, REFERENCE_LONGITUDE_DEGREES,
        REFERENCE_PRESSURE_MILLIBARS, REFERENCE_TEMPERATURE_CELSIUS,
    };
    use crate::{SpaDateTime, julian};
    use chrono::{TimeZone, Utc};

    const REFERENCE_DELTA_T_SECONDS: f64 = 67.0;

    /// `fmt::Write` sink that fails once a fixed byte budget is exhausted.
    struct FailingWriter {
        remaining: usize,
    }

    impl core::fmt::Write for FailingWriter {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            if s.len() > self.remaining {
                self.remaining = 0;
                Err(core::fmt::Error)
            } else {
                self.remaining -= s.len();
                Ok(())
            }
        }
    }

    fn reference_observer() -> Observer {
        Observer::try_new(
            REFERENCE_LATITUDE_DEGREES,
            REFERENCE_LONGITUDE_DEGREES,
            REFERENCE_ELEVATION_METRES,
            REFERENCE_PRESSURE_MILLIBARS,
            REFERENCE_TEMPERATURE_CELSIUS,
        )
        .unwrap()
    }

    fn reference_datetime() -> SpaDateTime<Utc> {
        SpaDateTime::new(Utc.with_ymd_and_hms(2003, 10, 17, 19, 30, 30).unwrap())
    }

    fn reference_solar_day() -> SolarDay<Utc> {
        SolarDay::compute_with_delta_t(
            &reference_datetime(),
            REFERENCE_DELTA_T_SECONDS,
            reference_observer(),
        )
    }

    fn fraction_to_clock_seconds(fraction_of_day: f64) -> i64 {
        #[allow(clippy::cast_possible_truncation)]
        {
            (fraction_of_day * 86_400.0).round() as i64
        }
    }

    #[test]
    fn solar_day_matches_table_a5_1() {
        let day = reference_solar_day();
        let utc_midnight = Utc.with_ymd_and_hms(2003, 10, 17, 0, 0, 0).unwrap();

        // Transit: 18:46:04.97 UT.
        let expected_transit = utc_midnight
            + chrono::TimeDelta::seconds(67_564)
            + chrono::TimeDelta::milliseconds(970);
        assert!((day.transit - expected_transit).num_milliseconds().abs() < 1_000);

        // Sunrise: 13:12:43.46 UT.
        let sunrise = day.sunrise.unwrap();
        let expected_sunrise = utc_midnight
            + chrono::TimeDelta::seconds(47_563)
            + chrono::TimeDelta::milliseconds(460);
        assert!((sunrise - expected_sunrise).num_milliseconds().abs() < 1_000);

        // Sunset: 00:20:19.19 UT on Oct 18.
        let sunset = day.sunset.unwrap();
        let expected_sunset = utc_midnight
            + chrono::TimeDelta::days(1)
            + chrono::TimeDelta::seconds(1_219)
            + chrono::TimeDelta::milliseconds(190);
        assert!((sunset - expected_sunset).num_milliseconds().abs() < 1_000);
    }

    #[test]
    fn solar_day_orders_sunrise_before_transit_before_sunset() {
        let day = reference_solar_day();
        let sunrise = day.sunrise.unwrap();
        let sunset = day.sunset.unwrap();
        assert!(sunrise < day.transit);
        assert!(day.transit < sunset);
    }

    #[test]
    fn solar_day_preserves_input_timezone() {
        let madrid_dt = SpaDateTime::new(
            chrono_tz::Europe::Madrid
                .with_ymd_and_hms(2003, 10, 17, 21, 30, 30)
                .unwrap(),
        );
        let day = SolarDay::compute_with_delta_t(
            &madrid_dt,
            REFERENCE_DELTA_T_SECONDS,
            reference_observer(),
        );

        let expected_sunrise = chrono_tz::Europe::Madrid
            .with_ymd_and_hms(2003, 10, 17, 15, 12, 43)
            .unwrap()
            + chrono::TimeDelta::milliseconds(460);
        assert!(
            (day.sunrise.unwrap() - expected_sunrise)
                .num_milliseconds()
                .abs()
                < 1_000
        );

        let expected_transit = chrono_tz::Europe::Madrid
            .with_ymd_and_hms(2003, 10, 17, 20, 46, 4)
            .unwrap()
            + chrono::TimeDelta::milliseconds(970);
        assert!((day.transit - expected_transit).num_milliseconds().abs() < 1_000);

        let expected_sunset = chrono_tz::Europe::Madrid
            .with_ymd_and_hms(2003, 10, 18, 2, 20, 19)
            .unwrap()
            + chrono::TimeDelta::milliseconds(190);
        assert!(
            (day.sunset.unwrap() - expected_sunset)
                .num_milliseconds()
                .abs()
                < 1_000
        );
    }

    #[test]
    fn solar_day_anchors_on_input_local_civil_date() {
        let madrid = chrono_tz::Europe::Madrid;
        let morning = SpaDateTime::new(madrid.with_ymd_and_hms(2003, 10, 17, 9, 0, 0).unwrap());
        let early = SpaDateTime::new(madrid.with_ymd_and_hms(2003, 10, 17, 1, 0, 0).unwrap());
        let day_morning = SolarDay::compute_with_delta_t(
            &morning,
            REFERENCE_DELTA_T_SECONDS,
            reference_observer(),
        );
        let day_early =
            SolarDay::compute_with_delta_t(&early, REFERENCE_DELTA_T_SECONDS, reference_observer());
        assert!(
            (day_morning.transit - day_early.transit)
                .num_milliseconds()
                .abs()
                < 1_000,
        );
        assert_eq!(
            day_morning.transit.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2003, 10, 17).unwrap(),
        );
    }

    #[test]
    fn solar_day_handles_ambiguous_local_midnight_on_dst_fall_back() {
        // Azores fall back at 01:00 LST on 2003-10-26: midnight is ambiguous.
        let azores = chrono_tz::Atlantic::Azores;
        let dt = SpaDateTime::new(azores.with_ymd_and_hms(2003, 10, 26, 12, 0, 0).unwrap());
        let observer = Observer::try_new(37.741, -25.668, 50.0, 1015.0, 18.0).unwrap();
        let day = SolarDay::compute_with_delta_t(&dt, REFERENCE_DELTA_T_SECONDS, observer);
        assert!(day.sunrise.unwrap() < day.transit);
        assert!(day.transit < day.sunset.unwrap());
    }

    #[test]
    fn solar_day_handles_skipped_local_midnight_on_dst_spring_forward() {
        // Sao_Paulo sprang forward at midnight 2017-10-15: no UT representation.
        let sao_paulo = chrono_tz::America::Sao_Paulo;
        let dt = SpaDateTime::new(sao_paulo.with_ymd_and_hms(2017, 10, 15, 12, 0, 0).unwrap());
        let observer = Observer::try_new(-23.533, -46.625, 760.0, 1010.0, 22.0).unwrap();
        let day = SolarDay::compute_with_delta_t(&dt, REFERENCE_DELTA_T_SECONDS, observer);
        assert!(day.sunrise.unwrap() < day.transit);
        assert!(day.transit < day.sunset.unwrap());
    }

    #[test]
    fn solar_day_propagates_dut1_into_event_times() {
        let zero = SolarDay::compute_with_delta_t(
            &reference_datetime(),
            REFERENCE_DELTA_T_SECONDS,
            reference_observer(),
        );
        let with_dut1 = SolarDay::compute_with_delta_t(
            &reference_datetime().try_with_dut1(0.5).unwrap(),
            REFERENCE_DELTA_T_SECONDS,
            reference_observer(),
        );
        let shift_ms = (with_dut1.transit - zero.transit).num_milliseconds();
        // UT1 leads UTC by DUT1, so the UTC clock reads the event earlier.
        assert!((shift_ms - -500).abs() < 50);
    }

    #[test]
    fn solar_day_polar_night_returns_none() {
        let polar = Observer::try_new(80.0, 0.0, 0.0, 1010.0, -20.0).unwrap();
        let solstice = SpaDateTime::new(Utc.with_ymd_and_hms(2026, 12, 21, 12, 0, 0).unwrap());
        let day = SolarDay::compute_with_delta_t(&solstice, 70.0, polar);
        assert!(day.sunrise.is_none());
        assert!(day.sunset.is_none());
    }

    #[test]
    fn solar_day_polar_day_returns_none() {
        let polar = Observer::try_new(80.0, 0.0, 0.0, 1010.0, 0.0).unwrap();
        let solstice = SpaDateTime::new(Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0).unwrap());
        let day = SolarDay::compute_with_delta_t(&solstice, 70.0, polar);
        assert!(day.sunrise.is_none());
        assert!(day.sunset.is_none());
    }

    #[test]
    fn approximate_sun_transit_time_is_linear_inside_range() {
        let baseline = approximate_sun_transit_time(200.0, -50.0, 50.0);
        for &d in &[-1.0_f64, -1e-3, 1e-6, 0.5] {
            assert!(
                (approximate_sun_transit_time(200.0 + d, -50.0, 50.0) - baseline - d / 360.0).abs()
                    < 1e-13,
            );
            assert!(
                (approximate_sun_transit_time(200.0, -50.0 + d, 50.0) - baseline + d / 360.0).abs()
                    < 1e-13,
            );
            assert!(
                (approximate_sun_transit_time(200.0, -50.0, 50.0 + d) - baseline + d / 360.0).abs()
                    < 1e-13,
            );
        }
    }

    #[test]
    fn approximate_sun_transit_time_wraps_into_unit_interval() {
        for &(alpha, sigma, nu) in &[
            (0.0_f64, 0.0, 0.0),
            (-720.0, 0.0, 0.0),
            (720.0, 0.0, 0.0),
            (0.0, 360.0, 0.0),
        ] {
            let m_0 = approximate_sun_transit_time(alpha, sigma, nu);
            assert!((0.0..1.0).contains(&m_0));
        }
    }

    #[test]
    fn sunrise_sunset_local_hour_angle_returns_none_for_polar() {
        assert!(
            sunrise_sunset_local_hour_angle(80.0, -23.0, SUN_ELEVATION_AT_HORIZON_DEGREES)
                .is_none(),
        );
        assert!(
            sunrise_sunset_local_hour_angle(80.0, 23.0, SUN_ELEVATION_AT_HORIZON_DEGREES).is_none(),
        );
    }

    #[test]
    fn sunrise_sunset_local_hour_angle_at_equator_equinox() {
        let h0 =
            sunrise_sunset_local_hour_angle(0.0, 0.0, SUN_ELEVATION_AT_HORIZON_DEGREES).unwrap();
        let expected = SUN_ELEVATION_AT_HORIZON_DEGREES
            .to_radians()
            .sin()
            .acos()
            .to_degrees();
        assert!((h0 - expected).abs() < 1e-12);
    }

    #[test]
    fn approximate_sunrise_and_sunset_are_symmetric_in_h0() {
        let baseline_sunrise = approximate_sunrise_time(0.5, 90.0);
        let baseline_sunset = approximate_sunset_time(0.5, 90.0);
        for &d in &[-1.0_f64, -1e-3, 1e-6, 1.0] {
            assert!(
                (approximate_sunrise_time(0.5, 90.0 + d) - baseline_sunrise + d / 360.0).abs()
                    < 1e-13,
            );
            assert!(
                (approximate_sunset_time(0.5, 90.0 + d) - baseline_sunset - d / 360.0).abs()
                    < 1e-13,
            );
        }
    }

    #[test]
    fn approximate_sunrise_and_sunset_wrap_into_unit_interval() {
        for &(m_0, h_0) in &[(0.05_f64, 90.0_f64), (0.95, 90.0), (0.0, 360.0)] {
            assert!((0.0..1.0).contains(&approximate_sunrise_time(m_0, h_0)));
            assert!((0.0..1.0).contains(&approximate_sunset_time(m_0, h_0)));
        }
    }

    #[test]
    fn sidereal_time_at_event_advances_by_diurnal_rate() {
        for &m in &[0.0_f64, 0.25, 0.5, 1.0] {
            let actual = sidereal_time_at_event(100.0, m);
            let expected = EARTH_SIDEREAL_DAILY_ROTATION_DEGREES.mul_add(m, 100.0);
            assert!((actual - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn delta_t_corrected_event_time_adds_seconds_per_day_offset() {
        for &m in &[0.0_f64, 0.5, 0.999] {
            assert!((delta_t_corrected_event_time(m, 67.0) - (m + 67.0 / 86_400.0)).abs() < 1e-15);
        }
    }

    #[test]
    fn interpolate_three_day_value_passes_through_each_node() {
        for &(n, expected) in &[(-1.0_f64, 10.0), (0.0, 11.0), (1.0, 12.0)] {
            assert!((interpolate_three_day_value(10.0, 11.0, 12.0, n) - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn interpolate_three_day_value_collapses_to_linear_for_constant_difference() {
        for &n in &[-0.7_f64, -0.1, 0.3, 0.99] {
            let actual = interpolate_three_day_value(0.0, 1.0, 2.0, n);
            assert!((actual - n.mul_add(1.0, 1.0)).abs() < 1e-12);
        }
    }

    #[test]
    fn interpolate_three_day_value_recovers_second_difference() {
        // (0, 1, 4): a=1, b=3, c=2. At n=0.5: 1 + 0.5·(1+3+2·0.5)/2 = 2.25.
        assert!((interpolate_three_day_value(0.0, 1.0, 4.0, 0.5) - 2.25).abs() < 1e-12);
    }

    #[test]
    fn interpolate_three_day_value_wraps_360_degree_difference() {
        // (358, 1, 4) simulates the vernal-equinox α roll-over.
        assert!((interpolate_three_day_value(358.0, 1.0, 4.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((interpolate_three_day_value(358.0, 1.0, 4.0, 0.5) - 2.5).abs() < 1e-12);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn wrap_interpolation_difference_threshold() {
        for &diff in &[-2.0_f64, -1.5, -1e-6, 0.0, 1e-6, 1.5, 2.0] {
            assert_eq!(wrap_interpolation_difference(diff), diff);
        }
        assert!((wrap_interpolation_difference(-359.0) - 1.0).abs() < 1e-12);
        assert!((wrap_interpolation_difference(361.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn event_local_hour_angle_wraps_into_signed_180() {
        for &(nu, sigma, alpha) in &[
            (0.0_f64, 0.0, 0.0),
            (300.0, 0.0, 0.0),
            (60.0, 0.0, 0.0),
            (-300.0, 0.0, 0.0),
            (1080.0, 0.0, 0.0),
        ] {
            let h_prime = event_local_hour_angle(nu, sigma, alpha);
            assert!(h_prime > -180.0 && h_prime <= 180.0);
        }
    }

    #[test]
    fn event_local_hour_angle_wraps_300_to_minus_60() {
        assert!((event_local_hour_angle(300.0, 0.0, 0.0) - -60.0).abs() < 1e-12);
        assert!((event_local_hour_angle(-300.0, 0.0, 0.0) - 60.0).abs() < 1e-12);
    }

    #[test]
    fn sun_altitude_at_event_reduces_at_meridian() {
        // At H' = 0: arcsin(cos(φ - δ)) = 90° - |φ - δ|.
        let altitude = sun_altitude_at_event(40.0, -10.0, 0.0);
        assert!((altitude - (90.0_f64 - 50.0_f64.abs())).abs() < 1e-12);
    }

    #[test]
    fn sun_transit_time_corrects_m0() {
        let baseline = sun_transit_time(0.5, 0.0);
        for &d in &[-180.0_f64, -1.0, 1e-3, 90.0] {
            assert!((sun_transit_time(0.5, d) - (baseline - d / 360.0)).abs() < 1e-13);
        }
    }

    #[test]
    fn sunrise_or_sunset_time_collapses_when_residual_altitude_vanishes() {
        for &(delta, phi, h_prime) in &[
            (-10.0_f64, 40.0, 80.0),
            (5.0, -30.0, -90.0),
            (0.0, 0.0, 90.0),
        ] {
            let m = 0.25;
            let result = sunrise_or_sunset_time(
                m,
                SUN_ELEVATION_AT_HORIZON_DEGREES,
                SUN_ELEVATION_AT_HORIZON_DEGREES,
                delta,
                phi,
                h_prime,
            );
            assert!((result - m).abs() < 1e-13);
        }
    }

    #[test]
    fn sunrise_or_sunset_time_scales_residual_by_one_over_360() {
        // (δ=0, φ=0, H'=90°): denominator = 360. Residual +1° → +1/360 day.
        let m = 0.25;
        let result = sunrise_or_sunset_time(m, 1.0, 0.0, 0.0, 0.0, 90.0);
        assert!((result - (m + 1.0 / 360.0)).abs() < 1e-13);
    }

    #[test]
    fn right_ascension_and_declination_matches_table_a5_1() {
        let jd = julian::calculate_julian_day(&reference_datetime());
        let jde = julian::calculate_julian_ephemeris_day(jd, REFERENCE_DELTA_T_SECONDS);
        let (alpha, delta) = right_ascension_and_declination(jde);
        assert!((alpha - 202.227_41).abs() < 1e-4);
        assert!((delta - -9.314_34).abs() < 1e-4);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn refined_event_fraction_dispatches_on_event_kind() {
        let observer = reference_observer();
        let utc_midnight = SpaDateTime::new(Utc.with_ymd_and_hms(2003, 10, 17, 0, 0, 0).unwrap());
        let jd_0 = julian::calculate_julian_day(&utc_midnight);
        let jde_0 = julian::calculate_julian_ephemeris_day(jd_0, REFERENCE_DELTA_T_SECONDS);
        let jce_0 = julian::calculate_julian_ephemeris_century(jde_0);
        let jme_0 = julian::calculate_julian_ephemeris_millennium(jce_0);
        let (delta_psi_0, delta_epsilon_0) =
            crate::nutation::nutation_in_longitude_and_obliquity(jce_0);
        let epsilon_0 = crate::obliquity::true_obliquity_of_ecliptic(jme_0, delta_epsilon_0);
        let nu = crate::sidereal::apparent_sidereal_time(
            crate::sidereal::mean_sidereal_time(jd_0),
            delta_psi_0,
            epsilon_0,
        );
        let (alpha_minus, delta_minus) = right_ascension_and_declination(jde_0 - 1.0);
        let (alpha_zero, delta_zero) = right_ascension_and_declination(jde_0);
        let (alpha_plus, delta_plus) = right_ascension_and_declination(jde_0 + 1.0);

        let m_0 = approximate_sun_transit_time(alpha_zero, observer.longitude(), nu);
        let h_0 = sunrise_sunset_local_hour_angle(
            observer.latitude(),
            delta_zero,
            SUN_ELEVATION_AT_HORIZON_DEGREES,
        )
        .unwrap();

        let transit = refined_event_fraction_of_day(
            EventKind::Transit,
            m_0,
            nu,
            observer,
            REFERENCE_DELTA_T_SECONDS,
            (alpha_minus, alpha_zero, alpha_plus),
            (delta_minus, delta_zero, delta_plus),
        );
        let sunrise = refined_event_fraction_of_day(
            EventKind::Sunrise,
            approximate_sunrise_time(m_0, h_0),
            nu,
            observer,
            REFERENCE_DELTA_T_SECONDS,
            (alpha_minus, alpha_zero, alpha_plus),
            (delta_minus, delta_zero, delta_plus),
        );
        let sunset = refined_event_fraction_of_day(
            EventKind::Sunset,
            approximate_sunset_time(m_0, h_0),
            nu,
            observer,
            REFERENCE_DELTA_T_SECONDS,
            (alpha_minus, alpha_zero, alpha_plus),
            (delta_minus, delta_zero, delta_plus),
        );

        let close = |actual: i64, expected: i64| (actual - expected).abs() <= 1;
        assert!(close(
            fraction_to_clock_seconds(transit.fraction_of_day),
            67_565
        ));
        assert!(close(
            fraction_to_clock_seconds(sunrise.fraction_of_day),
            47_563
        ));
        assert!(close(
            fraction_to_clock_seconds(sunset.fraction_of_day),
            1_219
        ));

        assert!(transit.local_hour_angle.abs() < 0.1);
        assert!(sunrise.local_hour_angle < 0.0);
        assert!(sunset.local_hour_angle > 0.0);
        assert!((transit.interpolated_declination - delta_zero).abs() < 1.0);
    }

    #[test]
    fn solar_day_unwraps_sunrise_to_previous_ut_day_for_east_observer() {
        let observer = Observer::try_new(-33.8, 150.0, 0.0, 1010.0, 18.0).unwrap();
        let utc_noon = SpaDateTime::new(Utc.with_ymd_and_hms(2026, 3, 20, 12, 0, 0).unwrap());
        let day = SolarDay::compute_with_delta_t(&utc_noon, 70.0, observer);
        let sunrise = day.sunrise.unwrap();
        assert!(sunrise < day.transit);
        assert!(day.transit < day.sunset.unwrap());
        assert_eq!(
            sunrise.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2026, 3, 19).unwrap(),
        );
        assert_eq!(
            day.transit.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2026, 3, 20).unwrap(),
        );
    }

    #[test]
    fn solar_day_display_renders_populated_events() {
        let rendered = format!("{}", reference_solar_day());
        assert!(rendered.starts_with("Sunrise:"));
        assert!(rendered.contains("Sun transit:"));
        assert!(rendered.contains("Sunset:"));
        assert!(!rendered.contains("none (polar day or polar night)"));
    }

    #[test]
    fn solar_day_display_marks_polar_for_missing_events() {
        let transit = Utc.with_ymd_and_hms(2026, 12, 21, 12, 0, 0).unwrap();
        let polar_day = SolarDay::<Utc> {
            transit,
            sunrise: None,
            sunset: None,
            sun_transit_altitude: -23.0,
            sunrise_hour_angle: None,
            sunset_hour_angle: None,
        };
        let rendered = format!("{polar_day}");
        assert_eq!(
            rendered.matches("none (polar day or polar night)").count(),
            4
        );
        assert!(rendered.contains("Sun transit:"));
        assert!(rendered.contains("Sun transit altitude:"));
    }

    #[test]
    fn solar_day_display_propagates_writer_errors() {
        let transit = Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0).unwrap();
        let configurations = [
            SolarDay::<Utc> {
                transit,
                sunrise: Some(transit - chrono::TimeDelta::hours(6)),
                sunset: Some(transit + chrono::TimeDelta::hours(6)),
                sun_transit_altitude: 65.0,
                sunrise_hour_angle: Some(-90.0),
                sunset_hour_angle: Some(90.0),
            },
            SolarDay::<Utc> {
                transit,
                sunrise: None,
                sunset: Some(transit + chrono::TimeDelta::hours(6)),
                sun_transit_altitude: 65.0,
                sunrise_hour_angle: None,
                sunset_hour_angle: Some(90.0),
            },
            SolarDay::<Utc> {
                transit,
                sunrise: Some(transit - chrono::TimeDelta::hours(6)),
                sunset: None,
                sun_transit_altitude: 65.0,
                sunrise_hour_angle: Some(-90.0),
                sunset_hour_angle: None,
            },
            SolarDay::<Utc> {
                transit,
                sunrise: None,
                sunset: None,
                sun_transit_altitude: 65.0,
                sunrise_hour_angle: None,
                sunset_hour_angle: None,
            },
        ];

        for day in &configurations {
            let rendered = format!("{day}");
            let total = rendered.len();
            for budget in 0..total {
                let mut writer = FailingWriter { remaining: budget };
                assert!(core::fmt::Write::write_fmt(&mut writer, format_args!("{day}")).is_err());
            }
            let mut writer = FailingWriter { remaining: total };
            assert!(core::fmt::Write::write_fmt(&mut writer, format_args!("{day}")).is_ok());
        }
    }
}
