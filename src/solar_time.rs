// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Sun transit (solar noon), sunrise and sunset times for a given civil day.
//!
//! Implements appendix A.2 of NREL/TP-560-34302. The procedure is anchored
//! on three values of the sun's geocentric right ascension and declination
//! evaluated at three equally-spaced TT instants 24 hours apart (the day
//! before the day of interest `D₋₁`, the day of interest `D₀`, the day
//! after `D₊₁`); the apparent sidereal time at Greenwich at the anchor;
//! the observer's geographic position; and `ΔT`. Equations A3 to A15 then
//! build first-order approximations of the three event times (transit
//! `m₀`, sunrise `m₁`, sunset `m₂`) from the day-`D₀` quantities, refine
//! them via Stirling interpolation of `(α, δ)` at the approximate event
//! times, and produce the final fractions of day `T`, `R`, `S` for
//! transit, sunrise and sunset respectively.
//!
//! Appendix A.2 phrases the anchor as `0 UT on D₀`. Section 3.1.1 of the
//! report extends the procedure to local time by subtracting the time
//! zone fraction from the JD, and the closing note of section A.3
//! confirms that adding the time zone back to the JD recovers the local
//! calendar date. The [`SolarDay`] orchestrator follows that
//! generalisation: `D₀` is the input's local civil date in its own
//! timezone, and the anchor is the UT instant of local midnight that
//! opens it. The three returned events fall on that local civil date
//! when the input timezone tracks the observer (within the
//! unwrap-relative-to-transit slack documented on [`SolarDay::compute`]).
//!
//! Sunrise and sunset only exist when the sun crosses the chosen horizon
//! elevation; at high latitudes near the solstices this fails (polar day
//! or polar night), in which case [`sunrise_sunset_local_hour_angle`]
//! and the [`SolarDay`] orchestrator return `None`.

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

/// Sun centre elevation `h'₀` at the visible horizon (degrees).
///
/// Reproduced verbatim from appendix A.2: the average horizon-level
/// atmospheric refraction `0.5667°` and the apparent solar disk radius
/// `0.26667°` jointly lift the sun's centre to `-0.8333°` at the
/// instants the upper limb visually grazes the geometric horizon.
pub const SUN_ELEVATION_AT_HORIZON_DEGREES: f64 = -0.8333;

/// Earth's sidereal rotation rate (`360.985647°` per UT day) used by
/// equation A7 to advance the apparent sidereal time forward into the
/// day. Reproduced verbatim from the appendix.
const EARTH_SIDEREAL_DAILY_ROTATION_DEGREES: f64 = 360.985_647;

/// Seconds per day, the divisor that converts `ΔT` (seconds) to a
/// fraction of a day in equation A8.
const SECONDS_PER_DAY: f64 = 86_400.0;

/// Half a rotation `180°`. Used by step A.2.11 to wrap `H'ᵢ` into
/// `(-180°, 180°]` and by the [`SolarDay`] orchestrator to centre the
/// sunrise and sunset events around the transit instant.
const HALF_REVOLUTION_DEGREES: f64 = 180.0;

/// Full rotation `360°`. Used by equations A3, A5, A6, A11, A13 and A14
/// as the period of the angular quantities they reduce to fractions of
/// a day.
const FULL_REVOLUTION_DEGREES: f64 = 360.0;

/// Threshold at which interpolation differences `α₀ − α₋₁`, `α₊₁ − α₀`
/// (and the equivalent for `δ`) are taken to be a `360°` wrap of the
/// underlying angle and reduced into `(-180°, 180°]` per step A.2.10.
/// `2°` comfortably exceeds the natural diurnal motion of both `α`
/// (≤ ~1°/day) and `δ` (≤ ~0.4°/day) for the sun, so the threshold
/// fires only on a wrap-around (typically once per year, near the
/// vernal equinox, when `α` rolls from `~360°` back to `~0°`).
const INTERPOLATION_WRAP_THRESHOLD_DEGREES: f64 = 2.0;

/// J2000.0 standard epoch in Julian Days, used as the zero point for
/// `JC` and `JCE` per equations 6 and 7.
const J2000_EPOCH_JD: f64 = 2_451_545.0;

/// Approximate sun transit time `m₀` (fraction of UT day from `0 UT`).
///
/// `geocentric_right_ascension_at_0tt` is `α₀` (degrees), the sun's
/// geocentric right ascension at 0 TT on `D₀`. `observer_longitude` is
/// `σ` (degrees, positive east of Greenwich, per section 3.11).
/// `apparent_sidereal_time_at_0ut` is `ν` (degrees), the apparent
/// sidereal time at Greenwich at 0 UT on `D₀`, as produced by
/// [`apparent_sidereal_time`].
///
/// Refer to appendix A.2, equation A3:
/// `m₀ = (α₀ − σ − ν) / 360`. The result is wrapped into `[0, 1)` per
/// step A.2.7 so that it expresses the transit time as a non-negative
/// fraction of the chosen UT day from its 00:00 instant.
///
/// # Examples
///
/// ```
/// use helioxide::solar_time::approximate_sun_transit_time;
///
/// // Plain in-range case: m₀ stays inside [0, 1).
/// let m0 = approximate_sun_transit_time(202.227_409, -105.1786, 100.0);
/// assert!((0.0..1.0).contains(&m0));
/// ```
///
/// [`apparent_sidereal_time`]: crate::sidereal::apparent_sidereal_time
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

/// Local hour angle `H₀` (degrees, in `[0°, 180°]`) corresponding to
/// the sun centre at the chosen `sun_horizon_elevation`.
///
/// Returns `None` for polar day or polar night, when the sun is always
/// above or always below the horizon for the given (latitude,
/// declination): per appendix A.2 the cosine argument falls outside
/// `[-1, 1]` in those cases.
///
/// `observer_latitude` is `φ` (degrees, positive north of the equator).
/// `geocentric_declination_at_0tt` is `δ₀` (degrees), the sun's
/// geocentric declination at 0 TT on `D₀`. `sun_horizon_elevation` is
/// `h'₀` (degrees); pass [`SUN_ELEVATION_AT_HORIZON_DEGREES`] for the
/// canonical `-0.8333°` of section A.2 (which already accounts for the
/// solar disk radius and the average horizon-level refraction).
///
/// Refer to appendix A.2, equation A4:
/// `H₀ = arccos((sin h'₀ − sin φ · sin δ₀) / (cos φ · cos δ₀))`.
/// `arccos` returns a value in `[0, π]` by construction, so the result
/// is already in `[0°, 180°]` after the radians-to-degrees conversion;
/// the explicit limit to `[0°, 180°]` mandated by step A.2.4 is
/// therefore the identity here.
///
/// # Examples
///
/// ```
/// use helioxide::solar_time::{
///     SUN_ELEVATION_AT_HORIZON_DEGREES, sunrise_sunset_local_hour_angle,
/// };
///
/// // Mid-latitude observer with the sun a few degrees south of the
/// // celestial equator: a regular sunrise/sunset day.
/// let h0 = sunrise_sunset_local_hour_angle(40.0, -10.0, SUN_ELEVATION_AT_HORIZON_DEGREES);
/// assert!(h0.is_some_and(|x| (0.0..=180.0).contains(&x)));
///
/// // Polar night: the sun never rises in midwinter at 80° N.
/// let polar = sunrise_sunset_local_hour_angle(80.0, -23.0, SUN_ELEVATION_AT_HORIZON_DEGREES);
/// assert!(polar.is_none());
/// ```
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

/// Approximate sunrise time `m₁` (fraction of UT day from `0 UT`).
///
/// `approximate_sun_transit_time` is `m₀`, as produced by
/// [`approximate_sun_transit_time`]. `sunrise_sunset_local_hour_angle`
/// is `H₀`, as produced by [`sunrise_sunset_local_hour_angle`].
///
/// Refer to appendix A.2, equation A5: `m₁ = m₀ − H₀ / 360`. The result
/// is wrapped into `[0, 1)` per step A.2.7. The wrap is mandatory: the
/// natural `m₁` may fall slightly below zero when the transit happens
/// in the early UT morning (i.e. for observers well east of Greenwich),
/// in which case `m₁_raw` represents a sunrise on `D₋₁` whose time of
/// day still lies in `[0, 1)` modulo a full revolution.
#[inline]
#[must_use]
pub fn approximate_sunrise_time(
    approximate_sun_transit_time: f64,
    sunrise_sunset_local_hour_angle: f64,
) -> f64 {
    (approximate_sun_transit_time - sunrise_sunset_local_hour_angle / FULL_REVOLUTION_DEGREES)
        .rem_euclid(1.0)
}

/// Approximate sunset time `m₂` (fraction of UT day from `0 UT`).
///
/// `approximate_sun_transit_time` is `m₀`, as produced by
/// [`approximate_sun_transit_time`]. `sunrise_sunset_local_hour_angle`
/// is `H₀`, as produced by [`sunrise_sunset_local_hour_angle`].
///
/// Refer to appendix A.2, equation A6: `m₂ = m₀ + H₀ / 360`. The result
/// is wrapped into `[0, 1)` per step A.2.7. The wrap is mandatory: the
/// natural `m₂` may fall slightly above one when the transit happens
/// in the late UT afternoon (i.e. for observers well west of
/// Greenwich), in which case `m₂_raw` represents a sunset on `D₊₁`
/// whose time of day still lies in `[0, 1)` modulo a full revolution.
#[inline]
#[must_use]
pub fn approximate_sunset_time(
    approximate_sun_transit_time: f64,
    sunrise_sunset_local_hour_angle: f64,
) -> f64 {
    (approximate_sun_transit_time + sunrise_sunset_local_hour_angle / FULL_REVOLUTION_DEGREES)
        .rem_euclid(1.0)
}

/// Apparent sidereal time at Greenwich at the approximate event time
/// `νᵢ` (degrees, signed, not wrapped).
///
/// `apparent_sidereal_time_at_0ut` is `ν` (degrees), the apparent
/// sidereal time at Greenwich at 0 UT on `D₀`, as produced by
/// [`apparent_sidereal_time`]. `approximate_event_time` is `mᵢ`
/// (fraction of UT day, in `[0, 1)` after step A.2.7).
///
/// Refer to appendix A.2, equation A7:
/// `νᵢ = ν + 360.985647 · mᵢ`. The result is left signed and unwrapped
/// because the only downstream consumer ([`event_local_hour_angle`])
/// folds it into a `(-180°, 180°]` wrap of its own; the diurnal rotation
/// rate is `360.985647° / day`, the same `(JD - JD_J2000)`-driven term
/// of equation 28.
///
/// [`apparent_sidereal_time`]: crate::sidereal::apparent_sidereal_time
#[inline]
#[must_use]
pub const fn sidereal_time_at_event(
    apparent_sidereal_time_at_0ut: f64,
    approximate_event_time: f64,
) -> f64 {
    apparent_sidereal_time_at_0ut + EARTH_SIDEREAL_DAILY_ROTATION_DEGREES * approximate_event_time
}

/// `ΔT`-corrected event time `nᵢ` (fraction of TT day from `0 TT`).
///
/// `approximate_event_time` is `mᵢ` (fraction of UT day, in `[0, 1)`
/// after step A.2.7). `delta_t_seconds` is `ΔT = TT − UT1` (seconds),
/// the same Earth-rotation correction consumed by
/// [`calculate_julian_ephemeris_day`].
///
/// Refer to appendix A.2, equation A8:
/// `nᵢ = mᵢ + ΔT / 86_400`. Used by [`interpolate_three_day_value`] as
/// the abscissa at which the three-point Stirling interpolation of `α`
/// and `δ` is evaluated, so that the interpolation reaches the actual
/// TT-clock instant of the event rather than its UT-clock approximation.
///
/// [`calculate_julian_ephemeris_day`]: crate::julian::calculate_julian_ephemeris_day
#[inline]
#[must_use]
pub const fn delta_t_corrected_event_time(
    approximate_event_time: f64,
    delta_t_seconds: f64,
) -> f64 {
    approximate_event_time + delta_t_seconds / SECONDS_PER_DAY
}

/// Three-point Stirling interpolation of a sun coordinate at the
/// approximate event time `nᵢ`.
///
/// `value_minus_one_day` is the coordinate at 0 TT on `D₋₁`,
/// `value_zero_day` at 0 TT on `D₀`, `value_plus_one_day` at 0 TT on
/// `D₊₁`. `delta_t_corrected_event_time` is `nᵢ` (fraction of TT day
/// from 0 TT on `D₀`), as produced by [`delta_t_corrected_event_time`].
///
/// Refer to appendix A.2, equations A9 and A10:
/// `value' = value₀ + nᵢ · (a + b + c · nᵢ) / 2`
/// where `a = value₀ − value₋₁`, `b = value₊₁ − value₀`, `c = b − a`.
/// At `nᵢ = 0` the interpolation collapses to `value₀`; at `nᵢ = 1` to
/// `value₊₁`; at `nᵢ = -1` to `value₋₁`.
///
/// Per step A.2.10, differences whose magnitude exceeds
/// `INTERPOLATION_WRAP_THRESHOLD_DEGREES` are taken to be a `360°`
/// wrap of the underlying angle and folded back into `(-180°, 180°]`
/// before the Stirling sum: this only fires for `α` near the vernal
/// equinox (when the wrap from `~360°` to `~0°` makes `b ≈ -359°`),
/// and is a no-op for `δ` which never wraps.
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

/// Wrap an interpolation difference into `(-180°, 180°]` per step
/// A.2.10. Differences smaller than `INTERPOLATION_WRAP_THRESHOLD_DEGREES`
/// pass through untouched (they cannot be a `360°` wrap of the
/// underlying angle since the natural diurnal motion is bounded by
/// `~1°/day`); larger differences are reduced to the closest
/// representative modulo `360°`.
fn wrap_interpolation_difference(diff: f64) -> f64 {
    if diff.abs() > INTERPOLATION_WRAP_THRESHOLD_DEGREES {
        FULL_REVOLUTION_DEGREES.mul_add(-(diff / FULL_REVOLUTION_DEGREES).round(), diff)
    } else {
        diff
    }
}

/// Topocentric local hour angle `H'ᵢ` at the approximate event time
/// (degrees, in `(-180°, 180°]`).
///
/// `sidereal_time_at_event` is `νᵢ` (degrees), as produced by
/// [`sidereal_time_at_event`]. `observer_longitude` is `σ` (degrees,
/// positive east of Greenwich). `interpolated_right_ascension` is the
/// sun's geocentric right ascension at the approximate event time
/// `α'ᵢ` (degrees), as produced by [`interpolate_three_day_value`]
/// applied to `α`.
///
/// Refer to appendix A.2, equation A11: `H'ᵢ = νᵢ + σ − α'ᵢ`. The
/// result is wrapped into `(-180°, 180°]` so that `T = m₀ − H'₀ / 360`
/// of equation A13 produces a transit refinement bounded by
/// `±0.5 fraction-of-day` instead of off by a full revolution.
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

/// Sun altitude at the approximate event time `hᵢ` (degrees, signed).
///
/// `observer_latitude` is `φ` (degrees, positive north of the equator).
/// `interpolated_declination` is the sun's geocentric declination at
/// the approximate event time `δ'ᵢ` (degrees), as produced by
/// [`interpolate_three_day_value`] applied to `δ`.
/// `event_local_hour_angle` is `H'ᵢ` (degrees), as produced by
/// [`event_local_hour_angle`].
///
/// Refer to appendix A.2, equation A12:
/// `hᵢ = arcsin(sin φ · sin δ'ᵢ + cos φ · cos δ'ᵢ · cos H'ᵢ)`. The
/// formula is the spherical law of cosines applied to the
/// observer-zenith / sun-zenith triangle; `arcsin` returns a value in
/// `[-π/2, π/2]` by construction, so `hᵢ` lies in `[-90°, 90°]` after
/// the radians-to-degrees conversion. Used by [`sunrise_or_sunset_time`]
/// as the residual altitude `hᵢ − h'₀` driving the secant correction.
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

/// Refined sun transit time `T` (fraction of day from `0 UT`).
///
/// `approximate_sun_transit_time` is `m₀`, as produced by
/// [`approximate_sun_transit_time`].
/// `event_local_hour_angle_at_transit` is `H'₀` (degrees), as produced
/// by [`event_local_hour_angle`] for `i = 0`.
///
/// Refer to appendix A.2, equation A13: `T = m₀ − H'₀ / 360`. The
/// `H'₀ / 360` correction is bounded by `±0.5 fraction-of-day` because
/// `H'₀` is wrapped into `(-180°, 180°]` upstream, so `T` lies in
/// `(m₀ − 0.5, m₀ + 0.5]`. In the natural transit configuration
/// (sun on the meridian) `H'₀` is small (a fraction of a degree),
/// so the correction barely shifts `m₀`.
#[inline]
#[must_use]
pub fn sun_transit_time(
    approximate_sun_transit_time: f64,
    event_local_hour_angle_at_transit: f64,
) -> f64 {
    approximate_sun_transit_time - event_local_hour_angle_at_transit / FULL_REVOLUTION_DEGREES
}

/// Refined sunrise or sunset time (fraction of day from `0 UT`).
///
/// `approximate_event_time` is `m₁` (sunrise) or `m₂` (sunset), as
/// produced by [`approximate_sunrise_time`] or [`approximate_sunset_time`].
/// `sun_altitude_at_event` is `hᵢ` (degrees), as produced by
/// [`sun_altitude_at_event`]. `sun_horizon_elevation` is `h'₀`
/// (degrees); pass [`SUN_ELEVATION_AT_HORIZON_DEGREES`] for the
/// canonical `-0.8333°` of section A.2.
/// `interpolated_declination` is `δ'ᵢ` (degrees), as produced by
/// [`interpolate_three_day_value`] applied to `δ`. `observer_latitude`
/// is `φ` (degrees). `event_local_hour_angle` is `H'ᵢ` (degrees), as
/// produced by [`event_local_hour_angle`].
///
/// Refer to appendix A.2, equations A14 and A15:
/// `R or S = mᵢ + (hᵢ − h'₀) / (360 · cos δ'ᵢ · cos φ · sin H'ᵢ)`.
/// The formula refines the approximate `mᵢ` by a Newton step of the
/// equation `h(t) = h'₀`: the correction `(hᵢ − h'₀)` is the residual
/// altitude at the approximate time, divided by the time derivative
/// `dh/dt = 360 · cos δ · cos φ · sin H` (degrees-of-altitude per
/// fraction-of-day). The denominator stays well away from zero outside
/// polar configurations, where the upstream `H₀` is `None` and the
/// orchestrator already short-circuits.
#[inline]
#[must_use]
#[allow(clippy::too_many_arguments)] // Each input mirrors a quantity defined by the appendix.
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

/// Sun transit (solar noon), sunrise and sunset for a single UT day.
///
/// `transit` is always populated: the sun crosses the local meridian
/// once per UT day at every observer position. `sunrise` and `sunset`
/// are populated only when the sun crosses the horizon on `D₀`; at high
/// latitudes near the solstices they are `None` (polar day or polar
/// night).
///
/// Beyond the three event instants, the struct also exposes the per-event
/// topocentric local hour angles `H'ᵢ` at sunrise and sunset (per
/// equation A11, signed and wrapped into `(-180°, 180°]`) and the sun
/// altitude at transit `h₀` (per equation A12).
#[derive(Debug, Clone, PartialEq)]
pub struct SolarDay<Tz: TimeZone> {
    /// Sun transit instant (solar noon at the observer's longitude).
    pub transit: DateTime<Tz>,
    /// Sunrise instant, or `None` for polar day/night.
    pub sunrise: Option<DateTime<Tz>>,
    /// Sunset instant, or `None` for polar day/night.
    pub sunset: Option<DateTime<Tz>>,
    /// Sun altitude at the transit instant (degrees, in `[-90°, 90°]`).
    /// Always populated: at transit `H' ≈ 0°`, so equation A12 collapses
    /// to `arcsin(sin φ · sin δ' + cos φ · cos δ') ≈ 90° − |φ − δ'|`,
    /// the geometric upper-culmination altitude. Negative values place
    /// the transit below the horizon (polar night).
    pub sun_transit_altitude: f64,
    /// Topocentric local hour angle `H'₁` at sunrise (degrees, signed,
    /// typically in `(-180°, 180°]`), per equation A11. `None` for polar
    /// day/night.
    pub sunrise_hour_angle: Option<f64>,
    /// Topocentric local hour angle `H'₂` at sunset (degrees, signed,
    /// typically in `(-180°, 180°]`), per equation A11. `None` for polar
    /// day/night.
    pub sunset_hour_angle: Option<f64>,
}

impl<Tz: TimeZone> SolarDay<Tz> {
    /// Run the appendix A.2 procedure end to end.
    ///
    /// `datetime` is any instant on the civil day of interest. Its local
    /// date in its own timezone defines `D₀`, and its DUT1 (`= UT1 − UTC`)
    /// is propagated to the internal Julian Day computation.
    /// `delta_t_seconds` is `ΔT = TT − UT1` (seconds); see
    /// [`SolarPosition::compute`] for current values. `observer`'s
    /// `latitude` and `longitude` drive the procedure; its `elevation`,
    /// `pressure` and `temperature` are unused because appendix A.2
    /// absorbs the average horizon-level refraction into the constant
    /// [`SUN_ELEVATION_AT_HORIZON_DEGREES`].
    ///
    /// The SPA reference is shifted from appendix A.2's `0 UT on D₀` to
    /// `0 local time on D₀ in the input's timezone`. Section 3.1.1
    /// expressly allows driving the JD chain from local time, and
    /// section A.3 confirms the local calendar date is the natural
    /// readout when the JD is shifted by the time zone fraction. The
    /// three returned events therefore fall on the input's local civil
    /// date when the input timezone tracks the observer; for inputs in
    /// a timezone far from the observer (offset gap close to half a day),
    /// the unwrap-relative-to-transit logic still enforces
    /// `sunrise < transit < sunset` at the observer, at the cost of
    /// pulling one of sunrise or sunset onto an adjacent calendar day on
    /// the input's wall clock.
    ///
    /// The returned [`SolarDay`] is rendered in the same timezone as the
    /// input `datetime`, so a caller passing a `Europe::Madrid` instant
    /// gets sunrise/transit/sunset projected onto Madrid wall clocks. To
    /// switch timezones, call [`DateTime::with_timezone`] on the
    /// individual fields.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use chrono::Utc;
    /// use chrono_tz::Tz;
    /// use helioxide::{Observer, SolarDay, SpaDateTime};
    ///
    /// let observer = Observer {
    ///     longitude: -0.490_68,
    ///     latitude: 38.346_02,
    ///     elevation: 3.0,
    ///     pressure: 1015.0,
    ///     temperature: 18.0,
    /// };
    /// let now = SpaDateTime::new(Utc::now().with_timezone(&Tz::Europe__Madrid));
    /// let day = SolarDay::compute(&now, 69.5, observer);
    /// println!("Solar noon: {}", day.transit);
    /// ```
    ///
    /// To anchor `D₀` on the UTC civil day instead of a local one, pass a
    /// [`DateTime<Utc>`] directly: its UTC date becomes `D₀`.
    ///
    /// ```
    /// use chrono::{NaiveDate, TimeZone, Utc};
    /// use helioxide::{Observer, SolarDay, SpaDateTime};
    /// # let observer = Observer {
    /// #     longitude: -0.490_68, latitude: 38.346_02,
    /// #     elevation: 3.0, pressure: 1015.0, temperature: 18.0,
    /// # };
    /// let noon_utc =
    ///     SpaDateTime::new(Utc.with_ymd_and_hms(2003, 10, 17, 12, 0, 0).unwrap());
    /// let day = SolarDay::compute(&noon_utc, 67.0, observer);
    /// assert_eq!(
    ///     day.transit.date_naive(),
    ///     NaiveDate::from_ymd_opt(2003, 10, 17).unwrap(),
    /// );
    /// ```
    ///
    /// [`DateTime<Utc>`]: chrono::DateTime
    /// [`SolarPosition::compute`]: crate::SolarPosition::compute
    #[must_use]
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_lines
    )]
    pub fn compute(datetime: &SpaDateTime<Tz>, delta_t_seconds: f64, observer: Observer) -> Self {
        let tz = datetime.datetime().timezone();
        let utc_anchor = local_civil_midnight_in_utc(datetime.datetime());
        let datetime_at_anchor = datetime.with_datetime(utc_anchor);

        // Step A.2.1 — apparent sidereal time at Greenwich at the anchor.
        let jd_0 = calculate_julian_day(&datetime_at_anchor);
        let jde_0 = calculate_julian_ephemeris_day(jd_0, delta_t_seconds);
        let jce_0 = calculate_julian_ephemeris_century(jde_0);
        let jme_0 = calculate_julian_ephemeris_millennium(jce_0);
        let (delta_psi_0, delta_epsilon_0) = nutation_in_longitude_and_obliquity(jce_0);
        let epsilon_0 = true_obliquity_of_ecliptic(jme_0, delta_epsilon_0);
        let nu = apparent_sidereal_time(mean_sidereal_time(jd_0), delta_psi_0, epsilon_0);

        // Step A.2.2 — (α, δ) at the anchor minus, equal to and plus 24 h (in TT).
        let (alpha_minus, delta_minus) = right_ascension_and_declination(jde_0 - 1.0);
        let (alpha_zero, delta_zero) = right_ascension_and_declination(jde_0);
        let (alpha_plus, delta_plus) = right_ascension_and_declination(jde_0 + 1.0);

        // Step A.2.3 — m₀.
        let m_0 = approximate_sun_transit_time(alpha_zero, observer.longitude, nu);

        // Step A.2.4 — H₀ (None for polar day/night).
        let h_0 = sunrise_sunset_local_hour_angle(
            observer.latitude,
            delta_zero,
            SUN_ELEVATION_AT_HORIZON_DEGREES,
        );

        // Step A.2.13 — refined transit time T plus the per-event H'/δ'
        // bundle that backs the sun_transit_altitude readout below.
        let transit_event = refined_event_fraction_of_day(
            EventKind::Transit,
            m_0,
            nu,
            observer,
            delta_t_seconds,
            (alpha_minus, alpha_zero, alpha_plus),
            (delta_minus, delta_zero, delta_plus),
        );

        // Steps A.2.5/A.2.6 + A.2.14/A.2.15 — refined sunrise R and sunset S
        // along with the per-event H'/δ' bundles that back the
        // sunrise_hour_angle/sunset_hour_angle readouts below.
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
        // around T, so the returned datetimes preserve the natural ordering
        // sunrise < transit < sunset across the anchored day boundaries.
        let transit_wrapped = transit_event.fraction_of_day.rem_euclid(1.0);
        let to_datetime = |fraction_of_day: f64| -> DateTime<Tz> {
            // Round to whole milliseconds: appendix A.2 publishes events to
            // hundredths of a second (Table A2.1), and milliseconds resolve
            // them comfortably without exposing transient floating-point
            // noise from the trailing decimals of the f64 fraction.
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

        // Sun transit altitude at the refined transit time. At transit
        // `H' ≈ 0°`, so this is essentially `arcsin(sin φ · sin δ' +
        // cos φ · cos δ')`, the upper-culmination altitude.
        let sun_transit_altitude = sun_altitude_at_event(
            observer.latitude,
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

/// Discriminator for the three appendix A.2 events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EventKind {
    Transit,
    Sunrise,
    Sunset,
}

/// Per-event quantities produced by the appendix A.2 refinement step.
///
/// Bundles the final fraction of UT day with the topocentric local hour
/// angle `H'` and the interpolated declination `δ'` evaluated at the
/// approximate event time. The latter two are surfaced on [`SolarDay`]
/// (`sunrise_hour_angle`, `sunset_hour_angle`, `sun_transit_altitude`)
/// without rerunning the whole interpolation downstream.
#[derive(Debug, Clone, Copy)]
struct RefinedEvent {
    /// Final fraction of UT day from `0 UT` (per equation A13/A14/A15).
    fraction_of_day: f64,
    /// Topocentric local hour angle `H'` at the approximate event time
    /// (degrees, in `(-180°, 180°]`, per step A.2.11).
    local_hour_angle: f64,
    /// Interpolated sun declination `δ'` at the approximate event time
    /// (degrees, signed).
    interpolated_declination: f64,
}

/// Refined per-event quantities for one of the three appendix A.2 events.
/// Bundles steps A.2.8 to A.2.12 plus the closing equation A13/A14/A15
/// so the [`SolarDay::compute`] orchestrator stays linear and so the
/// per-event `H'` and `δ'` are surfaced without rerunning the
/// interpolation downstream.
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
    let h_prime = event_local_hour_angle(nu_i, observer.longitude, alpha_prime);

    let fraction_of_day = match kind {
        EventKind::Transit => sun_transit_time(m, h_prime),
        EventKind::Sunrise | EventKind::Sunset => {
            let h_at_event = sun_altitude_at_event(observer.latitude, delta_prime, h_prime);
            sunrise_or_sunset_time(
                m,
                h_at_event,
                SUN_ELEVATION_AT_HORIZON_DEGREES,
                delta_prime,
                observer.latitude,
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

/// UT instant of local civil midnight at the start of `datetime`'s local
/// date in `datetime`'s timezone.
///
/// The [`SolarDay::compute`] anchor: the SPA `D₀` is the local civil day
/// that opens at this instant on the input's wall clock. Resolves the
/// chrono [`MappedLocalTime`] cases inline:
///
/// * `Single`: the dominant case (no DST transition lands on `00:00`).
/// * `Ambiguous`: a DST fall-back that loops over midnight. Both instants
///   share the same local date, so the earliest representation is
///   selected to keep the anchor deterministic.
/// * `None`: a DST spring-forward that skips over `00:00` entirely. The
///   naive local midnight is reinterpreted as a UT instant, so events
///   for that day can be off by the size of the DST gap on the input's
///   wall clock. Only a handful of historical zones trigger this case.
fn local_civil_midnight_in_utc<Tz: TimeZone>(datetime: &DateTime<Tz>) -> DateTime<Utc> {
    let local_midnight = datetime.date_naive().and_time(chrono::NaiveTime::MIN);
    match datetime.timezone().from_local_datetime(&local_midnight) {
        MappedLocalTime::Single(t) | MappedLocalTime::Ambiguous(t, _) => t.with_timezone(&Utc),
        MappedLocalTime::None => local_midnight.and_utc(),
    }
}

/// Sun's geocentric right ascension `α` (degrees, in `[0°, 360°)`) and
/// declination `δ` (degrees, signed) at the given Julian Ephemeris Day.
/// Bundles the chain `JDE → (JCE, JME) → (L, B, R) → (Θ, β) → (Δψ, Δε)
/// → ε → Δτ → λ → (α, δ)` so that the [`SolarDay::compute`]
/// orchestrator only re-runs it three times (one per day in the
/// `D₋₁, D₀, D₊₁` window) instead of plumbing every intermediate.
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

    /// ΔT for the Table A5.1 worked example (seconds), per section A.5.
    const REFERENCE_DELTA_T_SECONDS: f64 = 67.0;

    /// [`core::fmt::Write`] sink that accepts at most `remaining` bytes
    /// cumulatively and returns [`core::fmt::Error`] on any write that
    /// would exceed the budget. Used to fail the underlying writer at
    /// varying offsets so [`SolarDay::fmt`]'s `?` operators are
    /// exercised individually.
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
        Observer {
            longitude: REFERENCE_LONGITUDE_DEGREES,
            latitude: REFERENCE_LATITUDE_DEGREES,
            elevation: REFERENCE_ELEVATION_METRES,
            pressure: REFERENCE_PRESSURE_MILLIBARS,
            temperature: REFERENCE_TEMPERATURE_CELSIUS,
        }
    }

    /// Civil instant of the Table A5.1 worked example: 2003-10-17
    /// 12:30:30 LST at TZ = -7 h, i.e. 19:30:30 UTC.
    fn reference_datetime() -> SpaDateTime<Utc> {
        SpaDateTime::new(
            Utc.with_ymd_and_hms(2003, 10, 17, 19, 30, 30)
                .single()
                .expect("non-ambiguous reference instant"),
        )
    }

    fn reference_solar_day() -> SolarDay<Utc> {
        SolarDay::compute(
            &reference_datetime(),
            REFERENCE_DELTA_T_SECONDS,
            reference_observer(),
        )
    }

    /// Helper: convert a UT-day fraction (e.g. `0.78195`) to a "HH:MM:SS"
    /// string pinned at second resolution. Used to make assertion
    /// failures easier to read against the table A5.1 wall-clock values.
    fn fraction_to_clock_seconds(fraction_of_day: f64) -> i64 {
        #[allow(clippy::cast_possible_truncation)]
        {
            (fraction_of_day * 86_400.0).round() as i64
        }
    }

    /// `T`, `R`, `S` at the Table A5.1 reference instant must reproduce
    /// the published Transit `18:46:04.97 UT`, Sunrise `13:12:43.46 UT`
    /// and Sunset `00:20:19.19 UT`. The 1-second tolerance sits a
    /// hundred milliseconds above the trailing-digit precision of the
    /// published values and well above the 0.23 s ephemeris bias the
    /// appendix itself reports between SPA and the Almanac. Failure of
    /// any single field is a global integration red flag: a wrong sign
    /// on `H'₀ / 360` of equation A13, a swap of the
    /// (α₋₁, α₀, α₊₁) inputs to the Stirling interpolation, a wrong
    /// `cos δ' · cos φ · sin H'` denominator in A14/A15, or any
    /// upstream regression on `(L, B, R, Δψ, ε, Θ, β, λ, α, δ, ν)`
    /// would all surface here.
    #[test]
    fn solar_day_matches_table_a5_1() {
        let day = reference_solar_day();
        let utc_midnight = Utc
            .with_ymd_and_hms(2003, 10, 17, 0, 0, 0)
            .single()
            .unwrap();

        let expected_transit = utc_midnight
            + chrono::TimeDelta::seconds(67_564)
            + chrono::TimeDelta::milliseconds(970);
        assert!(
            (day.transit - expected_transit).num_milliseconds().abs() < 1_000,
            "transit mismatch at A5.1 reference: got {}",
            day.transit,
        );

        let sunrise = day.sunrise.expect("sunrise must exist at A5.1 reference");
        let expected_sunrise = utc_midnight
            + chrono::TimeDelta::seconds(47_563)
            + chrono::TimeDelta::milliseconds(460);
        assert!(
            (sunrise - expected_sunrise).num_milliseconds().abs() < 1_000,
            "sunrise mismatch at A5.1 reference: got {sunrise}",
        );

        let sunset = day.sunset.expect("sunset must exist at A5.1 reference");
        let expected_sunset = utc_midnight
            + chrono::TimeDelta::days(1)
            + chrono::TimeDelta::seconds(1_219)
            + chrono::TimeDelta::milliseconds(190);
        assert!(
            (sunset - expected_sunset).num_milliseconds().abs() < 1_000,
            "sunset mismatch at A5.1 reference: got {sunset}",
        );
    }

    /// The orchestrator must place `sunset > transit > sunrise`
    /// (absolute UTC), which holds *after* the wrap-relative-to-transit
    /// unwrap inside [`SolarDay::compute`] is applied. A regression in
    /// the unwrap (e.g. wrapping all three to `[0, 1)`, or unwrapping
    /// in the wrong sign) would surface here at the A5.1 reference,
    /// because A5.1 has a sunset that crosses into the next UT day and
    /// would otherwise come back as `00:20:19 UT on Oct 17` (before
    /// transit) instead of `Oct 18 00:20:19 UT` (after transit).
    #[test]
    fn solar_day_orders_sunrise_before_transit_before_sunset() {
        let day = reference_solar_day();
        let sunrise = day.sunrise.expect("sunrise must exist at A5.1 reference");
        let sunset = day.sunset.expect("sunset must exist at A5.1 reference");
        assert!(
            sunrise < day.transit,
            "sunrise {sunrise} must precede transit {}",
            day.transit
        );
        assert!(
            day.transit < sunset,
            "transit {} must precede sunset {sunset}",
            day.transit
        );
    }

    /// `SolarDay::compute` must return a result expressed in the same
    /// timezone as its input, so that callers passing a Madrid instant
    /// see Madrid wall clocks back. The civil time on Oct 17 2003 in
    /// Madrid was UTC+2 (the DST transition to UTC+1 happens on the
    /// last Sunday of October, i.e. Oct 26 2003), so the canonical
    /// Table A5.1 events on Madrid's clock are:
    ///
    /// * sunrise `13:12:43.46 UT` → `15:12:43.46 CEST`,
    /// * transit `18:46:04.97 UT` → `20:46:04.97 CEST`,
    /// * sunset  `00:20:19.19 UT` on Oct 18 → `02:20:19.19 CEST` on Oct 18.
    ///
    /// The 1-second tolerance absorbs the 0.23 s appendix-reported
    /// SPA-vs-Almanac bias and the sub-millisecond anchor-driven drift
    /// inherent to running the SPA from a Madrid-midnight anchor (a
    /// 2 h shift on the appendix A.2 `0 UT` reference) rather than the
    /// UT-midnight anchor used by Table A5.1's worked example.
    #[test]
    fn solar_day_preserves_input_timezone() {
        let madrid_dt = SpaDateTime::new(
            chrono_tz::Europe::Madrid
                .with_ymd_and_hms(2003, 10, 17, 21, 30, 30)
                .single()
                .expect("non-ambiguous CEST instant on the A5.1 reference date"),
        );
        let day = SolarDay::compute(&madrid_dt, REFERENCE_DELTA_T_SECONDS, reference_observer());

        let expected_sunrise = chrono_tz::Europe::Madrid
            .with_ymd_and_hms(2003, 10, 17, 15, 12, 43)
            .single()
            .unwrap()
            + chrono::TimeDelta::milliseconds(460);
        let sunrise = day
            .sunrise
            .expect("sunrise must exist at the A5.1 reference");
        assert!(
            (sunrise - expected_sunrise).num_milliseconds().abs() < 1_000,
            "Madrid sunrise must equal Table A5.1's sunrise on Madrid's wall clock, got {sunrise}",
        );

        let expected_transit = chrono_tz::Europe::Madrid
            .with_ymd_and_hms(2003, 10, 17, 20, 46, 4)
            .single()
            .unwrap()
            + chrono::TimeDelta::milliseconds(970);
        assert!(
            (day.transit - expected_transit).num_milliseconds().abs() < 1_000,
            "Madrid transit must equal Table A5.1's transit on Madrid's wall clock, got {}",
            day.transit,
        );

        let expected_sunset = chrono_tz::Europe::Madrid
            .with_ymd_and_hms(2003, 10, 18, 2, 20, 19)
            .single()
            .unwrap()
            + chrono::TimeDelta::milliseconds(190);
        let sunset = day.sunset.expect("sunset must exist at the A5.1 reference");
        assert!(
            (sunset - expected_sunset).num_milliseconds().abs() < 1_000,
            "Madrid sunset must equal Table A5.1's sunset on Madrid's wall clock, got {sunset}",
        );
    }

    /// `SolarDay::compute` must anchor `D₀` on the *local civil date* of
    /// the input, not on its UTC date. Two inputs whose local Madrid
    /// dates agree but whose UTC dates disagree must therefore produce
    /// the same events: at 01:00 CEST the UTC date has already rolled
    /// back to the previous day, while 09:00 CEST sits on the next UTC
    /// date. A UTC-anchored regression would surface here as a ~24 h
    /// gap between the two transits. The chosen 2003-10-17 reference
    /// year keeps Madrid in CEST (DST through Oct 26), so both inputs
    /// share an unambiguous +2 h offset.
    #[test]
    fn solar_day_anchors_on_input_local_civil_date() {
        let madrid = chrono_tz::Europe::Madrid;
        let morning = SpaDateTime::new(
            madrid
                .with_ymd_and_hms(2003, 10, 17, 9, 0, 0)
                .single()
                .unwrap(),
        );
        let early = SpaDateTime::new(
            madrid
                .with_ymd_and_hms(2003, 10, 17, 1, 0, 0)
                .single()
                .unwrap(),
        );

        let day_morning =
            SolarDay::compute(&morning, REFERENCE_DELTA_T_SECONDS, reference_observer());
        let day_early = SolarDay::compute(&early, REFERENCE_DELTA_T_SECONDS, reference_observer());

        let transit_diff_ms = (day_morning.transit - day_early.transit)
            .num_milliseconds()
            .abs();
        assert!(
            transit_diff_ms < 1_000,
            "transit must be invariant to the time-of-day on a single local civil date \
             (Madrid 2003-10-17): got {transit_diff_ms} ms between 09:00 and 01:00 CEST inputs",
        );
        assert_eq!(
            day_morning.transit.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2003, 10, 17).unwrap(),
            "transit on Madrid 2003-10-17 must report Madrid's Oct 17, got {}",
            day_morning.transit,
        );
    }

    /// `local_civil_midnight_in_utc` must select the
    /// [`MappedLocalTime::Ambiguous`] arm when a DST fall-back loops the
    /// local clock back over 00:00, producing two valid UT instants for
    /// the same local midnight. `Atlantic/Azores` fell back from Azores
    /// Summer Time (UTC+0) to Azores Standard Time (UTC-1) at 01:00 LST
    /// on 2003-10-26, which means the 00:00 → 01:00 LST hour is
    /// represented twice and `from_local_datetime(2003-10-26 00:00)`
    /// returns `Ambiguous`. The orchestrator must take the earliest
    /// representation, complete without panicking, and produce three
    /// ordered events at this mid-Atlantic latitude.
    #[test]
    fn solar_day_handles_ambiguous_local_midnight_on_dst_fall_back() {
        let azores = chrono_tz::Atlantic::Azores;
        let dt = SpaDateTime::new(
            azores
                .with_ymd_and_hms(2003, 10, 26, 12, 0, 0)
                .single()
                .expect("noon is unambiguous on the fall-back date"),
        );
        let observer = Observer {
            longitude: -25.668,
            latitude: 37.741,
            elevation: 50.0,
            pressure: 1015.0,
            temperature: 18.0,
        };
        let day = SolarDay::compute(&dt, REFERENCE_DELTA_T_SECONDS, observer);
        let sunrise = day
            .sunrise
            .expect("non-polar Azores latitude must produce a sunrise");
        let sunset = day
            .sunset
            .expect("non-polar Azores latitude must produce a sunset");
        assert!(
            sunrise < day.transit,
            "sunrise {sunrise} must precede transit {}",
            day.transit,
        );
        assert!(
            day.transit < sunset,
            "transit {} must precede sunset {sunset}",
            day.transit,
        );
    }

    /// `local_civil_midnight_in_utc` must fall back gracefully when a
    /// DST spring-forward skips over the entire 00:00 minute, leaving
    /// `from_local_datetime(midnight)` with no UT representation.
    /// `America/Sao_Paulo` sprang forward from Brazil Standard Time
    /// (UTC-3) to Brazil Summer Time (UTC-2) at midnight on 2017-10-15,
    /// so `from_local_datetime(2017-10-15 00:00)` returns `None`. The
    /// orchestrator must reinterpret the naive local midnight as a UT
    /// instant (the documented fallback), complete without panicking,
    /// and produce three ordered events for the SE-Brazil latitude.
    #[test]
    fn solar_day_handles_skipped_local_midnight_on_dst_spring_forward() {
        let sao_paulo = chrono_tz::America::Sao_Paulo;
        let dt = SpaDateTime::new(
            sao_paulo
                .with_ymd_and_hms(2017, 10, 15, 12, 0, 0)
                .single()
                .expect("noon is unambiguous on the spring-forward date"),
        );
        let observer = Observer {
            longitude: -46.625,
            latitude: -23.533,
            elevation: 760.0,
            pressure: 1010.0,
            temperature: 22.0,
        };
        let day = SolarDay::compute(&dt, REFERENCE_DELTA_T_SECONDS, observer);
        let sunrise = day
            .sunrise
            .expect("non-polar Sao_Paulo latitude must produce a sunrise");
        let sunset = day
            .sunset
            .expect("non-polar Sao_Paulo latitude must produce a sunset");
        assert!(
            sunrise < day.transit,
            "sunrise {sunrise} must precede transit {}",
            day.transit,
        );
        assert!(
            day.transit < sunset,
            "transit {} must precede sunset {sunset}",
            day.transit,
        );
    }

    /// `SolarDay::compute` must propagate DUT1 (`UT1 − UTC`, in
    /// seconds) into the Julian Day chain. UT1 leads UTC by DUT1
    /// seconds, so the same Earth-rotation-driven event reaches a
    /// fixed UT1 instant `DUT1` seconds *earlier* on the UTC clock:
    /// the returned UTC datetime must therefore shift by `−DUT1`.
    /// A regression that drops DUT1 inside the orchestrator would
    /// surface as a zero shift at the IERS upper bound `±1 s`, far
    /// outside the 50 ms tolerance.
    #[test]
    fn solar_day_propagates_dut1_into_event_times() {
        let zero = SolarDay::compute(
            &reference_datetime(),
            REFERENCE_DELTA_T_SECONDS,
            reference_observer(),
        );
        let with_dut1 = SolarDay::compute(
            &reference_datetime().try_with_dut1(0.5).unwrap(),
            REFERENCE_DELTA_T_SECONDS,
            reference_observer(),
        );
        let transit_shift_ms = (with_dut1.transit - zero.transit).num_milliseconds();
        assert!(
            (transit_shift_ms - -500).abs() < 50,
            "DUT1 = +0.5 s must shift transit (UTC) by ~-500 ms, got {transit_shift_ms} ms",
        );
    }

    /// At a high-latitude observer in midwinter the sun never rises:
    /// the orchestrator must return `None` for both sunrise and sunset
    /// while `transit` remains populated (the sun still crosses the
    /// meridian, just below the horizon). The chosen probe at 80° N on
    /// the December solstice 2026 falls comfortably inside polar
    /// night for the canonical `h'₀ = -0.8333°`. A regression in the
    /// `Option<H₀>` short-circuit (e.g. propagating a NaN from a
    /// `f64::acos` of an out-of-range argument) would surface here as
    /// either a finite sunrise/sunset or a panic.
    #[test]
    fn solar_day_polar_night_returns_none_for_sunrise_and_sunset() {
        let polar = Observer {
            longitude: 0.0,
            latitude: 80.0,
            elevation: 0.0,
            pressure: 1010.0,
            temperature: -20.0,
        };
        let solstice = SpaDateTime::new(
            Utc.with_ymd_and_hms(2026, 12, 21, 12, 0, 0)
                .single()
                .unwrap(),
        );
        let day = SolarDay::compute(&solstice, 70.0, polar);
        assert!(
            day.sunrise.is_none(),
            "polar night must have no sunrise, got {:?}",
            day.sunrise
        );
        assert!(
            day.sunset.is_none(),
            "polar night must have no sunset, got {:?}",
            day.sunset
        );
    }

    /// Symmetric counterpart: at the same high latitude in midsummer
    /// the sun never sets. Both [`Option`] arms of
    /// [`sunrise_sunset_local_hour_angle`] (argument `> 1` and
    /// argument `< -1`) must therefore short-circuit, since polar day
    /// and polar night each saturate one bound. Failure here while the
    /// midwinter test passes pins which arm regressed.
    #[test]
    fn solar_day_polar_day_returns_none_for_sunrise_and_sunset() {
        let polar = Observer {
            longitude: 0.0,
            latitude: 80.0,
            elevation: 0.0,
            pressure: 1010.0,
            temperature: 0.0,
        };
        let solstice = SpaDateTime::new(
            Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0)
                .single()
                .unwrap(),
        );
        let day = SolarDay::compute(&solstice, 70.0, polar);
        assert!(
            day.sunrise.is_none(),
            "polar day must have no sunrise, got {:?}",
            day.sunrise
        );
        assert!(
            day.sunset.is_none(),
            "polar day must have no sunset, got {:?}",
            day.sunset
        );
    }

    /// Equation A3 reduces `(α − σ − ν)` to a fraction of revolution
    /// then wraps to `[0, 1)`. Each of the three inputs has a `±1°`
    /// shift coefficient on the result of `±1/360` after wrap; a sign
    /// flip on any of them (e.g. `(α + σ − ν)` for σ as positive-west)
    /// would invert the dependency. The sweep keeps the pre-wrap value
    /// well inside `(0, 1)` so the small `δ` shifts do not interact
    /// with the wrap boundary, isolating the linear contributions.
    #[test]
    fn approximate_sun_transit_time_is_linear_in_each_input_inside_range() {
        let alpha = 200.0_f64;
        let sigma = -50.0_f64;
        let nu = 50.0_f64;
        let baseline = approximate_sun_transit_time(alpha, sigma, nu);
        for &delta in &[-1.0_f64, -1e-3, 1e-6, 0.5] {
            let from_alpha = approximate_sun_transit_time(alpha + delta, sigma, nu);
            let from_sigma = approximate_sun_transit_time(alpha, sigma + delta, nu);
            let from_nu = approximate_sun_transit_time(alpha, sigma, nu + delta);
            assert!(
                (from_alpha - baseline - delta / 360.0).abs() < 1e-13,
                "m₀ must shift by +δ/360 when α shifts by δ = {delta}: got {from_alpha} - {baseline}",
            );
            assert!(
                (from_sigma - baseline + delta / 360.0).abs() < 1e-13,
                "m₀ must shift by -δ/360 when σ shifts by δ = {delta}: got {from_sigma} - {baseline}",
            );
            assert!(
                (from_nu - baseline + delta / 360.0).abs() < 1e-13,
                "m₀ must shift by -δ/360 when ν shifts by δ = {delta}: got {from_nu} - {baseline}",
            );
        }
    }

    /// Step A.2.7 mandates `m₀ ∈ [0, 1)`. The sweep drives the raw
    /// `(α − σ − ν)/360` to both sides of zero and well past unity,
    /// exercising both branches of the modular wrap.
    #[test]
    fn approximate_sun_transit_time_is_wrapped_into_unit_interval() {
        for &(alpha, sigma, nu) in &[
            (0.0_f64, 0.0, 0.0), // raw 0: pass-through.
            (-720.0, 0.0, 0.0),  // raw -2: negative-remainder branch, > one revolution.
            (720.0, 0.0, 0.0),   // raw +2: positive-remainder branch, > one revolution.
            (0.0, 360.0, 0.0),   // raw -1: negative-remainder branch, exactly one revolution.
        ] {
            let m_0 = approximate_sun_transit_time(alpha, sigma, nu);
            assert!(
                (0.0..1.0).contains(&m_0),
                "m₀ escaped [0, 1) for (α, σ, ν) = ({alpha}, {sigma}, {nu}): got {m_0}",
            );
        }
    }

    /// Equation A4's cosine argument `(sin h'₀ − sin φ · sin δ) /
    /// (cos φ · cos δ)` falls outside `[-1, 1]` at high latitudes near
    /// the solstices: that is the polar configuration. The two probes
    /// pin both arms of the `Option` short-circuit. The chosen
    /// `(φ = 80°, δ = -23°, h'₀ = -0.8333°)` puts the argument well
    /// above `1` (polar night), and `(φ = 80°, δ = 23°, h'₀ = -0.8333°)`
    /// puts it below `-1` (polar day).
    #[test]
    fn sunrise_sunset_local_hour_angle_returns_none_at_polar_configurations() {
        assert!(
            sunrise_sunset_local_hour_angle(80.0, -23.0, SUN_ELEVATION_AT_HORIZON_DEGREES)
                .is_none(),
            "polar night argument must exceed 1 and yield None",
        );
        assert!(
            sunrise_sunset_local_hour_angle(80.0, 23.0, SUN_ELEVATION_AT_HORIZON_DEGREES).is_none(),
            "polar day argument must drop below -1 and yield None",
        );
    }

    /// At the equator `(φ = 0°)` and equinox `(δ = 0°)` with the
    /// sun-centre cutoff `h'₀ = -0.8333°`, equation A4 reduces to
    /// `arccos(sin h'₀)`. With `sin(-0.8333°) ≈ -0.014542`, the
    /// expected `H₀` is `arccos(-0.014542) ≈ 90.8333°` — i.e. the
    /// horizon-level configuration where the day length exceeds 12 h
    /// by twice the refraction-plus-disk lift. A regression that
    /// dropped either the `sin h'₀` term or the polar guard would
    /// surface here as either a NaN propagation or a `90°` answer.
    #[test]
    fn sunrise_sunset_local_hour_angle_at_equator_equinox_equals_arccos_of_sin_horizon() {
        let h0 = sunrise_sunset_local_hour_angle(0.0, 0.0, SUN_ELEVATION_AT_HORIZON_DEGREES)
            .expect("non-polar configuration must produce a finite H₀");
        let expected = SUN_ELEVATION_AT_HORIZON_DEGREES
            .to_radians()
            .sin()
            .acos()
            .to_degrees();
        assert!(
            (h0 - expected).abs() < 1e-12,
            "H₀ at φ = 0°, δ = 0° must equal arccos(sin h'₀) ≈ {expected}°: got {h0}",
        );
    }

    /// Equations A5 and A6 are mirror images: shifting `H₀` by `δ`
    /// shifts `m₁` by `-δ/360` and `m₂` by `+δ/360`. A swap of A5/A6
    /// (sunrise/sunset confusion) would invert the signs and surface
    /// here. The baseline `(m₀ = 0.5, H₀ = 90°)` keeps the pre-wrap
    /// values comfortably inside `(0, 1)` so the small `δ` does not
    /// trip the wrap.
    #[test]
    fn approximate_sunrise_and_sunset_times_are_symmetric_in_hour_angle() {
        let m_0 = 0.5_f64;
        let h_0 = 90.0_f64;
        let baseline_sunrise = approximate_sunrise_time(m_0, h_0);
        let baseline_sunset = approximate_sunset_time(m_0, h_0);
        for &delta in &[-1.0_f64, -1e-3, 1e-6, 1.0] {
            let sunrise = approximate_sunrise_time(m_0, h_0 + delta);
            let sunset = approximate_sunset_time(m_0, h_0 + delta);
            assert!(
                (sunrise - baseline_sunrise + delta / 360.0).abs() < 1e-13,
                "m₁ must shift by -δ/360 when H₀ shifts by δ = {delta}: got {sunrise} vs baseline {baseline_sunrise}",
            );
            assert!(
                (sunset - baseline_sunset - delta / 360.0).abs() < 1e-13,
                "m₂ must shift by +δ/360 when H₀ shifts by δ = {delta}: got {sunset} vs baseline {baseline_sunset}",
            );
        }
    }

    /// Step A.2.7 mandates `m₁, m₂ ∈ [0, 1)`. The sweep combines
    /// `(m₀, H₀)` configurations whose raw `m₀ ± H₀/360` lands either
    /// just below zero or just above one, exercising both wrap arms.
    #[test]
    fn approximate_sunrise_and_sunset_times_are_wrapped_into_unit_interval() {
        for &(m_0, h_0) in &[
            (0.05_f64, 90.0_f64), // sunrise raw -0.2: negative-remainder.
            (0.95, 90.0),         // sunset raw  +1.2: positive-remainder.
            (0.0, 360.0),         // sunrise raw -1.0 / sunset raw +1.0: full-revolution boundary.
        ] {
            let sunrise = approximate_sunrise_time(m_0, h_0);
            let sunset = approximate_sunset_time(m_0, h_0);
            assert!(
                (0.0..1.0).contains(&sunrise),
                "m₁ escaped [0, 1) for (m₀, H₀) = ({m_0}, {h_0}): got {sunrise}",
            );
            assert!(
                (0.0..1.0).contains(&sunset),
                "m₂ escaped [0, 1) for (m₀, H₀) = ({m_0}, {h_0}): got {sunset}",
            );
        }
    }

    /// Equation A7 advances ν by the published diurnal rate. At
    /// `m_i = 0` the result must collapse to the upstream `ν`; at
    /// `m_i = 1` it must lead by exactly the canonical `360.985647°`
    /// shift; at `m_i = 0.5` by half of that. A typo in the rate
    /// constant would surface as a measurable shift here even when the
    /// A5.1 reference still passes (because the reference probes a
    /// single `m_0 ≈ 0.78` configuration).
    #[test]
    fn sidereal_time_at_event_advances_by_published_diurnal_rate() {
        let nu = 100.0_f64;
        for &m in &[0.0_f64, 0.25, 0.5, 1.0] {
            let advanced = sidereal_time_at_event(nu, m);
            let expected = EARTH_SIDEREAL_DAILY_ROTATION_DEGREES.mul_add(m, nu);
            assert!(
                (advanced - expected).abs() < 1e-12,
                "νᵢ at m = {m} must equal ν + 360.985647·m: got {advanced} vs expected {expected}",
            );
        }
    }

    /// Equation A8 is `nᵢ = mᵢ + ΔT/86400`. A regression with the
    /// wrong divisor (e.g. `3600` for hours) would surface as a
    /// 24-fold shift here. The probe runs at the canonical
    /// `ΔT = 67 s` and several `m` values to rule out a fixed-shift
    /// bug.
    #[test]
    fn delta_t_corrected_event_time_adds_seconds_per_day_normalised_offset() {
        for &m in &[0.0_f64, 0.5, 0.999] {
            let n = delta_t_corrected_event_time(m, 67.0);
            let expected = m + 67.0 / 86_400.0;
            assert!(
                (n - expected).abs() < 1e-15,
                "nᵢ at m = {m} must equal m + ΔT/86400: got {n} vs expected {expected}",
            );
        }
    }

    /// Stirling interpolation at the three boundary nodes must
    /// reproduce the input values exactly: `n = -1 → value₋₁`,
    /// `n = 0 → value₀`, `n = 1 → value₊₁`. This pins the formula's
    /// shape (second-degree polynomial passing through the three
    /// points) independently of the wrap-difference logic, since the
    /// chosen `(value₋₁, value₀, value₊₁) = (10, 11, 12)` keeps every
    /// difference inside the no-wrap branch.
    #[test]
    fn interpolate_three_day_value_passes_through_each_node() {
        let v_minus = 10.0_f64;
        let v_zero = 11.0_f64;
        let v_plus = 12.0_f64;
        for &(n, expected) in &[(-1.0_f64, v_minus), (0.0, v_zero), (1.0, v_plus)] {
            let interpolated = interpolate_three_day_value(v_minus, v_zero, v_plus, n);
            assert!(
                (interpolated - expected).abs() < 1e-12,
                "interpolation at n = {n} must yield {expected}: got {interpolated}",
            );
        }
    }

    /// On a linear sequence (constant first difference, zero second
    /// difference) the Stirling interpolation must collapse to plain
    /// linear interpolation: `value₀ + n · (value₊₁ − value₀)`. A
    /// regression in the `c · n` second-difference coefficient would
    /// fail here because the linear ramp leaves `c = 0` exactly,
    /// isolating any spurious quadratic contribution.
    #[test]
    fn interpolate_three_day_value_collapses_to_linear_for_constant_first_difference() {
        let v_minus = 0.0_f64;
        let v_zero = 1.0_f64;
        let v_plus = 2.0_f64;
        for &n in &[-0.7_f64, -0.1, 0.3, 0.99] {
            let interpolated = interpolate_three_day_value(v_minus, v_zero, v_plus, n);
            let expected = n.mul_add(v_plus - v_zero, v_zero);
            assert!(
                (interpolated - expected).abs() < 1e-12,
                "linear-input interpolation at n = {n} must equal {expected}: got {interpolated}",
            );
        }
    }

    /// On a quadratic sequence (constant second difference) the
    /// interpolation reaches its non-degenerate form. The chosen
    /// `(0, 1, 4)` has `a = 1, b = 3, c = 2`. At `n = 0.5` the
    /// expected output is `1 + 0.5 · (1 + 3 + 2 · 0.5) / 2 = 2.25`.
    /// This isolates the second-difference handling: a regression
    /// dropping the `c · n` term would yield `2.0` (linear estimate).
    #[test]
    fn interpolate_three_day_value_recovers_second_difference_at_half_step() {
        let result = interpolate_three_day_value(0.0, 1.0, 4.0, 0.5);
        let expected = 2.25_f64;
        assert!(
            (result - expected).abs() < 1e-12,
            "Stirling at n = 0.5 on (0, 1, 4) must equal 2.25: got {result}",
        );
    }

    /// Step A.2.10 mandates folding `360°` wraps in the differences
    /// `a, b`. The probe `(α₋₁, α₀, α₊₁) = (358°, 1°, 4°)` simulates
    /// the vernal-equinox crossing where `α` rolls from below `360°`
    /// down to small positive values. The natural `(a, b)` are
    /// `(-357°, 3°)`; after the wrap, `a' = 3°`, recovering the
    /// physical diurnal motion. The interpolation at `n = 0` must
    /// then yield `1°` exactly. Without the wrap, `a` would dominate
    /// the Stirling sum and the result at `n = 0.5` (probed below)
    /// would be `~−176°` instead of `~2.5°`.
    #[test]
    fn interpolate_three_day_value_wraps_360_degree_difference_in_alpha() {
        let v_minus = 358.0_f64;
        let v_zero = 1.0_f64;
        let v_plus = 4.0_f64;
        // At n = 0 we trivially get value₀, but the wrap must still
        // allow it without polluting the constant term: any leak from
        // the unwrapped differences would propagate at n ≠ 0.
        let at_zero = interpolate_three_day_value(v_minus, v_zero, v_plus, 0.0);
        assert!(
            (at_zero - v_zero).abs() < 1e-12,
            "interpolation at n=0 must equal value₀: got {at_zero}"
        );
        // At n = 0.5 the expected result is value₀ + 0.5·(3 + 3 + 0·0.5)/2 = 2.5.
        let at_half = interpolate_three_day_value(v_minus, v_zero, v_plus, 0.5);
        assert!(
            (at_half - 2.5).abs() < 1e-12,
            "wrap must recover the diurnal motion: at n = 0.5 expected 2.5°, got {at_half}",
        );
    }

    /// `wrap_interpolation_difference` must be the identity for any
    /// input below the 2° threshold (the natural sun motion never
    /// exceeds it) and must collapse a near-`±360°` wrap to a small
    /// representative. Direct equality is meaningful below the
    /// threshold: the function returns its input verbatim.
    #[test]
    #[allow(clippy::float_cmp)]
    fn wrap_interpolation_difference_threshold_behaviour() {
        for &diff in &[-2.0_f64, -1.5, -1e-6, 0.0, 1e-6, 1.5, 2.0] {
            assert_eq!(
                wrap_interpolation_difference(diff),
                diff,
                "differences within the 2° threshold must pass through",
            );
        }
        assert!(
            (wrap_interpolation_difference(-359.0) - 1.0).abs() < 1e-12,
            "wrap of -359° must recover +1°",
        );
        assert!(
            (wrap_interpolation_difference(361.0) - 1.0).abs() < 1e-12,
            "wrap of +361° must recover +1°",
        );
    }

    /// Step A.2.11 mandates `H'ᵢ ∈ (-180°, 180°]`. The sweep covers
    /// raw inputs that originate above `+360°` and below `-360°` (the
    /// pre-wrap "limit to ±360°" branch the appendix mentions) plus a
    /// near-`+180°` and a near-`-180°` boundary so the post-wrap fold
    /// is also exercised.
    #[test]
    fn event_local_hour_angle_wraps_into_signed_180() {
        for &(nu, sigma, alpha) in &[
            (0.0_f64, 0.0, 0.0), // raw 0: pass-through.
            (300.0, 0.0, 0.0),   // raw 300°: high half, must fold to -60°.
            (60.0, 0.0, 0.0),    // raw 60°: low half, pass-through.
            (-300.0, 0.0, 0.0),  // raw -300°: pre-wrap below -360, must fold to +60°.
            (1080.0, 0.0, 0.0),  // raw 1080°: three full revolutions above zero.
        ] {
            let h_prime = event_local_hour_angle(nu, sigma, alpha);
            assert!(
                (-180.0 < h_prime) && (h_prime <= 180.0),
                "H'ᵢ escaped (-180°, 180°] for (ν, σ, α) = ({nu}, {sigma}, {alpha}): got {h_prime}",
            );
        }
    }

    /// At a 300° raw `H'ᵢ`, the wrap must fold to `-60°`; the inverse
    /// boundary at `-300°` must fold to `+60°`. This pins both branches
    /// of the wrap explicitly: a typo replacing `-360.0` by `+360.0`
    /// would invert the sign on at least one probe.
    #[test]
    fn event_local_hour_angle_wraps_300_degrees_to_minus_60() {
        let positive = event_local_hour_angle(300.0, 0.0, 0.0);
        assert!(
            (positive - -60.0).abs() < 1e-12,
            "raw 300° must fold to -60°: got {positive}",
        );
        let negative = event_local_hour_angle(-300.0, 0.0, 0.0);
        assert!(
            (negative - 60.0).abs() < 1e-12,
            "raw -300° must fold to +60°: got {negative}",
        );
    }

    /// Equation A12 collapses to `arcsin(sin φ · sin δ) = δ` at
    /// `H'ᵢ = ±90°` (sun's hour angle perpendicular to the meridian)
    /// only when `φ = 0`; a more useful structural anchor is `H'ᵢ = 0°`
    /// (sun on meridian), where equation A12 reduces to
    /// `arcsin(sin φ · sin δ + cos φ · cos δ) = arcsin(cos(φ − δ))
    /// = 90° − |φ − δ|`. The probe at `(φ, δ) = (40°, -10°)` therefore
    /// expects `90° − 50° = 40°`, pinning the formula at the meridian
    /// crossing.
    #[test]
    fn sun_altitude_at_event_reduces_to_complement_of_zenith_distance_at_meridian() {
        let altitude = sun_altitude_at_event(40.0, -10.0, 0.0);
        let expected = 90.0_f64 - (40.0_f64 - -10.0_f64).abs();
        assert!(
            (altitude - expected).abs() < 1e-12,
            "h at meridian must equal 90° - |φ - δ|: expected {expected}°, got {altitude}",
        );
    }

    /// Equation A13 is `T = m₀ - H'₀ / 360`. Shifting `H'₀` by `δ`
    /// must shift `T` by `-δ/360`. A regression dropping the divisor
    /// or flipping the sign would surface here even when the A5.1
    /// reference test still passes (the reference probes a single
    /// `H'₀ ≈ -0.001°` configuration, where small typos can hide).
    #[test]
    fn sun_transit_time_corrects_m0_by_minus_h_prime_over_360() {
        let m_0 = 0.5_f64;
        let baseline = sun_transit_time(m_0, 0.0);
        for &delta in &[-180.0_f64, -1.0, 1e-3, 90.0] {
            let shifted = sun_transit_time(m_0, delta);
            let expected = baseline - delta / 360.0;
            assert!(
                (shifted - expected).abs() < 1e-13,
                "T must shift by -δ/360 when H'₀ shifts by δ = {delta}: got {shifted} vs expected {expected}",
            );
        }
    }

    /// `sunrise_or_sunset_time` of equation A14/A15 is a Newton step
    /// of `h(t) = h'₀`. When the residual altitude `hᵢ − h'₀` is zero
    /// the correction must vanish, leaving `mᵢ` unchanged. This pins
    /// the formula's structure: a typo dropping the residual altogether
    /// (returning `mᵢ + 1/(360 · …)`) would fail here. The sweep
    /// exercises non-trivial `(δ, φ, H')` so the denominator is finite
    /// and any leak from a non-zero correction surfaces.
    #[test]
    fn sunrise_or_sunset_time_collapses_to_input_when_residual_altitude_vanishes() {
        for &(delta, phi, h_prime) in &[
            (-10.0_f64, 40.0, 80.0),
            (5.0, -30.0, -90.0),
            (0.0, 0.0, 90.0),
        ] {
            let m = 0.25_f64;
            let h = SUN_ELEVATION_AT_HORIZON_DEGREES;
            let result =
                sunrise_or_sunset_time(m, h, SUN_ELEVATION_AT_HORIZON_DEGREES, delta, phi, h_prime);
            assert!(
                (result - m).abs() < 1e-13,
                "h = h'₀ must keep the time at m: at (δ, φ, H') = ({delta}, {phi}, {h_prime}) got {result} vs expected {m}",
            );
        }
    }

    /// A non-zero residual `hᵢ − h'₀ = +1°` with a denominator close
    /// to `+360° · 1 · 1 · 1 = 360°` (which is `(δ, φ, H') ≈ (0, 0, 90°)`)
    /// must add `1/360` of a day to `mᵢ`. This pins the explicit
    /// `360 · cos δ · cos φ · sin H'` denominator: a typo using
    /// `cos H'` instead of `sin H'`, or omitting the `360` factor,
    /// would yield a completely different correction.
    #[test]
    fn sunrise_or_sunset_time_scales_residual_by_one_over_360_at_unit_denominator() {
        let m = 0.25_f64;
        // (δ = 0, φ = 0, H' = 90°): cos δ · cos φ · sin H' = 1; residual = +1°.
        let result = sunrise_or_sunset_time(m, 1.0, 0.0, 0.0, 0.0, 90.0);
        let expected = m + 1.0 / 360.0;
        assert!(
            (result - expected).abs() < 1e-13,
            "residual +1° at unit denominator must shift m by +1/360: got {result} vs expected {expected}",
        );
    }

    /// `right_ascension_and_declination` is the private chain
    /// that bundles the `JDE → (α, δ)` pipeline. At the Table A5.1
    /// reference instant, calling it on the published `JDE` must
    /// reproduce the published `α ≈ 202.22741°` and `δ ≈ -9.31434°`.
    /// Failure pins a typo inside the bundling itself (e.g. swapping
    /// `geocentric_right_ascension`'s argument order, or feeding `JC`
    /// instead of `JCE` to `nutation_in_longitude_and_obliquity`).
    /// The published `JDE = JD + ΔT / 86_400` is reconstructed from
    /// the fixture's UTC instant rather than the report's printed
    /// six-decimal `JD`, to share the bit-for-bit alignment that the
    /// rest of the per-section tests use.
    #[test]
    fn right_ascension_and_declination_matches_table_a5_1() {
        let dt = reference_datetime();
        let jd = julian::calculate_julian_day(&dt);
        let jde = julian::calculate_julian_ephemeris_day(jd, REFERENCE_DELTA_T_SECONDS);

        let (alpha, delta) = right_ascension_and_declination(jde);
        assert!(
            (alpha - 202.227_41).abs() < 1e-4,
            "α at A5.1 reference must equal 202.22741°: got {alpha}",
        );
        assert!(
            (delta - -9.314_34).abs() < 1e-4,
            "δ at A5.1 reference must equal -9.31434°: got {delta}",
        );
    }

    /// `refined_event_fraction_of_day` dispatches on [`EventKind`].
    /// The dispatch must select the transit branch for `EventKind::Transit`
    /// and the sunrise/sunset branch for the other two. This pins the
    /// `match` arms by re-running the full computation for each kind
    /// at the A5.1 reference and comparing against the unwrapped
    /// fractions reported in `solar_day_matches_table_a5_1`.
    #[test]
    #[allow(clippy::similar_names)] // `jde_0`, `jce_0`, `jme_0` mirror the appendix's notation.
    fn refined_event_fraction_dispatches_correctly_on_event_kind() {
        let observer = reference_observer();

        // 0 UT on D₀ (the day the A5.1 reference instant lives in).
        let utc_midnight = SpaDateTime::new(
            Utc.with_ymd_and_hms(2003, 10, 17, 0, 0, 0)
                .single()
                .unwrap(),
        );
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

        let m_0 = approximate_sun_transit_time(alpha_zero, observer.longitude, nu);
        let h_0 = sunrise_sunset_local_hour_angle(
            observer.latitude,
            delta_zero,
            SUN_ELEVATION_AT_HORIZON_DEGREES,
        )
        .expect("non-polar A5.1 reference must produce a finite H₀");

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

        // Compare against Table A5.1 values converted from UT clock to
        // seconds-since-midnight on D₀. A 1-second tolerance absorbs the
        // 0.23 s appendix-reported bias and any sub-second drift accumulated
        // through the upstream chain. Failure here means the EventKind
        // dispatch routed to the wrong branch (e.g. transit equation A13
        // instead of A14/A15).
        let assert_close = |actual: i64, expected: i64, label: &str| {
            assert!(
                (actual - expected).abs() <= 1,
                "{label}: got {actual} s, expected {expected} ± 1",
            );
        };
        assert_close(
            fraction_to_clock_seconds(transit.fraction_of_day),
            67_565,
            "transit at A5.1",
        );
        assert_close(
            fraction_to_clock_seconds(sunrise.fraction_of_day),
            47_563,
            "sunrise at A5.1",
        );
        // Sunset's *raw* refined fraction is the wrapped time-of-day on
        // D₀ (00:20:19 UT); the [`SolarDay::compute`] orchestrator is
        // what later lifts it onto Oct 18 by unwrapping relative to the
        // transit. This test is checking the dispatch only, so it
        // compares to the wrapped 1 219 s.
        assert_close(
            fraction_to_clock_seconds(sunset.fraction_of_day),
            1_219,
            "sunset at A5.1",
        );

        // Per-event H' must be small at transit (sun on the meridian by
        // construction) and roughly mirror around it at sunrise/sunset.
        // This pins the new RefinedEvent fields against a known operating
        // point, ensuring the orchestrator wires them through unchanged.
        assert!(
            transit.local_hour_angle.abs() < 0.1,
            "H' at transit must be near zero: got {}",
            transit.local_hour_angle,
        );
        assert!(
            sunrise.local_hour_angle < 0.0,
            "H' at sunrise must be negative (east of meridian): got {}",
            sunrise.local_hour_angle,
        );
        assert!(
            sunset.local_hour_angle > 0.0,
            "H' at sunset must be positive (west of meridian): got {}",
            sunset.local_hour_angle,
        );

        // δ' at transit must agree with δ_zero within the diurnal motion
        // of the sun's declination (≤ 0.4°/day). This pins the
        // interpolated_declination wiring on the new RefinedEvent struct.
        assert!(
            (transit.interpolated_declination - delta_zero).abs() < 1.0,
            "δ' at transit must lie within ~1° of δ₀: got {} vs {}",
            transit.interpolated_declination,
            delta_zero,
        );
    }

    /// At an extreme east longitude where transit lands in the early
    /// UT morning, the sunrise of the same observer-day happens on
    /// the *previous* UT day. The orchestrator's unwrap-relative-to-
    /// transit logic must then shift the wrapped sunrise fraction
    /// down by `1` (the `raw > 0.5` branch of the unwrap), which
    /// places the returned sunrise datetime before `utc_midnight`.
    /// A regression that dropped the high-side branch (or always
    /// added `+1`) would surface here as a sunrise on the same UT
    /// day, hours later than the transit.
    #[test]
    fn solar_day_unwraps_sunrise_to_previous_ut_day_for_east_longitude_observer() {
        // Observer near Sydney (+150° E) at the autumn equinox.
        let observer = Observer {
            longitude: 150.0,
            latitude: -33.8,
            elevation: 0.0,
            pressure: 1010.0,
            temperature: 18.0,
        };
        let utc_noon = SpaDateTime::new(
            Utc.with_ymd_and_hms(2026, 3, 20, 12, 0, 0)
                .single()
                .unwrap(),
        );
        let day = SolarDay::compute(&utc_noon, 70.0, observer);
        let sunrise = day
            .sunrise
            .expect("autumn equinox at -33.8° N must have a sunrise");
        let sunset = day
            .sunset
            .expect("autumn equinox at -33.8° N must have a sunset");

        assert!(
            sunrise < day.transit,
            "sunrise {sunrise} must precede transit {}",
            day.transit
        );
        assert!(
            day.transit < sunset,
            "transit {} must precede sunset {sunset}",
            day.transit
        );
        // Transit at +150° E happens around 02:00 UT, sunrise the previous UT
        // calendar day around 19:00 UT. Pin the sunrise UTC date to D₋₁ so a
        // future regression that dropped the cross-day unwrap surfaces.
        assert_eq!(
            sunrise.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2026, 3, 19).unwrap(),
            "sunrise must land on the previous UT calendar day, got {sunrise}",
        );
        assert_eq!(
            day.transit.date_naive(),
            chrono::NaiveDate::from_ymd_opt(2026, 3, 20).unwrap(),
            "transit must land on D₀, got {}",
            day.transit,
        );
    }

    /// `SolarDay`'s [`fmt::Display`] impl must render all three events
    /// readable in the populated case. Pinning the rendered string at
    /// a non-polar reference catches a regression where any of the
    /// three labels were dropped or the line ordering reordered.
    #[test]
    fn solar_day_display_renders_populated_events_with_labels() {
        let day = reference_solar_day();
        let rendered = format!("{day}");
        assert!(
            rendered.starts_with("Sunrise:"),
            "Display must lead with the Sunrise line: got {rendered}"
        );
        assert!(
            rendered.contains("Sun transit:"),
            "Display must include a Sun transit line: got {rendered}"
        );
        assert!(
            rendered.contains("Sunset:"),
            "Display must include a Sunset line: got {rendered}"
        );
        assert!(
            !rendered.contains("none (polar day or polar night)"),
            "non-polar Display must not advertise the polar-day fallback: got {rendered}",
        );
    }

    /// `SolarDay`'s [`fmt::Display`] impl must surface the
    /// `none (polar day or polar night)` placeholder for all four
    /// [`Option::None`] arms (sunrise time, sunset time, sunrise hour
    /// angle, sunset hour angle). This pins the `match` arms inside the
    /// [`fmt::Display`] impl that the non-polar test above cannot
    /// exercise. The construction uses a hand-rolled [`SolarDay`] (rather
    /// than running the full orchestrator at a polar location) so the
    /// Display path is pinned independently of the upstream polar guard.
    #[test]
    fn solar_day_display_marks_polar_day_for_missing_sunrise_or_sunset() {
        let transit = Utc
            .with_ymd_and_hms(2026, 12, 21, 12, 0, 0)
            .single()
            .unwrap();
        let polar_day = SolarDay::<Utc> {
            transit,
            sunrise: None,
            sunset: None,
            sun_transit_altitude: -23.0,
            sunrise_hour_angle: None,
            sunset_hour_angle: None,
        };
        let rendered = format!("{polar_day}");
        // All four polar arms (sunrise time, sunset time, sunrise hour
        // angle, sunset hour angle) must surface the placeholder line.
        let occurrences = rendered.matches("none (polar day or polar night)").count();
        assert_eq!(
            occurrences, 4,
            "polar-day Display must mark sunrise/sunset and their hour angles as none: \
             got {rendered}",
        );
        assert!(
            rendered.contains("Sun transit:"),
            "polar-day Display must still surface the transit line: got {rendered}",
        );
        assert!(
            rendered.contains("Sun transit altitude:"),
            "polar-day Display must still surface the transit altitude line: got {rendered}",
        );
    }

    /// `Display` must propagate every [`fmt::Error`] returned by the
    /// underlying writer back to the caller via the `?` operator after
    /// each `writeln!`. Walking a byte budget from `0` to one short of
    /// the full rendered length forces the writer to fail at every
    /// point inside [`SolarDay::fmt`], pinning the `?` propagation at
    /// every offset for each of the four [`Option`] arms (`Some` and
    /// `None` for sunrise; `Some` and `None` for sunset). A `.unwrap()`
    /// left in place of a `?` would let the error be silently dropped
    /// at the corresponding budget.
    #[test]
    fn solar_day_display_propagates_writer_errors() {
        let transit = Utc
            .with_ymd_and_hms(2026, 6, 21, 12, 0, 0)
            .single()
            .unwrap();
        let configurations = [
            // Sunrise = Some, Sunset = Some.
            SolarDay::<Utc> {
                transit,
                sunrise: Some(transit - chrono::TimeDelta::hours(6)),
                sunset: Some(transit + chrono::TimeDelta::hours(6)),
                sun_transit_altitude: 65.0,
                sunrise_hour_angle: Some(-90.0),
                sunset_hour_angle: Some(90.0),
            },
            // Sunrise = None, Sunset = Some — pins the polar `?` on the sunrise line.
            SolarDay::<Utc> {
                transit,
                sunrise: None,
                sunset: Some(transit + chrono::TimeDelta::hours(6)),
                sun_transit_altitude: 65.0,
                sunrise_hour_angle: None,
                sunset_hour_angle: Some(90.0),
            },
            // Sunrise = Some, Sunset = None — pins the polar `?` on the sunset line.
            SolarDay::<Utc> {
                transit,
                sunrise: Some(transit - chrono::TimeDelta::hours(6)),
                sunset: None,
                sun_transit_altitude: 65.0,
                sunrise_hour_angle: Some(-90.0),
                sunset_hour_angle: None,
            },
            // Both None — full polar configuration.
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
            let total_bytes = rendered.len();
            assert!(total_bytes > 0, "rendered SolarDay must not be empty");

            for budget in 0..total_bytes {
                let mut writer = FailingWriter { remaining: budget };
                let result = core::fmt::Write::write_fmt(&mut writer, format_args!("{day}"));
                assert!(
                    result.is_err(),
                    "Display must surface fmt::Error at byte budget {budget} of {total_bytes}",
                );
            }

            let mut writer = FailingWriter {
                remaining: total_bytes,
            };
            let result = core::fmt::Write::write_fmt(&mut writer, format_args!("{day}"));
            assert!(
                result.is_ok(),
                "Display must succeed when the writer accepts every byte",
            );
        }
    }
}
