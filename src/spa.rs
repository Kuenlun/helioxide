// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! Orchestrator running the full SPA pipeline end to end.

use core::fmt;

use chrono::TimeZone;
use thiserror::Error;

use crate::{
    SpaDateTime, apparent, equation_of_time, equatorial, geocentric, heliocentric, horizontal,
    hour_angle, incidence, julian, nutation, obliquity, parallax, sidereal,
};

/// Geographic and atmospheric description of the observation site.
///
/// Latitude positive north (section 3.12.2), longitude positive east
/// (section 3.11), elevation in metres (section 3.12.3), pressure in
/// millibars and temperature in degrees Celsius (equation 42).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observer {
    longitude: f64,
    latitude: f64,
    elevation: f64,
    pressure: f64,
    temperature: f64,
}

impl Observer {
    /// `1010 mbar`. Collapses equation 42's pressure ratio to one.
    pub const REFERENCE_PRESSURE_MILLIBARS: f64 = horizontal::STANDARD_PRESSURE_MILLIBARS;

    /// `10 °C` (`283 K`). Collapses equation 42's temperature ratio to one.
    pub const REFERENCE_TEMPERATURE_CELSIUS: f64 =
        horizontal::REFERENCE_TEMPERATURE_KELVIN - horizontal::KELVIN_OFFSET_FROM_CELSIUS;

    /// ICAO/ISA sea-level pressure (ISO 2533:1975): `1013.25 mbar`.
    pub const ISA_PRESSURE_MILLIBARS: f64 = 1013.25;

    /// ICAO/ISA sea-level temperature (ISO 2533:1975): `15 °C`.
    pub const ISA_TEMPERATURE_CELSIUS: f64 = 15.0;

    /// Open lower bound on temperature: equation 42's `273 + T` vanishes at
    /// `T = -273 °C` exactly (the paper uses `273`, not `273.15`).
    pub const TEMPERATURE_FLOOR_CELSIUS_EXCLUSIVE: f64 = -horizontal::KELVIN_OFFSET_FROM_CELSIUS;

    /// Build with explicit atmosphere.
    ///
    /// # Errors
    /// * `latitude` finite, in `[-90°, 90°]`.
    /// * `longitude` finite, in `[-180°, 180°]`.
    /// * `elevation` finite (no range).
    /// * `pressure` finite, strictly positive.
    /// * `temperature` finite, strictly above [`Self::TEMPERATURE_FLOOR_CELSIUS_EXCLUSIVE`].
    #[inline]
    pub fn try_new(
        latitude: f64,
        longitude: f64,
        elevation: f64,
        pressure: f64,
        temperature: f64,
    ) -> Result<Self, ObserverError> {
        if !latitude.is_finite() || !(-90.0..=90.0).contains(&latitude) {
            return Err(ObserverError::InvalidLatitude(latitude));
        }
        if !longitude.is_finite() || !(-180.0..=180.0).contains(&longitude) {
            return Err(ObserverError::InvalidLongitude(longitude));
        }
        if !elevation.is_finite() {
            return Err(ObserverError::InvalidElevation(elevation));
        }
        if !pressure.is_finite() || pressure <= 0.0 {
            return Err(ObserverError::InvalidPressure(pressure));
        }
        if !temperature.is_finite() || temperature <= Self::TEMPERATURE_FLOOR_CELSIUS_EXCLUSIVE {
            return Err(ObserverError::InvalidTemperature(temperature));
        }
        Ok(Self {
            longitude,
            latitude,
            elevation,
            pressure,
            temperature,
        })
    }

    /// Build with the SPA paper's reference atmosphere (`1010 mbar`, `10 °C`).
    ///
    /// # Errors
    /// Same variants as [`Self::try_new`] for the three geographic arguments.
    #[inline]
    pub fn try_with_reference_atmosphere(
        latitude: f64,
        longitude: f64,
        elevation: f64,
    ) -> Result<Self, ObserverError> {
        Self::try_new(
            latitude,
            longitude,
            elevation,
            Self::REFERENCE_PRESSURE_MILLIBARS,
            Self::REFERENCE_TEMPERATURE_CELSIUS,
        )
    }

    /// Build a sea-level observer using the ICAO/ISA atmosphere.
    ///
    /// # Errors
    /// [`ObserverError::InvalidLatitude`] or [`ObserverError::InvalidLongitude`]
    /// when out of range.
    #[inline]
    pub fn try_at_sea_level_isa(latitude: f64, longitude: f64) -> Result<Self, ObserverError> {
        Self::try_new(
            latitude,
            longitude,
            0.0,
            Self::ISA_PRESSURE_MILLIBARS,
            Self::ISA_TEMPERATURE_CELSIUS,
        )
    }

    #[inline]
    #[must_use]
    pub const fn latitude(&self) -> f64 {
        self.latitude
    }

    #[inline]
    #[must_use]
    pub const fn longitude(&self) -> f64 {
        self.longitude
    }

    #[inline]
    #[must_use]
    pub const fn elevation(&self) -> f64 {
        self.elevation
    }

    #[inline]
    #[must_use]
    pub const fn pressure(&self) -> f64 {
        self.pressure
    }

    #[inline]
    #[must_use]
    pub const fn temperature(&self) -> f64 {
        self.temperature
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ObserverError {
    #[error("latitude {0}° must lie in [-90°, 90°] and be finite")]
    InvalidLatitude(f64),
    #[error("longitude {0}° must lie in [-180°, 180°] and be finite")]
    InvalidLongitude(f64),
    #[error("elevation {0} m must be finite")]
    InvalidElevation(f64),
    #[error("pressure {0} mbar must be > 0 mbar and finite")]
    InvalidPressure(f64),
    #[error(
        "temperature {0} °C must be > -273 °C and finite \
         (equation 42's denominator 273 + T vanishes at -273 °C)"
    )]
    InvalidTemperature(f64),
}

/// Tilted surface for the angle-of-incidence calculation (section 3.16).
///
/// `slope ∈ [0°, 180°]`, `azimuth_rotation ∈ [-180°, 180°]` measured
/// westward from south (same convention as `Γ`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Surface {
    slope: f64,
    azimuth_rotation: f64,
}

impl Surface {
    /// # Errors
    /// `slope` in `[0°, 180°]` and finite, `azimuth_rotation` in `[-180°, 180°]`
    /// and finite.
    #[inline]
    pub fn try_new(slope: f64, azimuth_rotation: f64) -> Result<Self, SurfaceError> {
        if !slope.is_finite() || !(0.0..=180.0).contains(&slope) {
            return Err(SurfaceError::InvalidSlope(slope));
        }
        if !azimuth_rotation.is_finite() || !(-180.0..=180.0).contains(&azimuth_rotation) {
            return Err(SurfaceError::InvalidAzimuthRotation(azimuth_rotation));
        }
        Ok(Self {
            slope,
            azimuth_rotation,
        })
    }

    /// Horizontal collector facing up.
    #[inline]
    #[must_use]
    pub const fn horizontal() -> Self {
        Self {
            slope: 0.0,
            azimuth_rotation: 0.0,
        }
    }

    #[inline]
    #[must_use]
    pub const fn slope(&self) -> f64 {
        self.slope
    }

    #[inline]
    #[must_use]
    pub const fn azimuth_rotation(&self) -> f64 {
        self.azimuth_rotation
    }
}

impl Default for Surface {
    #[inline]
    fn default() -> Self {
        Self::horizontal()
    }
}

#[derive(Debug, Clone, PartialEq, Error)]
pub enum SurfaceError {
    #[error("slope {0}° must lie in [0°, 180°] and be finite")]
    InvalidSlope(f64),
    #[error("azimuth rotation {0}° must lie in [-180°, 180°] and be finite")]
    InvalidAzimuthRotation(f64),
}

/// Full output of the SPA pipeline. Fields are ordered by paper section.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarPosition {
    pub julian_day: f64,
    pub julian_ephemeris_day: f64,
    pub julian_century: f64,
    pub julian_ephemeris_century: f64,
    pub julian_ephemeris_millennium: f64,

    pub earth_heliocentric_longitude: f64,
    pub earth_heliocentric_latitude: f64,
    pub earth_radius_vector: f64,

    pub geocentric_longitude: f64,
    pub geocentric_latitude: f64,

    /// `X₀..X₄` (degrees, raw). Equations 15 to 19.
    pub mean_elongation_moon_sun: f64,
    pub mean_anomaly_sun: f64,
    pub mean_anomaly_moon: f64,
    pub argument_latitude_moon: f64,
    pub ascending_longitude_moon: f64,

    pub nutation_in_longitude: f64,
    pub nutation_in_obliquity: f64,

    /// `ε₀` (arc seconds). Equation 24.
    pub mean_obliquity_arcseconds: f64,
    pub true_obliquity: f64,

    pub aberration_correction: f64,
    pub apparent_sun_longitude: f64,

    pub mean_sidereal_time: f64,
    pub apparent_sidereal_time: f64,

    pub geocentric_right_ascension: f64,
    pub geocentric_declination: f64,

    pub observer_local_hour_angle: f64,

    pub equatorial_horizontal_parallax: f64,
    pub parallax_in_right_ascension: f64,
    pub topocentric_right_ascension: f64,
    pub topocentric_declination: f64,

    pub topocentric_local_hour_angle: f64,

    pub topocentric_elevation_unrefracted: f64,
    pub atmospheric_refraction: f64,
    pub topocentric_elevation_corrected: f64,
    pub topocentric_zenith: f64,

    pub astronomers_azimuth: f64,
    pub astronomers_azimuth_signed: f64,
    pub topocentric_azimuth: f64,

    pub surface_incidence: f64,

    pub equation_of_time: f64,
}

impl SolarPosition {
    /// Run the full SPA pipeline, resolving `ΔT` via
    /// [`crate::delta_t::delta_t_seconds_for_datetime`] (observed USNO value
    /// when available, polynomial approximation otherwise). Pass
    /// [`Surface::horizontal`] when only the topocentric position is needed.
    /// Use [`Self::compute_with_delta_t`] to pin a specific `ΔT`.
    #[must_use]
    pub fn compute<Tz: TimeZone>(
        datetime: &SpaDateTime<Tz>,
        observer: Observer,
        surface: Surface,
    ) -> Self {
        let delta_t = crate::delta_t::delta_t_seconds_for_datetime(datetime.datetime());
        Self::compute_with_delta_t(datetime, delta_t, observer, surface)
    }

    /// Run the full SPA pipeline with an explicit `ΔT = TT − UT1` (seconds,
    /// ≈ 69.1 s in 2026).
    ///
    /// Reach for this when reproducing NREL reference cases or honouring an
    /// IERS bulletin value; otherwise [`Self::compute`] picks the best
    /// available `ΔT` automatically.
    #[must_use]
    #[allow(clippy::many_single_char_names, clippy::similar_names)]
    pub fn compute_with_delta_t<Tz: TimeZone>(
        datetime: &SpaDateTime<Tz>,
        delta_t: f64,
        observer: Observer,
        surface: Surface,
    ) -> Self {
        let jd = julian::calculate_julian_day(datetime);
        let jde = julian::calculate_julian_ephemeris_day(jd, delta_t);
        let jc = julian::calculate_julian_century(jd);
        let jce = julian::calculate_julian_ephemeris_century(jde);
        let jme = julian::calculate_julian_ephemeris_millennium(jce);

        let l = heliocentric::earth_heliocentric_longitude(jme);
        let b = heliocentric::earth_heliocentric_latitude(jme);
        let r = heliocentric::earth_radius_vector(jme);

        let theta = geocentric::geocentric_longitude(l);
        let beta = geocentric::geocentric_latitude(b);

        let [x0, x1, x2, x3, x4] = nutation::fundamental_arguments(jce);
        let (delta_psi, delta_epsilon) = nutation::nutation_in_longitude_and_obliquity(jce);
        // Evaluate equation 24 once and apply equation 25 inline.
        let epsilon0_arcseconds = obliquity::mean_obliquity_of_ecliptic_arcseconds(jme);
        let epsilon = epsilon0_arcseconds / 3600.0 + delta_epsilon;

        let delta_tau = apparent::aberration_correction(r);
        let lambda = apparent::apparent_sun_longitude(theta, delta_psi, delta_tau);

        let nu0 = sidereal::mean_sidereal_time(jd);
        let nu = sidereal::apparent_sidereal_time(nu0, delta_psi, epsilon);

        let alpha = equatorial::geocentric_right_ascension(lambda, beta, epsilon);
        let delta = equatorial::geocentric_declination(lambda, beta, epsilon);

        let h = hour_angle::observer_local_hour_angle(nu, observer.longitude(), alpha);

        let xi = parallax::equatorial_horizontal_parallax(r);
        let topocentric = parallax::topocentric_equatorial_coordinates(
            alpha,
            delta,
            h,
            xi,
            observer.latitude(),
            observer.elevation(),
        );

        let h_prime =
            hour_angle::topocentric_local_hour_angle(h, topocentric.parallax_in_right_ascension);

        let e0 = horizontal::topocentric_elevation_without_refraction(
            observer.latitude(),
            topocentric.declination,
            h_prime,
        );
        let delta_e =
            horizontal::atmospheric_refraction(e0, observer.pressure(), observer.temperature());
        let e_corrected = horizontal::topocentric_elevation_corrected(e0, delta_e);
        let zenith = horizontal::topocentric_zenith_angle(e0, delta_e);

        let gamma =
            horizontal::astronomers_azimuth(h_prime, observer.latitude(), topocentric.declination);
        let gamma_signed = horizontal::astronomers_azimuth_signed(gamma);
        let azimuth = horizontal::topocentric_azimuth_angle(gamma);

        let incidence_angle = incidence::surface_incidence_angle(
            zenith,
            gamma,
            surface.slope(),
            surface.azimuth_rotation(),
        );

        let m = equation_of_time::sun_mean_longitude(jme);
        let eot = equation_of_time::equation_of_time(m, alpha, delta_psi, epsilon);

        Self {
            julian_day: jd,
            julian_ephemeris_day: jde,
            julian_century: jc,
            julian_ephemeris_century: jce,
            julian_ephemeris_millennium: jme,
            earth_heliocentric_longitude: l,
            earth_heliocentric_latitude: b,
            earth_radius_vector: r,
            geocentric_longitude: theta,
            geocentric_latitude: beta,
            mean_elongation_moon_sun: x0,
            mean_anomaly_sun: x1,
            mean_anomaly_moon: x2,
            argument_latitude_moon: x3,
            ascending_longitude_moon: x4,
            nutation_in_longitude: delta_psi,
            nutation_in_obliquity: delta_epsilon,
            mean_obliquity_arcseconds: epsilon0_arcseconds,
            true_obliquity: epsilon,
            aberration_correction: delta_tau,
            apparent_sun_longitude: lambda,
            mean_sidereal_time: nu0,
            apparent_sidereal_time: nu,
            geocentric_right_ascension: alpha,
            geocentric_declination: delta,
            observer_local_hour_angle: h,
            equatorial_horizontal_parallax: xi,
            parallax_in_right_ascension: topocentric.parallax_in_right_ascension,
            topocentric_right_ascension: topocentric.right_ascension,
            topocentric_declination: topocentric.declination,
            topocentric_local_hour_angle: h_prime,
            topocentric_elevation_unrefracted: e0,
            atmospheric_refraction: delta_e,
            topocentric_elevation_corrected: e_corrected,
            topocentric_zenith: zenith,
            astronomers_azimuth: gamma,
            astronomers_azimuth_signed: gamma_signed,
            topocentric_azimuth: azimuth,
            surface_incidence: incidence_angle,
            equation_of_time: eot,
        }
    }
}

impl fmt::Display for SolarPosition {
    #[allow(clippy::too_many_lines)]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Julian Day: {}", self.julian_day)?;
        writeln!(f, "Julian Ephemeris Day: {}", self.julian_ephemeris_day)?;
        writeln!(f, "Julian Century: {}", self.julian_century)?;
        writeln!(
            f,
            "Julian Ephemeris Century: {}",
            self.julian_ephemeris_century
        )?;
        writeln!(
            f,
            "Julian Ephemeris Millennium: {}",
            self.julian_ephemeris_millennium
        )?;
        writeln!(
            f,
            "Earth heliocentric longitude L: {}°",
            self.earth_heliocentric_longitude
        )?;
        writeln!(
            f,
            "Earth heliocentric latitude B: {}°",
            self.earth_heliocentric_latitude
        )?;
        writeln!(f, "Earth radius vector R: {} AU", self.earth_radius_vector)?;
        writeln!(
            f,
            "Sun geocentric longitude Θ: {}°",
            self.geocentric_longitude
        )?;
        writeln!(
            f,
            "Sun geocentric latitude β: {}°",
            self.geocentric_latitude
        )?;
        writeln!(
            f,
            "Mean elongation (moon-sun) X₀: {}°",
            self.mean_elongation_moon_sun
        )?;
        writeln!(f, "Mean anomaly (sun) X₁: {}°", self.mean_anomaly_sun)?;
        writeln!(f, "Mean anomaly (moon) X₂: {}°", self.mean_anomaly_moon)?;
        writeln!(
            f,
            "Argument latitude (moon) X₃: {}°",
            self.argument_latitude_moon
        )?;
        writeln!(
            f,
            "Ascending longitude (moon) X₄: {}°",
            self.ascending_longitude_moon
        )?;
        writeln!(
            f,
            "Nutation in longitude Δψ: {}°",
            self.nutation_in_longitude
        )?;
        writeln!(
            f,
            "Nutation in obliquity Δε: {}°",
            self.nutation_in_obliquity
        )?;
        writeln!(
            f,
            "Ecliptic mean obliquity ε₀: {}\"",
            self.mean_obliquity_arcseconds
        )?;
        writeln!(
            f,
            "True obliquity of the ecliptic ε: {}°",
            self.true_obliquity
        )?;
        writeln!(
            f,
            "Aberration correction Δτ: {}°",
            self.aberration_correction
        )?;
        writeln!(
            f,
            "Apparent sun longitude λ: {}°",
            self.apparent_sun_longitude
        )?;
        writeln!(
            f,
            "Mean sidereal time at Greenwich ν₀: {}°",
            self.mean_sidereal_time
        )?;
        writeln!(
            f,
            "Apparent sidereal time at Greenwich ν: {}°",
            self.apparent_sidereal_time
        )?;
        writeln!(
            f,
            "Sun geocentric right ascension α: {}°",
            self.geocentric_right_ascension
        )?;
        writeln!(
            f,
            "Sun geocentric declination δ: {}°",
            self.geocentric_declination
        )?;
        writeln!(
            f,
            "Observer local hour angle H: {}°",
            self.observer_local_hour_angle
        )?;
        writeln!(
            f,
            "Sun equatorial horizontal parallax ξ: {}°",
            self.equatorial_horizontal_parallax
        )?;
        writeln!(
            f,
            "Sun right ascension parallax Δα: {}°",
            self.parallax_in_right_ascension
        )?;
        writeln!(
            f,
            "Topocentric sun right ascension α': {}°",
            self.topocentric_right_ascension
        )?;
        writeln!(
            f,
            "Topocentric sun declination δ': {}°",
            self.topocentric_declination
        )?;
        writeln!(
            f,
            "Topocentric local hour angle H': {}°",
            self.topocentric_local_hour_angle
        )?;
        writeln!(
            f,
            "Topocentric elevation without refraction e₀: {}°",
            self.topocentric_elevation_unrefracted
        )?;
        writeln!(
            f,
            "Atmospheric refraction Δe: {}°",
            self.atmospheric_refraction
        )?;
        writeln!(
            f,
            "Topocentric elevation (corrected) e: {}°",
            self.topocentric_elevation_corrected
        )?;
        writeln!(
            f,
            "Topocentric zenith angle θ: {}°",
            self.topocentric_zenith
        )?;
        writeln!(
            f,
            "Topocentric astronomers' azimuth Γ: {}°",
            self.astronomers_azimuth
        )?;
        writeln!(
            f,
            "Topocentric azimuth (westward from south, signed) Γ′: {}°",
            self.astronomers_azimuth_signed
        )?;
        writeln!(f, "Topocentric azimuth Φ: {}°", self.topocentric_azimuth)?;
        writeln!(f, "Surface incidence angle I: {}°", self.surface_incidence)?;
        write!(f, "Equation of time E: {} min", self.equation_of_time)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::{Observer, ObserverError, SolarPosition, Surface, SurfaceError};
    use crate::SpaDateTime;
    use crate::test_fixtures::{
        REFERENCE_ELEVATION_METRES, REFERENCE_LATITUDE_DEGREES, REFERENCE_LONGITUDE_DEGREES,
        REFERENCE_PRESSURE_MILLIBARS, REFERENCE_TEMPERATURE_CELSIUS,
    };
    use chrono::{TimeZone, Utc};
    use core::fmt::{self, Write};

    const REFERENCE_DELTA_T_SECONDS: f64 = 67.0;
    const REFERENCE_SURFACE_SLOPE_DEGREES: f64 = 30.0;
    const REFERENCE_SURFACE_AZIMUTH_ROTATION_DEGREES: f64 = -10.0;

    fn reference_datetime() -> SpaDateTime<Utc> {
        SpaDateTime::new(Utc.with_ymd_and_hms(2003, 10, 17, 19, 30, 30).unwrap())
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

    fn reference_surface() -> Surface {
        Surface::try_new(
            REFERENCE_SURFACE_SLOPE_DEGREES,
            REFERENCE_SURFACE_AZIMUTH_ROTATION_DEGREES,
        )
        .unwrap()
    }

    fn reference_position() -> SolarPosition {
        SolarPosition::compute_with_delta_t(
            &reference_datetime(),
            REFERENCE_DELTA_T_SECONDS,
            reference_observer(),
            reference_surface(),
        )
    }

    #[test]
    fn compute_matches_table_a5_1() {
        let p = reference_position();

        assert!((p.julian_day - 2_452_930.312_847).abs() < 1e-6);

        assert!((p.earth_heliocentric_longitude - 24.018_261_691_7).abs() < 1e-6);
        assert!((p.earth_heliocentric_latitude - -0.000_101_121_9).abs() < 1e-9);
        assert!((p.earth_radius_vector - 0.996_542_297_4).abs() < 1e-9);

        assert!((p.geocentric_longitude - 204.018_261_691_7).abs() < 1e-6);
        assert!((p.geocentric_latitude - 0.000_101_121_9).abs() < 1e-9);

        assert!((p.nutation_in_longitude - -0.003_998_40).abs() < 1e-8);
        assert!((p.nutation_in_obliquity - 0.001_666_57).abs() < 1e-8);

        let epsilon0_round_trip = (23.440_465_f64 - 0.001_666_57) * 3600.0;
        assert!((p.mean_obliquity_arcseconds - epsilon0_round_trip).abs() < 0.1);
        assert!((p.true_obliquity - 23.440_465).abs() < 1e-6);

        assert!((p.apparent_sun_longitude - 204.008_551_928_1).abs() < 1e-6);

        assert!((p.mean_sidereal_time - 318.515_578).abs() < 1e-4);
        assert!((p.apparent_sidereal_time - 318.511_910).abs() < 1e-4);

        assert!((p.geocentric_right_ascension - 202.227_41).abs() < 1e-4);
        assert!((p.geocentric_declination - -9.314_34).abs() < 1e-4);

        assert!((p.observer_local_hour_angle - 11.105_900).abs() < 1e-4);

        assert!((p.topocentric_right_ascension - 202.227_04).abs() < 1e-4);
        assert!((p.topocentric_declination - -9.316_179).abs() < 1e-4);

        assert!((p.topocentric_local_hour_angle - 11.106_29).abs() < 1e-4);

        assert!((p.topocentric_zenith - 50.111_62).abs() < 1e-4);
        assert!((p.topocentric_elevation_corrected - (90.0_f64 - 50.111_62)).abs() < 1e-4);
        assert!((p.astronomers_azimuth - 14.340_24).abs() < 1e-4);
        assert!((p.astronomers_azimuth_signed - 14.340_24).abs() < 1e-4);
        assert!((p.topocentric_azimuth - 194.340_24).abs() < 1e-4);

        assert!((p.surface_incidence - 25.187_00).abs() < 1e-4);

        assert!((p.equation_of_time - 14.641_503).abs() < 1e-4);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn compute_wires_intermediates_to_section_functions() {
        use crate::{
            apparent, equation_of_time, horizontal, julian, nutation, obliquity, parallax,
        };

        let dt = reference_datetime();
        let observer = reference_observer();
        let p = SolarPosition::compute_with_delta_t(
            &dt,
            REFERENCE_DELTA_T_SECONDS,
            observer,
            reference_surface(),
        );

        let jd = julian::calculate_julian_day(&dt);
        let jde = julian::calculate_julian_ephemeris_day(jd, REFERENCE_DELTA_T_SECONDS);
        let jc = julian::calculate_julian_century(jd);
        let jce = julian::calculate_julian_ephemeris_century(jde);
        let jme = julian::calculate_julian_ephemeris_millennium(jce);
        assert_eq!(p.julian_ephemeris_day, jde);
        assert_eq!(p.julian_century, jc);
        assert_eq!(p.julian_ephemeris_century, jce);
        assert_eq!(p.julian_ephemeris_millennium, jme);

        let [x0, x1, x2, x3, x4] = nutation::fundamental_arguments(jce);
        assert_eq!(p.mean_elongation_moon_sun, x0);
        assert_eq!(p.mean_anomaly_sun, x1);
        assert_eq!(p.mean_anomaly_moon, x2);
        assert_eq!(p.argument_latitude_moon, x3);
        assert_eq!(p.ascending_longitude_moon, x4);

        assert_eq!(
            p.mean_obliquity_arcseconds,
            obliquity::mean_obliquity_of_ecliptic_arcseconds(jme),
        );

        let r = p.earth_radius_vector;
        assert_eq!(p.aberration_correction, apparent::aberration_correction(r));
        assert_eq!(
            p.equatorial_horizontal_parallax,
            parallax::equatorial_horizontal_parallax(r)
        );

        let xi = parallax::equatorial_horizontal_parallax(r);
        let topocentric = parallax::topocentric_equatorial_coordinates(
            p.geocentric_right_ascension,
            p.geocentric_declination,
            p.observer_local_hour_angle,
            xi,
            observer.latitude(),
            observer.elevation(),
        );
        assert_eq!(
            p.parallax_in_right_ascension,
            topocentric.parallax_in_right_ascension
        );

        let e0 = horizontal::topocentric_elevation_without_refraction(
            observer.latitude(),
            p.topocentric_declination,
            p.topocentric_local_hour_angle,
        );
        let delta_e =
            horizontal::atmospheric_refraction(e0, observer.pressure(), observer.temperature());
        assert_eq!(p.topocentric_elevation_unrefracted, e0);
        assert_eq!(p.atmospheric_refraction, delta_e);
        assert_eq!(
            p.topocentric_elevation_corrected,
            horizontal::topocentric_elevation_corrected(e0, delta_e),
        );

        assert_eq!(
            p.astronomers_azimuth_signed,
            horizontal::astronomers_azimuth_signed(p.astronomers_azimuth),
        );

        let m = equation_of_time::sun_mean_longitude(jme);
        assert_eq!(
            p.equation_of_time,
            equation_of_time::equation_of_time(
                m,
                p.geocentric_right_ascension,
                p.nutation_in_longitude,
                p.true_obliquity,
            ),
        );
    }

    struct FailingWriter {
        remaining: usize,
    }

    impl Write for FailingWriter {
        fn write_str(&mut self, s: &str) -> fmt::Result {
            if s.len() > self.remaining {
                self.remaining = 0;
                Err(fmt::Error)
            } else {
                self.remaining -= s.len();
                Ok(())
            }
        }
    }

    #[test]
    fn display_propagates_writer_errors() {
        let position = reference_position();
        let rendered = format!("{position}");
        let total = rendered.len();
        for budget in 0..total {
            let mut writer = FailingWriter { remaining: budget };
            assert!(write!(&mut writer, "{position}").is_err());
        }
        let mut writer = FailingWriter { remaining: total };
        assert!(write!(&mut writer, "{position}").is_ok());
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn observer_constants_match_published_references() {
        assert_eq!(
            Observer::REFERENCE_PRESSURE_MILLIBARS,
            crate::horizontal::STANDARD_PRESSURE_MILLIBARS,
        );
        assert_eq!(Observer::REFERENCE_PRESSURE_MILLIBARS, 1010.0);
        assert_eq!(
            Observer::REFERENCE_TEMPERATURE_CELSIUS,
            crate::horizontal::REFERENCE_TEMPERATURE_KELVIN
                - crate::horizontal::KELVIN_OFFSET_FROM_CELSIUS,
        );
        assert_eq!(Observer::REFERENCE_TEMPERATURE_CELSIUS, 10.0);
        assert_eq!(Observer::ISA_PRESSURE_MILLIBARS, 1013.25);
        assert_eq!(Observer::ISA_TEMPERATURE_CELSIUS, 15.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn observer_try_with_reference_atmosphere_uses_paper_constants() {
        let obs = Observer::try_with_reference_atmosphere(38.346_02, -0.490_68, 3.0).unwrap();
        assert_eq!(obs.latitude(), 38.346_02);
        assert_eq!(obs.longitude(), -0.490_68);
        assert_eq!(obs.elevation(), 3.0);
        assert_eq!(obs.pressure(), Observer::REFERENCE_PRESSURE_MILLIBARS);
        assert_eq!(obs.temperature(), Observer::REFERENCE_TEMPERATURE_CELSIUS);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn observer_try_at_sea_level_isa_uses_isa_constants() {
        let obs = Observer::try_at_sea_level_isa(40.0, -3.0).unwrap();
        assert_eq!(obs.latitude(), 40.0);
        assert_eq!(obs.longitude(), -3.0);
        assert_eq!(obs.elevation(), 0.0);
        assert_eq!(obs.pressure(), Observer::ISA_PRESSURE_MILLIBARS);
        assert_eq!(obs.temperature(), Observer::ISA_TEMPERATURE_CELSIUS);
    }

    #[test]
    fn reference_atmosphere_collapses_equation_42_to_saemundsson() {
        let obs = Observer::try_with_reference_atmosphere(0.0, 0.0, 0.0).unwrap();
        for &e0 in &[0.0_f64, 10.0, 45.0, 89.0] {
            let actual =
                crate::horizontal::atmospheric_refraction(e0, obs.pressure(), obs.temperature());
            let aux = e0 + 10.3 / (e0 + 5.11);
            let expected = 1.02 / (60.0 * aux.to_radians().tan());
            assert!((actual - expected).abs() < 1e-15);
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn observer_temperature_floor_matches_equation_42_offset() {
        assert_eq!(
            Observer::TEMPERATURE_FLOOR_CELSIUS_EXCLUSIVE,
            -crate::horizontal::KELVIN_OFFSET_FROM_CELSIUS,
        );
        assert_eq!(Observer::TEMPERATURE_FLOOR_CELSIUS_EXCLUSIVE, -273.0);
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn observer_try_new_round_trips_through_accessors() {
        let obs = Observer::try_new(39.742_476, -105.1786, 1830.14, 820.0, 11.0).unwrap();
        assert_eq!(obs.latitude(), 39.742_476);
        assert_eq!(obs.longitude(), -105.1786);
        assert_eq!(obs.elevation(), 1830.14);
        assert_eq!(obs.pressure(), 820.0);
        assert_eq!(obs.temperature(), 11.0);
    }

    #[test]
    fn observer_accepts_inclusive_bounds() {
        for &lat in &[-90.0_f64, 90.0] {
            for &lon in &[-180.0_f64, 180.0] {
                assert!(Observer::try_new(lat, lon, 0.0, 1013.25, 15.0).is_ok());
            }
        }
    }

    #[test]
    fn observer_rejects_invalid_latitude() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            90.000_001,
            -90.000_001,
            180.0,
        ] {
            assert!(matches!(
                Observer::try_new(bad, 0.0, 0.0, 1013.25, 15.0),
                Err(ObserverError::InvalidLatitude(_)),
            ));
        }
    }

    #[test]
    fn observer_rejects_invalid_longitude() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            180.000_001,
            -180.000_001,
            360.0,
        ] {
            assert!(matches!(
                Observer::try_new(0.0, bad, 0.0, 1013.25, 15.0),
                Err(ObserverError::InvalidLongitude(_)),
            ));
        }
    }

    #[test]
    fn observer_rejects_non_finite_elevation() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(matches!(
                Observer::try_new(0.0, 0.0, bad, 1013.25, 15.0),
                Err(ObserverError::InvalidElevation(_)),
            ));
        }
    }

    #[test]
    fn observer_rejects_invalid_pressure() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -1.0,
            -1013.25,
        ] {
            assert!(matches!(
                Observer::try_new(0.0, 0.0, 0.0, bad, 15.0),
                Err(ObserverError::InvalidPressure(_)),
            ));
        }
    }

    #[test]
    fn observer_rejects_invalid_temperature() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -273.0,
            -273.000_001,
            -1e6,
        ] {
            assert!(matches!(
                Observer::try_new(0.0, 0.0, 0.0, 1013.25, bad),
                Err(ObserverError::InvalidTemperature(_)),
            ));
        }
    }

    #[test]
    fn observer_try_with_reference_atmosphere_rejects_invalid_geographics() {
        assert!(matches!(
            Observer::try_with_reference_atmosphere(f64::NAN, 0.0, 0.0),
            Err(ObserverError::InvalidLatitude(_)),
        ));
        assert!(matches!(
            Observer::try_with_reference_atmosphere(0.0, f64::NAN, 0.0),
            Err(ObserverError::InvalidLongitude(_)),
        ));
        assert!(matches!(
            Observer::try_with_reference_atmosphere(0.0, 0.0, f64::NAN),
            Err(ObserverError::InvalidElevation(_)),
        ));
    }

    #[test]
    fn observer_try_at_sea_level_isa_rejects_invalid_geographics() {
        assert!(matches!(
            Observer::try_at_sea_level_isa(f64::NAN, 0.0),
            Err(ObserverError::InvalidLatitude(_)),
        ));
        assert!(matches!(
            Observer::try_at_sea_level_isa(0.0, f64::NAN),
            Err(ObserverError::InvalidLongitude(_)),
        ));
    }

    #[test]
    fn observer_error_display_mentions_field_and_value() {
        for (err, field) in [
            (ObserverError::InvalidLatitude(91.0), "latitude"),
            (ObserverError::InvalidLongitude(181.0), "longitude"),
            (ObserverError::InvalidElevation(f64::NAN), "elevation"),
            (ObserverError::InvalidPressure(-1.0), "pressure"),
            (ObserverError::InvalidTemperature(-300.0), "temperature"),
        ] {
            assert!(format!("{err}").contains(field));
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn surface_try_new_round_trips_through_accessors() {
        let surface = Surface::try_new(30.0, -10.0).unwrap();
        assert_eq!(surface.slope(), 30.0);
        assert_eq!(surface.azimuth_rotation(), -10.0);
    }

    #[test]
    fn surface_accepts_inclusive_bounds() {
        for &slope in &[0.0_f64, 180.0] {
            for &azimuth in &[-180.0_f64, 180.0] {
                assert!(Surface::try_new(slope, azimuth).is_ok());
            }
        }
    }

    #[test]
    fn surface_rejects_invalid_slope() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.000_001,
            180.000_001,
            360.0,
        ] {
            assert!(matches!(
                Surface::try_new(bad, 0.0),
                Err(SurfaceError::InvalidSlope(_)),
            ));
        }
    }

    #[test]
    fn surface_rejects_invalid_azimuth_rotation() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            180.000_001,
            -180.000_001,
            360.0,
        ] {
            assert!(matches!(
                Surface::try_new(0.0, bad),
                Err(SurfaceError::InvalidAzimuthRotation(_)),
            ));
        }
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn surface_horizontal_and_default_are_equivalent() {
        let h = Surface::horizontal();
        assert_eq!(h.slope(), 0.0);
        assert_eq!(h.azimuth_rotation(), 0.0);
        assert_eq!(Surface::default(), h);
    }

    #[test]
    fn surface_error_display_mentions_field_and_value() {
        for (err, field) in [
            (SurfaceError::InvalidSlope(-1.0), "slope"),
            (SurfaceError::InvalidAzimuthRotation(200.0), "azimuth"),
        ] {
            assert!(format!("{err}").contains(field));
        }
    }
}
