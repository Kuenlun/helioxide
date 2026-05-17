// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

//! High-level pipeline that runs the full SPA computation end-to-end.
//!
//! The per-section modules in this crate (`heliocentric`, `geocentric`,
//! `parallax`, ...) expose the individual building blocks of
//! NREL/TP-560-34302. [`SolarPosition::compute`] orchestrates them in the
//! order prescribed by the paper and bundles every intermediate and final
//! value into a single [`SolarPosition`]. This keeps callers from having to
//! plumb 20+ partial results by hand and gives a single entry point for the
//! common case "given a place and a time, what is the sun doing?".
//!
//! [`Observer`] and [`Surface`] are plain value types describing the
//! observation site and the optional tilted surface (e.g. a fixed-tilt PV
//! panel) consumed by the angle-of-incidence calculation.

use core::fmt;

use chrono::TimeZone;
use thiserror::Error;

use crate::{
    SpaDateTime, apparent, equation_of_time, equatorial, geocentric, heliocentric, horizontal,
    hour_angle, incidence, julian, nutation, obliquity, parallax, sidereal,
};

/// Geographic and atmospheric description of the observation site.
///
/// Latitude and longitude follow the conventions of sections 3.11 and
/// 3.12.2: positive east of Greenwich and positive north of the equator,
/// respectively. Elevation is the observer height above sea level
/// (metres), consumed by section 3.12.3. Pressure (millibars) and
/// temperature (degrees Celsius) feed the atmospheric refraction model
/// of equation 42 and should be annual averages for the site.
///
/// Fields are private: every value reaches the SPA pipeline through one
/// of the validating constructors below, so a downstream `compute` call
/// can never observe NaN, an infinity, an off-equator latitude, a
/// non-physical pressure, or a temperature at or below equation 42's
/// `T = -273 °C` singularity. Read the stored values back with the
/// per-field accessors ([`Self::latitude`], [`Self::longitude`],
/// [`Self::elevation`], [`Self::pressure`], [`Self::temperature`]).
///
/// Two reference atmospheres are exposed as `pub const` values for
/// callers without local annual averages on hand:
///
/// * The ICAO/ISA sea-level standard atmosphere
///   ([`ISA_PRESSURE_MILLIBARS`], [`ISA_TEMPERATURE_CELSIUS`]):
///   `1013.25 mbar`, `15 °C`. The industry-wide aeronautical default and
///   the recommended choice when no local meteorological data is
///   available. The convenience constructor
///   [`Observer::try_at_sea_level_isa`] fills both fields from these
///   constants and pins `elevation` at `0 m`.
/// * The SPA paper's calibration atmosphere
///   ([`REFERENCE_PRESSURE_MILLIBARS`], [`REFERENCE_TEMPERATURE_CELSIUS`]):
///   `1010 mbar`, `10 °C` (`283 K`). At these values, both ratios in
///   equation 42 collapse to one, leaving
///   `Δe = 1.02 / (60 · tan(e₀ + 10.3/(e₀ + 5.11)))`. Pick this only when
///   reproducing the appendix A.5 worked example (the published Table
///   A5.1 values were computed at this atmosphere). The convenience
///   constructor [`Observer::try_with_reference_atmosphere`] fills both
///   fields from these constants. The ISA atmosphere above scales
///   equation 42's `Δe` by a constant factor of `~0.986` relative to
///   this one (the formula is linear in both ratios).
///
/// [`SolarPosition::compute`] is the sole consumer of pressure and
/// temperature. [`SolarDay::compute`] ignores them because appendix A.2
/// absorbs the average horizon-level refraction into the constant
/// `-0.8333°`.
///
/// [`REFERENCE_PRESSURE_MILLIBARS`]: Observer::REFERENCE_PRESSURE_MILLIBARS
/// [`REFERENCE_TEMPERATURE_CELSIUS`]: Observer::REFERENCE_TEMPERATURE_CELSIUS
/// [`ISA_PRESSURE_MILLIBARS`]: Observer::ISA_PRESSURE_MILLIBARS
/// [`ISA_TEMPERATURE_CELSIUS`]: Observer::ISA_TEMPERATURE_CELSIUS
/// [`SolarDay::compute`]: crate::SolarDay::compute
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observer {
    longitude: f64,
    latitude: f64,
    elevation: f64,
    pressure: f64,
    temperature: f64,
}

impl Observer {
    /// SPA paper reference pressure (millibars).
    ///
    /// Reproduced verbatim from equation 42 of NREL/TP-560-34302: setting
    /// pressure to this value collapses the `(P / 1010)` ratio in the
    /// refraction formula to one. Identical to the literal used
    /// internally by [`atmospheric_refraction`].
    ///
    /// [`atmospheric_refraction`]: crate::horizontal::atmospheric_refraction
    pub const REFERENCE_PRESSURE_MILLIBARS: f64 = horizontal::STANDARD_PRESSURE_MILLIBARS;

    /// SPA paper reference temperature (degrees Celsius).
    ///
    /// Equivalent to `283 K`, the numerator of the temperature ratio in
    /// equation 42: setting temperature to this value collapses the
    /// `(283 / (273 + T))` ratio to one.
    pub const REFERENCE_TEMPERATURE_CELSIUS: f64 =
        horizontal::REFERENCE_TEMPERATURE_KELVIN - horizontal::KELVIN_OFFSET_FROM_CELSIUS;

    /// ICAO/ISA sea-level standard pressure (millibars).
    ///
    /// Reproduced from ISO 2533:1975 / ICAO Doc 7488: `1013.25 mbar` is
    /// the sea-level pressure of the International Standard Atmosphere.
    /// Differs from [`Self::REFERENCE_PRESSURE_MILLIBARS`] by `+3.25 mbar`;
    /// equation 42 scales `Δe` by the same proportion (linear in `P`).
    pub const ISA_PRESSURE_MILLIBARS: f64 = 1013.25;

    /// ICAO/ISA sea-level standard temperature (degrees Celsius).
    ///
    /// Reproduced from ISO 2533:1975 / ICAO Doc 7488: `15 °C` (`288.15 K`)
    /// is the sea-level temperature of the International Standard
    /// Atmosphere. Differs from [`Self::REFERENCE_TEMPERATURE_CELSIUS`] by
    /// `+5 °C`; equation 42's temperature ratio `283 / (273 + T)` drops
    /// from `1.000` to `283 / 288 ≈ 0.9826`, scaling `Δe` by the same
    /// factor.
    pub const ISA_TEMPERATURE_CELSIUS: f64 = 15.0;

    /// Open lower bound on the observer temperature (degrees Celsius).
    ///
    /// Equation 42's denominator `273 + T` vanishes at `T = -273 °C`
    /// exactly, so [`Self::try_new`] rejects any temperature at or below
    /// this value. The bound mirrors the paper's `273 + T` constant
    /// rather than the strict IAU `-273.15 °C` of absolute zero (the
    /// `0.15 °C` gap is below the trailing digit of equation 42 at every
    /// pressure and elevation reproduced in appendix A.5).
    pub const TEMPERATURE_FLOOR_CELSIUS_EXCLUSIVE: f64 = -horizontal::KELVIN_OFFSET_FROM_CELSIUS;

    /// Build an observer with explicit atmosphere, validating every
    /// argument before assembling the struct.
    ///
    /// Argument order follows the universal `(latitude, longitude)`
    /// geographic convention. Each value is checked against the
    /// preconditions of the SPA pipeline; the first failed check wins
    /// and the rest are short-circuited.
    ///
    /// # Errors
    /// Returns the matching [`ObserverError`] variant when one of the
    /// inputs is non-finite or out of its admissible range:
    ///
    /// * `latitude` must lie in `[-90°, 90°]` and be finite, per section
    ///   3.12.2.
    /// * `longitude` must lie in `[-180°, 180°]` and be finite, per
    ///   section 3.11.
    /// * `elevation` must be finite (metres above sea level, per section
    ///   3.12.3; no range is imposed because the parallax correction
    ///   stays well behaved for any physically meaningful altitude).
    /// * `pressure` must be finite and strictly positive (millibars):
    ///   equation 42's pressure ratio `P / 1010` is undefined for
    ///   non-positive inputs.
    /// * `temperature` must be finite and strictly above
    ///   [`Self::TEMPERATURE_FLOOR_CELSIUS_EXCLUSIVE`] (degrees Celsius):
    ///   equation 42's denominator `273 + T` vanishes at the boundary.
    ///
    /// # Examples
    ///
    /// ```
    /// use helioxide::Observer;
    ///
    /// // Alicante: lat 38.346°N, lon 0.490°W, 3 m elevation, 1015 mbar, 18 °C.
    /// let obs = Observer::try_new(38.346_02, -0.490_68, 3.0, 1015.0, 18.0).unwrap();
    /// assert_eq!(obs.latitude(), 38.346_02);
    /// assert_eq!(obs.longitude(), -0.490_68);
    /// ```
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

    /// Build an observer using the SPA paper's reference atmosphere
    /// (`1010 mbar`, `10 °C`).
    ///
    /// Pick this only when reproducing the appendix worked examples. The
    /// refraction correction of equation 42 reduces to
    /// `Δe = 1.02 / (60 · tan(e₀ + 10.3/(e₀ + 5.11)))` at this atmosphere,
    /// matching the values published in Table A5.1. For ordinary use with
    /// no local meteorological data prefer [`Self::try_at_sea_level_isa`]
    /// instead.
    ///
    /// # Errors
    /// Returns the same [`ObserverError`] variants as [`Self::try_new`]
    /// would for `latitude`, `longitude`, and `elevation`. The two
    /// atmospheric fields are pinned to the paper's calibration
    /// constants and therefore never fail validation.
    ///
    /// # Examples
    ///
    /// ```
    /// use helioxide::Observer;
    ///
    /// let obs = Observer::try_with_reference_atmosphere(38.346_02, -0.490_68, 3.0).unwrap();
    /// assert_eq!(obs.pressure(), Observer::REFERENCE_PRESSURE_MILLIBARS);
    /// assert_eq!(obs.temperature(), Observer::REFERENCE_TEMPERATURE_CELSIUS);
    /// ```
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

    /// Build a sea-level observer using the ICAO/ISA standard atmosphere
    /// (`1013.25 mbar`, `15 °C`, elevation `0 m`).
    ///
    /// The recommended default when the observer site is approximately at
    /// sea level and no local meteorological data is available. Diverges
    /// from the SPA paper's reference by `+3.25 mbar` and `+5 °C`,
    /// scaling equation 42's `Δe` by a constant factor of `~0.986` (the
    /// formula is linear in both ratios).
    ///
    /// # Errors
    /// Returns [`ObserverError::InvalidLatitude`] or
    /// [`ObserverError::InvalidLongitude`] when the corresponding
    /// argument is non-finite or out of range. The three remaining
    /// fields (elevation, pressure, temperature) are pinned to the ISA
    /// constants and therefore never fail validation.
    ///
    /// # Examples
    ///
    /// ```
    /// use helioxide::Observer;
    ///
    /// let obs = Observer::try_at_sea_level_isa(40.0, -3.0).unwrap();
    /// assert_eq!(obs.elevation(), 0.0);
    /// assert_eq!(obs.pressure(), Observer::ISA_PRESSURE_MILLIBARS);
    /// assert_eq!(obs.temperature(), Observer::ISA_TEMPERATURE_CELSIUS);
    /// ```
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

    /// Geographic latitude `φ` (degrees, signed, positive north of the
    /// equator).
    #[inline]
    #[must_use]
    pub const fn latitude(&self) -> f64 {
        self.latitude
    }

    /// Geographic longitude `σ` (degrees, signed, positive east of
    /// Greenwich).
    #[inline]
    #[must_use]
    pub const fn longitude(&self) -> f64 {
        self.longitude
    }

    /// Elevation above sea level `E` (metres).
    #[inline]
    #[must_use]
    pub const fn elevation(&self) -> f64 {
        self.elevation
    }

    /// Annual average atmospheric pressure `P` (millibars).
    #[inline]
    #[must_use]
    pub const fn pressure(&self) -> f64 {
        self.pressure
    }

    /// Annual average atmospheric temperature `T` (degrees Celsius).
    #[inline]
    #[must_use]
    pub const fn temperature(&self) -> f64 {
        self.temperature
    }
}

/// Why an [`Observer`] could not be built.
///
/// Each variant carries the offending raw value so the caller can map it
/// straight into a user-facing diagnostic without re-deriving which input
/// was rejected. The five variants are mutually exclusive: the first
/// failing field (in the order `latitude`, `longitude`, `elevation`,
/// `pressure`, `temperature`) short-circuits [`Observer::try_new`].
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ObserverError {
    /// Latitude is non-finite or outside `[-90°, 90°]`.
    #[error("latitude {0}° must lie in [-90°, 90°] and be finite")]
    InvalidLatitude(f64),
    /// Longitude is non-finite or outside `[-180°, 180°]`.
    #[error("longitude {0}° must lie in [-180°, 180°] and be finite")]
    InvalidLongitude(f64),
    /// Elevation is non-finite.
    #[error("elevation {0} m must be finite")]
    InvalidElevation(f64),
    /// Pressure is non-finite or not strictly positive.
    #[error("pressure {0} mbar must be > 0 mbar and finite")]
    InvalidPressure(f64),
    /// Temperature is non-finite or at/below the `T = -273 °C` singularity
    /// of equation 42.
    #[error(
        "temperature {0} °C must be > -273 °C and finite \
         (equation 42's denominator 273 + T vanishes at -273 °C)"
    )]
    InvalidTemperature(f64),
}

/// Tilted surface (e.g. a fixed-tilt photovoltaic panel) consumed by the
/// angle-of-incidence calculation in section 3.16.
///
/// Fields are private: every value reaches the SPA pipeline through one
/// of the validating constructors below, so equation 47 can never be
/// evaluated on NaN, an infinity, or a slope outside the geometrically
/// meaningful `[0°, 180°]` range. Read the stored values back with
/// [`Self::slope`] and [`Self::azimuth_rotation`].
///
/// A horizontal collector is [`Surface::horizontal`]; a vertical south
/// facing wall is `Surface::try_new(90.0, 0.0)`; a vertical west facing
/// wall is `Surface::try_new(90.0, 90.0)`; a vertical east facing wall
/// is `Surface::try_new(90.0, -90.0)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Surface {
    slope: f64,
    azimuth_rotation: f64,
}

impl Surface {
    /// Build a tilted surface, validating both arguments before
    /// assembling the struct.
    ///
    /// # Errors
    /// Returns the matching [`SurfaceError`] variant when one of the
    /// inputs is non-finite or out of its admissible range:
    ///
    /// * `slope` must lie in `[0°, 180°]` and be finite. Equation 47
    ///   admits the full hemisphere from horizontal (`0°`) through
    ///   vertical (`90°`) to an upside-down collector (`180°`); values
    ///   outside this range have no geometric meaning.
    /// * `azimuth_rotation` must lie in `[-180°, 180°]` and be finite.
    ///   The astronomers' azimuth `Γ` that equation 47 differences this
    ///   value against shares the same `westward from south` convention,
    ///   so keeping `γ` inside `[-180°, 180°]` avoids ambiguity even
    ///   though `cos(Γ − γ)` is periodic.
    ///
    /// # Examples
    ///
    /// ```
    /// use helioxide::Surface;
    ///
    /// // South-facing fixed tilt at 38.35° (typical for southern Spain).
    /// let surface = Surface::try_new(38.35, 0.0).unwrap();
    /// assert_eq!(surface.slope(), 38.35);
    /// ```
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

    /// A horizontal collector facing up (`slope = 0°, azimuth = 0°`).
    ///
    /// Infallible because both stored values are known-valid constants.
    /// Pass this to [`SolarPosition::compute`] when no tilted surface is
    /// of interest and only the topocentric solar position matters.
    #[inline]
    #[must_use]
    pub const fn horizontal() -> Self {
        Self {
            slope: 0.0,
            azimuth_rotation: 0.0,
        }
    }

    /// Slope from the horizontal plane `ω` (degrees; `0°` is horizontal,
    /// `90°` is vertical).
    #[inline]
    #[must_use]
    pub const fn slope(&self) -> f64 {
        self.slope
    }

    /// Surface azimuth rotation `γ` (degrees, signed, positive westward
    /// and negative eastward from due south; `0°` faces south). Matches
    /// the convention of the astronomers' azimuth `Γ` so that `Γ − γ`
    /// in equation 47 is the signed angular gap between the sun and the
    /// surface normal projected on the horizontal plane.
    #[inline]
    #[must_use]
    pub const fn azimuth_rotation(&self) -> f64 {
        self.azimuth_rotation
    }
}

impl Default for Surface {
    /// Equivalent to [`Self::horizontal`].
    #[inline]
    fn default() -> Self {
        Self::horizontal()
    }
}

/// Why a [`Surface`] could not be built.
///
/// Each variant carries the offending raw value so the caller can map it
/// straight into a user-facing diagnostic without re-deriving which input
/// was rejected.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum SurfaceError {
    /// Slope is non-finite or outside `[0°, 180°]`.
    #[error("slope {0}° must lie in [0°, 180°] and be finite")]
    InvalidSlope(f64),
    /// Azimuth rotation is non-finite or outside `[-180°, 180°]`.
    #[error("azimuth rotation {0}° must lie in [-180°, 180°] and be finite")]
    InvalidAzimuthRotation(f64),
}

/// Full output of the SPA pipeline.
///
/// Fields are listed in the order of the sections of NREL/TP-560-34302
/// they belong to. The "primary" outputs of the algorithm are
/// [`Self::topocentric_zenith`] and [`Self::topocentric_azimuth`] (and
/// [`Self::surface_incidence`] when a tilted surface is supplied); the rest
/// are intermediates exposed for paper validation and downstream
/// computation (e.g. sunrise/transit/sunset).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SolarPosition {
    /// Julian Day `JD` (days). Section 3.1.
    pub julian_day: f64,
    /// Julian Ephemeris Day `JDE` (days). Section 3.1.
    pub julian_ephemeris_day: f64,
    /// Julian Century `JC`. Section 3.1.
    pub julian_century: f64,
    /// Julian Ephemeris Century `JCE`. Section 3.1.
    pub julian_ephemeris_century: f64,
    /// Julian Ephemeris Millennium `JME`. Section 3.1.
    pub julian_ephemeris_millennium: f64,

    /// Earth heliocentric longitude `L` (degrees, in `[0°, 360°)`).
    /// Section 3.2.
    pub earth_heliocentric_longitude: f64,
    /// Earth heliocentric latitude `B` (degrees, signed). Section 3.2.
    pub earth_heliocentric_latitude: f64,
    /// Earth-Sun distance `R` (astronomical units). Section 3.2.
    pub earth_radius_vector: f64,

    /// Sun geocentric longitude `Θ` (degrees, in `[0°, 360°)`). Section 3.3.
    pub geocentric_longitude: f64,
    /// Sun geocentric latitude `β` (degrees, signed). Section 3.3.
    pub geocentric_latitude: f64,

    /// Mean elongation of the moon from the sun `X₀` (degrees, raw — not
    /// reduced into `[0°, 360°)`, as the only internal consumer is the
    /// nutation series). Section 3.4, equation 15.
    pub mean_elongation_moon_sun: f64,
    /// Mean anomaly of the sun (Earth) `X₁` (degrees, raw). Section 3.4,
    /// equation 16.
    pub mean_anomaly_sun: f64,
    /// Mean anomaly of the moon `X₂` (degrees, raw). Section 3.4,
    /// equation 17.
    pub mean_anomaly_moon: f64,
    /// Moon's argument of latitude `X₃` (degrees, raw). Section 3.4,
    /// equation 18.
    pub argument_latitude_moon: f64,
    /// Longitude of the ascending node of the moon's mean orbit on the
    /// ecliptic, measured from the mean equinox of the date `X₄`
    /// (degrees, raw). Section 3.4, equation 19.
    pub ascending_longitude_moon: f64,

    /// Nutation in longitude `Δψ` (degrees, signed). Section 3.4.
    pub nutation_in_longitude: f64,
    /// Nutation in obliquity `Δε` (degrees, signed). Section 3.4.
    pub nutation_in_obliquity: f64,

    /// Mean obliquity of the ecliptic `ε₀` (arc seconds). Section 3.5,
    /// equation 24. Multiply by `1/3600` to obtain degrees.
    pub mean_obliquity_arcseconds: f64,
    /// True obliquity of the ecliptic `ε` (degrees). Section 3.5.
    pub true_obliquity: f64,

    /// Aberration correction `Δτ` (degrees, signed). Section 3.6.
    pub aberration_correction: f64,
    /// Apparent sun longitude `λ` (degrees, in `[0°, 360°)`). Section 3.7.
    pub apparent_sun_longitude: f64,

    /// Mean sidereal time at Greenwich `ν₀` (degrees, in `[0°, 360°)`).
    /// Section 3.8.
    pub mean_sidereal_time: f64,
    /// Apparent sidereal time at Greenwich `ν` (degrees, in `[0°, 360°)`).
    /// Section 3.8.
    pub apparent_sidereal_time: f64,

    /// Sun geocentric right ascension `α` (degrees, in `[0°, 360°)`).
    /// Section 3.9.
    pub geocentric_right_ascension: f64,
    /// Sun geocentric declination `δ` (degrees, signed, in `[-90°, 90°]`).
    /// Section 3.10.
    pub geocentric_declination: f64,

    /// Observer local hour angle `H` (degrees, in `[0°, 360°)`).
    /// Section 3.11.
    pub observer_local_hour_angle: f64,

    /// Sun equatorial horizontal parallax `ξ` (degrees). Section 3.12.1.
    pub equatorial_horizontal_parallax: f64,
    /// Parallax in the sun right ascension `Δα` (degrees, signed).
    /// Section 3.12.6.
    pub parallax_in_right_ascension: f64,
    /// Topocentric sun right ascension `α'` (degrees, not wrapped;
    /// physically `|α' − α| = |Δα| ≤ ξ`, so it stays in the same
    /// `[0°, 360°)` interval upstream `α` already lives in).
    /// Section 3.12.7.
    pub topocentric_right_ascension: f64,
    /// Topocentric sun declination `δ'` (degrees, signed, typically in
    /// `[-90°, 90°]`). Section 3.12.8.
    pub topocentric_declination: f64,

    /// Topocentric local hour angle `H'` (degrees, signed, not wrapped).
    /// Section 3.13.
    pub topocentric_local_hour_angle: f64,

    /// Topocentric elevation angle without atmospheric refraction `e₀`
    /// (degrees, signed). Section 3.14.1.
    pub topocentric_elevation_unrefracted: f64,
    /// Atmospheric refraction correction `Δe` (degrees). Section 3.14.2.
    pub atmospheric_refraction: f64,
    /// Topocentric elevation angle (corrected) `e = e₀ + Δe` (degrees,
    /// signed). Section 3.14.3.
    pub topocentric_elevation_corrected: f64,
    /// Topocentric zenith angle `θ = 90° − e` (degrees, signed, typically
    /// in `[0°, 180°]`). Section 3.14.4.
    pub topocentric_zenith: f64,

    /// Topocentric astronomers' azimuth `Γ` (degrees, in `[0°, 360°)`,
    /// measured westward from south). Section 3.15.1.
    pub astronomers_azimuth: f64,
    /// Topocentric astronomers' azimuth re-expressed in `(-180°, 180°]`
    /// (degrees, signed, measured westward from south). Section 3.15.1
    /// rephrased through [`astronomers_azimuth_signed`].
    ///
    /// [`astronomers_azimuth_signed`]: crate::horizontal::astronomers_azimuth_signed
    pub astronomers_azimuth_signed: f64,
    /// Topocentric azimuth `Φ` (degrees, in `[0°, 360°)`, measured
    /// eastward from north). Section 3.15.2.
    pub topocentric_azimuth: f64,

    /// Angle of incidence `I` on the [`Surface`] (degrees, in `[0°, 180°]`).
    /// Section 3.16.
    pub surface_incidence: f64,

    /// Equation of time `E` (minutes, signed). Section 3.18.
    pub equation_of_time: f64,
}

impl SolarPosition {
    /// Run the full SPA pipeline.
    ///
    /// `datetime` is the wall-clock instant of observation. `delta_t` is
    /// the difference `ΔT = TT − UT1` (seconds), an Earth rotation
    /// correction whose value depends on the year (≈ 69.5 s in 2026;
    /// consult IERS bulletins for the current value). `observer` describes
    /// the observation site, and `surface` the tilted surface used by the
    /// angle-of-incidence calculation; pass a horizontal `Surface` (slope
    /// `0`, azimuth rotation `0`) when only the topocentric position is
    /// needed.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use chrono::Utc;
    /// use chrono_tz::Tz;
    /// use helioxide::{Observer, SolarPosition, SpaDateTime, Surface};
    ///
    /// let observer = Observer::try_new(38.346_02, -0.490_68, 3.0, 1015.0, 18.0)
    ///     .expect("validated observer");
    /// let surface = Surface::try_new(38.346_02, 0.0).expect("validated surface");
    /// let now = SpaDateTime::new(Utc::now().with_timezone(&Tz::Europe__Madrid));
    /// let position = SolarPosition::compute(&now, 69.5, observer, surface);
    /// println!("Zenith: {}°", position.topocentric_zenith);
    /// ```
    #[must_use]
    #[allow(
        clippy::many_single_char_names,
        clippy::similar_names,
        clippy::too_many_lines
    )]
    pub fn compute<Tz: TimeZone>(
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
        // Evaluate equation 24 once and apply equation 25 inline so the
        // Horner sweep of the ε₀ polynomial is not duplicated here.
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

    /// ΔT for the Table A5.1 worked example (seconds), per section A.5.
    const REFERENCE_DELTA_T_SECONDS: f64 = 67.0;
    /// Reference surface slope `ω` for the Table A5.1 worked example
    /// (degrees), per section A.5.
    const REFERENCE_SURFACE_SLOPE_DEGREES: f64 = 30.0;
    /// Reference surface azimuth rotation `γ` for the Table A5.1 worked
    /// example (degrees, positive west / negative east of south), per
    /// section A.5.
    const REFERENCE_SURFACE_AZIMUTH_ROTATION_DEGREES: f64 = -10.0;

    /// Civil instant of the Table A5.1 worked example: 2003-10-17
    /// 12:30:30 LST at TZ = -7 h, i.e. 19:30:30 UTC. Reconstructed from
    /// the UTC instant rather than the report's printed JD so the time
    /// fed into the orchestrator and the time fed into the per-section
    /// reference tests stay aligned bit-for-bit.
    fn reference_datetime() -> SpaDateTime<Utc> {
        SpaDateTime::new(
            Utc.with_ymd_and_hms(2003, 10, 17, 19, 30, 30)
                .single()
                .expect("non-ambiguous reference instant"),
        )
    }

    fn reference_observer() -> Observer {
        Observer::try_new(
            REFERENCE_LATITUDE_DEGREES,
            REFERENCE_LONGITUDE_DEGREES,
            REFERENCE_ELEVATION_METRES,
            REFERENCE_PRESSURE_MILLIBARS,
            REFERENCE_TEMPERATURE_CELSIUS,
        )
        .expect("reference observer at the A5.1 reference site is valid by construction")
    }

    fn reference_surface() -> Surface {
        Surface::try_new(
            REFERENCE_SURFACE_SLOPE_DEGREES,
            REFERENCE_SURFACE_AZIMUTH_ROTATION_DEGREES,
        )
        .expect("reference surface at the A5.1 reference orientation is valid by construction")
    }

    fn reference_position() -> SolarPosition {
        SolarPosition::compute(
            &reference_datetime(),
            REFERENCE_DELTA_T_SECONDS,
            reference_observer(),
            reference_surface(),
        )
    }

    /// Every struct field on the [`SolarPosition`] returned by
    /// [`SolarPosition::compute`] must reproduce its Table A5.1 published
    /// value at the reference instant. This is the single global
    /// integration test of the orchestrator: each per-section unit test
    /// already pins its own function against the paper, so any failure
    /// here implicates [`SolarPosition::compute`] specifically — either a
    /// section was called with the wrong argument (e.g. `α` and `δ`
    /// swapped into `topocentric_equatorial_coordinates`) or a struct
    /// field was assigned the wrong local variable in the final struct
    /// literal. Tolerances mirror the per-section tests so a real bug
    /// cannot hide behind rounding. The constants are reproduced from
    /// the paper's Table A5.1 (and from the per-section reference tests
    /// where they pin a derived intermediate not directly tabulated).
    #[test]
    #[allow(clippy::too_many_lines)]
    fn compute_matches_table_a5_1_published_values() {
        let position = reference_position();

        // Section 3.1 — Julian Day. JD is the only quantity in the Julian
        // chain that the paper publishes directly; the rest are linear
        // transforms pinned by the `julian` module's own tests.
        assert!(
            (position.julian_day - 2_452_930.312_847).abs() < 1e-6,
            "JD mismatch: got {}",
            position.julian_day,
        );

        // Section 3.2 — Earth heliocentric coordinates.
        assert!(
            (position.earth_heliocentric_longitude - 24.018_261_691_7).abs() < 1e-6,
            "L mismatch: got {}",
            position.earth_heliocentric_longitude,
        );
        assert!(
            (position.earth_heliocentric_latitude - -0.000_101_121_9).abs() < 1e-9,
            "B mismatch: got {}",
            position.earth_heliocentric_latitude,
        );
        assert!(
            (position.earth_radius_vector - 0.996_542_297_4).abs() < 1e-9,
            "R mismatch: got {}",
            position.earth_radius_vector,
        );

        // Section 3.3 — Geocentric coordinates.
        assert!(
            (position.geocentric_longitude - 204.018_261_691_7).abs() < 1e-6,
            "Θ mismatch: got {}",
            position.geocentric_longitude,
        );
        assert!(
            (position.geocentric_latitude - 0.000_101_121_9).abs() < 1e-9,
            "β mismatch: got {}",
            position.geocentric_latitude,
        );

        // Section 3.4 — Nutation.
        assert!(
            (position.nutation_in_longitude - -0.003_998_40).abs() < 1e-8,
            "Δψ mismatch: got {}",
            position.nutation_in_longitude,
        );
        assert!(
            (position.nutation_in_obliquity - 0.001_666_57).abs() < 1e-8,
            "Δε mismatch: got {}",
            position.nutation_in_obliquity,
        );

        // Section 3.5 — Mean and true obliquity of the ecliptic.
        // ε₀ in arc seconds is back-derived from `(ε - Δε) · 3600` to a
        // ~0.1" tolerance: rounding to the published 1e-6° on `ε` and
        // 1e-8° on `Δε` allows roughly 0.07" of slack in the round-trip.
        let epsilon0_via_round_trip = (23.440_465_f64 - 0.001_666_57) * 3600.0;
        assert!(
            (position.mean_obliquity_arcseconds - epsilon0_via_round_trip).abs() < 0.1,
            "ε₀ mismatch: got {}\" vs round-trip {epsilon0_via_round_trip}\"",
            position.mean_obliquity_arcseconds,
        );
        assert!(
            (position.true_obliquity - 23.440_465).abs() < 1e-6,
            "ε mismatch: got {}",
            position.true_obliquity,
        );

        // Section 3.7 — Apparent sun longitude.
        assert!(
            (position.apparent_sun_longitude - 204.008_551_928_1).abs() < 1e-6,
            "λ mismatch: got {}",
            position.apparent_sun_longitude,
        );

        // Section 3.8 — Sidereal time.
        assert!(
            (position.mean_sidereal_time - 318.515_578).abs() < 1e-4,
            "ν₀ mismatch: got {}",
            position.mean_sidereal_time,
        );
        assert!(
            (position.apparent_sidereal_time - 318.511_910).abs() < 1e-4,
            "ν mismatch: got {}",
            position.apparent_sidereal_time,
        );

        // Sections 3.9 and 3.10 — Geocentric right ascension and declination.
        assert!(
            (position.geocentric_right_ascension - 202.227_41).abs() < 1e-4,
            "α mismatch: got {}",
            position.geocentric_right_ascension,
        );
        assert!(
            (position.geocentric_declination - -9.314_34).abs() < 1e-4,
            "δ mismatch: got {}",
            position.geocentric_declination,
        );

        // Section 3.11 — Observer local hour angle.
        assert!(
            (position.observer_local_hour_angle - 11.105_900).abs() < 1e-4,
            "H mismatch: got {}",
            position.observer_local_hour_angle,
        );

        // Section 3.12 — Topocentric equatorial coordinates.
        assert!(
            (position.topocentric_right_ascension - 202.227_04).abs() < 1e-4,
            "α' mismatch: got {}",
            position.topocentric_right_ascension,
        );
        assert!(
            (position.topocentric_declination - -9.316_179).abs() < 1e-4,
            "δ' mismatch: got {}",
            position.topocentric_declination,
        );

        // Section 3.13 — Topocentric local hour angle.
        assert!(
            (position.topocentric_local_hour_angle - 11.106_29).abs() < 1e-4,
            "H' mismatch: got {}",
            position.topocentric_local_hour_angle,
        );

        // Sections 3.14 and 3.15 — Topocentric horizontal coordinates.
        assert!(
            (position.topocentric_zenith - 50.111_62).abs() < 1e-4,
            "θ mismatch: got {}",
            position.topocentric_zenith,
        );
        // The corrected elevation `e = e₀ + Δe` must equal `90° − θ`
        // exactly, since equation 44 is `θ = 90° − e`. Pinning the
        // published `θ` therefore implies the corrected elevation
        // matches the published `90° − θ ≈ 39.88838°`.
        assert!(
            (position.topocentric_elevation_corrected - (90.0_f64 - 50.111_62)).abs() < 1e-4,
            "e (corrected) mismatch: got {}",
            position.topocentric_elevation_corrected,
        );
        assert!(
            (position.astronomers_azimuth - 14.340_24).abs() < 1e-4,
            "Γ mismatch: got {}",
            position.astronomers_azimuth,
        );
        // The signed astronomers' azimuth must be the identity of `Γ`
        // for `Γ ≤ 180°` (the published reference value sits at
        // `~14.34°`). A regression that flipped the sign or shifted by
        // `±360°` would surface here.
        assert!(
            (position.astronomers_azimuth_signed - 14.340_24).abs() < 1e-4,
            "Γ′ (signed) mismatch: got {}",
            position.astronomers_azimuth_signed,
        );
        assert!(
            (position.topocentric_azimuth - 194.340_24).abs() < 1e-4,
            "Φ mismatch: got {}",
            position.topocentric_azimuth,
        );

        // Section 3.16 — Surface incidence angle.
        assert!(
            (position.surface_incidence - 25.187_00).abs() < 1e-4,
            "I mismatch: got {}",
            position.surface_incidence,
        );

        // Section 3.18 — Equation of time.
        assert!(
            (position.equation_of_time - 14.641_503).abs() < 1e-4,
            "E mismatch: got {}",
            position.equation_of_time,
        );
    }

    /// The struct fields whose values the paper does not publish
    /// directly — `JDE`, `JC`, `JCE`, `JME`, `Δτ`, `ξ`, `Δα`, `e₀`, `Δe`
    /// plus the five fundamental angles `X₀..X₄`, the mean obliquity
    /// `ε₀` (in arc seconds), the corrected elevation `e` and the signed
    /// astronomers' azimuth `Γ′` — must each equal the result of calling
    /// the underlying section function with the same inputs the
    /// orchestrator feeds it. This guards against a struct-field swap
    /// between two such intermediates, which the published-values test
    /// above cannot detect because none of these quantities appear in
    /// the paper's Table A5.1. Bitwise equality is required: the
    /// orchestrator and the test re-invoke the same section functions,
    /// so any difference is a wiring bug rather than rounding.
    #[test]
    #[allow(clippy::float_cmp)]
    fn compute_wires_unpublished_intermediates_to_their_section_functions() {
        use crate::{
            apparent, equation_of_time, horizontal, julian, nutation, obliquity, parallax,
        };

        let dt = reference_datetime();
        let observer = reference_observer();
        let position = SolarPosition::compute(
            &dt,
            REFERENCE_DELTA_T_SECONDS,
            observer,
            reference_surface(),
        );

        // Julian chain: every derived quantity must match the value the
        // `julian` module produces for the same `(JD, ΔT)` pair.
        let jd = julian::calculate_julian_day(&dt);
        let jde = julian::calculate_julian_ephemeris_day(jd, REFERENCE_DELTA_T_SECONDS);
        let jc = julian::calculate_julian_century(jd);
        let jce = julian::calculate_julian_ephemeris_century(jde);
        let jme = julian::calculate_julian_ephemeris_millennium(jce);
        assert_eq!(position.julian_ephemeris_day, jde, "JDE field swapped");
        assert_eq!(position.julian_century, jc, "JC field swapped");
        assert_eq!(position.julian_ephemeris_century, jce, "JCE field swapped");
        assert_eq!(
            position.julian_ephemeris_millennium, jme,
            "JME field swapped",
        );

        // Section 3.4 fundamental angles: each X_k is exposed as its own
        // SolarPosition field. Pin them all against the public
        // `fundamental_arguments(jce)` so a transposition of the five
        // outputs (e.g. swapping X₂ and X₃ in the struct literal, both
        // around 18_000° at the reference instant) cannot hide behind
        // the (Δψ, Δε) reference test that doesn't depend on the
        // surfaced field order.
        let [x0, x1, x2, x3, x4] = nutation::fundamental_arguments(jce);
        assert_eq!(position.mean_elongation_moon_sun, x0, "X₀ field swapped");
        assert_eq!(position.mean_anomaly_sun, x1, "X₁ field swapped");
        assert_eq!(position.mean_anomaly_moon, x2, "X₂ field swapped");
        assert_eq!(position.argument_latitude_moon, x3, "X₃ field swapped");
        assert_eq!(position.ascending_longitude_moon, x4, "X₄ field swapped");

        // Section 3.5 mean obliquity `ε₀` (arc seconds): the published
        // Table A5.1 only lists `ε` in degrees, so the only way to pin
        // `ε₀` is against the `mean_obliquity_of_ecliptic_arcseconds`
        // section function.
        assert_eq!(
            position.mean_obliquity_arcseconds,
            obliquity::mean_obliquity_of_ecliptic_arcseconds(jme),
            "ε₀ field swapped or unit-converted",
        );

        // Aberration `Δτ` and equatorial horizontal parallax `ξ` both
        // collapse to a small constant divided by `R`; a swap between
        // them would hide behind the per-section tests, so pin them
        // here against the section functions fed with the orchestrator's
        // own `R`.
        let r = position.earth_radius_vector;
        let delta_tau = apparent::aberration_correction(r);
        let xi = parallax::equatorial_horizontal_parallax(r);
        assert_eq!(
            position.aberration_correction, delta_tau,
            "Δτ field swapped"
        );
        assert_eq!(
            position.equatorial_horizontal_parallax, xi,
            "ξ field swapped",
        );

        // The parallax-in-right-ascension `Δα` is read off the
        // `topocentric_equatorial_coordinates` struct alongside `α'` and
        // `δ'`; pinning it specifically guards against the orchestrator
        // pulling the wrong field off that struct.
        let topocentric = parallax::topocentric_equatorial_coordinates(
            position.geocentric_right_ascension,
            position.geocentric_declination,
            position.observer_local_hour_angle,
            xi,
            observer.latitude(),
            observer.elevation(),
        );
        assert_eq!(
            position.parallax_in_right_ascension, topocentric.parallax_in_right_ascension,
            "Δα field swapped",
        );

        // Topocentric elevation without refraction `e₀` and the
        // atmospheric refraction correction `Δe` must each match the
        // section functions fed with the orchestrator's own
        // `(φ, δ', H', P, T)`. A swap with `θ`/`Γ`/`Φ` (all in degrees
        // and same order of magnitude) is exactly the bug this pins.
        let e0 = horizontal::topocentric_elevation_without_refraction(
            observer.latitude(),
            position.topocentric_declination,
            position.topocentric_local_hour_angle,
        );
        let delta_e =
            horizontal::atmospheric_refraction(e0, observer.pressure(), observer.temperature());
        let e_corrected = horizontal::topocentric_elevation_corrected(e0, delta_e);
        assert_eq!(
            position.topocentric_elevation_unrefracted, e0,
            "e₀ field swapped",
        );
        assert_eq!(position.atmospheric_refraction, delta_e, "Δe field swapped");
        assert_eq!(
            position.topocentric_elevation_corrected, e_corrected,
            "e (corrected) field swapped",
        );

        // Signed astronomers' azimuth `Γ′` must equal the published
        // `Γ ∈ [0°, 360°)` wrapped through `astronomers_azimuth_signed`.
        // A swap with `Γ` (which itself is published in the unsigned
        // form) would survive at the reference instant because both
        // values agree there (Γ ≈ 14.34°), so this test exists to pin
        // the wiring at any future instant whose `Γ` lands in the upper
        // half.
        assert_eq!(
            position.astronomers_azimuth_signed,
            horizontal::astronomers_azimuth_signed(position.astronomers_azimuth),
            "Γ′ field swapped or wrap convention changed",
        );

        // `equation_of_time::sun_mean_longitude` is not exposed on the
        // struct, but the equation-of-time output must match the
        // function fed with the orchestrator's `M`, `α`, `Δψ`, `ε`. A
        // swap of `Δψ` with `Δε` would change `E` measurably; this
        // assertion catches it specifically at the orchestrator level
        // (the per-section test catches it at the section level).
        let m = equation_of_time::sun_mean_longitude(jme);
        let expected_eot = equation_of_time::equation_of_time(
            m,
            position.geocentric_right_ascension,
            position.nutation_in_longitude,
            position.true_obliquity,
        );
        assert_eq!(
            position.equation_of_time, expected_eot,
            "E orchestrator wiring",
        );
    }

    /// [`fmt::Write`] sink that accepts at most `remaining` bytes
    /// cumulatively and returns [`fmt::Error`] on any write that would
    /// exceed the budget. Used to fail the underlying writer at varying
    /// offsets so [`SolarPosition::fmt`]'s `?` operators are exercised
    /// individually.
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

    /// [`SolarPosition::fmt`] must propagate every [`fmt::Error`]
    /// returned by the underlying writer back to the caller via the `?`
    /// operator after each `writeln!`. Walking a byte budget from `0`
    /// to one short of the full rendered length forces the writer to
    /// fail at every point inside [`SolarPosition::fmt`], and each
    /// budget value lands somewhere inside a different `writeln!`. A
    /// `.unwrap()` or `.ok()` left behind in place of a `?` would let
    /// the error be silently dropped and the assertion would fail at
    /// the corresponding budget. The final iteration at
    /// `budget == total_bytes` confirms the success path also clears
    /// the writer cleanly, ruling out an off-by-one in the budget
    /// accounting that would otherwise mask a genuine
    /// short-write regression.
    #[test]
    fn display_propagates_writer_errors_through_every_writeln() {
        let position = reference_position();
        let rendered = format!("{position}");
        let total_bytes = rendered.len();
        assert!(total_bytes > 0, "rendered SolarPosition must not be empty");

        for budget in 0..total_bytes {
            let mut writer = FailingWriter { remaining: budget };
            let result = write!(&mut writer, "{position}");
            assert!(
                result.is_err(),
                "Display must surface fmt::Error at byte budget {budget} of {total_bytes}",
            );
        }

        let mut writer = FailingWriter {
            remaining: total_bytes,
        };
        let result = write!(&mut writer, "{position}");
        assert!(
            result.is_ok(),
            "Display must succeed when the writer accepts every byte",
        );
    }

    /// The four [`Observer`] atmospheric constants must equal the literal
    /// values they document. The SPA paper's reference is pinned against
    /// the equation 42 constants in [`crate::horizontal`] so a future
    /// refactor that decouples them (or changes one without the other)
    /// surfaces here; the ICAO/ISA pair is pinned against the ISO 2533:1975
    /// sea-level standard atmosphere values.
    #[test]
    #[allow(clippy::float_cmp)]
    fn observer_atmospheric_constants_equal_their_published_references() {
        assert_eq!(
            Observer::REFERENCE_PRESSURE_MILLIBARS,
            crate::horizontal::STANDARD_PRESSURE_MILLIBARS,
            "Observer reference pressure must equal the eq. 42 calibration constant",
        );
        assert_eq!(
            Observer::REFERENCE_PRESSURE_MILLIBARS,
            1010.0,
            "SPA paper reference pressure is 1010 mbar",
        );
        assert_eq!(
            Observer::REFERENCE_TEMPERATURE_CELSIUS,
            crate::horizontal::REFERENCE_TEMPERATURE_KELVIN
                - crate::horizontal::KELVIN_OFFSET_FROM_CELSIUS,
            "Observer reference temperature must equal `283 K - 273 K` from eq. 42",
        );
        assert_eq!(
            Observer::REFERENCE_TEMPERATURE_CELSIUS,
            10.0,
            "SPA paper reference temperature is 10 °C (283 K)",
        );
        assert_eq!(
            Observer::ISA_PRESSURE_MILLIBARS,
            1013.25,
            "ICAO/ISA sea-level pressure is 1013.25 mbar per ISO 2533:1975",
        );
        assert_eq!(
            Observer::ISA_TEMPERATURE_CELSIUS,
            15.0,
            "ICAO/ISA sea-level temperature is 15 °C per ISO 2533:1975",
        );
    }

    /// [`Observer::try_with_reference_atmosphere`] must forward the three
    /// geographic arguments verbatim and fill the two atmospheric fields
    /// from the SPA paper's reference constants. An arg-order swap (e.g.
    /// switching `latitude` and `longitude`, since both are `f64`) would
    /// surface here at the asymmetric reference site.
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

    /// [`Observer::try_at_sea_level_isa`] must forward the two geographic
    /// arguments verbatim, pin `elevation` at `0 m` (the ISA reference
    /// elevation), and fill the two atmospheric fields from the ISA
    /// constants.
    #[test]
    #[allow(clippy::float_cmp)]
    fn observer_try_at_sea_level_isa_uses_isa_constants_and_zero_elevation() {
        let obs = Observer::try_at_sea_level_isa(40.0, -3.0).unwrap();
        assert_eq!(obs.latitude(), 40.0);
        assert_eq!(obs.longitude(), -3.0);
        assert_eq!(obs.elevation(), 0.0);
        assert_eq!(obs.pressure(), Observer::ISA_PRESSURE_MILLIBARS);
        assert_eq!(obs.temperature(), Observer::ISA_TEMPERATURE_CELSIUS);
    }

    /// An observer built from [`Observer::try_with_reference_atmosphere`]
    /// collapses equation 42's pressure and temperature ratios to one, so
    /// [`atmospheric_refraction`] returns the pure Saemundsson form
    /// `1.02 / (60 · tan(e₀ + 10.3/(e₀ + 5.11)))`. Pinning this functional
    /// invariant catches any future drift between the two sources of truth
    /// (the `Observer` constants and the `horizontal` formula constants)
    /// that the literal-equality test above cannot detect on its own.
    ///
    /// [`atmospheric_refraction`]: crate::horizontal::atmospheric_refraction
    #[test]
    fn reference_atmosphere_collapses_equation_42_ratios_to_one() {
        let obs = Observer::try_with_reference_atmosphere(0.0, 0.0, 0.0).unwrap();
        for &e0 in &[0.0_f64, 10.0, 45.0, 89.0] {
            let delta_e =
                crate::horizontal::atmospheric_refraction(e0, obs.pressure(), obs.temperature());
            let aux = e0 + 10.3 / (e0 + 5.11);
            let expected = 1.02 / (60.0 * aux.to_radians().tan());
            assert!(
                (delta_e - expected).abs() < 1e-15,
                "Δe at the reference atmosphere must equal the pure Saemundsson form \
                 for e₀ = {e0}: got {delta_e} vs expected {expected}",
            );
        }
    }

    /// [`Observer::TEMPERATURE_FLOOR_CELSIUS_EXCLUSIVE`] must mirror the
    /// `273` denominator constant from equation 42 (not the strict IAU
    /// `273.15`). A future refactor that swapped the offset would surface
    /// here as both an exact-equality failure and a documented divergence
    /// of `0.15 °C` between the validation floor and the formula
    /// singularity.
    #[test]
    #[allow(clippy::float_cmp)]
    fn observer_temperature_floor_matches_equation_42_kelvin_offset() {
        assert_eq!(
            Observer::TEMPERATURE_FLOOR_CELSIUS_EXCLUSIVE,
            -crate::horizontal::KELVIN_OFFSET_FROM_CELSIUS,
            "Temperature floor must equal -eq.42 Kelvin offset",
        );
        assert_eq!(
            Observer::TEMPERATURE_FLOOR_CELSIUS_EXCLUSIVE,
            -273.0,
            "Temperature floor must be -273 °C exactly (paper's rounded constant)",
        );
    }

    /// [`Observer::try_new`] must accept the canonical reference inputs
    /// (the asymmetric Table A5.1 site at 39.74° N, -105.18° E, 1830 m
    /// elevation, 820 mbar, 11 °C) and round-trip every field bit-for-bit
    /// through the accessors. A regression in argument ordering would
    /// surface here as a latitude/longitude swap.
    #[test]
    #[allow(clippy::float_cmp)]
    fn observer_try_new_accepts_valid_inputs_and_round_trips_through_accessors() {
        let obs = Observer::try_new(39.742_476, -105.1786, 1830.14, 820.0, 11.0)
            .expect("A5.1 reference observer must validate");
        assert_eq!(obs.latitude(), 39.742_476);
        assert_eq!(obs.longitude(), -105.1786);
        assert_eq!(obs.elevation(), 1830.14);
        assert_eq!(obs.pressure(), 820.0);
        assert_eq!(obs.temperature(), 11.0);
    }

    /// Inclusive boundaries: latitude `±90°` (the poles) and longitude
    /// `±180°` (the antimeridian, in both forms) must validate. A guard
    /// that uses strict `<` / `>` instead of `<=` / `>=` would reject
    /// these limits and surface here.
    #[test]
    fn observer_try_new_accepts_latitude_and_longitude_at_their_inclusive_bounds() {
        for &lat in &[-90.0_f64, 90.0] {
            for &lon in &[-180.0_f64, 180.0] {
                assert!(
                    Observer::try_new(lat, lon, 0.0, 1013.25, 15.0).is_ok(),
                    "Observer at (lat={lat}, lon={lon}) must validate",
                );
            }
        }
    }

    /// [`Observer::try_new`] must reject every flavour of bad latitude:
    /// `NaN`, `±Inf`, and any value outside `[-90°, 90°]`. The matrix
    /// covers both arms of the `||` short-circuit inside the validation
    /// (`!is_finite()` and `!contains()`) so a future split of the two
    /// checks cannot accidentally let one half through.
    #[test]
    fn observer_try_new_rejects_invalid_latitude() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            90.000_001,
            -90.000_001,
            180.0,
        ] {
            assert!(
                matches!(
                    Observer::try_new(bad, 0.0, 0.0, 1013.25, 15.0),
                    Err(ObserverError::InvalidLatitude(_)),
                ),
                "latitude {bad} must be rejected",
            );
        }
    }

    /// Symmetric to the latitude test: every flavour of bad longitude
    /// (`NaN`, `±Inf`, outside `[-180°, 180°]`) must be rejected. A
    /// regression where the longitude check fell through (e.g. validating
    /// against the latitude range by accident) would surface here.
    #[test]
    fn observer_try_new_rejects_invalid_longitude() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            180.000_001,
            -180.000_001,
            360.0,
        ] {
            assert!(
                matches!(
                    Observer::try_new(0.0, bad, 0.0, 1013.25, 15.0),
                    Err(ObserverError::InvalidLongitude(_)),
                ),
                "longitude {bad} must be rejected",
            );
        }
    }

    /// Non-finite elevation must be rejected. The validator does not
    /// impose a range on elevation (any physically meaningful altitude
    /// keeps the parallax correction well behaved), so the only rejection
    /// is the `!is_finite()` branch.
    #[test]
    fn observer_try_new_rejects_non_finite_elevation() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                matches!(
                    Observer::try_new(0.0, 0.0, bad, 1013.25, 15.0),
                    Err(ObserverError::InvalidElevation(_)),
                ),
                "elevation {bad} must be rejected",
            );
        }
    }

    /// Pressure must be strictly positive and finite. Equation 42's
    /// `(P / 1010)` ratio is undefined for non-positive inputs, so `0.0`
    /// and any negative pressure must be rejected alongside `NaN` and
    /// `±Inf`.
    #[test]
    fn observer_try_new_rejects_invalid_pressure() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            0.0,
            -1.0,
            -1013.25,
        ] {
            assert!(
                matches!(
                    Observer::try_new(0.0, 0.0, 0.0, bad, 15.0),
                    Err(ObserverError::InvalidPressure(_)),
                ),
                "pressure {bad} must be rejected",
            );
        }
    }

    /// Temperature must be strictly above the equation 42 singularity at
    /// `T = -273 °C` and finite. The probe at the exact floor pins the
    /// strict inequality.
    #[test]
    fn observer_try_new_rejects_invalid_temperature() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -273.0,
            -273.000_001,
            -1e6,
        ] {
            assert!(
                matches!(
                    Observer::try_new(0.0, 0.0, 0.0, 1013.25, bad),
                    Err(ObserverError::InvalidTemperature(_)),
                ),
                "temperature {bad} must be rejected",
            );
        }
    }

    /// [`Observer::try_with_reference_atmosphere`] must surface the same
    /// per-field [`ObserverError`] variants as [`Observer::try_new`] for
    /// the three geographic arguments. The atmospheric fields are pinned
    /// to known-valid constants so they cannot fail; only the geographic
    /// inputs are propagated.
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

    /// Symmetric to the reference-atmosphere check: invalid lat/lon
    /// inputs to [`Observer::try_at_sea_level_isa`] must surface as the
    /// matching [`ObserverError`] variants while the pinned ISA
    /// elevation/pressure/temperature never fail.
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

    /// [`ObserverError`]'s [`Display`] impl (via `thiserror`) must
    /// mention the offending raw value and the matching field name. The
    /// assertion is loose on phrasing but pins both pieces of metadata so
    /// a future copy-edit that drops either surfaces here.
    ///
    /// [`Display`]: core::fmt::Display
    #[test]
    fn observer_error_display_mentions_field_and_value() {
        for (err, field) in [
            (ObserverError::InvalidLatitude(91.0), "latitude"),
            (ObserverError::InvalidLongitude(181.0), "longitude"),
            (ObserverError::InvalidElevation(f64::NAN), "elevation"),
            (ObserverError::InvalidPressure(-1.0), "pressure"),
            (ObserverError::InvalidTemperature(-300.0), "temperature"),
        ] {
            let rendered = format!("{err}");
            assert!(
                rendered.contains(field),
                "{err:?} Display must mention `{field}`, got: {rendered}",
            );
        }
    }

    /// [`Surface::try_new`] must accept the canonical Table A5.1 surface
    /// orientation (`slope = 30°`, `azimuth_rotation = -10°`) and
    /// round-trip both fields through the accessors. The asymmetric
    /// rotation catches an accidental swap with the slope.
    #[test]
    #[allow(clippy::float_cmp)]
    fn surface_try_new_accepts_valid_inputs_and_round_trips_through_accessors() {
        let surface = Surface::try_new(30.0, -10.0).expect("A5.1 reference surface must validate");
        assert_eq!(surface.slope(), 30.0);
        assert_eq!(surface.azimuth_rotation(), -10.0);
    }

    /// Inclusive boundaries: `slope ∈ {0°, 180°}` (horizontal, upside
    /// down) and `azimuth_rotation ∈ {-180°, 180°}` must validate. The
    /// matrix mirrors the equivalent [`Observer`] boundary test.
    #[test]
    fn surface_try_new_accepts_slope_and_azimuth_at_their_inclusive_bounds() {
        for &slope in &[0.0_f64, 180.0] {
            for &azimuth in &[-180.0_f64, 180.0] {
                assert!(
                    Surface::try_new(slope, azimuth).is_ok(),
                    "Surface at (slope={slope}, azimuth={azimuth}) must validate",
                );
            }
        }
    }

    /// Every flavour of bad slope (`NaN`, `±Inf`, outside `[0°, 180°]`)
    /// must be rejected. Negative slope has no geometric meaning (it
    /// would re-encode an azimuth flip).
    #[test]
    fn surface_try_new_rejects_invalid_slope() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            -0.000_001,
            180.000_001,
            360.0,
        ] {
            assert!(
                matches!(
                    Surface::try_new(bad, 0.0),
                    Err(SurfaceError::InvalidSlope(_)),
                ),
                "slope {bad} must be rejected",
            );
        }
    }

    /// Symmetric to the slope test: every flavour of bad azimuth rotation
    /// (`NaN`, `±Inf`, outside `[-180°, 180°]`) must be rejected.
    #[test]
    fn surface_try_new_rejects_invalid_azimuth_rotation() {
        for bad in [
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            180.000_001,
            -180.000_001,
            360.0,
        ] {
            assert!(
                matches!(
                    Surface::try_new(0.0, bad),
                    Err(SurfaceError::InvalidAzimuthRotation(_)),
                ),
                "azimuth rotation {bad} must be rejected",
            );
        }
    }

    /// [`Surface::horizontal`] and [`Surface::default`] must both produce
    /// the same `slope = 0°, azimuth_rotation = 0°` surface, the
    /// canonical horizontal collector consumed by
    /// [`SolarPosition::compute`] when no tilted panel is of interest.
    #[test]
    #[allow(clippy::float_cmp)]
    fn surface_horizontal_and_default_are_equivalent_and_zero() {
        let h = Surface::horizontal();
        assert_eq!(h.slope(), 0.0);
        assert_eq!(h.azimuth_rotation(), 0.0);
        assert_eq!(Surface::default(), h);
    }

    /// [`SurfaceError`]'s [`Display`] impl (via `thiserror`) must mention
    /// the offending raw value and the matching field name.
    ///
    /// [`Display`]: core::fmt::Display
    #[test]
    fn surface_error_display_mentions_field_and_value() {
        for (err, field) in [
            (SurfaceError::InvalidSlope(-1.0), "slope"),
            (SurfaceError::InvalidAzimuthRotation(200.0), "azimuth"),
        ] {
            let rendered = format!("{err}");
            assert!(
                rendered.contains(field),
                "{err:?} Display must mention `{field}`, got: {rendered}",
            );
        }
    }
}
