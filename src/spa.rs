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

use crate::{
    SpaDateTime, apparent, equation_of_time, equatorial, geocentric, heliocentric, horizontal,
    hour_angle, incidence, julian, nutation, obliquity, parallax, sidereal,
};

/// Geographic and atmospheric description of the observation site.
///
/// `longitude` and `latitude` follow the conventions of sections 3.11 and
/// 3.12.2: positive east of Greenwich and positive north of the equator,
/// respectively. `elevation` is the observer height above sea level
/// (metres), consumed by section 3.12.3. `pressure` (millibars) and
/// `temperature` (degrees Celsius) feed the atmospheric refraction model
/// of equation 42 and should be annual averages for the site.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Observer {
    /// Geographic longitude `σ` (degrees, signed, positive east of
    /// Greenwich).
    pub longitude: f64,
    /// Geographic latitude `φ` (degrees, signed, positive north of the
    /// equator).
    pub latitude: f64,
    /// Elevation above sea level `E` (metres).
    pub elevation: f64,
    /// Annual average atmospheric pressure `P` (millibars).
    pub pressure: f64,
    /// Annual average atmospheric temperature `T` (degrees Celsius).
    pub temperature: f64,
}

/// Tilted surface (e.g. a fixed-tilt photovoltaic panel) consumed by the
/// angle-of-incidence calculation in section 3.16.
///
/// A horizontal collector is `Surface { slope: 0.0, azimuth_rotation: 0.0 }`;
/// a vertical south-facing wall is `Surface { slope: 90.0, azimuth_rotation: 0.0 }`;
/// a vertical west-facing wall is `Surface { slope: 90.0, azimuth_rotation: 90.0 }`;
/// a vertical east-facing wall is `Surface { slope: 90.0, azimuth_rotation: -90.0 }`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Surface {
    /// Slope from the horizontal plane `ω` (degrees; `0°` is horizontal,
    /// `90°` is vertical).
    pub slope: f64,
    /// Surface azimuth rotation `γ` (degrees, signed, positive westward
    /// and negative eastward from due south; `0°` faces south). Matches
    /// the convention of the astronomers' azimuth `Γ` so that `Γ − γ`
    /// in equation 47 is the signed angular gap between the sun and the
    /// surface normal projected on the horizontal plane.
    pub azimuth_rotation: f64,
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
    /// let observer = Observer {
    ///     longitude: -0.490_68,
    ///     latitude: 38.346_02,
    ///     elevation: 3.0,
    ///     pressure: 1015.0,
    ///     temperature: 18.0,
    /// };
    /// let surface = Surface { slope: 38.346_02, azimuth_rotation: 0.0 };
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

        let h = hour_angle::observer_local_hour_angle(nu, observer.longitude, alpha);

        let xi = parallax::equatorial_horizontal_parallax(r);
        let topocentric = parallax::topocentric_equatorial_coordinates(
            alpha,
            delta,
            h,
            xi,
            observer.latitude,
            observer.elevation,
        );

        let h_prime =
            hour_angle::topocentric_local_hour_angle(h, topocentric.parallax_in_right_ascension);

        let e0 = horizontal::topocentric_elevation_without_refraction(
            observer.latitude,
            topocentric.declination,
            h_prime,
        );
        let delta_e =
            horizontal::atmospheric_refraction(e0, observer.pressure, observer.temperature);
        let e_corrected = horizontal::topocentric_elevation_corrected(e0, delta_e);
        let zenith = horizontal::topocentric_zenith_angle(e0, delta_e);

        let gamma =
            horizontal::astronomers_azimuth(h_prime, observer.latitude, topocentric.declination);
        let gamma_signed = horizontal::astronomers_azimuth_signed(gamma);
        let azimuth = horizontal::topocentric_azimuth_angle(gamma);

        let incidence_angle = incidence::surface_incidence_angle(
            zenith,
            gamma,
            surface.slope,
            surface.azimuth_rotation,
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
    use super::{Observer, SolarPosition, Surface};
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
        Observer {
            longitude: REFERENCE_LONGITUDE_DEGREES,
            latitude: REFERENCE_LATITUDE_DEGREES,
            elevation: REFERENCE_ELEVATION_METRES,
            pressure: REFERENCE_PRESSURE_MILLIBARS,
            temperature: REFERENCE_TEMPERATURE_CELSIUS,
        }
    }

    fn reference_surface() -> Surface {
        Surface {
            slope: REFERENCE_SURFACE_SLOPE_DEGREES,
            azimuth_rotation: REFERENCE_SURFACE_AZIMUTH_ROTATION_DEGREES,
        }
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
            observer.latitude,
            observer.elevation,
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
            observer.latitude,
            position.topocentric_declination,
            position.topocentric_local_hour_angle,
        );
        let delta_e =
            horizontal::atmospheric_refraction(e0, observer.pressure, observer.temperature);
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
}
