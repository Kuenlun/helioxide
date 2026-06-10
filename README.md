# helioxide

[![CI](https://github.com/Kuenlun/helioxide/actions/workflows/rust.yml/badge.svg?branch=master)](https://github.com/Kuenlun/helioxide/actions/workflows/rust.yml)
[![codecov](https://codecov.io/gh/Kuenlun/helioxide/branch/master/graph/badge.svg)](https://codecov.io/gh/Kuenlun/helioxide)
[![Crates.io](https://img.shields.io/crates/v/helioxide.svg)](https://crates.io/crates/helioxide)
[![Docs.rs](https://docs.rs/helioxide/badge.svg)](https://docs.rs/helioxide)
[![License: MIT OR Apache-2.0](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

A pure Rust implementation of the NREL Solar Position Algorithm (SPA) for
high-precision solar calculations, faithful to Reda & Andreas,
*Solar Position Algorithm for Solar Radiation Applications*
(NREL/TP-560-34302, 2008 revision).

## Features

- **Full SPA pipeline** — Julian day through topocentric zenith and azimuth
  (sections 3.1 to 3.15, every intermediate quantity exposed on
  [`SolarPosition`]), incidence on tilted collectors (3.16), equation of
  time (appendix A.1), and sunrise, solar noon and sunset on [`SolarDay`]
  (appendix A.2).
- **Reference precision** — zenith and azimuth carry the paper's stated
  uncertainty of ±0.0003° over the years −2000 to 6000. Every periodic term
  of Tables A4.2 and A4.3 is transcribed digit-for-digit, and the test suite
  pins the appendix A.5 worked example and the Table A4.1 Julian-day cases.
- **Automatic ΔT** — [`SolarPosition::compute`] and [`SolarDay::compute`]
  resolve `ΔT = TT − UT1` from the embedded USNO observed monthly table
  (linearly interpolated) and fall back to the Espenak–Meeus polynomials
  outside the observed window; `compute_with_delta_t` pins an explicit value.
- **Timezone-aware civil days** — inputs are `chrono` datetimes in any
  timezone; [`SolarDay`] anchors on the input's local civil date and renders
  events back in the same timezone, handling DST gaps and overlaps.
- **Valid by construction** — [`Observer`], [`Surface`] and [`SpaDateTime`]
  validate their domains at the boundary, so the numeric pipeline itself is
  infallible. `#![forbid(unsafe_code)]`, no panics, 100% branch coverage.

[`SolarPosition`]: https://docs.rs/helioxide/latest/helioxide/spa/struct.SolarPosition.html
[`SolarPosition::compute`]: https://docs.rs/helioxide/latest/helioxide/spa/struct.SolarPosition.html#method.compute
[`SolarDay`]: https://docs.rs/helioxide/latest/helioxide/solar_time/struct.SolarDay.html
[`SolarDay::compute`]: https://docs.rs/helioxide/latest/helioxide/solar_time/struct.SolarDay.html#method.compute
[`Observer`]: https://docs.rs/helioxide/latest/helioxide/spa/struct.Observer.html
[`Surface`]: https://docs.rs/helioxide/latest/helioxide/spa/struct.Surface.html
[`SpaDateTime`]: https://docs.rs/helioxide/latest/helioxide/time/struct.SpaDateTime.html

## Quick start

```rust
use chrono::{TimeZone, Utc};
use helioxide::{Observer, SolarDay, SolarPosition, SpaDateTime, Surface};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // NREL reference case: Golden, Colorado, 2003-10-17 12:30:30 MST (19:30:30 UT).
    let datetime = SpaDateTime::new(Utc.with_ymd_and_hms(2003, 10, 17, 19, 30, 30).unwrap());
    let observer = Observer::try_new(39.742476, -105.1786, 1830.14, 820.0, 11.0)?;

    // Topocentric solar position with the paper's ΔT = 67 s pinned;
    // `SolarPosition::compute` resolves ΔT automatically instead.
    let position = SolarPosition::compute_with_delta_t(&datetime, 67.0, observer);
    assert!((position.topocentric_zenith - 50.11162).abs() < 1e-4);
    assert!((position.topocentric_azimuth - 194.34024).abs() < 1e-4);

    // Angle of incidence on a collector tilted 30°, rotated 10° east of south.
    // One computed position serves any number of surface orientations.
    let surface = Surface::try_new(30.0, -10.0)?;
    assert!((position.surface_incidence(surface) - 25.18700).abs() < 1e-4);

    // Sunrise, solar noon and sunset on the civil day of the input,
    // `None` on polar day or polar night.
    let day = SolarDay::compute_with_delta_t(&datetime, 67.0, observer);
    assert!(day.sunrise.unwrap() < day.transit);
    assert!(day.transit < day.sunset.unwrap());
    Ok(())
}
```

Each pipeline stage (heliocentric, geocentric, nutation, parallax, …) also
lives in its own module as a documented free function keyed to the paper's
equation numbers, so partial computations and cross-checks against the
report are straightforward.

## License

Licensed under either of

- Apache License, Version 2.0
  ([LICENSE-APACHE](https://github.com/Kuenlun/helioxide/blob/master/LICENSE-APACHE)
  or <https://www.apache.org/licenses/LICENSE-2.0>)
- MIT license
  ([LICENSE-MIT](https://github.com/Kuenlun/helioxide/blob/master/LICENSE-MIT)
  or <https://opensource.org/licenses/MIT>)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in the work by you, as defined in the
Apache-2.0 license, shall be dual licensed as above, without any
additional terms or conditions.
