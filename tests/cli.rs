// SPDX-License-Identifier: MIT OR Apache-2.0
// helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
// Copyright (c) 2026 Juan Luis Leal Contreras (Kuenlun)

use std::process::Command;

#[test]
fn binary_runs_and_prints_expected_output_markers() -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_helioxide")).output()?;

    assert!(
        output.status.success(),
        "binary exited with status {:?}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );

    let stdout = String::from_utf8(output.stdout)?;
    for marker in [
        "Now:",
        "Julian Day:",
        "Julian Ephemeris Day:",
        "Julian Century:",
        "Julian Ephemeris Century:",
        "Julian Ephemeris Millennium:",
        "Earth heliocentric longitude L:",
        "Earth heliocentric latitude B:",
        "Earth radius vector R:",
        "Sun geocentric longitude Θ:",
        "Sun geocentric latitude β:",
        "Nutation in longitude Δψ:",
        "Nutation in obliquity Δε:",
        "True obliquity of the ecliptic ε:",
        "Aberration correction Δτ:",
        "Apparent sun longitude λ:",
        "Mean sidereal time at Greenwich ν₀:",
        "Apparent sidereal time at Greenwich ν:",
        "Sun geocentric right ascension α:",
        "Sun geocentric declination δ:",
        "Observer local hour angle H:",
        "Sun equatorial horizontal parallax ξ:",
        "Sun right ascension parallax Δα:",
        "Topocentric sun right ascension α':",
        "Topocentric sun declination δ':",
        "Topocentric local hour angle H':",
        "Topocentric elevation without refraction e₀:",
        "Atmospheric refraction Δe:",
        "Topocentric zenith angle θ:",
        "Topocentric astronomers' azimuth Γ:",
        "Topocentric azimuth Φ:",
        "Surface incidence angle I:",
        "Equation of time E:",
    ] {
        assert!(
            stdout.contains(marker),
            "stdout missing marker {marker:?}\nfull stdout:\n{stdout}",
        );
    }

    Ok(())
}
