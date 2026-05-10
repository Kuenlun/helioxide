/*!
helioxide - Rust implementation of NREL Solar Position Algorithm (SPA)
Copyright (C) 2026  Juan Luis Leal Contreras (Kuenlun)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/

use std::process::Command;

#[test]
fn binary_runs_and_logs_expected_output_markers() -> Result<(), Box<dyn std::error::Error>> {
    let output = Command::new(env!("CARGO_BIN_EXE_helioxide"))
        .env("RUST_LOG", "trace")
        .output()?;

    assert!(
        output.status.success(),
        "binary exited with status {:?}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr),
    );

    let stderr = String::from_utf8(output.stderr)?;
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
    ] {
        assert!(
            stderr.contains(marker),
            "stderr missing marker {marker:?}\nfull stderr:\n{stderr}",
        );
    }

    Ok(())
}
