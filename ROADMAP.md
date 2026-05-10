# SPA Implementation Roadmap

Section 3 of NREL/TP-560-34302 is fully implemented in `src/` (3.2
through 3.16). Section A.1 (`M`, `E`) is covered by
`equation_of_time.rs`, and section A.3 (`JD → calendar date`) by
`calculate_calendar_date_from_julian_day` in `julian.rs`. What remains
is section A.2.

## Remaining

| Section | Quantity                   | Module            |
|---------|----------------------------|-------------------|
| A.2     | Sunrise / transit / sunset | `solar_events.rs` |

A.2 drives sections 3.2 to 3.5 and 3.9 three times per call, and
consumes section A.3 to format the wall-clock outputs.
