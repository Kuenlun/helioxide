use thiserror::Error;

#[derive(Error, Debug)]
pub enum HelioxideError {
    #[error("Failed to parse timezone: {0}")]
    TimezoneParseError(#[from] chrono_tz::ParseError),
}
