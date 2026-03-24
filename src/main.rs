use chrono::{DateTime, Datelike, Timelike, Utc};
use chrono_tz::{OffsetComponents, Tz};
use log::{debug, info};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    let now = DateTimeWithDUT1 {
        datetime: Utc::now().with_timezone(&Tz::Europe__Madrid),
        dut1: 0.0,
    };
    info!("Now: {now:?}");

    let julian_day = calculate_julian_day(&now);
    debug!("Julian Day: {julian_day}");
}

#[derive(Debug)]
struct DateTimeWithDUT1 {
    datetime: DateTime<Tz>,
    dut1: f64,
}

fn calculate_julian_day(datetime: &DateTimeWithDUT1) -> f64 {
    let mut year: i32 = datetime.datetime.year();
    let mut month: u32 = datetime.datetime.month();
    let day: u32 = datetime.datetime.day();
    let hour: u32 = datetime.datetime.hour();
    let minute: u32 = datetime.datetime.minute();
    let seconds: f64 = f64::from(datetime.datetime.second())
        + f64::from(datetime.datetime.nanosecond()) / 1_000_000_000.0;
    let dut1: f64 = datetime.dut1;
    let tz_offset_s: f64 = datetime
        .datetime
        .offset()
        .base_utc_offset()
        .as_seconds_f64();

    if month < 3 {
        month += 12;
        year -= 1;
    }

    let day_decimal = f64::from(day)
        + (f64::from(hour) + (f64::from(minute) + (seconds + dut1 - tz_offset_s) / 60.0) / 60.0)
            / 24.0;
    let mut julian_day = ((365.25 * f64::from(year + 4716)).trunc())
        + ((30.6001 * f64::from(month + 1)).trunc())
        + day_decimal
        - 1524.5;
    if julian_day > 2_299_160.0 {
        let a = (f64::from(year) / 100.0).trunc();
        julian_day += 2.0 - a + ((a / 4.0).trunc());
    }
    julian_day
}
