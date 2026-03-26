use chrono::{DateTime, Datelike, Timelike, Utc};
use chrono_tz::{OffsetComponents, Tz};
use log::{debug, info};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    let now = DateTimeWithDUT1::new(Utc::now().with_timezone(&Tz::Europe__Madrid));
    info!("Now: {now:?}");

    let julian_day = calculate_julian_day(&now);
    debug!("Julian Day: {julian_day}");
}

#[derive(Debug)]
struct DateTimeWithDUT1 {
    datetime: DateTime<Tz>,
    dut1: f64,
}

impl DateTimeWithDUT1 {
    const fn new(datetime: DateTime<Tz>) -> Self {
        Self {
            datetime,
            dut1: 0.0,
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn test_calculate_julian_day() {
        // (year, month, day, hour, minute, second, exp_julian_day)
        let test_cases = [
            (2000, 1, 1, 12, 0, 0, 2_451_545.0),
            (1999, 1, 1, 0, 0, 0, 2_451_179.5),
            (1987, 1, 27, 0, 0, 0, 2_446_822.5),
            (1987, 6, 19, 12, 0, 0, 2_446_966.0),
            (1988, 1, 27, 0, 0, 0, 2_447_187.5),
            (1988, 6, 19, 12, 0, 0, 2_447_332.0),
            (1900, 1, 1, 0, 0, 0, 2_415_020.5),
            (1600, 1, 1, 0, 0, 0, 2_305_447.5),
            (1600, 12, 31, 0, 0, 0, 2_305_812.5),
            (837, 4, 10, 7, 12, 0, 2_026_871.8),
            (-123, 12, 31, 0, 0, 0, 1_676_496.5),
            (-122, 1, 1, 0, 0, 0, 1_676_497.5),
            (-1000, 7, 12, 12, 0, 0, 1_356_001.0),
            // (-1000, 2, 29, 0, 0, 0, 1_355_866.5), // TODO: Chrono panics here. The proleptic Gregorian calendar does not consider year -1000 a leap year
            (-1001, 8, 17, 21, 36, 0, 1_355_671.4),
            (-4712, 1, 1, 12, 0, 0, 0.0),
        ];

        for (y, m, d, h, min, s, exp_julian_day) in test_cases {
            let dt = chrono_tz::UTC.with_ymd_and_hms(y, m, d, h, min, s).unwrap(); //("Failed to create valid datetime from test case data");

            let datetime = DateTimeWithDUT1::new(dt);
            let julian_day = calculate_julian_day(&datetime);

            assert!(
                (julian_day - exp_julian_day).abs() < 1e-6,
                "Julian day mismatch for {y}-{m:02}-{d:02} {h:02}:{min:02}:{s:02}. Expected: {exp_julian_day}, Got: {julian_day}"
            );
        }
    }
}
