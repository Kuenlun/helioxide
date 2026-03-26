use chrono::Utc;
use chrono_tz::Tz;
use helioxide::{DateTimeWithDUT1, calculate_julian_day};
use log::{debug, info};

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    let now = DateTimeWithDUT1::new(Utc::now().with_timezone(&Tz::Europe__Madrid));
    info!("Now: {now:?}");

    let julian_day = calculate_julian_day(&now);
    debug!("Julian Day: {julian_day}");
}
