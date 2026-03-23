mod error;

use chrono::Utc;
use chrono_tz::Tz;
use log::info;

use crate::error::HelioxideError;

fn main() -> Result<(), HelioxideError> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("trace")).init();

    let tz: Tz = "Europe/Madrid".parse()?;
    let now = Utc::now().with_timezone(&tz);
    info!("Now: {now:?}");

    Ok(())
}
