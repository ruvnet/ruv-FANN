//! Calendar arithmetic in the original IANA timezone, retaining UTC instants.
use super::{advance, Error, Frequency, Result};
use chrono::{DateTime, Days, Duration, LocalResult, Offset, TimeZone, Utc};
use chrono_tz::Tz;
use polars::prelude::DataType;

pub(super) fn advance_zoned(
    timestamp: DateTime<Utc>,
    frequency: &Frequency,
    periods: usize,
    dtype: &DataType,
) -> Result<DateTime<Utc>> {
    let DataType::Datetime(_, Some(zone)) = dtype else {
        return advance(timestamp, frequency, periods);
    };
    let timezone: Tz = zone
        .parse()
        .map_err(|_| Error::data(format!("Unsupported IANA timestamp timezone '{zone}'")))?;
    if let Frequency::Custom(spec) = frequency {
        let digits = spec.bytes().take_while(u8::is_ascii_digit).count();
        let (count, suffix) = spec.split_at(digits);
        let count = count
            .parse::<usize>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| Error::config("Custom frequency needs a positive period count"))?;
        let base = Frequency::from_str(suffix)?;
        if matches!(base, Frequency::Custom(_)) {
            return Err(Error::config(format!("Unsupported frequency '{spec}'")));
        }
        return advance_zoned(
            timestamp,
            &base,
            periods
                .checked_mul(count)
                .ok_or_else(|| Error::data("Frequency offset overflow"))?,
            dtype,
        );
    }
    if !matches!(
        frequency,
        Frequency::Daily
            | Frequency::Weekly
            | Frequency::Monthly
            | Frequency::Quarterly
            | Frequency::Yearly
            | Frequency::BusinessDaily
            | Frequency::BusinessMonthly
            | Frequency::BusinessQuarterly
    ) {
        // Sub-day frequencies describe elapsed time, including repeated/skipped
        // wall-clock hours at daylight-saving transitions.
        return advance(timestamp, frequency, periods);
    }
    // Avoid DateTime::naive_local's panic near chrono's representable limits.
    let utc = timestamp.naive_utc();
    let offset = timezone
        .offset_from_utc_datetime(&utc)
        .fix()
        .local_minus_utc();
    let local = utc
        .checked_add_signed(Duration::seconds(i64::from(offset)))
        .ok_or_else(|| Error::data("Local timestamp overflow"))?;
    let future_local = match frequency {
        Frequency::Daily | Frequency::Weekly => {
            let days = periods
                .checked_mul(if matches!(frequency, Frequency::Weekly) {
                    7
                } else {
                    1
                })
                .and_then(|value| u64::try_from(value).ok())
                .ok_or_else(|| Error::data("Calendar day offset overflow"))?;
            local
                .checked_add_days(Days::new(days))
                .ok_or_else(|| Error::data("Local calendar timestamp overflow"))?
        }
        // Reuse checked Months and business-day logic on a wall-clock anchor.
        // This pseudo-UTC value is only an arithmetic carrier, never an instant
        // returned to callers or stored in model history.
        _ => advance(local.and_utc(), frequency, periods)?.naive_utc(),
    };
    match timezone.from_local_datetime(&future_local) {
        LocalResult::Single(value) => Ok(value.with_timezone(&Utc)),
        LocalResult::Ambiguous(_, _) => Err(Error::data(
            "Calendar offset resolves to an ambiguous local timestamp",
        )),
        LocalResult::None => Err(Error::data(
            "Calendar offset resolves to a nonexistent or out-of-range local timestamp",
        )),
    }
}
