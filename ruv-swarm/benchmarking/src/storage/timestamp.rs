//! Read the SQLite timestamp representations accepted by the former backend.
use chrono::{DateTime, NaiveDateTime, Utc};
use rusqlite::types::{FromSql, FromSqlError, FromSqlResult, ValueRef};

pub(super) struct SqlTimestamp(pub DateTime<Utc>);

impl FromSql for SqlTimestamp {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let timestamp = match value {
            ValueRef::Integer(seconds) => DateTime::from_timestamp(seconds, 0),
            ValueRef::Real(julian) => {
                let seconds = (julian - 2_440_587.5) * 86_400.0;
                if !seconds.is_finite() || seconds < i64::MIN as f64 || seconds >= i64::MAX as f64 {
                    None
                } else {
                    // Retain SQLx's integer/fractional split for Julian dates.
                    DateTime::from_timestamp(
                        seconds.trunc() as i64,
                        (seconds.fract().abs() * 1e9) as u32,
                    )
                }
            }
            ValueRef::Text(bytes) => parse_text(
                std::str::from_utf8(bytes).map_err(|error| FromSqlError::Other(Box::new(error)))?,
            ),
            _ => return Err(FromSqlError::InvalidType),
        };
        timestamp
            .map(Self)
            .ok_or_else(|| FromSqlError::Other("invalid benchmark timestamp".into()))
    }
}

fn parse_text(value: &str) -> Option<DateTime<Utc>> {
    if let Ok(timestamp) = DateTime::parse_from_rfc3339(value) {
        return Some(timestamp.with_timezone(&Utc));
    }
    for pattern in [
        "%F %T%.f",
        "%F %R",
        "%F %RZ",
        "%F %R%:z",
        "%F %T%.fZ",
        "%F %T%.f%:z",
        "%FT%R",
        "%FT%RZ",
        "%FT%R%:z",
        "%FT%T%.f",
        "%FT%T%.fZ",
        "%FT%T%.f%:z",
    ] {
        if let Ok(timestamp) = DateTime::parse_from_str(value, pattern) {
            return Some(timestamp.with_timezone(&Utc));
        }
        if let Ok(timestamp) = NaiveDateTime::parse_from_str(value, pattern) {
            return Some(timestamp.and_utc());
        }
    }
    None
}
