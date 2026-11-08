//! Checked conversion helpers for core data frames.
use super::*;

pub(super) fn read_times(column: &Column) -> NeuroDivergentResult<Vec<DateTime<Utc>>> {
    let normalized = match column.dtype() {
        DataType::Date | DataType::Int64 | DataType::Int32 => column
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .map_err(|e| ErrorBuilder::data(format!("Invalid timestamps: {e}")).build())?,
        DataType::Datetime(_, _) => column.clone(),
        _ => {
            return Err(data_error!(
                "Timestamp column must be Date, Datetime, or signed Unix milliseconds"
            ))
        }
    };
    let datetime = normalized
        .datetime()
        .map_err(|e| ErrorBuilder::data(e.to_string()).build())?;
    let unit = datetime.time_unit();
    datetime
        .into_iter()
        .map(|value| {
            value
                .and_then(|timestamp| match unit {
                    TimeUnit::Nanoseconds => Some(DateTime::from_timestamp_nanos(timestamp)),
                    TimeUnit::Microseconds => DateTime::from_timestamp_micros(timestamp),
                    TimeUnit::Milliseconds => DateTime::from_timestamp_millis(timestamp),
                })
                .ok_or_else(|| data_error!("Null or out-of-range timestamp"))
        })
        .collect()
}

pub(super) fn read_numeric<T: Float>(column: &Column) -> NeuroDivergentResult<Vec<T>> {
    let values = column
        .cast(&DataType::Float64)
        .map_err(|e| ErrorBuilder::data(format!("Expected numeric column: {e}")).build())?;
    values
        .f64()
        .map_err(|e| ErrorBuilder::data(e.to_string()).build())?
        .into_iter()
        .map(|value| {
            value
                .and_then(T::from)
                .filter(|value| value.is_finite())
                .ok_or_else(|| data_error!("Null, non-finite, or unrepresentable numeric value"))
        })
        .collect()
}

pub(super) fn exogenous_matrix<T: Float>(
    frame: &DataFrame,
    names: &[String],
) -> NeuroDivergentResult<Option<Array2<T>>> {
    if names.is_empty() {
        return Ok(None);
    }
    let mut matrix = Array2::zeros((frame.height(), names.len()));
    for (index, name) in names.iter().enumerate() {
        let values = read_numeric(
            frame
                .column(name)
                .map_err(|e| ErrorBuilder::data(e.to_string()).build())?,
        )?;
        matrix.column_mut(index).assign(&Array1::from(values));
    }
    Ok(Some(matrix))
}

pub(super) fn infer_frequency(times: &[DateTime<Utc>]) -> Option<String> {
    let delta = times.get(1)?.signed_duration_since(times[0]);
    if delta <= chrono::Duration::zero() || times.windows(2).any(|pair| pair[1] - pair[0] != delta)
    {
        return None;
    }
    for (label, duration) in [
        ("W", chrono::Duration::weeks(1)),
        ("D", chrono::Duration::days(1)),
        ("H", chrono::Duration::hours(1)),
        ("min", chrono::Duration::minutes(1)),
        ("s", chrono::Duration::seconds(1)),
        ("ms", chrono::Duration::milliseconds(1)),
        ("us", chrono::Duration::microseconds(1)),
        ("ns", chrono::Duration::nanoseconds(1)),
    ] {
        if delta == duration {
            return Some(label.into());
        }
    }
    delta.num_nanoseconds().map(|value| format!("{value}ns"))
}
