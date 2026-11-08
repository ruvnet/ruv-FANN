//! Checked chronological data preparation shared by forecasting and backtesting.
use crate::{
    Frequency, NeuroDivergentError as Error, NeuroDivergentResult as Result, TimeSeriesDataFrame,
};
use chrono::{DateTime, Datelike, Duration, Months, Utc, Weekday};
use num_traits::Float;
use polars::prelude::*;
use std::collections::BTreeMap;

#[path = "calendar.rs"]
mod calendar;
use calendar::advance_zoned;

#[derive(Debug, Clone)]
pub(crate) struct SeriesHistory {
    pub id: String,
    pub times: Vec<DateTime<Utc>>,
    pub values: Vec<f64>,
    pub indices: Vec<IdxSize>,
}

#[derive(Debug, Clone)]
pub(crate) struct Timeline {
    pub series: BTreeMap<String, SeriesHistory>,
    pub timestamp_type: DataType,
}

pub(crate) fn timestamps(column: &Column) -> Result<Vec<DateTime<Utc>>> {
    let (physical, unit) = match column.dtype() {
        DataType::Datetime(unit, _) => (column.cast(&DataType::Int64), *unit),
        DataType::Date => (
            column
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .and_then(|c| c.cast(&DataType::Int64)),
            TimeUnit::Milliseconds,
        ),
        DataType::Int64 | DataType::Int32 => {
            (column.cast(&DataType::Int64), TimeUnit::Milliseconds)
        }
        _ => {
            return Err(Error::data(
                "Timestamp column must be Date, Datetime, or signed Unix milliseconds",
            ))
        }
    };
    let physical = physical.map_err(|e| Error::data(format!("Invalid timestamps: {e}")))?;
    physical
        .i64()
        .map_err(|e| Error::data(e.to_string()))?
        .into_iter()
        .map(|value| {
            value
                .and_then(|value| match unit {
                    TimeUnit::Nanoseconds => Some(DateTime::from_timestamp_nanos(value)),
                    TimeUnit::Microseconds => DateTime::from_timestamp_micros(value),
                    TimeUnit::Milliseconds => DateTime::from_timestamp_millis(value),
                })
                .ok_or_else(|| Error::data("Null or out-of-range timestamp"))
        })
        .collect()
}

pub(crate) fn timestamp_column(
    name: &str,
    values: &[DateTime<Utc>],
    dtype: &DataType,
) -> Result<Column> {
    let (unit, output_type) = match dtype {
        DataType::Datetime(unit, zone) => (*unit, DataType::Datetime(*unit, zone.clone())),
        DataType::Date => (
            TimeUnit::Milliseconds,
            DataType::Datetime(TimeUnit::Milliseconds, None),
        ),
        DataType::Int64 | DataType::Int32 => (TimeUnit::Milliseconds, DataType::Int64),
        _ => return Err(Error::data("Unsupported timestamp representation")),
    };
    let raw = values
        .iter()
        .map(|value| match unit {
            TimeUnit::Nanoseconds => value
                .timestamp_nanos_opt()
                .ok_or_else(|| Error::data("Nanosecond timestamp is out of range")),
            TimeUnit::Microseconds if value.timestamp_subsec_nanos() % 1_000 == 0 => {
                Ok(value.timestamp_micros())
            }
            TimeUnit::Milliseconds if value.timestamp_subsec_nanos() % 1_000_000 == 0 => {
                Ok(value.timestamp_millis())
            }
            _ => Err(Error::data(
                "Forecast frequency exceeds timestamp column precision",
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    let physical = Int64Chunked::from_vec(name.into(), raw);
    // Casting Int64 to Datetime discards the timezone in Polars 0.45. Construct
    // its logical datetime array explicitly to preserve the caller's metadata.
    match output_type {
        DataType::Datetime(unit, zone) => {
            Ok(physical.into_datetime(unit, zone).into_series().into())
        }
        DataType::Int64 => Ok(physical.into_series().into()),
        _ => Err(Error::data("Unsupported timestamp representation")),
    }
}

impl Timeline {
    pub fn read<T: Float>(data: &TimeSeriesDataFrame<T>, frequency: &Frequency) -> Result<Self> {
        data.schema.validate_dataframe(&data.data)?;
        if data.data.height() == 0 {
            return Err(Error::data("Time series data is empty"));
        }
        if data
            .frequency
            .as_ref()
            .is_some_and(|found| found != frequency)
        {
            return Err(Error::data(
                "Data frequency does not match forecasting frequency",
            ));
        }
        let ids = data
            .data
            .column(&data.schema.unique_id_col)
            .and_then(|column| column.str())
            .map_err(|_| Error::data("Series identifiers must be strings"))?;
        let timestamp = data
            .data
            .column(&data.schema.ds_col)
            .map_err(|e| Error::data(e.to_string()))?;
        let times = timestamps(timestamp)?;
        let targets = data
            .data
            .column(&data.schema.y_col)
            .and_then(|c| c.cast(&DataType::Float64))
            .map_err(|_| Error::data("Targets must be numeric"))?;
        let targets = targets.f64().map_err(|e| Error::data(e.to_string()))?;
        let mut grouped = BTreeMap::<String, BTreeMap<DateTime<Utc>, (f64, IdxSize)>>::new();
        for (index, time) in times.into_iter().enumerate() {
            let id = ids
                .get(index)
                .filter(|id| !id.trim().is_empty())
                .ok_or_else(|| Error::data("Null or empty series identifier"))?;
            let value = targets
                .get(index)
                .filter(|value| value.is_finite())
                .ok_or_else(|| Error::data("Null or non-finite target"))?;
            if T::from(value).is_none_or(|value| !value.is_finite()) {
                return Err(Error::data(
                    "Target cannot be represented by the model numeric type",
                ));
            }
            let index =
                IdxSize::try_from(index).map_err(|_| Error::data("Too many observations"))?;
            if grouped
                .entry(id.to_owned())
                .or_default()
                .insert(time, (value, index))
                .is_some()
            {
                return Err(Error::data(format!(
                    "Duplicate timestamp for series '{id}'"
                )));
            }
        }
        let mut series = BTreeMap::new();
        for (id, observations) in grouped {
            let mut history = SeriesHistory {
                id: id.clone(),
                times: Vec::new(),
                values: Vec::new(),
                indices: Vec::new(),
            };
            for (time, (value, index)) in observations {
                if history.times.is_empty()
                    && advance_zoned(time, frequency, 0, timestamp.dtype())? != time
                {
                    return Err(Error::data(format!(
                        "Timestamp is not aligned to frequency for series '{id}'"
                    )));
                }
                if let Some(first) = history.times.first() {
                    if advance_zoned(*first, frequency, history.times.len(), timestamp.dtype())?
                        != time
                    {
                        return Err(Error::data(format!(
                            "Irregular timestamps for series '{id}'"
                        )));
                    }
                }
                history.times.push(time);
                history.values.push(value);
                history.indices.push(index);
            }
            series.insert(id, history);
        }
        Ok(Self {
            series,
            timestamp_type: timestamp.dtype().clone(),
        })
    }

    pub fn normalized<T: Float>(
        &self,
        data: &TimeSeriesDataFrame<T>,
        frequency: &Frequency,
    ) -> Result<TimeSeriesDataFrame<T>> {
        let indices = self
            .series
            .values()
            .flat_map(|series| series.indices.iter().copied())
            .collect();
        let frame = data
            .data
            .take(&IdxCa::from_vec("index".into(), indices))
            .map_err(|e| Error::data(format!("Cannot sort observations: {e}")))?;
        TimeSeriesDataFrame::from_polars(frame, data.schema.clone(), Some(frequency.clone()))
    }

    pub fn future(
        &self,
        frequency: &Frequency,
        horizon: usize,
    ) -> Result<Vec<(String, DateTime<Utc>)>> {
        if horizon == 0 || horizon > 1_000_000 {
            return Err(Error::config("Forecast horizon must be in 1..=1000000"));
        }
        let count = self
            .series
            .len()
            .checked_mul(horizon)
            .filter(|count| *count <= 10_000_000)
            .ok_or_else(|| Error::config("Forecast exceeds 10000000 rows"))?;
        let mut keys = Vec::with_capacity(count);
        for series in self.series.values() {
            for step in 0..horizon {
                let index = series
                    .times
                    .len()
                    .checked_add(step)
                    .ok_or_else(|| Error::data("Forecast index overflow"))?;
                keys.push((
                    series.id.clone(),
                    advance_zoned(series.times[0], frequency, index, &self.timestamp_type)?,
                ));
            }
        }
        Ok(keys)
    }
}

/// Advance from an original calendar anchor, preserving month-end alignment.
pub(crate) fn advance(
    timestamp: DateTime<Utc>,
    frequency: &Frequency,
    periods: usize,
) -> Result<DateTime<Utc>> {
    let calendar = |months: usize, business: bool| -> Result<DateTime<Utc>> {
        let months = u32::try_from(months).map_err(|_| Error::data("Calendar offset overflow"))?;
        let shifted = timestamp
            .checked_add_months(Months::new(months))
            .ok_or_else(|| Error::data("Timestamp overflow"))?;
        let month_end = last_day(timestamp)? == timestamp.day();
        let mut shifted = if month_end || business {
            shifted
                .with_day(last_day(shifted)?)
                .ok_or_else(|| Error::data("Invalid month end"))?
        } else {
            shifted
        };
        if business {
            while matches!(shifted.weekday(), Weekday::Sat | Weekday::Sun) {
                shifted = shifted
                    .checked_sub_signed(Duration::days(1))
                    .ok_or_else(|| Error::data("Timestamp overflow"))?;
            }
        }
        Ok(shifted)
    };
    let scaled = |factor: usize| {
        periods
            .checked_mul(factor)
            .ok_or_else(|| Error::data("Calendar offset overflow"))
    };
    match frequency {
        Frequency::Monthly => calendar(periods, false),
        Frequency::Quarterly => calendar(scaled(3)?, false),
        Frequency::Yearly => calendar(scaled(12)?, false),
        Frequency::BusinessMonthly => calendar(periods, true),
        Frequency::BusinessQuarterly => calendar(scaled(3)?, true),
        Frequency::BusinessDaily => {
            if matches!(timestamp.weekday(), Weekday::Sat | Weekday::Sun) {
                return Err(Error::data(
                    "Business-daily series cannot start on a weekend",
                ));
            }
            let weeks = periods / 5;
            let days = weeks
                .checked_mul(7)
                .and_then(|days| i64::try_from(days).ok())
                .ok_or_else(|| Error::data("Business offset overflow"))?;
            let offset =
                Duration::try_days(days).ok_or_else(|| Error::data("Business offset overflow"))?;
            let mut next = timestamp
                .checked_add_signed(offset)
                .ok_or_else(|| Error::data("Timestamp overflow"))?;
            for _ in 0..periods % 5 {
                loop {
                    next = next
                        .checked_add_signed(Duration::days(1))
                        .ok_or_else(|| Error::data("Timestamp overflow"))?;
                    if !matches!(next.weekday(), Weekday::Sat | Weekday::Sun) {
                        break;
                    }
                }
            }
            Ok(next)
        }
        Frequency::Custom(spec) => {
            let digits = spec.bytes().take_while(u8::is_ascii_digit).count();
            let (count, suffix) = spec.split_at(digits);
            let count = count
                .parse::<usize>()
                .ok()
                .filter(|count| *count > 0)
                .ok_or_else(|| {
                    Error::config(format!(
                        "Unsupported frequency '{spec}'; use a positive count and known unit"
                    ))
                })?;
            let base = Frequency::from_str(suffix)?;
            if matches!(base, Frequency::Custom(_)) {
                return Err(Error::config(format!("Unsupported frequency '{spec}'")));
            }
            advance(
                timestamp,
                &base,
                periods
                    .checked_mul(count)
                    .ok_or_else(|| Error::data("Frequency overflow"))?,
            )
        }
        _ => {
            let periods =
                i32::try_from(periods).map_err(|_| Error::data("Frequency offset is too large"))?;
            let offset = frequency
                .duration()
                .checked_mul(periods)
                .ok_or_else(|| Error::data("Frequency overflow"))?;
            timestamp
                .checked_add_signed(offset)
                .ok_or_else(|| Error::data("Timestamp overflow"))
        }
    }
}

fn last_day(timestamp: DateTime<Utc>) -> Result<u32> {
    let first = timestamp
        .with_day(1)
        .ok_or_else(|| Error::data("Invalid calendar anchor"))?;
    let next = first
        .checked_add_months(Months::new(1))
        .ok_or_else(|| Error::data("Calendar overflow"))?;
    Ok((next - Duration::days(1)).day())
}

#[cfg(test)]
#[path = "calendar_regression_tests.rs"]
mod calendar_regression_tests;
