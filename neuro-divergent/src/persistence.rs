//! Versioned, bounded checkpoints for supported fitted models and training context.
use crate::timeline::{timestamp_column, Timeline};
use crate::{
    BaseModel, Frequency, NaiveForecaster, NeuralForecast, NeuroDivergentError as Error,
    NeuroDivergentResult as Result, ScalerType, TimeSeriesDataFrame, TimeSeriesSchema,
};
use chrono::{DateTime, SecondsFormat, Utc};
use num_traits::Float;
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
};

const LIMIT: usize = 64 * 1024 * 1024;
static COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Serialize, Deserialize)]
struct Snapshot {
    version: u32,
    numeric_epsilon: u64,
    frequency: Frequency,
    schema: TimeSeriesSchema,
    timestamp_unit: String,
    time_zone: Option<String>,
    rows: Vec<Row>,
    models: Vec<SavedModel>,
    scaler_type: Option<ScalerType>,
    scales: crate::scaling::Scales,
}
#[derive(Serialize, Deserialize)]
struct Row {
    id: String,
    timestamp: String,
    value: f64,
}
#[derive(Serialize, Deserialize)]
struct SavedModel {
    name: String,
    horizon: usize,
    kind: String,
    bytes: Vec<u8>,
}

pub(crate) fn save<T: Float + Send + Sync>(
    pipeline: &NeuralForecast<T>,
    path: &Path,
) -> Result<()> {
    if !pipeline.is_fitted() {
        return Err(Error::serialization("Only fitted pipelines can be saved"));
    }
    let history = pipeline
        .training_data
        .as_ref()
        .ok_or_else(|| Error::serialization("Missing training context"))?;
    if !history.schema.static_features.is_empty() || !history.schema.exogenous_features.is_empty() {
        return Err(Error::serialization(
            "Checkpoint mapping for exogenous/static schemas is unsupported",
        ));
    }
    let mut models = Vec::with_capacity(pipeline.models.len());
    for model in &pipeline.models {
        let kind = model.model_type().to_ascii_lowercase();
        if !matches!(kind.as_str(), "naive" | "rnn") {
            return Err(Error::serialization(
                "Persistence is unsupported for this custom model",
            ));
        }
        let bytes = model.save_state()?;
        if bytes.len() > LIMIT {
            return Err(Error::serialization("Model checkpoint exceeds 64 MiB"));
        }
        models.push(SavedModel {
            name: model.name().to_owned(),
            horizon: model.horizon(),
            kind,
            bytes,
        });
    }
    let timeline = Timeline::read(history, &pipeline.frequency)?;
    let (timestamp_unit, time_zone) = match &timeline.timestamp_type {
        DataType::Datetime(TimeUnit::Nanoseconds, zone) => {
            ("ns", zone.as_ref().map(ToString::to_string))
        }
        DataType::Datetime(TimeUnit::Microseconds, zone) => {
            ("us", zone.as_ref().map(ToString::to_string))
        }
        DataType::Datetime(TimeUnit::Milliseconds, zone) => {
            ("ms", zone.as_ref().map(ToString::to_string))
        }
        DataType::Date => ("date", None),
        DataType::Int32 => ("integer32", None),
        _ => ("integer64", None),
    };
    let rows = timeline
        .series
        .values()
        .flat_map(|series| {
            series
                .times
                .iter()
                .zip(&series.values)
                .map(|(time, value)| Row {
                    id: series.id.clone(),
                    timestamp: time.to_rfc3339_opts(SecondsFormat::Nanos, true),
                    value: *value,
                })
        })
        .collect();
    let state = Snapshot {
        version: 1,
        numeric_epsilon: epsilon::<T>()?,
        frequency: pipeline.frequency.clone(),
        schema: history.schema.clone(),
        timestamp_unit: timestamp_unit.into(),
        time_zone,
        rows,
        models,
        scaler_type: pipeline.local_scaler_type,
        scales: pipeline.scales.clone(),
    };
    let bytes = serde_json::to_vec(&state).map_err(|e| Error::serialization(e.to_string()))?;
    if bytes.len() > LIMIT {
        return Err(Error::serialization("Pipeline checkpoint exceeds 64 MiB"));
    }
    atomic_write(path, &bytes)
}

pub(crate) fn load<T>(path: &Path) -> Result<NeuralForecast<T>>
where
    T: Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static,
{
    let file = open_checkpoint(path)?;
    let mut bytes = Vec::new();
    file.take((LIMIT + 1) as u64).read_to_end(&mut bytes)?;
    if bytes.len() > LIMIT {
        return Err(Error::serialization("Pipeline checkpoint exceeds 64 MiB"));
    }
    let snapshot: Snapshot =
        serde_json::from_slice(&bytes).map_err(|e| Error::serialization(e.to_string()))?;
    if snapshot.version != 1 || snapshot.numeric_epsilon != epsilon::<T>()? {
        return Err(Error::serialization(
            "Unsupported checkpoint version or numeric precision",
        ));
    }
    let mut models: Vec<Box<dyn BaseModel<T>>> = Vec::new();
    for saved in snapshot.models {
        let model: Box<dyn BaseModel<T>> = match saved.kind.as_str() {
            "naive" => Box::new(NaiveForecaster::<T>::from_state(&saved.bytes)?),
            #[cfg(feature = "models")]
            "rnn" => Box::new(crate::ModelAdapter::<
                neuro_divergent_models::models::RNN<T>,
                T,
            >::from_state(&saved.bytes)?),
            _ => {
                return Err(Error::serialization(
                    "Unsupported model checkpoint implementation",
                ))
            }
        };
        if model.name() != saved.name || model.horizon() != saved.horizon || !model.is_fitted() {
            return Err(Error::serialization(
                "Checkpoint model metadata/state mismatch",
            ));
        }
        models.push(model);
    }
    let dtype = match snapshot.timestamp_unit.as_str() {
        "ns" => DataType::Datetime(TimeUnit::Nanoseconds, snapshot.time_zone.map(Into::into)),
        "us" => DataType::Datetime(TimeUnit::Microseconds, snapshot.time_zone.map(Into::into)),
        "ms" => DataType::Datetime(TimeUnit::Milliseconds, snapshot.time_zone.map(Into::into)),
        "date" => DataType::Date,
        "integer32" => DataType::Int32,
        "integer64" => DataType::Int64,
        _ => {
            return Err(Error::serialization(
                "Unknown checkpoint timestamp representation",
            ))
        }
    };
    let mut ids = Vec::with_capacity(snapshot.rows.len());
    let mut times = Vec::with_capacity(snapshot.rows.len());
    let mut values = Vec::with_capacity(snapshot.rows.len());
    for row in snapshot.rows {
        ids.push(row.id);
        times.push(
            DateTime::parse_from_rfc3339(&row.timestamp)
                .map_err(|e| Error::serialization(e.to_string()))?
                .with_timezone(&Utc),
        );
        values.push(row.value);
    }
    let mut time_column = timestamp_column(&snapshot.schema.ds_col, &times, &dtype)?;
    if matches!(dtype, DataType::Date | DataType::Int32) {
        time_column = time_column
            .cast(&dtype)
            .map_err(|e| Error::serialization(e.to_string()))?;
    }
    let frame = DataFrame::new(vec![
        Series::new(snapshot.schema.unique_id_col.clone().into(), ids).into(),
        time_column,
        Series::new(snapshot.schema.y_col.clone().into(), values).into(),
    ])
    .map_err(|e| Error::serialization(e.to_string()))?;
    let history =
        TimeSeriesDataFrame::from_polars(frame, snapshot.schema, Some(snapshot.frequency.clone()))?;
    let timeline = Timeline::read(&history, &snapshot.frequency)?;
    if snapshot.scales.len() != timeline.series.len()
        || snapshot.scales.iter().any(|(id, scale)| {
            !timeline.series.contains_key(id)
                || !scale.location.is_finite()
                || !scale.scale.is_finite()
                || scale.scale <= 0.0
        })
    {
        return Err(Error::serialization("Invalid checkpoint scaler state"));
    }
    let mut pipeline = NeuralForecast::new(models, snapshot.frequency)?;
    if history.schema.all_columns().iter().any(|column| {
        pipeline
            .models
            .iter()
            .any(|model| model.name() == column.as_str())
    }) {
        return Err(Error::serialization(
            "Checkpoint model name collides with data schema",
        ));
    }
    pipeline.local_scaler_type = snapshot.scaler_type;
    pipeline.training_schema = Some(history.schema.clone());
    pipeline.training_data = Some(history);
    pipeline.scales = snapshot.scales;
    pipeline.is_fitted = true;
    Ok(pipeline)
}

fn epsilon<T: Float>() -> Result<u64> {
    T::epsilon()
        .to_f64()
        .map(f64::to_bits)
        .ok_or_else(|| Error::serialization("Unknown numeric precision"))
}

fn validate_file(metadata: &std::fs::Metadata) -> Result<()> {
    if !metadata.is_file() {
        return Err(Error::serialization("Checkpoint must be a regular file"));
    }
    if metadata.len() > LIMIT as u64 {
        return Err(Error::serialization("Pipeline checkpoint exceeds 64 MiB"));
    }
    Ok(())
}

fn open_checkpoint(path: &Path) -> Result<File> {
    // Reject static special files before opening on every platform.
    validate_file(&std::fs::metadata(path)?)?;
    open_after_preflight(path)
}

fn open_after_preflight(path: &Path) -> Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        // A checked regular path may be replaced by a FIFO before open. A
        // nonblocking open followed by fstat rejects that race without hanging.
        options.custom_flags(libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    validate_file(&file.metadata()?)?;
    Ok(file)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .ok_or_else(|| Error::serialization("Checkpoint needs a filename"))?
        .to_string_lossy();
    for _ in 0..8 {
        let temp = parent.join(format!(
            ".{name}.{}.{}.tmp",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = match options.open(&temp) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(error.into()),
        };
        let result = (|| {
            file.write_all(bytes)?;
            file.sync_all()?;
            std::fs::rename(&temp, path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&temp);
        }
        return result;
    }
    Err(Error::serialization(
        "Cannot allocate a unique checkpoint temporary file",
    ))
}

#[cfg(test)]
#[path = "persistence_regression_tests.rs"]
mod regression_tests;
