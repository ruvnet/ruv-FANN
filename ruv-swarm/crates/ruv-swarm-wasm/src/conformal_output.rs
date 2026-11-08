//! Borrow interval fields directly instead of allocating per-row JSON maps.
//! Field order matches the published default serde_json map encoding.
use ruv_fann::calibration::{CalibrationGroup, IntervalRecord, ResidualCalibrator};
use serde::{ser::SerializeSeq, Serialize, Serializer};

pub(super) struct Groups<'a>(pub &'a ResidualCalibrator<i64>);
impl Serialize for Groups<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(None)?;
        for group in self.0.groups() {
            sequence.serialize_element(&GroupView::from(group))?;
        }
        sequence.end()
    }
}

#[derive(Serialize)]
struct GroupView<'a> {
    first_outcome: i64,
    horizon: usize,
    last_outcome: i64,
    model: &'a str,
    radii: &'a [f64],
    samples: usize,
    series: &'a str,
}
impl<'a> From<&'a CalibrationGroup<i64>> for GroupView<'a> {
    fn from(value: &'a CalibrationGroup<i64>) -> Self {
        Self {
            first_outcome: value.first_outcome,
            horizon: value.horizon,
            last_outcome: value.last_outcome,
            model: &value.model,
            radii: &value.radii,
            samples: value.samples,
            series: &value.series,
        }
    }
}

pub(super) struct Intervals<'a>(pub &'a [IntervalRecord<i64>]);
impl Serialize for Intervals<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.len()))?;
        for interval in self.0 {
            sequence.serialize_element(&IntervalView::from(interval))?;
        }
        sequence.end()
    }
}

#[derive(Serialize)]
struct IntervalView<'a> {
    calibration_as_of: i64,
    calibration_samples: usize,
    coverage: f64,
    forecast_origin: i64,
    horizon: usize,
    lower: f64,
    model: &'a str,
    point: f64,
    series: &'a str,
    timestamp: i64,
    upper: f64,
}
impl<'a> From<&'a IntervalRecord<i64>> for IntervalView<'a> {
    fn from(value: &'a IntervalRecord<i64>) -> Self {
        Self {
            calibration_as_of: value.calibration_as_of,
            calibration_samples: value.calibration_samples,
            coverage: value.coverage,
            forecast_origin: value.forecast_origin,
            horizon: value.horizon,
            lower: value.lower,
            model: &value.model,
            point: value.point,
            series: &value.series,
            timestamp: value.timestamp,
            upper: value.upper,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrowed_intervals_preserve_json_numbers_escaping_and_fields() {
        for point in [0., -0., 2f64.powi(98), 1e-200] {
            let value = IntervalRecord {
                series: "雪\n\"\\".into(),
                model: "m\u{0000}".into(),
                timestamp: 123,
                horizon: 3,
                coverage: 0.9,
                point,
                lower: point - 1.,
                upper: point + 1.,
                calibration_samples: 19,
                forecast_origin: 120,
                calibration_as_of: 100,
            };
            let old = serde_json::json!([{"series":value.series,"model":value.model,
                "timestamp":value.timestamp,"horizon":value.horizon,"coverage":value.coverage,
                "point":value.point,"lower":value.lower,"upper":value.upper,
                "calibration_samples":value.calibration_samples,"forecast_origin":value.forecast_origin,
                "calibration_as_of":value.calibration_as_of}]);
            let actual =
                super::super::wire::serialize(&Intervals(std::slice::from_ref(&value))).unwrap();
            assert_eq!(actual, serde_json::to_string(&old).unwrap());
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&actual).unwrap(),
                old
            );
            // The shipped configuration uses sorted serde_json map keys.
            assert!(actual.starts_with("[{\"calibration_as_of\":100,"));
        }
    }
}
