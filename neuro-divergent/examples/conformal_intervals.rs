//! Historical residual intervals evaluated on untouched AirPassengers observations.
//!
//! Numeric observations: R datasets::AirPassengers, monthly international airline
//! passengers (thousands), 1949--1960; Box, Jenkins & Reinsel (1994).
//! https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/AirPassengers.html
//! Values checked against https://vincentarelbundock.github.io/Rdatasets/csv/datasets/AirPassengers.csv
//! The datasets package is part of R (distributed under GPL-2 | GPL-3).
//! Only numerical observations are reproduced here; no R implementation is copied.
//! This single-series evaluation is not evidence of SOTA accuracy or a temporal
//! coverage guarantee. Overlapping calibration windows have dependent errors.
use chrono::{DateTime, Months, TimeZone, Utc};
use neuro_divergent::{
    ConformalCalibrator, ConformalConfig, CrossValidationConfig, Frequency, NaiveForecaster,
    NeuralForecast, TimeSeriesDataFrame, TimeSeriesSchema,
};
use polars::prelude::*;
use std::collections::BTreeMap;

const PASSENGERS: [u32; 144] = [
    112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126, 141, 135, 125, 149, 170,
    170, 158, 133, 114, 140, 145, 150, 178, 163, 172, 178, 199, 199, 184, 162, 146, 166, 171, 180,
    193, 181, 183, 218, 230, 242, 209, 191, 172, 194, 196, 196, 236, 235, 229, 243, 264, 272, 237,
    211, 180, 201, 204, 188, 235, 227, 234, 264, 302, 293, 259, 229, 203, 229, 242, 233, 267, 269,
    270, 315, 364, 347, 312, 274, 237, 278, 284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271,
    306, 315, 301, 356, 348, 355, 422, 465, 467, 404, 347, 305, 336, 340, 318, 362, 348, 363, 435,
    491, 505, 404, 359, 310, 337, 360, 342, 406, 396, 420, 472, 548, 559, 463, 407, 362, 405, 417,
    391, 419, 461, 472, 535, 622, 606, 508, 461, 390, 432,
];
fn month(index: usize) -> DateTime<Utc> {
    Utc.with_ymd_and_hms(1949, 1, 1, 0, 0, 0)
        .unwrap()
        .checked_add_months(Months::new(index as u32))
        .unwrap()
}
fn history(end: usize) -> anyhow::Result<TimeSeriesDataFrame<f64>> {
    Ok(TimeSeriesDataFrame::from_polars(
        df! {
            "unique_id" => vec!["AirPassengers";end],
            "ds" => (0..end).map(|i| month(i).timestamp_millis()).collect::<Vec<_>>(),
            "y" => PASSENGERS[..end].iter().map(|v| *v as f64).collect::<Vec<_>>(),
        }?,
        TimeSeriesSchema::default(),
        Some(Frequency::Monthly),
    )?)
}
#[derive(Default)]
struct Scores {
    n: usize,
    covered: usize,
    width: f64,
    interval_score: f64,
}
fn main() -> anyhow::Result<()> {
    let h = 3;
    let mut pipeline = NeuralForecast::new(
        vec![
            Box::new(NaiveForecaster::<f64>::new("LastValue", h)?),
            Box::new(NaiveForecaster::<f64>::seasonal("Seasonal12", h, 12)?),
        ],
        Frequency::Monthly,
    )?;
    // Model selection and calibration never see the final 12 months (1960).
    // Eighteen past origins give n=18 per model/h: ranks16 and18 for80%/90%.
    let cv = pipeline.cross_validation(
        history(132)?,
        CrossValidationConfig::new(18, h)
            .with_step_size(1)
            .with_refit(true),
    )?;
    let calibration =
        ConformalCalibrator::fit(&cv, ConformalConfig::new(month(131), h, vec![0.8, 0.9])?)?;
    println!("AirPassengers: calibration outcomes through {};18 errors/model/horizon; final12 months held out.", month(131));
    let mut scores: BTreeMap<(String, u32, u32), Scores> = BTreeMap::new();
    for end in (132..144).step_by(h) {
        // Later origins may use earlier observations from the evaluation year;
        // calibration stays frozen. No future evaluation outcome is fitted.
        pipeline.reset()?;
        pipeline.fit(history(end)?)?;
        let points = pipeline.predict()?;
        let origins = [("AirPassengers".into(), month(end - 1))]
            .into_iter()
            .collect();
        let intervals = calibration.apply(&points, &origins)?;
        let table = &intervals.data;
        for row in 0..table.height() {
            let step = table.column("horizon")?.u32()?.get(row).unwrap();
            let model = table.column("model")?.str()?.get(row).unwrap();
            let coverage = table.column("coverage")?.f64()?.get(row).unwrap();
            let lo = table.column("lower")?.f64()?.get(row).unwrap();
            let hi = table.column("upper")?.f64()?.get(row).unwrap();
            let actual = PASSENGERS[end + step as usize - 1] as f64;
            assert_eq!(
                table.column("ds")?.i64()?.get(row),
                Some(month(end + step as usize - 1).timestamp_millis())
            );
            let alpha = 1. - coverage;
            let width = hi - lo;
            let interval_score =
                width + 2. / alpha * (lo - actual).max(0.) + 2. / alpha * (actual - hi).max(0.);
            let score = scores
                .entry((model.into(), step, (coverage * 100.).round() as u32))
                .or_default();
            score.n += 1;
            score.covered += usize::from(actual >= lo && actual <= hi);
            score.width += width;
            score.interval_score += interval_score;
        }
    }
    println!("model,horizon,target_coverage,observations,covered,empirical_coverage,mean_width,mean_interval_score");
    for ((model, horizon, level), s) in scores {
        println!(
            "{model},{horizon},{level}%,{},{},{:.3},{:.3},{:.3}",
            s.n,
            s.covered,
            s.covered as f64 / s.n as f64,
            s.width / s.n as f64,
            s.interval_score / s.n as f64
        );
    }
    println!("Only4 held-out observations per model/horizon: coverage estimates are coarse and may miss their target.");
    Ok(())
}
