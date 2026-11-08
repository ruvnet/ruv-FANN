//! A Python-style data contract, not a claim of runtime/API parity with Python.
//! Required columns have the same roles: string series id, timestamp, target.
//! Only the implemented baseline and RNN adapter are used here.
use anyhow::Result;
use neuro_divergent::{
    CrossValidationConfig, Frequency, NaiveForecaster, NeuralForecast, TimeSeriesDataFrame,
    TimeSeriesSchema,
};
use polars::prelude::*;

pub fn run() -> Result<()> {
    let df = df!("store"=>vec!["one";12],"timestamp"=>(0..12).map(|i|1_704_067_200_000_i64+i*86_400_000).collect::<Vec<_>>(),"sales"=>(0..12).map(|i|i as f64).collect::<Vec<_>>())?;
    let data = TimeSeriesDataFrame::<f64>::from_polars(
        df,
        TimeSeriesSchema::new("store", "timestamp", "sales"),
        Some(Frequency::Daily),
    )?;
    let mut nf = NeuralForecast::new(
        vec![Box::new(NaiveForecaster::new("Naive", 2)?)],
        Frequency::Daily,
    )?;
    nf.fit(data.clone())?;
    let future = nf.predict()?;
    assert_eq!(
        future.data.get_column_names_str(),
        vec!["store", "timestamp", "Naive"]
    );
    assert_eq!(future.data.height(), 2);
    let cv = nf.cross_validation(data, CrossValidationConfig::new(2, 2).with_step_size(2))?;
    assert!(cv.data.column("cutoff").is_ok());
    println!("Python-style column migration: forecast and CV contracts verified.");
    Ok(())
}
