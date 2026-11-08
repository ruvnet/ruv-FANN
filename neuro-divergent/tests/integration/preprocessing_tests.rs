//! Train-only scaling, inverse forecasts and causal feature generation.
use super::support::*;
use chrono::{Duration, TimeZone, Utc};
use neuro_divergent_data::preprocessing::{MinMaxScaler, RobustScaler, Scaler, StandardScaler};
use neuro_divergent_data::{features::FeatureEngine, TimeSeriesDatasetBuilder};
use polars::prelude::*;

#[test]
fn fitted_scalers_roundtrip_training_and_unseen_observations() -> TestResult {
    let train = [10., 20., 30., 40., 50.];
    let unseen = [5., 25., 100.];
    let mut scalers: Vec<Box<dyn Scaler<f64>>> = vec![
        Box::new(StandardScaler::new()),
        Box::new(MinMaxScaler::new()),
        Box::new(RobustScaler::new()),
    ];
    for scaler in &mut scalers {
        assert!(!scaler.is_fitted());
        scaler.fit(&train)?;
        let transformed = scaler.transform(&unseen)?;
        let recovered = scaler.inverse_transform(&transformed)?;
        for (a, b) in recovered.iter().zip(unseen) {
            assert!((a - b).abs() < 1e-10);
        }
    }
    Ok(())
}

#[test]
fn scaler_is_fitted_on_past_observations_and_does_not_clip_holdout() -> TestResult {
    let mut scaler = MinMaxScaler::new();
    scaler.fit(&[1., 2., 3.])?;
    assert_eq!(scaler.transform(&[1000.])?, vec![499.5]);
    assert_eq!(scaler.data_max(), Some(3.));
    Ok(())
}

#[test]
fn scaled_forecasting_inverse_transform_returns_original_target_units() -> TestResult {
    let mut data = frame(1, 12);
    let original = values(&data.data, "y");
    let mut scaler = StandardScaler::<f64>::new();
    let scaled = scaler.fit_transform(&original)?;
    data.data.with_column(Series::new("y".into(), scaled))?;
    let mut nf = baseline(3);
    nf.fit(data)?;
    let inverse = scaler.inverse_transform(&values(&nf.predict()?.data, "Naive"))?;
    for y in inverse {
        assert!((y - 11.).abs() < 1e-10);
    }
    Ok(())
}

#[test]
fn scalers_reject_nonfinite_and_empty_training_data() {
    let mut scaler = StandardScaler::<f64>::new();
    assert!(scaler.fit(&[]).is_err());
    assert!(scaler.fit(&[1., f64::NAN]).is_err());
    assert!(scaler.transform(&[1.]).is_err());
}

#[test]
fn causal_lag_and_rolling_features_do_not_see_the_current_target() -> TestResult {
    let start = Utc.timestamp_millis_opt(START).unwrap();
    let make = |outlier| {
        TimeSeriesDatasetBuilder::new("one".into())
            .with_frequency("D".into())
            .with_values(vec![1., 2., 3., outlier, 5., 6.])
            .with_timestamps((0..6).map(|i| start + Duration::days(i)).collect())
            .build()
            .unwrap()
    };
    let engine = FeatureEngine::new()
        .with_lag_features(vec![1])
        .with_rolling_features(vec![3]);
    let a = engine.generate_features(&make(4.))?;
    let b = engine.generate_features(&make(1e9))?;
    assert_eq!(a.timestamps[0], start + Duration::days(3));
    assert_eq!(a.get_feature("lag_1").unwrap()[0], 3.);
    assert_eq!(a.get_feature("rolling_mean_3").unwrap()[0], 2.);
    assert_eq!(
        a.get_feature("lag_1").unwrap()[0],
        b.get_feature("lag_1").unwrap()[0]
    );
    assert_eq!(
        a.get_feature("rolling_mean_3").unwrap()[0],
        b.get_feature("rolling_mean_3").unwrap()[0]
    );
    Ok(())
}

#[test]
fn constant_series_scaling_and_forecasting_remain_finite() -> TestResult {
    let mut scaler = StandardScaler::<f64>::new();
    let scaled = scaler.fit_transform(&[7.; 10])?;
    assert!(scaled.iter().all(|x| x.is_finite()));
    let mut data = frame(1, 10);
    data.data.with_column(Series::new("y".into(), scaled))?;
    let mut nf = baseline(2);
    nf.fit(data)?;
    assert_eq!(
        scaler.inverse_transform(&values(&nf.predict()?.data, "Naive"))?,
        vec![7.; 2]
    );
    Ok(())
}

#[test]
fn data_cleaning_can_be_explicit_without_fitting_on_missing_targets() -> TestResult {
    let mut data = frame(1, 5);
    data.data.with_column(Series::new(
        "y".into(),
        &[Some(1.), None, Some(3.), None, Some(5.)],
    ))?;
    // A caller chooses forward-fill before handing observations to the model.
    let mut previous = 0.;
    let filled: Vec<_> = data
        .data
        .column("y")?
        .f64()?
        .into_iter()
        .map(|x| {
            if let Some(x) = x {
                previous = x;
            }
            previous
        })
        .collect();
    assert_eq!(filled, vec![1., 1., 3., 3., 5.]);
    data.data.with_column(Series::new("y".into(), filled))?;
    let mut nf = baseline(2);
    nf.fit(data)?;
    assert_eq!(values(&nf.predict()?.data, "Naive"), vec![5.; 2]);
    Ok(())
}

#[test]
fn facade_local_scalers_return_each_series_forecasts_in_original_units() -> TestResult {
    use neuro_divergent::{Frequency, NaiveForecaster, NeuralForecast, ScalerType};
    for scaler in ScalerType::all() {
        let mut nf = NeuralForecast::builder()
            .with_model(Box::new(NaiveForecaster::<f64>::new("Naive", 2)?))
            .with_frequency(Frequency::Daily)
            .with_local_scaler(*scaler)
            .build()?;
        nf.fit(frame(3, 12))?;
        let forecast = nf.predict_on(frame(3, 14))?;
        let keyed = rows(&forecast.data, "Naive");
        for id in 0..3 {
            for step in 14..16 {
                assert!(
                    (keyed[&(format!("series_{id}"), START + step * DAY)]
                        - (id as f64 * 100. + 13.))
                        .abs()
                        < 1e-10
                );
            }
        }
    }
    Ok(())
}
