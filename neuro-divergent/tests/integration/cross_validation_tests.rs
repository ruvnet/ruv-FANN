//! Rolling-origin evaluation uses actual observed holdouts and past-only fits.
use super::support::*;
use neuro_divergent::{CrossValidationConfig, Frequency, NeuralForecast};
use polars::prelude::*;

#[test]
fn rolling_windows_have_exact_cutoffs_observed_targets_and_naive_errors() -> TestResult {
    let mut nf = baseline(2);
    let cv = nf.cross_validation(
        frame(2, 12),
        CrossValidationConfig::new(3, 2).with_step_size(2),
    )?;
    assert_eq!(cv.data.height(), 12);
    assert_eq!(cv.models, vec!["Naive"]);
    assert_eq!(cv.cutoffs.len(), 3);
    let cutoff = times(&cv.data, "cutoff");
    let ds = times(&cv.data, "ds");
    let y = values(&cv.data, "y");
    let predictions = values(&cv.data, "Naive");
    let ids = cv.data.column("unique_id")?.str()?;
    for i in 0..cv.data.height() {
        let id: usize = ids.get(i).unwrap().trim_start_matches("series_").parse()?;
        let t = (ds[i] - START) / DAY;
        assert!(ds[i] > cutoff[i]);
        assert!([1, 2].contains(&((ds[i] - cutoff[i]) / DAY)));
        assert_eq!(y[i], id as f64 * 100.0 + t as f64);
        assert_eq!(
            predictions[i],
            id as f64 * 100.0 + ((cutoff[i] - START) / DAY) as f64
        );
    }
    assert_eq!(
        cv.cutoffs
            .iter()
            .map(|x| x.timestamp_millis())
            .collect::<Vec<_>>(),
        vec![START + 5 * DAY, START + 7 * DAY, START + 9 * DAY]
    );
    Ok(())
}

#[test]
fn no_refit_cannot_reuse_weights_fitted_on_future_observations() -> TestResult {
    let mut data = frame(1, 12);
    data.data.with_column(Series::new(
        "y".into(),
        vec![0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 1000., 2000.],
    ))?;
    let mut nf = NeuralForecast::new(
        vec![Box::new(DriftForecaster::new("Drift", 2))],
        Frequency::Daily,
    )?;
    nf.fit(data.clone())?; // Deliberately contaminate its pre-CV fit.
    let cv = nf.cross_validation(
        data,
        CrossValidationConfig::new(3, 2)
            .with_step_size(2)
            .with_refit(false),
    )?;
    // Initial past ends at index5: fitted drift=1, despite huge final targets.
    let cutoff = times(&cv.data, "cutoff");
    let ds = times(&cv.data, "ds");
    let p = values(&cv.data, "Drift");
    for i in 0..p.len() {
        assert_eq!(
            p[i],
            ((cutoff[i] - START) / DAY) as f64 + ((ds[i] - cutoff[i]) / DAY) as f64
        );
    }
    Ok(())
}

#[test]
fn explicit_test_size_matches_equivalent_window_definition() -> TestResult {
    let mut a = baseline(3);
    let mut b = baseline(3);
    let config = CrossValidationConfig::new(2, 3).with_step_size(2);
    let expected = a.cross_validation(frame(1, 16), config.clone())?;
    let actual = b.cross_validation(frame(1, 16), config.with_test_size(5))?;
    assert!(expected.data.equals_missing(&actual.data));
    Ok(())
}

#[test]
fn overlapping_windows_preserve_duplicate_observation_keys_across_cutoffs() -> TestResult {
    let mut nf = baseline(3);
    let cv = nf.cross_validation(
        frame(1, 10),
        CrossValidationConfig::new(3, 3).with_step_size(1),
    )?;
    assert_eq!(cv.data.height(), 9);
    assert_eq!(
        times(&cv.data, "ds")
            .into_iter()
            .filter(|t| *t == START + 7 * DAY)
            .count(),
        3
    );
    assert_eq!(cv.cutoffs.len(), 3);
    Ok(())
}

#[test]
fn unequal_series_lengths_use_their_own_observed_cutoffs() -> TestResult {
    let mut data = frame(2, 12);
    let mask = data
        .data
        .column("unique_id")?
        .str()?
        .into_iter()
        .zip(times(&data.data, "ds"))
        .map(|(id, t)| id != Some("series_1") || t < START + 9 * DAY)
        .collect::<Vec<_>>();
    data.data = data
        .data
        .filter(&BooleanChunked::from_slice("".into(), &mask))?;
    let mut nf = baseline(2);
    let cv = nf.cross_validation(data, CrossValidationConfig::new(2, 2).with_step_size(2))?;
    assert_eq!(cv.data.height(), 8);
    let ids = cv.data.column("unique_id")?.str()?;
    let cutoffs = times(&cv.data, "cutoff");
    for (i, id) in ids.into_iter().enumerate() {
        let expected = if id == Some("series_0") {
            [7, 9]
        } else {
            [4, 6]
        };
        assert!(expected.contains(&((cutoffs[i] - START) / DAY)));
    }
    Ok(())
}

#[test]
fn cross_validation_rejects_invalid_or_unavailable_windows() {
    for config in [
        CrossValidationConfig::new(0, 2),
        CrossValidationConfig::new(1, 0),
        CrossValidationConfig::new(1, 2).with_step_size(0),
        CrossValidationConfig::new(4, 2).with_test_size(2),
        CrossValidationConfig::new(1, 3),
    ] {
        assert!(baseline(2).cross_validation(frame(1, 6), config).is_err());
    }
    assert!(baseline(2)
        .cross_validation(frame(1, 3), CrossValidationConfig::new(3, 2))
        .is_err());
}

#[test]
fn cv_outputs_can_be_exported_and_grouped_by_real_cutoff() -> TestResult {
    let cv = baseline(2).cross_validation(frame(1, 12), CrossValidationConfig::new(2, 2))?;
    let one = cv.get_cutoff_results(cv.cutoffs[0])?;
    assert_eq!(one.height(), 2);
    let dir = tempfile::tempdir()?;
    let path = dir.path().join("cv.csv");
    cv.to_csv(&path)?;
    let text = std::fs::read_to_string(path)?;
    assert!(text.lines().next().unwrap().contains("cutoff"));
    assert_eq!(text.lines().count(), 5);
    Ok(())
}

#[test]
fn metrics_by_model_and_summary_are_computed_from_observed_holdouts() -> TestResult {
    use neuro_divergent::NaiveForecaster;
    let mut nf = NeuralForecast::new(
        vec![
            Box::new(NaiveForecaster::<f64>::new("Naive", 2)?),
            Box::new(DriftForecaster::new("Drift", 2)),
        ],
        Frequency::Daily,
    )?;
    let cv = nf.cross_validation(
        frame(2, 12),
        CrossValidationConfig::new(3, 2).with_step_size(2),
    )?;
    let metrics = cv.metrics_by_model()?;
    assert_eq!(metrics["Naive"].mae, 1.5);
    assert_eq!(metrics["Naive"].mse, 2.5);
    assert_eq!(metrics["Drift"].mse, 0.);
    let summary = cv.summary_stats()?;
    assert_eq!(summary.height(), 2);
    assert!(summary.column("model").is_ok());
    assert_eq!(
        values(&summary, "mse").iter().copied().fold(0., f64::max),
        2.5
    );
    Ok(())
}
