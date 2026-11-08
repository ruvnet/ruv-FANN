//! Persistence checks compare forecasts, state and independent continued use.
use super::support::*;
use neuro_divergent::{Frequency, NaiveForecaster, NeuralForecast};
use std::fs;

#[test]
fn fitted_baseline_roundtrip_preserves_history_and_predictions() -> TestResult {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("baseline.json");
    let mut nf = baseline(3);
    nf.fit(frame(3, 12))?;
    let expected = nf.predict()?;
    nf.save(&path)?;
    assert!(path.is_file());
    let loaded = NeuralForecast::<f64>::load(&path)?;
    assert!(loaded.is_fitted());
    assert_eq!(loaded.frequency(), Frequency::Daily);
    assert_eq!(loaded.model_names(), nf.model_names());
    assert_eq!(
        rows(&loaded.predict()?.data, "Naive"),
        rows(&expected.data, "Naive")
    );
    assert_eq!(
        values(&loaded.predict_on(frame(3, 14))?.data, "Naive"),
        values(&nf.predict_on(frame(3, 14))?.data, "Naive")
    );
    Ok(())
}

#[test]
fn ensemble_roundtrip_restores_each_supported_model() -> TestResult {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("ensemble.json");
    let mut nf = NeuralForecast::new(
        vec![
            Box::new(NaiveForecaster::<f64>::new("Short", 2)?),
            Box::new(NaiveForecaster::<f64>::new("Other", 2)?),
        ],
        Frequency::Daily,
    )?;
    nf.fit(frame(2, 8))?;
    nf.save(&path)?;
    let loaded = NeuralForecast::<f64>::load(path)?;
    for name in nf.model_names() {
        assert_eq!(
            rows(&loaded.predict()?.data, &name),
            rows(&nf.predict()?.data, &name)
        );
    }
    Ok(())
}

#[test]
fn loaded_state_can_be_reset_and_refitted_without_changing_original() -> TestResult {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("incremental.json");
    let mut original = baseline(2);
    original.fit(frame(1, 10))?;
    original.save(&path)?;
    let mut loaded = NeuralForecast::<f64>::load(path)?;
    loaded.reset()?;
    loaded.fit(frame(1, 14))?;
    assert_eq!(values(&loaded.predict()?.data, "Naive"), vec![13.; 2]);
    assert_eq!(values(&original.predict()?.data, "Naive"), vec![9.; 2]);
    Ok(())
}

#[test]
fn malformed_and_missing_snapshots_are_rejected() -> TestResult {
    let directory = tempfile::tempdir()?;
    assert!(NeuralForecast::<f64>::load(directory.path().join("missing.json")).is_err());
    for (index, bytes) in [b"".as_slice(), b"not-json", b"{}", b"{\"version\":999999}"]
        .into_iter()
        .enumerate()
    {
        let path = directory.path().join(format!("corrupt-{index}.json"));
        fs::write(&path, bytes)?;
        assert!(NeuralForecast::<f64>::load(&path).is_err());
    }
    Ok(())
}

#[test]
fn unsupported_custom_model_save_fails_without_a_misleading_snapshot() -> TestResult {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("unsupported.json");
    let mut nf = NeuralForecast::new(
        vec![Box::new(DriftForecaster::new("Drift", 2))],
        Frequency::Daily,
    )?;
    nf.fit(frame(1, 8))?;
    assert!(nf.save(&path).is_err());
    assert!(!path.exists());
    assert_finite_forecast(&nf.predict()?, 1, 2);
    Ok(())
}

#[test]
fn simultaneous_snapshots_have_independent_paths_and_predictions() -> TestResult {
    let directory = tempfile::tempdir()?;
    let paths: Vec<_> = (0..4)
        .map(|i| directory.path().join(format!("model-{i}.json")))
        .collect();
    std::thread::scope(|scope| {
        let tasks: Vec<_> = paths
            .iter()
            .enumerate()
            .map(|(i, path)| {
                scope.spawn(move || {
                    let mut nf = baseline(2);
                    nf.fit(frame(1, 8 + i)).unwrap();
                    nf.save(path).unwrap();
                })
            })
            .collect();
        for task in tasks {
            task.join().unwrap();
        }
    });
    for (i, path) in paths.iter().enumerate() {
        let loaded = NeuralForecast::<f64>::load(path)?;
        assert_eq!(
            values(&loaded.predict()?.data, "Naive"),
            vec![7.0 + i as f64; 2]
        );
    }
    Ok(())
}

#[cfg(feature = "models")]
#[test]
fn real_trained_rnn_snapshot_preserves_forecasts_and_fitted_state() -> TestResult {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("rnn.json");
    let mut nf = neural(3);
    nf.fit(frame(2, 16))?;
    let expected = nf.predict()?;
    nf.save(&path)?;
    let loaded = NeuralForecast::<f64>::load(path)?;
    assert!(loaded.get_model("RNN").unwrap().is_fitted());
    let actual = loaded.predict()?;
    assert_finite_forecast(&actual, 2, 3);
    let expected = rows(&expected.data, "RNN");
    let actual = rows(&actual.data, "RNN");
    assert_eq!(
        expected.keys().collect::<Vec<_>>(),
        actual.keys().collect::<Vec<_>>()
    );
    for (key, value) in expected {
        assert!((value - actual[&key]).abs() < 1e-12);
    }
    Ok(())
}

#[test]
fn valid_snapshot_version_and_model_metadata_are_checked_on_load() -> TestResult {
    let directory = tempfile::tempdir()?;
    let path = directory.path().join("valid.json");
    let mut nf = baseline(2);
    nf.fit(frame(1, 8))?;
    nf.save(&path)?;
    let original: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
    let mut wrong_version = original.clone();
    wrong_version["version"] = serde_json::json!(u32::MAX);
    let changed = directory.path().join("wrong-version.json");
    fs::write(&changed, serde_json::to_vec(&wrong_version)?)?;
    assert!(NeuralForecast::<f64>::load(&changed).is_err());
    let mut wrong_name = original;
    wrong_name["models"][0]["name"] = serde_json::json!("Different");
    fs::write(&changed, serde_json::to_vec(&wrong_name)?)?;
    assert!(NeuralForecast::<f64>::load(changed).is_err());
    assert!(NeuralForecast::<f64>::load(path).is_ok());
    Ok(())
}
