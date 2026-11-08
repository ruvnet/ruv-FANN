//! Deterministic shape/resource checks; timings are observations, not speed claims.
#[path = "../support/mod.rs"]
mod support;
use std::time::Instant;
use support::*;

#[test]
fn panel_size_scales_forecast_rows_without_losing_keys() -> TestResult {
    for n in [1, 8, 64, 256] {
        let input = frame(n, 32);
        let started = Instant::now();
        let result = baseline(4).fit_predict(input)?;
        assert_finite_forecast(&result, n, 4);
        assert_eq!(rows(&result.data, "Naive").len(), n * 4);
        eprintln!("{n} series x32 observations: {:?}", started.elapsed());
    }
    Ok(())
}

#[test]
fn input_storage_and_forecast_storage_scale_linearly_with_rows() -> TestResult {
    for n in [64, 512, 4096] {
        let input = frame(1, n);
        let requested = input.data.estimated_size();
        assert!(
            requested < n * 128,
            "columnar input exceeded 128 bytes per observation"
        );
        let mut nf = baseline(8);
        nf.fit(input)?;
        let output = nf.predict()?;
        assert_eq!(output.data.height(), 8);
        assert!(output.data.estimated_size() < 8 * 128);
    }
    Ok(())
}

#[test]
fn many_short_series_and_one_long_series_are_both_supported() -> TestResult {
    for (series, length) in [(1, 32_768), (1024, 16)] {
        let mut nf = baseline(2);
        nf.fit(frame(series, length))?;
        assert_finite_forecast(&nf.predict()?, series, 2);
    }
    Ok(())
}

#[test]
fn serial_and_independent_threaded_workflows_are_equivalent() -> TestResult {
    let expected = baseline(3).fit_predict(frame(8, 32))?;
    let handles: Vec<_> = (0..4)
        .map(|_| std::thread::spawn(|| baseline(3).fit_predict(frame(8, 32)).unwrap()))
        .collect();
    for handle in handles {
        assert_eq!(
            rows(&handle.join().unwrap().data, "Naive"),
            rows(&expected.data, "Naive")
        );
    }
    Ok(())
}

#[cfg(feature = "models")]
#[test]
fn actual_rnn_width_and_horizon_sweep_returns_finite_requested_shapes() -> TestResult {
    use neuro_divergent::models::{BaseModel, TimeSeriesInput};
    for (hidden, horizon) in [(3, 1), (5, 3), (8, 4), (16, 2)] {
        let mut model = rnn(horizon, hidden, 5);
        let data = model_dataset(12, 8, horizon);
        let metrics = model.fit(&data)?;
        assert!(metrics.final_loss.is_finite());
        let inputs: Vec<_> = (0..8).map(|_| TimeSeriesInput::new(vec![0.1; 8])).collect();
        let outputs = model.predict_batch(&inputs)?;
        assert_eq!(outputs.len(), 8);
        for output in outputs {
            assert_eq!(output.forecasts.len(), horizon);
            assert!(output.forecasts.iter().all(|x| x.is_finite()));
        }
    }
    Ok(())
}

#[test]
fn horizon_size_scales_only_the_number_of_returned_future_rows() -> TestResult {
    for h in [1, 7, 28, 128] {
        let result = baseline(h).fit_predict(frame(8, 32))?;
        assert_finite_forecast(&result, 8, h);
        let ts = times(&result.data, "ds");
        assert_eq!(ts.iter().max(), Some(&(START + (31 + h as i64) * DAY)));
    }
    Ok(())
}
