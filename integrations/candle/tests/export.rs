use candle_core::{Device, Tensor};
use candle_nn::Linear;
use ruv_fann::{ActivationFunction as A, PreparedNetwork};
use ruv_fann_candle::{
    export_dense_f32_json, export_dense_f64_json, prepare_f32, prepare_f64,
    CandleAdapterError as E, LinearStage,
};

#[test]
fn actual_candle_f64_json_round_trip_matches_direct_snapshot(
) -> Result<(), Box<dyn std::error::Error>> {
    let layer = Linear::new(
        Tensor::new(
            &[[1.2345678901234567f64, -0.7], [0.25, 0.125]],
            &Device::Cpu,
        )?,
        Some(Tensor::new(&[0.3f64, -0.25], &Device::Cpu)?),
    );
    let stages = [LinearStage {
        layer: &layer,
        activation: A::Tanh,
        steepness: 0.75,
    }];
    let direct = prepare_f64(&stages)?;
    let json = export_dense_f64_json(&stages)?;
    assert!(json.len() <= 4 * 1024 * 1024);
    let value: serde_json::Value = serde_json::from_str(&json)?;
    assert_eq!(value["schema"], "ruv-fann.dense-layers/1");
    let parsed = PreparedNetwork::<f64>::from_dense_layers_json(&json)?;
    let mut a = [0.; 2];
    let mut b = [0.; 2];
    direct.run_into(&[0.125, -0.375], &mut a, &mut direct.workspace()?)?;
    parsed.run_into(&[0.125, -0.375], &mut b, &mut parsed.workspace()?)?;
    assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
    Ok(())
}

#[test]
fn f32_json_round_trip_uses_the_matching_f32_parser() -> Result<(), Box<dyn std::error::Error>> {
    let layer = Linear::new(Tensor::new(&[[0.1f32, 0.2]], &Device::Cpu)?, None);
    let stages = [LinearStage {
        layer: &layer,
        activation: A::Linear,
        steepness: 1.,
    }];
    let direct = prepare_f32(&stages)?;
    let parsed = PreparedNetwork::<f32>::from_dense_layers_json(&export_dense_f32_json(&stages)?)?;
    let mut a = [0.];
    let mut b = [0.];
    direct.run_into(&[0.3, 0.4], &mut a, &mut direct.workspace()?)?;
    parsed.run_into(&[0.3, 0.4], &mut b, &mut parsed.workspace()?)?;
    assert_eq!(a.map(f32::to_bits), b.map(f32::to_bits));
    Ok(())
}

#[test]
fn large_logical_tensor_json_fails_without_returning_truncated_output(
) -> Result<(), Box<dyn std::error::Error>> {
    let weight =
        Tensor::new(&[[0.12345678901234567f64]], &Device::Cpu)?.broadcast_as((1, 400_000))?;
    let layer = Linear::new(weight, None);
    assert!(matches!(
        export_dense_f64_json(&[LinearStage {
            layer: &layer,
            activation: A::Linear,
            steepness: 1.
        }]),
        Err(E::Budget {
            resource: "dense JSON bytes",
            limit: 4_194_304,
            ..
        })
    ));
    Ok(())
}

#[test]
fn provably_oversized_json_fails_before_reading_nan_storage(
) -> Result<(), Box<dyn std::error::Error>> {
    let weight = Tensor::new(&[[f64::NAN]], &Device::Cpu)?.broadcast_as((1450, 1450))?;
    let layer = Linear::new(weight, None);
    assert!(matches!(
        export_dense_f64_json(&[LinearStage {
            layer: &layer,
            activation: A::Linear,
            steepness: 1.
        }]),
        Err(E::Budget {
            resource: "dense JSON bytes",
            required: 4_205_000,
            limit: 4_194_304
        })
    ));
    Ok(())
}
