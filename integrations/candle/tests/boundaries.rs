use candle_core::{DType, Device, Tensor};
use candle_nn::Linear;
use ruv_fann::{ActivationFunction as A, PreparationLimits};
use ruv_fann_candle::{
    prepare_f32, prepare_f32_with_limits, prepare_f64, CandleAdapterError as E, LinearStage,
};
type TestResult = Result<(), Box<dyn std::error::Error>>;

fn stage(layer: &Linear) -> LinearStage<'_, f32> {
    LinearStage {
        layer,
        activation: A::Linear,
        steepness: 1.,
    }
}

#[test]
fn wrong_rank_zero_dimensions_and_bias_shapes_fail() -> TestResult {
    let rank = Linear::new(Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?, None);
    assert!(matches!(
        prepare_f32(&[stage(&rank)]),
        Err(E::Invalid { stage: 0, .. })
    ));
    let zero = Linear::new(
        Tensor::from_vec(Vec::<f32>::new(), (0, 3), &Device::Cpu)?,
        None,
    );
    assert!(matches!(
        prepare_f32(&[stage(&zero)]),
        Err(E::Invalid { .. })
    ));
    for bias in [
        Tensor::new(&[1f32, 2., 3.], &Device::Cpu)?,
        Tensor::new(&[[1f32, 2.]], &Device::Cpu)?,
    ] {
        let layer = Linear::new(
            Tensor::new(&[[1f32, 2.], [3., 4.]], &Device::Cpu)?,
            Some(bias),
        );
        assert!(matches!(
            prepare_f32(&[stage(&layer)]),
            Err(E::Invalid { .. })
        ));
    }
    Ok(())
}

#[test]
fn strict_precision_rejects_weight_and_bias_casting() -> TestResult {
    let f64_weight = Linear::new(Tensor::new(&[[1f64, 2.]], &Device::Cpu)?, None);
    assert!(matches!(
        prepare_f32(&[stage(&f64_weight)]),
        Err(E::DType {
            expected: DType::F32,
            actual: DType::F64,
            ..
        })
    ));
    let f32_weight = Linear::new(Tensor::new(&[[1f32, 2.]], &Device::Cpu)?, None);
    assert!(matches!(
        prepare_f64(&[LinearStage {
            layer: &f32_weight,
            activation: A::Linear,
            steepness: 1.
        }]),
        Err(E::DType {
            expected: DType::F64,
            actual: DType::F32,
            ..
        })
    ));
    let mixed = Linear::new(
        Tensor::new(&[[1f32, 2.]], &Device::Cpu)?,
        Some(Tensor::new(&[0f64], &Device::Cpu)?),
    );
    assert!(matches!(
        prepare_f32(&[stage(&mixed)]),
        Err(E::DType { tensor: "bias", .. })
    ));
    Ok(())
}

#[test]
fn all_stage_metadata_is_checked_before_any_numeric_copy() -> TestResult {
    let first = Linear::new(Tensor::new(&[[f32::NAN]], &Device::Cpu)?, None);
    let second = Linear::new(Tensor::new(&[[1f64]], &Device::Cpu)?, None);
    assert!(matches!(
        prepare_f32(&[stage(&first), stage(&second)]),
        Err(E::DType { stage: 1, .. })
    ));
    let wrong_inputs = Linear::new(Tensor::new(&[[1f32, 2.]], &Device::Cpu)?, None);
    assert!(matches!(
        prepare_f32(&[stage(&first), stage(&wrong_inputs)]),
        Err(E::Invalid { stage: 1, .. })
    ));
    Ok(())
}

#[test]
fn resource_preflight_rejects_large_broadcasts_before_reading_nan_storage() -> TestResult {
    let backing = Tensor::new(&[[f32::NAN]], &Device::Cpu)?;
    let layer = Linear::new(backing.broadcast_as((2, 4096))?, None);
    let limit = PreparationLimits {
        max_bytes: 128,
        ..PreparationLimits::default()
    };
    assert!(matches!(
        prepare_f32_with_limits(&[stage(&layer)], limit),
        Err(E::Budget {
            resource: "Candle temporary and packed requested bytes",
            ..
        })
    ));
    let limit = PreparationLimits {
        max_connections: 10,
        ..PreparationLimits::default()
    };
    assert!(matches!(
        prepare_f32_with_limits(&[stage(&layer)], limit),
        Err(E::Budget {
            resource: "Candle connections",
            ..
        })
    ));
    let limit = PreparationLimits {
        max_neurons: 10,
        ..PreparationLimits::default()
    };
    assert!(matches!(
        prepare_f32_with_limits(&[stage(&layer)], limit),
        Err(E::Budget {
            resource: "Candle neurons",
            ..
        })
    ));
    Ok(())
}

#[test]
fn numerical_parameters_and_empty_oversized_stacks_are_rejected() -> TestResult {
    assert!(matches!(prepare_f32(&[]), Err(E::Invalid { .. })));
    let layer = Linear::new(Tensor::new(&[[1f32]], &Device::Cpu)?, None);
    let stages = vec![stage(&layer); 65];
    assert!(matches!(
        prepare_f32(&stages),
        Err(E::Budget {
            resource: "Candle stages",
            ..
        })
    ));
    assert!(matches!(
        prepare_f32(&[LinearStage {
            steepness: f32::INFINITY,
            ..stage(&layer)
        }]),
        Err(E::Invalid { .. })
    ));
    let nan = Linear::new(Tensor::new(&[[1f32, f32::NAN]], &Device::Cpu)?, None);
    assert!(matches!(
        prepare_f32(&[stage(&nan)]),
        Err(E::NonFinite {
            tensor: "weight",
            index: 1,
            ..
        })
    ));
    let inf = Linear::new(
        Tensor::new(&[[1f32]], &Device::Cpu)?,
        Some(Tensor::new(&[f32::INFINITY], &Device::Cpu)?),
    );
    assert!(matches!(
        prepare_f32(&[stage(&inf)]),
        Err(E::NonFinite {
            tensor: "bias",
            index: 0,
            ..
        })
    ));
    Ok(())
}
