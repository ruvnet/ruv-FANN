//! A dense first layer reads the caller's inputs in place. That must not change
//! a single bit of the result, the error reporting, or the workspace contract.

use ruv_fann::{ActivationFunction, DenseLayer, PreparedError, PreparedNetwork};

fn sigmoid64(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn sigmoid32(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

fn value(seed: &mut u64) -> f64 {
    *seed ^= *seed << 13;
    *seed ^= *seed >> 7;
    *seed ^= *seed << 17;
    ((*seed >> 11) as f64 / (1u64 << 53) as f64) * 4.0 - 2.0
}

#[test]
fn single_layer_with_bias_matches_ordered_arithmetic_bit_for_bit() {
    let mut seed = 0x1234_5678_9ABC_DEF1u64;
    for inputs in [1usize, 2, 7, 10, 33] {
        let w64: Vec<f64> = (0..inputs).map(|_| value(&mut seed)).collect();
        let b64 = [value(&mut seed)];
        let w32: Vec<f32> = w64.iter().map(|&v| v as f32).collect();
        let b32 = [b64[0] as f32];
        let layer64 = [DenseLayer {
            inputs,
            outputs: 1,
            weights: &w64,
            bias: Some(&b64),
            activation: ActivationFunction::Sigmoid,
            steepness: 1.0,
        }];
        let layer32 = [DenseLayer {
            inputs,
            outputs: 1,
            weights: &w32,
            bias: Some(&b32),
            activation: ActivationFunction::Sigmoid,
            steepness: 1.0,
        }];
        let n64 = PreparedNetwork::from_dense_layers(&layer64).unwrap();
        let n32 = PreparedNetwork::from_dense_layers(&layer32).unwrap();
        let (mut ws64, mut ws32) = (n64.workspace().unwrap(), n32.workspace().unwrap());
        for _ in 0..500 {
            let x64: Vec<f64> = (0..inputs).map(|_| value(&mut seed)).collect();
            let x32: Vec<f32> = x64.iter().map(|&v| v as f32).collect();
            let (mut o64, mut o32) = ([0.0f64], [0.0f32]);
            n64.run_into(&x64, &mut o64, &mut ws64).unwrap();
            n32.run_into(&x32, &mut o32, &mut ws32).unwrap();
            let e64 = sigmoid64(x64.iter().zip(&w64).fold(0.0, |s, (a, w)| s + a * w) + b64[0]);
            let e32 = sigmoid32(x32.iter().zip(&w32).fold(0.0, |s, (a, w)| s + a * w) + b32[0]);
            assert_eq!(o64[0].to_bits(), e64.to_bits());
            assert_eq!(o32[0].to_bits(), e32.to_bits());
        }
    }
}

#[test]
fn bias_free_and_two_layer_networks_match_ordered_arithmetic() {
    let mut seed = 0xDEAD_BEEF_0BAD_F00Du64;
    let wh: Vec<f32> = (0..12).map(|_| value(&mut seed) as f32).collect(); // 3 x 4, no bias
    let wo: Vec<f32> = (0..3).map(|_| value(&mut seed) as f32).collect();
    let bo = [value(&mut seed) as f32];
    let net = PreparedNetwork::from_dense_layers(&[
        DenseLayer {
            inputs: 4,
            outputs: 3,
            weights: &wh,
            bias: None,
            activation: ActivationFunction::ReLU,
            steepness: 1.0,
        },
        DenseLayer {
            inputs: 3,
            outputs: 1,
            weights: &wo,
            bias: Some(&bo),
            activation: ActivationFunction::Sigmoid,
            steepness: 1.0,
        },
    ])
    .unwrap();
    let mut ws = net.workspace().unwrap();
    for _ in 0..500 {
        let x: Vec<f32> = (0..4).map(|_| value(&mut seed) as f32).collect();
        let mut out = [0.0f32];
        net.run_into(&x, &mut out, &mut ws).unwrap();
        let hidden: Vec<f32> = (0..3)
            .map(|r| {
                x.iter()
                    .zip(&wh[r * 4..r * 4 + 4])
                    .fold(0.0f32, |s, (a, w)| s + a * w)
                    .max(0.0)
            })
            .collect();
        let expected =
            sigmoid32(hidden.iter().zip(&wo).fold(0.0f32, |s, (a, w)| s + a * w) + bo[0]);
        assert_eq!(out[0].to_bits(), expected.to_bits());
    }
}

#[test]
fn errors_and_output_commit_are_unchanged() {
    let w = [1.0f32, 1.0];
    let b = [0.0f32];
    let net = PreparedNetwork::from_dense_layers(&[DenseLayer {
        inputs: 2,
        outputs: 1,
        weights: &w,
        bias: Some(&b),
        activation: ActivationFunction::Linear,
        steepness: 1.0,
    }])
    .unwrap();
    let mut ws = net.workspace().unwrap();
    let mut out = [7.0f32];
    assert!(matches!(
        net.run_into(&[0.0, f32::NAN], &mut out, &mut ws),
        Err(PreparedError::NonFiniteInput { index: 1 })
    ));
    assert!(matches!(
        net.run_into(&[f32::MAX, f32::MAX], &mut out, &mut ws),
        Err(PreparedError::NumericOverflow {
            layer: 1,
            neuron: 0
        })
    ));
    assert_eq!(
        out[0], 7.0,
        "a failed run must not touch the caller's outputs"
    );
    assert!(matches!(
        net.run_into(&[0.0], &mut out, &mut ws),
        Err(PreparedError::InputSizeMismatch { .. })
    ));
    // The caller's inputs are read in place and never modified.
    let x = [1.5f32, -0.5];
    net.run_into(&x, &mut out, &mut ws).unwrap();
    assert_eq!((out[0], x), (1.0, [1.5, -0.5]));
}
