use super::*;

fn point_losses() -> Vec<Box<dyn LossFunction<f64>>> {
    vec![
        Box::new(MAELoss::new()),
        Box::new(MSELoss::new()),
        Box::new(RMSELoss::new()),
        Box::new(MAPELoss::new()),
        Box::new(SMAPELoss::new()),
        Box::new(MASELoss::new(1.0)),
        Box::new(PinballLoss::new(0.5)),
        Box::new(HuberLoss::new(1.0)),
        Box::new(QuantileLoss::new(vec![0.5])),
        Box::new(ScaledLoss::new(Loss::MSE(MSELoss::new()), 2.0)),
        Box::new(SeasonalLoss::new(Loss::MSE(MSELoss::new()), vec![1.0, 0.5])),
    ]
}

#[test]
fn public_point_losses_reject_empty_mismatched_and_nonfinite_inputs() {
    for loss in point_losses() {
        for (prediction, target) in [
            (vec![], vec![]),
            (vec![1.0, 2.0], vec![1.0]),
            (vec![f64::NAN], vec![1.0]),
            (vec![1.0], vec![f64::INFINITY]),
        ] {
            assert!(
                loss.forward(&prediction, &target).is_err(),
                "{} forward",
                loss.name()
            );
            assert!(
                loss.backward(&prediction, &target).is_err(),
                "{} backward",
                loss.name()
            );
        }
    }
}

#[test]
fn distribution_losses_validate_shapes_values_and_domains_before_indexing() {
    let losses: Vec<Box<dyn LossFunction<f64>>> = vec![
        Box::new(NegativeLogLikelihoodLoss::new()),
        Box::new(GaussianNLLLoss::new()),
        Box::new(CRPSLoss::new()),
    ];
    for loss in losses {
        for (prediction, target) in [
            (vec![], vec![]),
            (vec![0.1], vec![0.2]),
            (vec![0.1, f64::NAN], vec![0.2]),
            (vec![0.1, 1.0], vec![f64::INFINITY]),
        ] {
            assert!(loss.forward(&prediction, &target).is_err());
            assert!(loss.backward(&prediction, &target).is_err());
        }
    }
    for std in [0.0, -1.0] {
        assert!(CRPSLoss::new().forward(&[0.1, std], &[0.2]).is_err());
        assert!(CRPSLoss::new().backward(&[0.1, std], &[0.2]).is_err());
    }
    assert!(PoissonNLLLoss::new().forward(&[0.1], &[-1.0]).is_err());
    assert!(NegativeBinomialNLLLoss::new(0.0)
        .backward(&[0.1], &[1.0])
        .is_err());
}

#[test]
fn loss_parameters_and_overflow_are_checked() {
    let losses: Vec<Box<dyn LossFunction<f64>>> = vec![
        Box::new(MAPELoss::with_epsilon(0.0)),
        Box::new(SMAPELoss::with_epsilon(f64::NAN)),
        Box::new(MASELoss::new(-1.0)),
        Box::new(PinballLoss::new(1.5)),
        Box::new(HuberLoss::new(0.0)),
        Box::new(QuantileLoss::new(vec![])),
        Box::new(QuantileLoss::new(vec![f64::NAN])),
        Box::new(ScaledLoss::new(Loss::MSE(MSELoss::new()), 0.0)),
        Box::new(SeasonalLoss::new(Loss::MSE(MSELoss::new()), vec![f64::NAN])),
    ];
    for loss in losses {
        assert!(
            loss.forward(&[0.1], &[0.2]).is_err(),
            "{} forward",
            loss.name()
        );
        assert!(
            loss.backward(&[0.1], &[0.2]).is_err(),
            "{} backward",
            loss.name()
        );
    }
    assert!(MSELoss::new().forward(&[f64::MAX], &[-f64::MAX]).is_err());
    assert!(MSELoss::new().backward(&[f64::MAX], &[-f64::MAX]).is_err());
}

#[test]
fn point_loss_and_gaussian_crps_gradients_match_finite_differences() {
    let mut losses = point_losses();
    losses.push(Box::new(CRPSLoss::new()));
    losses.push(Box::new(GaussianNLLLoss::new()));
    for loss in losses {
        let target = if matches!(loss.name(), "CRPS" | "GaussianNLL") {
            vec![0.7]
        } else {
            vec![0.7, -0.4]
        };
        let predictions = vec![1.2, 0.8];
        let gradients = loss.backward(&predictions, &target).unwrap();
        for index in 0..predictions.len() {
            let mut perturbed = predictions.clone();
            perturbed[index] += 1e-6;
            let plus = loss.forward(&perturbed, &target).unwrap();
            perturbed[index] -= 2e-6;
            let minus = loss.forward(&perturbed, &target).unwrap();
            let numerical = (plus - minus) / 2e-6;
            assert!(
                (gradients[index] - numerical).abs() < 3e-6,
                "{} index={index} analytical={} numerical={numerical}",
                loss.name(),
                gradients[index]
            );
        }
    }
}

#[test]
fn percentage_losses_use_zero_subgradients_at_an_exact_fit() {
    for loss in [
        Box::new(MAPELoss::new()) as Box<dyn LossFunction<f64>>,
        Box::new(SMAPELoss::new()),
        Box::new(MASELoss::new(1.0)),
    ] {
        assert_eq!(
            loss.backward(&[0.0, 1.0], &[0.0, 1.0]).unwrap(),
            vec![0.0, 0.0]
        );
    }
}

#[test]
fn smape_differentiates_the_constant_epsilon_denominator() {
    let loss = SMAPELoss::<f64>::with_epsilon(1e-8);
    let gradient = loss.backward(&[3e-10], &[1e-10]).unwrap()[0];
    assert!((gradient / 1e10 - 1.0).abs() < 1e-14);
}
