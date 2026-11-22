use ruv_fann_wsp_quant_guard::{
    solve_one_coordinate, Edit, IntBox, Ledger, Model, Patch, Protected, Target,
};

fn model(width: usize, scale: i32) -> Model {
    Model {
        weights: vec![vec![0; width]],
        bias: vec![0],
        scales: vec![scale],
        output_min: vec![-128],
        output_max: vec![127],
        weight_min: -128,
        weight_max: 127,
        prefix_id: "synthetic-frozen-prefix-v1".into(),
    }
}

fn anchors(width: usize) -> Protected {
    Protected {
        points: (0..width)
            .map(|j| {
                let mut x = vec![0; width];
                x[j] = 1;
                x
            })
            .collect(),
        boxes: vec![],
    }
}

fn patch(ledger: &Ledger, col: usize, value: i32) -> Patch {
    Patch {
        expected_parent: ledger.head().into(),
        edits: vec![Edit { row: 0, col, value }],
    }
}

#[test]
fn floor_is_not_truncation_for_negative_accumulators() {
    let mut m = model(1, 4);
    m.weights[0][0] = 1;
    assert_eq!(m.predict(&[-1]).unwrap(), vec![-1]);
    assert_eq!(m.predict(&[-4]).unwrap(), vec![-1]);
    assert_eq!(m.predict(&[-5]).unwrap(), vec![-2]);
}

#[test]
fn full_rank_anchors_admit_nonzero_integer_edits() {
    let mut ledger = Ledger::new(model(2, 8), anchors(2)).unwrap();
    let p = solve_one_coordinate(
        &ledger,
        &[Target {
            input: vec![8, 8],
            output: vec![1],
        }],
    )
    .unwrap()
    .unwrap();
    let c = ledger.certify(&p).unwrap();
    ledger.apply(&p, &c).unwrap();
    assert_eq!(ledger.model().predict(&[1, 0]).unwrap(), vec![0]);
    assert_eq!(ledger.model().predict(&[0, 1]).unwrap(), vec![0]);
    assert_eq!(ledger.model().predict(&[8, 8]).unwrap(), vec![1]);
    assert_ne!(ledger.model().weights, ledger.baseline().weights);
}

#[test]
fn no_one_coordinate_solution_does_not_imply_global_infeasibility() {
    let mut ledger = Ledger::new(model(2, 8), anchors(2)).unwrap();
    let targets = [Target {
        input: vec![1, 1],
        output: vec![1],
    }];
    assert!(solve_one_coordinate(&ledger, &targets).unwrap().is_none());
    let p = Patch {
        expected_parent: ledger.head().into(),
        edits: vec![
            Edit {
                row: 0,
                col: 0,
                value: 4,
            },
            Edit {
                row: 0,
                col: 1,
                value: 4,
            },
        ],
    };
    let c = ledger.certify(&p).unwrap();
    ledger.apply(&p, &c).unwrap();
    assert_eq!(ledger.model().predict(&[1, 1]).unwrap(), vec![1]);
}

#[test]
fn certificate_tampering_is_rejected_atomically() {
    let mut ledger = Ledger::new(model(2, 8), anchors(2)).unwrap();
    let p = patch(&ledger, 0, 1);
    let mut c = ledger.certify(&p).unwrap();
    let head = ledger.head().to_string();
    c.candidate_hash.push('0');
    assert!(ledger.verify(&p, &c).is_err());
    assert!(ledger.apply(&p, &c).is_err());
    assert_eq!(ledger.head(), head);
    assert_eq!(ledger.revision(), 0);
    assert_eq!(ledger.model().weights, vec![vec![0, 0]]);
}

#[test]
fn certificate_binds_patch_and_protected_context() {
    let first = Ledger::new(model(2, 8), anchors(2)).unwrap();
    let second = Ledger::new(
        model(2, 8),
        Protected {
            points: vec![vec![1, 0]],
            boxes: vec![],
        },
    )
    .unwrap();
    let p = patch(&first, 0, 1);
    let c = first.certify(&p).unwrap();
    let mut other = p.clone();
    other.edits[0].value = 2;
    assert!(first.verify(&other, &c).is_err());
    other = patch(&second, 0, 1);
    assert!(second.verify(&other, &c).is_err());
}

#[test]
fn replay_is_rejected_even_after_rollback() {
    let mut ledger = Ledger::new(model(1, 8), anchors(1)).unwrap();
    let p = patch(&ledger, 0, 1);
    let c = ledger.certify(&p).unwrap();
    ledger.apply(&p, &c).unwrap();
    assert!(ledger.apply(&p, &c).is_err());
    let receipt = ledger.rollback().unwrap();
    assert_eq!(receipt.revision, 2);
    assert_eq!(ledger.model(), ledger.baseline());
    assert!(ledger.apply(&p, &c).is_err());
    assert!(ledger.rollback().is_err());
}

#[test]
fn sequence_uses_original_cells_and_rejects_cumulative_drift() {
    let mut ledger = Ledger::new(model(1, 8), anchors(1)).unwrap();
    for value in 1..8 {
        let p = patch(&ledger, 0, value);
        let c = ledger.certify(&p).unwrap();
        ledger.apply(&p, &c).unwrap();
    }
    let head = ledger.head().to_string();
    let p = patch(&ledger, 0, 8);
    assert!(ledger.certify(&p).is_err());
    assert_eq!(ledger.head(), head);
    assert_eq!(ledger.model().predict(&[1]).unwrap(), vec![0]);
}

#[test]
fn saturation_cells_have_one_unbounded_side() {
    let mut m = model(1, 4);
    m.output_min = vec![0];
    m.output_max = vec![2];
    let low = Ledger::new(m.clone(), anchors(1)).unwrap();
    assert!(low.certify(&patch(&low, 0, -128)).is_ok());
    assert!(low.certify(&patch(&low, 0, 3)).is_ok());
    assert!(low.certify(&patch(&low, 0, 4)).is_err());
    m.weights[0][0] = 8;
    let high = Ledger::new(m, anchors(1)).unwrap();
    assert!(high.certify(&patch(&high, 0, 127)).is_ok());
    assert!(high.certify(&patch(&high, 0, 7)).is_err());
}

#[test]
fn degenerate_quantizer_still_checks_integer_overflow() {
    let mut m = model(1, 1);
    m.output_min = vec![3];
    m.output_max = vec![3];
    let ledger = Ledger::new(
        m,
        Protected {
            points: vec![vec![i32::MAX]],
            boxes: vec![],
        },
    )
    .unwrap();
    assert!(ledger.certify(&patch(&ledger, 0, 1)).is_ok());
    assert!(ledger.certify(&patch(&ledger, 0, 2)).is_err());
}

#[test]
fn box_certificate_protects_every_integer_point() {
    let mut m = model(2, 16);
    m.bias[0] = 4;
    let domain = IntBox {
        lower: vec![-1, -2],
        upper: vec![2, 1],
    };
    let mut ledger = Ledger::new(
        m,
        Protected {
            points: vec![],
            boxes: vec![domain],
        },
    )
    .unwrap();
    let p = patch(&ledger, 0, 3);
    let c = ledger.certify(&p).unwrap();
    ledger.apply(&p, &c).unwrap();
    for x in -1..=2 {
        for y in -2..=1 {
            assert_eq!(
                ledger.model().predict(&[x, y]).unwrap(),
                ledger.baseline().predict(&[x, y]).unwrap()
            );
        }
    }
    assert!(ledger.certify(&patch(&ledger, 0, 5)).is_err());
}

#[test]
fn varying_baseline_box_is_rejected() {
    let mut m = model(1, 2);
    m.weights[0][0] = 1;
    assert!(Ledger::new(
        m,
        Protected {
            points: vec![],
            boxes: vec![IntBox {
                lower: vec![0],
                upper: vec![2]
            }]
        }
    )
    .is_err());
}

#[test]
fn int32_intermediate_overflow_rejected_despite_final_cancellation() {
    let mut m = model(2, 8);
    m.weights[0] = vec![1, -1];
    m.bias[0] = i32::MAX;
    assert!(m.predict(&[1, 1]).is_err());
}

#[test]
fn int32_multiplication_overflow_is_rejected_before_cancellation() {
    let mut m = model(1, 8);
    m.weights[0][0] = 2;
    m.bias[0] = -i32::MAX;
    assert!(m.predict(&[i32::MAX]).is_err());
}

#[test]
fn solver_excludes_product_overflow_even_when_final_sum_fits() {
    let mut m = model(1, i32::MAX);
    m.bias[0] = -i32::MAX;
    let ledger = Ledger::new(m, Protected::default()).unwrap();
    // Weight 2 would have a representable final sum but an invalid product.
    let target = Target {
        input: vec![i32::MAX],
        output: vec![1],
    };
    assert!(solve_one_coordinate(&ledger, &[target]).unwrap().is_none());
}

#[test]
fn malformed_model_and_domains_fail_closed() {
    let mut m = model(1, 0);
    assert!(m.validate().is_err());
    m.scales[0] = 1;
    m.weights.push(vec![]);
    assert!(m.validate().is_err());
    let mut m = model(1, 1);
    m.weight_min = -129;
    assert!(m.validate().is_err());
    let m = model(1, 1);
    assert!(m.predict(&[]).is_err());
    assert!(Ledger::new(
        m.clone(),
        Protected {
            points: vec![vec![0, 0]],
            boxes: vec![]
        }
    )
    .is_err());
    assert!(Ledger::new(
        m,
        Protected {
            points: vec![],
            boxes: vec![IntBox {
                lower: vec![1],
                upper: vec![0]
            }]
        }
    )
    .is_err());
}

#[test]
fn patch_rejects_noop_duplicates_and_out_of_range() {
    let ledger = Ledger::new(model(1, 8), anchors(1)).unwrap();
    assert!(ledger.certify(&patch(&ledger, 0, 0)).is_err());
    assert!(ledger.certify(&patch(&ledger, 0, 128)).is_err());
    assert!(ledger.certify(&patch(&ledger, 1, 1)).is_err());
    let mut p = patch(&ledger, 0, 1);
    p.edits.push(Edit {
        row: 0,
        col: 0,
        value: 2,
    });
    assert!(ledger.certify(&p).is_err());
}

#[test]
fn unrelated_output_rows_cannot_be_ignored_by_solver() {
    let mut m = model(1, 8);
    m.weights.push(vec![0]);
    m.bias.push(0);
    m.scales.push(8);
    m.output_min.push(-128);
    m.output_max.push(127);
    let ledger = Ledger::new(m, anchors(1)).unwrap();
    assert!(solve_one_coordinate(
        &ledger,
        &[Target {
            input: vec![8],
            output: vec![1, 1]
        }]
    )
    .unwrap()
    .is_none());
}

#[test]
fn target_shape_and_range_are_validated() {
    let ledger = Ledger::new(model(1, 8), anchors(1)).unwrap();
    assert!(solve_one_coordinate(&ledger, &[]).is_err());
    assert!(solve_one_coordinate(
        &ledger,
        &[Target {
            input: vec![1],
            output: vec![]
        }]
    )
    .is_err());
    assert!(solve_one_coordinate(
        &ledger,
        &[Target {
            input: vec![1],
            output: vec![128]
        }]
    )
    .is_err());
}

#[test]
fn empty_protection_is_explicitly_an_unconstrained_control() {
    let ledger = Ledger::new(model(1, 8), Protected::default()).unwrap();
    let c = ledger.certify(&patch(&ledger, 0, 127)).unwrap();
    assert_eq!(c.protected_point_count, 0);
    assert_eq!(c.protected_box_count, 0);
}

#[test]
fn unknown_json_fields_are_not_silently_ignored() {
    let mut value = serde_json::to_value(model(1, 8)).unwrap();
    value["rounding"] = serde_json::json!("truncate");
    assert!(serde_json::from_value::<Model>(value).is_err());
}
