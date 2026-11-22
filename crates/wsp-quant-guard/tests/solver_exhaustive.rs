use ruv_fann_wsp_quant_guard::{
    solve_one_coordinate, Edit, IntBox, Ledger, Model, Patch, Protected, Target,
};

fn next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 32
}

fn number(state: &mut u64, bound: i32) -> i32 {
    (next(state) % (2 * bound + 1) as u64) as i32 - bound
}

#[test]
fn interval_search_matches_exhaustive_integer_replacements() {
    let mut state = 0xCE11_2026_u64;
    let mut feasible = 0;
    let mut examined = 0;
    for case in 0..1000 {
        let rows = if case % 5 == 0 { 2 } else { 1 };
        let width = 3;
        let m = Model {
            weights: (0..rows)
                .map(|_| (0..width).map(|_| number(&mut state, 3)).collect())
                .collect(),
            bias: (0..rows).map(|_| number(&mut state, 3)).collect(),
            scales: (0..rows)
                .map(|_| (next(&mut state) % 8 + 1) as i32)
                .collect(),
            output_min: vec![-3; rows],
            output_max: vec![3; rows],
            weight_min: -3,
            weight_max: 3,
            prefix_id: "fixed-prefix".into(),
        };
        let protected = if case % 3 == 0 {
            Protected {
                points: vec![],
                boxes: vec![IntBox {
                    lower: vec![0, -1, 0],
                    upper: vec![1, 0, 0],
                }],
            }
        } else {
            Protected {
                points: vec![vec![1, 0, 0], vec![0, -1, 0]],
                boxes: vec![],
            }
        };
        let ledger = match Ledger::new(m.clone(), protected) {
            Ok(l) => l,
            Err(_) => continue,
        };
        let targets = vec![Target {
            input: (0..width).map(|_| number(&mut state, 4)).collect(),
            output: (0..rows).map(|_| number(&mut state, 3)).collect(),
        }];
        let mut expected = None;
        for row in 0..rows {
            for col in 0..width {
                for value in -3..=3 {
                    let old = m.weights[row][col];
                    if old == value {
                        continue;
                    }
                    let p = Patch {
                        expected_parent: ledger.head().into(),
                        edits: vec![Edit { row, col, value }],
                    };
                    if ledger.certify(&p).is_err() {
                        continue;
                    }
                    let mut candidate = m.clone();
                    candidate.weights[row][col] = value;
                    if targets
                        .iter()
                        .any(|t| candidate.predict(&t.input).unwrap() != t.output)
                    {
                        continue;
                    }
                    let item = ((value - old).abs() as i64, row, col, value);
                    if expected.as_ref().is_none_or(|old| item < *old) {
                        expected = Some(item);
                    }
                }
            }
        }
        let actual = solve_one_coordinate(&ledger, &targets).unwrap().map(|p| {
            let e = &p.edits[0];
            (
                (e.value - m.weights[e.row][e.col]).abs() as i64,
                e.row,
                e.col,
                e.value,
            )
        });
        assert_eq!(actual, expected, "case {case}");
        if actual.is_some() {
            feasible += 1;
        }
        examined += 1;
    }
    assert!(
        examined > 700,
        "must test enough nonconstant-box-filtered cases"
    );
    assert!(feasible > 50, "must include nontrivial feasible cases");
}

#[test]
fn hundred_distinct_effective_patches_preserve_full_rank_anchors() {
    let width = 128;
    let model = Model {
        weights: vec![vec![0; width]],
        bias: vec![0],
        scales: vec![128],
        output_min: vec![-128],
        output_max: vec![127],
        weight_min: -128,
        weight_max: 127,
        prefix_id: "identity-128".into(),
    };
    let points = (0..width)
        .map(|j| {
            let mut x = vec![0; width];
            x[j] = 1;
            x
        })
        .collect();
    let mut ledger = Ledger::new(
        model,
        Protected {
            points,
            boxes: vec![],
        },
    )
    .unwrap();
    let mut prior_targets: Vec<Target> = Vec::new();
    for index in 0..100 {
        let mut input = vec![0; width];
        input[index] = 128;
        let target = Target {
            input,
            output: vec![1],
        };
        assert_eq!(ledger.model().predict(&target.input).unwrap(), vec![0]);
        let p = solve_one_coordinate(&ledger, std::slice::from_ref(&target))
            .unwrap()
            .unwrap();
        assert_eq!(p.edits[0].col, index);
        assert_eq!(p.edits[0].value, 1);
        let c = ledger.certify(&p).unwrap();
        ledger.apply(&p, &c).unwrap();
        prior_targets.push(target);
        for old in &prior_targets {
            assert_eq!(ledger.model().predict(&old.input).unwrap(), old.output);
        }
    }
    assert_eq!(ledger.revision(), 100);
    assert_eq!(
        ledger.model().weights[0]
            .iter()
            .filter(|v| **v != 0)
            .count(),
        100
    );
    for x in &ledger.protected().points {
        assert_eq!(ledger.model().predict(x).unwrap(), vec![0]);
    }
}
