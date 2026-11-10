use ruv_fann::PreparedNetwork;
use ruv_fann_esp32_c5::{score, CAP_Q10, MARGIN_Q10, Q10_ONE};

#[test]
fn matches_prepared_ruvfann() {
    let reference =
        PreparedNetwork::<f64>::from_dense_layers_json(include_str!("../model.json")).unwrap();
    let mut workspace = reference.workspace().unwrap();
    let scale = f64::from(Q10_ONE);
    for local in (0..=Q10_ONE).step_by(32) {
        for adjacent in (0..=Q10_ONE).step_by(32) {
            let inputs = [f64::from(local) / scale, f64::from(adjacent) / scale];
            let mut expected = [0.0];
            reference
                .run_into(&inputs, &mut expected, &mut workspace)
                .unwrap();
            assert_eq!(
                f64::from(score(local, adjacent).unwrap()) / scale,
                expected[0]
            );
        }
    }
}

#[test]
fn handles_both_hinges_and_rejects_out_of_range_inputs() {
    assert_eq!(score(MARGIN_Q10, 0), Some(0));
    assert_eq!(score(MARGIN_Q10 + 1, 0), Some(1));
    assert_eq!(score(MARGIN_Q10 + CAP_Q10, 0), Some(CAP_Q10));
    assert_eq!(score(Q10_ONE, 0), Some(CAP_Q10));
    assert_eq!(score(Q10_ONE + 1, 0), None);
    assert_eq!(score(0, u16::MAX), None);
}
