use super::*;
fn expected(values: &[f64], levels: &[f64]) -> Result<Vec<f64>> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mut result = Vec::new();
    for c in levels {
        let r = ((sorted.len() + 1) as f64 * c).ceil() as usize;
        if r == 0 || r > sorted.len() {
            return Err(error("insufficient"));
        }
        result.push(sorted[r - 1]);
    }
    Ok(result)
}
fn assert_reference(values: &[f64], levels: &[f64]) {
    let reference = expected(values, levels);
    let actual = radii(&mut values.to_vec(), levels);
    match (reference, actual) {
        (Ok(reference), Ok(actual)) => assert_eq!(
            reference.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        ),
        (Err(_), Err(_)) => {}
        _ => panic!("selection/reference success mismatch"),
    }
}
#[test]
fn selector_matches_full_sort_at_thresholds_with_ties_and_extreme_values() {
    for n in [
        0, 1, 2, 9, 10, 19, 20, 50, 255, 256, 511, 512, 513, 1023, 1024, 1025, 3000, 100000,
    ] {
        let mut seed = 123456_u64;
        let values = (0..n)
            .map(|i| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                match i % 13 {
                    0 => -0.,
                    1 => 0.,
                    2 => f64::from_bits(1),
                    3 => f64::MIN_POSITIVE,
                    4 => f64::MAX,
                    5 => -f64::MAX,
                    _ => ((seed >> 32) % 17) as f64 / 8.,
                }
            })
            .collect::<Vec<_>>();
        for levels in [
            vec![0.9],
            vec![0.01, 0.5, 0.8, 0.95],
            (1..=16).map(|i| 0.95 * i as f64 / 16.).collect(),
        ] {
            assert_reference(&values, &levels);
            let mut reversed = values.clone();
            reversed.reverse();
            assert_reference(&reversed, &levels);
        }
    }
}
#[test]
fn sorted_reverse_and_constant_vectors_preserve_exact_order_statistics() {
    let levels = (1..=16).map(|i| 0.95 * i as f64 / 16.).collect::<Vec<_>>();
    for n in [10, 50, 512, 513, 3000] {
        let mut values = (0..n).map(|i| i as f64 / 16.).collect::<Vec<_>>();
        assert_reference(&values, &levels);
        values.reverse();
        assert_reference(&values, &levels);
        assert_reference(&vec![3.25; n], &levels);
    }
}
#[test]
fn insufficient_corrected_rank_remains_an_error_for_both_paths() {
    for n in [9, 511, 512, 513, 1023, 1024, 1025] {
        let coverage = 1. - 0.5 / (n + 1) as f64;
        let error = radii(&mut vec![1.; n], &[coverage])
            .unwrap_err()
            .to_string();
        assert!(error.contains(&format!("required rank={}", n + 1)));
    }
}
