//! Exact corrected order statistics: sort small groups, select large groups.
use super::{error, Result};

fn rank(n: usize, coverage: f64) -> Result<usize> {
    let rank = ((n + 1) as f64 * coverage).ceil() as usize;
    if rank == 0 || rank > n {
        return Err(error(format!(
            "Insufficient calibration samples: n={n}, coverage={coverage}, required rank={rank}"
        )));
    }
    Ok(rank - 1)
}
pub(super) fn radii(errors: &mut [f64], coverages: &[f64]) -> Result<Vec<f64>> {
    // Native kernel measurements favor full sorting for tiny score vectors,
    // especially when many levels collapse onto the same small set of ranks.
    // 1024 leaves a clear measured margin even with all 16 requested levels.
    if errors.len() < 1024 {
        errors.sort_by(f64::total_cmp);
        let mut output = Vec::with_capacity(coverages.len());
        for coverage in coverages {
            output.push(errors[rank(errors.len(), *coverage)?]);
        }
        return Ok(output);
    }
    let ranks = coverages
        .iter()
        .map(|c| rank(errors.len(), *c))
        .collect::<Result<Vec<_>>>()?;
    let mut unique = ranks.clone();
    unique.dedup(); // Coverage levels are validated and sorted by the caller.
    select_required(errors, 0, &unique);
    Ok(ranks.into_iter().map(|index| errors[index]).collect())
}
fn select_required(values: &mut [f64], offset: usize, ranks: &[usize]) {
    if ranks.is_empty() {
        return;
    }
    let middle = ranks.len() / 2;
    let index = ranks[middle];
    let (left, _, right) = values.select_nth_unstable_by(index - offset, f64::total_cmp);
    select_required(left, offset, &ranks[..middle]);
    select_required(right, index + 1, &ranks[middle + 1..]);
}
#[cfg(test)]
#[path = "calibration_quantile_tests.rs"]
mod tests;
