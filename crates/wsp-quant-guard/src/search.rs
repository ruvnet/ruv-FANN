use crate::model::{IntBox, MAX_DOMAINS};
use crate::{fail, Edit, Ledger, Model, Patch, Result};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Target {
    pub input: Vec<i32>,
    pub output: Vec<i32>,
}

// A finite integer interval. All calculations stay far inside i128 because
// model dimensions, integer types and int8 weight ranges are bounded.
#[derive(Clone, Copy)]
struct Interval {
    lo: i128,
    hi: i128,
}

impl Interval {
    fn ge(&mut self, coefficient: i128, rhs: i128) {
        if coefficient > 0 {
            self.lo = self.lo.max(-(-rhs).div_euclid(coefficient));
        } else if coefficient < 0 {
            self.hi = self.hi.min((-rhs).div_euclid(-coefficient));
        } else if rhs > 0 {
            self.lo = 1;
            self.hi = 0;
        }
    }

    fn le(&mut self, coefficient: i128, rhs: i128) {
        self.ge(-coefficient, -rhs);
    }

    fn contains_any(&self) -> bool {
        self.lo <= self.hi
    }
}

/// Intersect an interval with every accumulator prefix and the final output
/// cell for a box. Splitting the replacement weight by sign makes extrema
/// affine in that weight, avoiding exponential corner enumeration.
fn constrain(
    interval: &mut Interval,
    model: &Model,
    row: usize,
    col: usize,
    domain: &IntBox,
    output: i32,
    nonnegative: bool,
) -> Result<()> {
    let mut lower = model.bias[row] as i128;
    let mut upper = lower;
    let mut low_coefficient = 0;
    let mut high_coefficient = 0;
    for j in 0..model.weights[row].len() {
        if j == col {
            low_coefficient = if nonnegative {
                domain.lower[j]
            } else {
                domain.upper[j]
            } as i128;
            high_coefficient = if nonnegative {
                domain.upper[j]
            } else {
                domain.lower[j]
            } as i128;
            interval.ge(low_coefficient, i32::MIN as i128);
            interval.le(high_coefficient, i32::MAX as i128);
        } else {
            let a = model.weights[row][j] as i128 * domain.lower[j] as i128;
            let b = model.weights[row][j] as i128 * domain.upper[j] as i128;
            if a.min(b) < i32::MIN as i128 || a.max(b) > i32::MAX as i128 {
                interval.lo = 1;
                interval.hi = 0;
            }
            lower += a.min(b);
            upper += a.max(b);
        }
        interval.ge(low_coefficient, i32::MIN as i128 - lower);
        interval.le(high_coefficient, i32::MAX as i128 - upper);
    }
    let (cell_lower, cell_upper) = model.cell(row, output)?;
    interval.ge(low_coefficient, cell_lower - lower);
    interval.le(high_coefficient, cell_upper - upper);
    Ok(())
}

/// Find a single changed weight satisfying all exact target outputs and all
/// protected domains. `None` means no such one-coordinate edit exists, not
/// that arbitrary multiweight editing is impossible. The nearest feasible
/// replacement is chosen, then row/column/value break ties deterministically.
pub fn solve_one_coordinate(ledger: &Ledger, targets: &[Target]) -> Result<Option<Patch>> {
    let model = ledger.model();
    if targets.is_empty() || targets.len() > MAX_DOMAINS {
        return fail("target count must be 1..4096");
    }
    let rows = model.weights.len();
    let width = model.weights[0].len();
    // Bound this intentionally simple exhaustive coordinate search.
    let work = 2
        * rows as u128
        * width as u128
        * width as u128
        * (targets.len() + ledger.domains.len()) as u128
        + rows as u128 * rows as u128 * width as u128 * targets.len() as u128;
    if work > 100_000_000 {
        return fail("one-coordinate search work budget exceeded");
    }
    let mut target_domains = Vec::with_capacity(targets.len());
    for target in targets {
        let domain = IntBox::point(&target.input);
        domain.validate(width)?;
        if target.output.len() != rows {
            return fail("target output shape mismatch");
        }
        for row in 0..rows {
            model.cell(row, target.output[row])?;
        }
        target_domains.push(domain);
    }
    let mut best: Option<(i64, usize, usize, i32)> = None;
    for row in 0..rows {
        // One changed row cannot repair mismatches in a different row.
        let mut others_ok = true;
        for (target, domain) in targets.iter().zip(&target_domains) {
            for other in 0..rows {
                if other != row {
                    match model.bounds(other, domain) {
                        Ok((a, _)) if model.quantize(other, a) == target.output[other] => {}
                        _ => others_ok = false,
                    }
                }
            }
        }
        if !others_ok {
            continue;
        }
        for col in 0..width {
            for nonnegative in [false, true] {
                let mut interval = Interval {
                    lo: model.weight_min as i128,
                    hi: model.weight_max as i128,
                };
                if nonnegative {
                    interval.lo = interval.lo.max(0);
                } else {
                    interval.hi = interval.hi.min(-1);
                }
                for (domain, outputs) in ledger.domains.iter().zip(&ledger.outputs) {
                    constrain(
                        &mut interval,
                        model,
                        row,
                        col,
                        domain,
                        outputs[row],
                        nonnegative,
                    )?;
                }
                for (target, domain) in targets.iter().zip(&target_domains) {
                    constrain(
                        &mut interval,
                        model,
                        row,
                        col,
                        domain,
                        target.output[row],
                        nonnegative,
                    )?;
                }
                if !interval.contains_any() {
                    continue;
                }
                let current = model.weights[row][col] as i128;
                let closest = current.clamp(interval.lo, interval.hi);
                let candidates = if closest == current {
                    vec![current - 1, current + 1]
                } else {
                    vec![closest]
                };
                for value in candidates {
                    if value < interval.lo || value > interval.hi || value == current {
                        continue;
                    }
                    let item = ((value - current).abs() as i64, row, col, value as i32);
                    if best.as_ref().is_none_or(|old| item < *old) {
                        best = Some(item);
                    }
                }
            }
        }
    }
    if let Some((_, row, col, value)) = best {
        let patch = Patch {
            expected_parent: ledger.head().into(),
            edits: vec![Edit { row, col, value }],
        };
        // Independent full verifier remains authoritative over synthesis.
        ledger.certify(&patch)?;
        Ok(Some(patch))
    } else {
        Ok(None)
    }
}
