use crate::{fail, Error, Result};
use serde::{Deserialize, Serialize};

pub(crate) const MAX_DIMENSION: usize = 4096;
pub(crate) const MAX_PARAMETERS: usize = 1_048_576;
pub(crate) const MAX_DOMAINS: usize = 4096;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub weights: Vec<Vec<i32>>,
    pub bias: Vec<i32>,
    pub scales: Vec<i32>,
    pub output_min: Vec<i32>,
    pub output_max: Vec<i32>,
    pub weight_min: i32,
    pub weight_max: i32,
    /// Caller supplied identity. This crate cannot authenticate the actual prefix.
    pub prefix_id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct IntBox {
    pub lower: Vec<i32>,
    pub upper: Vec<i32>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct Protected {
    pub points: Vec<Vec<i32>>,
    pub boxes: Vec<IntBox>,
}

impl IntBox {
    pub(crate) fn point(input: &[i32]) -> Self {
        Self {
            lower: input.to_vec(),
            upper: input.to_vec(),
        }
    }

    pub(crate) fn validate(&self, width: usize) -> Result<()> {
        if self.lower.len() != width || self.upper.len() != width {
            return fail("domain shape mismatch");
        }
        if self.lower.iter().zip(&self.upper).any(|(a, b)| a > b) {
            return fail("domain lower bound exceeds upper bound");
        }
        Ok(())
    }
}

impl Protected {
    pub(crate) fn domains(&self, width: usize) -> Result<Vec<IntBox>> {
        let count = self
            .points
            .len()
            .checked_add(self.boxes.len())
            .ok_or_else(|| Error("too many domains".into()))?;
        if count > MAX_DOMAINS || count as u128 * width as u128 > 10_000_000 {
            return fail("protected domain size budget exceeded");
        }
        let mut domains: Vec<IntBox> = self.points.iter().map(|p| IntBox::point(p)).collect();
        domains.extend(self.boxes.iter().cloned());
        for domain in &domains {
            domain.validate(width)?;
        }
        Ok(domains)
    }
}

impl Model {
    pub fn validate(&self) -> Result<()> {
        let rows = self.weights.len();
        let width = self.weights.first().map_or(0, Vec::len);
        if rows == 0
            || rows > 256
            || width == 0
            || width > MAX_DIMENSION
            || rows.checked_mul(width).unwrap_or(usize::MAX) > MAX_PARAMETERS
        {
            return fail("invalid or excessive model shape");
        }
        if self.weights.iter().any(|r| r.len() != width)
            || [&self.bias, &self.scales, &self.output_min, &self.output_max]
                .iter()
                .any(|v| v.len() != rows)
        {
            return fail("model vector shape mismatch");
        }
        if self.weight_min < -128 || self.weight_max > 127 || self.weight_min > self.weight_max {
            return fail("weight bounds must be a nonempty subset of int8");
        }
        if self
            .weights
            .iter()
            .flatten()
            .any(|w| *w < self.weight_min || *w > self.weight_max)
        {
            return fail("weight outside declared bounds");
        }
        if self.scales.iter().any(|s| *s <= 0)
            || self
                .output_min
                .iter()
                .zip(&self.output_max)
                .any(|(a, b)| a > b)
        {
            return fail("invalid quantizer");
        }
        if self.prefix_id.is_empty() || self.prefix_id.len() > 4096 {
            return fail("prefix identity must contain 1..4096 bytes");
        }
        Ok(())
    }

    pub fn predict(&self, input: &[i32]) -> Result<Vec<i32>> {
        self.validate()?;
        let domain = IntBox::point(input);
        domain.validate(self.weights[0].len())?;
        (0..self.weights.len())
            .map(|row| {
                let (lower, _) = self.bounds(row, &domain)?;
                Ok(self.quantize(row, lower))
            })
            .collect()
    }

    pub(crate) fn quantize(&self, row: usize, accumulator: i128) -> i32 {
        accumulator
            .div_euclid(self.scales[row] as i128)
            .clamp(self.output_min[row] as i128, self.output_max[row] as i128) as i32
    }

    /// The exact accumulator preimage of an output, clipped to int32 arithmetic.
    pub(crate) fn cell(&self, row: usize, output: i32) -> Result<(i128, i128)> {
        let lo = self.output_min[row];
        let hi = self.output_max[row];
        if output < lo || output > hi {
            return fail("requested output outside quantizer range");
        }
        let scale = self.scales[row] as i128;
        let lower = if output == lo {
            i32::MIN as i128
        } else {
            output as i128 * scale
        };
        let upper = if output == hi {
            i32::MAX as i128
        } else {
            (output as i128 + 1) * scale - 1
        };
        Ok((lower.max(i32::MIN as i128), upper.min(i32::MAX as i128)))
    }

    /// Exact extrema for every ordered accumulator prefix over an integer box.
    /// Requiring all prefixes to fit also rejects cancellation after overflow.
    pub(crate) fn bounds(&self, row: usize, domain: &IntBox) -> Result<(i128, i128)> {
        let mut lower = self.bias[row] as i128;
        let mut upper = lower;
        for ((w, lo), hi) in self.weights[row]
            .iter()
            .zip(&domain.lower)
            .zip(&domain.upper)
        {
            let a = (*w as i128)
                .checked_mul(*lo as i128)
                .ok_or_else(|| Error("product overflow".into()))?;
            let b = (*w as i128)
                .checked_mul(*hi as i128)
                .ok_or_else(|| Error("product overflow".into()))?;
            if a.min(b) < i32::MIN as i128 || a.max(b) > i32::MAX as i128 {
                return fail("int32 product overflow");
            }
            lower = lower
                .checked_add(a.min(b))
                .ok_or_else(|| Error("accumulator overflow".into()))?;
            upper = upper
                .checked_add(a.max(b))
                .ok_or_else(|| Error("accumulator overflow".into()))?;
            if lower < i32::MIN as i128 || upper > i32::MAX as i128 {
                return fail("int32 intermediate accumulator overflow");
            }
        }
        Ok((lower, upper))
    }

    pub(crate) fn protected_outputs(&self, domains: &[IntBox]) -> Result<Vec<Vec<i32>>> {
        domains
            .iter()
            .map(|domain| {
                (0..self.weights.len())
                    .map(|row| {
                        let (lower, upper) = self.bounds(row, domain)?;
                        let q = self.quantize(row, lower);
                        if q != self.quantize(row, upper) {
                            return fail("protected box does not have constant baseline output");
                        }
                        Ok(q)
                    })
                    .collect()
            })
            .collect()
    }

    pub(crate) fn preserves(&self, domains: &[IntBox], expected: &[Vec<i32>]) -> Result<()> {
        for (domain, outputs) in domains.iter().zip(expected) {
            for (row, output) in outputs.iter().enumerate() {
                let (lower, upper) = self.bounds(row, domain)?;
                let (cell_lower, cell_upper) = self.cell(row, *output)?;
                if lower < cell_lower || upper > cell_upper {
                    return fail("patch changes a protected quantized output");
                }
            }
        }
        Ok(())
    }
}
