//! Shared budgets for text formats, checked before allocation.

use super::{IoError, IoResult};
use std::io::BufRead;

pub(super) const MODEL_BYTES: u64 = 256 * 1024 * 1024;
pub(super) const TRAINING_BYTES: u64 = 1024 * 1024 * 1024;
const LINE_BYTES: u64 = 16 * 1024 * 1024;

/// Reads at most the remaining budget plus a sentinel byte. A missing newline
/// cannot grow the line buffer without bound, including for ignored fields.
pub(super) fn read_line<R: BufRead>(
    reader: &mut R,
    line: &mut String,
    remaining: &mut u64,
) -> IoResult<usize> {
    let limit = (*remaining).min(LINE_BYTES);
    let bytes = std::io::Read::take(&mut *reader, limit.saturating_add(1)).read_line(line)?;
    if bytes as u64 > limit {
        return Err(IoError::InvalidFileFormat(
            "Text input exceeds its byte or line limit".into(),
        ));
    }
    *remaining -= bytes as u64;
    Ok(bytes)
}

/// Accounts for the owned vectors and values implied by an untrusted header.
pub(super) fn training_shape(samples: usize, inputs: usize, outputs: usize) -> IoResult<()> {
    let bytes = inputs
        .checked_add(outputs)
        .and_then(|width| width.checked_mul(std::mem::size_of::<f32>()))
        .and_then(|values| values.checked_add(2 * std::mem::size_of::<Vec<f32>>()))
        .and_then(|row| row.checked_mul(samples));
    if inputs == 0
        || outputs == 0
        || !matches!(bytes, Some(bytes) if bytes as u64 <= TRAINING_BYTES)
    {
        return Err(IoError::InvalidTrainingData(
            "Training dimensions exceed the memory budget or are zero".into(),
        ));
    }
    Ok(())
}

pub(super) fn network_shape<T: num_traits::Float>(sizes: &[usize], budget: u64) -> IoResult<()> {
    let mut allocation = 0usize;
    for (index, &width) in sizes.iter().enumerate() {
        let count = width.checked_add(usize::from(index == 0 || index + 1 < sizes.len()));
        let neuron_bytes =
            count.and_then(|count| count.checked_mul(std::mem::size_of::<crate::Neuron<T>>()));
        let connections = if index == 0 {
            Some(0)
        } else {
            sizes[index - 1]
                .checked_add(1)
                .and_then(|previous| previous.checked_mul(width))
                .and_then(|count| count.checked_mul(std::mem::size_of::<crate::Connection<T>>()))
        };
        let required = neuron_bytes
            .zip(connections)
            .and_then(|(a, b)| a.checked_add(b));
        allocation = required
            .and_then(|bytes| allocation.checked_add(bytes))
            .filter(|&bytes| bytes as u64 <= budget)
            .ok_or_else(|| {
                IoError::InvalidNetwork("Network dimensions exceed the memory budget".into())
            })?;
        if width == 0 {
            return Err(IoError::InvalidNetwork(
                "Layer sizes must be positive".into(),
            ));
        }
    }
    Ok(())
}
