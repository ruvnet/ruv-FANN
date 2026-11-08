//! Fixed-capacity batched inference, with independent samples contiguous in
//! scratch. The public input/output layout remains ordinary row-major samples.
use super::{add, budget, bytes, reserved, PreparedError, PreparedNetwork};
use num_traits::Float;
use std::mem::size_of;
#[path = "prepared_batch_simd.rs"]
mod kernels;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// Supported batch arithmetic. Sealed to f32/f64; single-sample inference retains
/// its existing generic Float API. SIMD is explicit only on wasm32+simd128.
pub trait BatchFloat: Float + sealed::Sealed {
    #[doc(hidden)]
    fn parallel_min_rows() -> usize;
    #[doc(hidden)]
    fn accumulate(sums: &mut [Self], values: &[Self], weight: Self);
}
impl BatchFloat for f32 {
    #[inline(always)]
    fn parallel_min_rows() -> usize {
        4
    }
    #[inline(always)]
    fn accumulate(sums: &mut [Self], values: &[Self], weight: Self) {
        kernels::accumulate_f32(sums, values, weight);
    }
}
impl BatchFloat for f64 {
    #[inline(always)]
    fn parallel_min_rows() -> usize {
        if cfg!(all(target_arch = "wasm32", target_feature = "simd128")) {
            2
        } else {
            4
        }
    }
    #[inline(always)]
    fn accumulate(sums: &mut [Self], values: &[Self], weight: Self) {
        kernels::accumulate_f64(sums, values, weight);
    }
}

/// Reusable caller-owned scratch. A wider workspace can serve another prepared
/// network. Contents may change on error; parameter snapshots are immutable.
#[derive(Debug)]
pub struct BatchWorkspace<T: BatchFloat> {
    front: Vec<T>,
    back: Vec<T>,
    max_rows: usize,
    width: usize,
    estimated_bytes: usize,
}

impl<T: BatchFloat> BatchWorkspace<T> {
    pub fn max_rows(&self) -> usize {
        self.max_rows
    }
    /// Requested buffer payload plus this object; excludes allocator overhead
    /// and spare capacity. Does not include the immutable prepared snapshot.
    pub fn estimated_bytes(&self) -> usize {
        self.estimated_bytes
    }
}

impl<T: BatchFloat> PreparedNetwork<T> {
    /// Check the exact requested workspace bytes without allocating. Capacity
    /// must be positive; multiplication/addition overflow returns SizeOverflow.
    pub fn batch_workspace_bytes(&self, max_rows: usize) -> Result<usize, PreparedError> {
        if max_rows == 0 {
            return Err(PreparedError::BudgetExceeded {
                resource: "batch capacity",
                required: 1,
                limit: 0,
            });
        }
        add(
            size_of::<BatchWorkspace<T>>(),
            bytes(bytes(bytes(self.width, max_rows)?, 2)?, size_of::<T>())?,
        )
    }

    /// Allocate two feature-major scratch buffers once. The checked byte budget
    /// is enforced before either allocation; no temporary per-row vectors exist.
    pub fn batch_workspace(
        &self,
        max_rows: usize,
        max_bytes: usize,
    ) -> Result<BatchWorkspace<T>, PreparedError> {
        let estimated_bytes = self.batch_workspace_bytes(max_rows)?;
        budget("batch workspace bytes", estimated_bytes, max_bytes)?;
        let cells = bytes(self.width, max_rows)?;
        let mut front = reserved(cells)?;
        let mut back = reserved(cells)?;
        front.resize(cells, T::zero());
        back.resize(cells, T::zero());
        Ok(BatchWorkspace {
            front,
            back,
            max_rows,
            width: self.width,
            estimated_bytes,
        })
    }

    /// Infer row-major samples without steady-state allocation. Every sample
    /// visits weights in the original order, including sparse/duplicate sources.
    /// Outputs are committed only after all rows/layers succeed. Zero rows is a
    /// no-op with empty slices. Error input indices refer to flattened input.
    pub fn run_batch_into(
        &self,
        inputs: &[T],
        outputs: &mut [T],
        rows: usize,
        workspace: &mut BatchWorkspace<T>,
    ) -> Result<(), PreparedError> {
        let input_len = bytes(rows, self.inputs)?;
        let output_len = bytes(rows, self.outputs)?;
        if inputs.len() != input_len {
            return Err(PreparedError::InputSizeMismatch {
                expected: input_len,
                actual: inputs.len(),
            });
        }
        if outputs.len() != output_len {
            return Err(PreparedError::OutputSizeMismatch {
                expected: output_len,
                actual: outputs.len(),
            });
        }
        budget("batch rows", rows, workspace.max_rows)?;
        if rows == 0 {
            return Ok(());
        }
        if workspace.width < self.width {
            return Err(PreparedError::WorkspaceTooSmall {
                required: bytes(self.width, workspace.max_rows)?,
                actual: workspace.front.len(),
            });
        }
        if let Some(index) = inputs.iter().position(|value| !value.is_finite()) {
            return Err(PreparedError::NonFiniteInput { index });
        }
        let stride = workspace.max_rows;
        let (mut source, mut target) = (&mut workspace.front, &mut workspace.back);
        for (sample, input) in inputs.chunks_exact(self.inputs).enumerate() {
            for (feature, &value) in input.iter().enumerate() {
                source[feature * stride + sample] = value;
            }
        }
        if self.input_bias {
            source[self.inputs * stride..self.inputs * stride + rows].fill(T::one());
        }
        for (layer_index, layer) in self.layers.iter().enumerate() {
            for (neuron, row) in self.rows[layer.row_start..layer.row_start + layer.row_count]
                .iter()
                .enumerate()
            {
                let sums = &mut target[neuron * stride..neuron * stride + rows];
                let weights = &self.weights[row.start..row.start + row.len];
                if rows < T::parallel_min_rows() {
                    // A local accumulator avoids a memory update per weight
                    // when there are too few samples to fill parallel lanes.
                    for (sample, sum) in sums.iter_mut().enumerate() {
                        *sum = if row.index_start == usize::MAX {
                            weights
                                .iter()
                                .enumerate()
                                .fold(T::zero(), |sum, (feature, &weight)| {
                                    sum + source[feature * stride + sample] * weight
                                })
                        } else {
                            self.indices[row.index_start..row.index_start + row.len]
                                .iter()
                                .zip(weights)
                                .fold(T::zero(), |sum, (&feature, &weight)| {
                                    sum + source[feature * stride + sample] * weight
                                })
                        };
                    }
                } else {
                    sums.fill(T::zero());
                    for (index, &weight) in weights.iter().enumerate() {
                        let feature = if row.index_start == usize::MAX {
                            index
                        } else {
                            self.indices[row.index_start + index]
                        };
                        let values = &source[feature * stride..feature * stride + rows];
                        T::accumulate(sums, values, weight);
                    }
                }
                for sum in sums {
                    if !sum.is_finite() {
                        return Err(PreparedError::NumericOverflow {
                            layer: layer_index + 1,
                            neuron,
                        });
                    }
                    let value = row.activation.apply(*sum, row.steepness);
                    if !value.is_finite() {
                        return Err(PreparedError::NumericOverflow {
                            layer: layer_index + 1,
                            neuron,
                        });
                    }
                    *sum = value;
                }
            }
            if layer.bias {
                target[layer.row_count * stride..layer.row_count * stride + rows].fill(T::one());
            }
            std::mem::swap(&mut source, &mut target);
        }
        // All fallible work is complete, so transposition cannot expose a prefix
        // of successful rows when any other row has invalid numeric output.
        for (sample, output) in outputs.chunks_exact_mut(self.outputs).enumerate() {
            for (feature, value) in output.iter_mut().enumerate() {
                *value = source[feature * stride + sample];
            }
        }
        Ok(())
    }
}
