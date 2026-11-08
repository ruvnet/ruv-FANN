//! Tiled CPU reference for memory-efficient scaled dot-product attention
//!
//! Implements the tiled, I/O-aware attention algorithm from Dao et al. (2022/2023).
//! Instead of materializing the full N×N attention matrix in memory, Flash Attention
//! processes scores in tiles and keeps running softmax statistics in host vectors,
//! without materializing the full S = Q·K^T matrix.
//!
//! This uses O(N) auxiliary memory without materializing the score matrix.
//! It is not a fused GPU FlashAttention-2 kernel and has no measured HBM benefit.
//!
//! Reference: "FlashAttention-2: Faster Attention with Better Parallelism
//!             and Work Partitioning" — Tri Dao, 2023

use std::fmt;

/// Configuration for Flash Attention computation.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Head dimension (d_k). Typical: 64, 128.
    pub head_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Query tile rows for the CPU reference outer loop.
    pub block_size_q: usize,
    /// Tile size for the K/V (key/value) inner loop.
    pub block_size_kv: usize,
    /// Whether to apply causal (autoregressive) masking.
    pub causal: bool,
    /// Softmax scaling factor. Default: 1/sqrt(d_k).
    pub scale: Option<f32>,
    /// Dropout probability (0.0 = no dropout).
    pub dropout_p: f32,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            head_dim: 64,
            num_heads: 8,
            block_size_q: 64,
            block_size_kv: 64,
            causal: false,
            scale: None,
            dropout_p: 0.0,
        }
    }
}

impl FlashAttentionConfig {
    /// Create config with head_dim and num_heads.
    pub fn new(head_dim: usize, num_heads: usize) -> Self {
        Self {
            head_dim,
            num_heads,
            ..Default::default()
        }
    }

    /// Set causal masking.
    pub fn with_causal(mut self, causal: bool) -> Self {
        self.causal = causal;
        self
    }

    /// Set dropout probability.
    pub fn with_dropout(mut self, p: f32) -> Self {
        self.dropout_p = p;
        self
    }

    /// Effective softmax scale factor.
    pub fn softmax_scale(&self) -> f32 {
        self.scale.unwrap_or(1.0 / (self.head_dim as f32).sqrt())
    }
}

/// Result of a Flash Attention forward pass.
#[derive(Debug, Clone)]
pub struct FlashAttentionOutput {
    /// Output tensor: (batch, num_heads, seq_len, head_dim) flattened row-major.
    pub output: Vec<f32>,
    /// Log-sum-exp per (batch, head, query_row) for backward pass.
    pub logsumexp: Vec<f32>,
    /// Total FLOPs performed.
    pub flops: u64,
    /// Peak score-row scratch allocation in bytes (host memory, excluding outputs).
    pub peak_sram_bytes: usize,
}

/// Tiled CPU attention reference engine.
///
/// Processes attention in tiles to achieve O(N) memory instead of O(N²).
pub struct FlashAttention {
    config: FlashAttentionConfig,
}

impl FlashAttention {
    /// Create a new Flash Attention instance.
    pub fn new(config: FlashAttentionConfig) -> Self {
        Self { config }
    }

    /// Forward pass: compute scaled dot-product attention.
    ///
    /// # Arguments
    /// * `q` — Query tensor, shape (seq_len_q, head_dim), row-major.
    /// * `k` — Key tensor,   shape (seq_len_kv, head_dim), row-major.
    /// * `v` — Value tensor,  shape (seq_len_kv, head_dim), row-major.
    ///
    /// # Returns
    /// `FlashAttentionOutput` with the attention output and statistics.
    pub fn forward(&self, q: &[f32], k: &[f32], v: &[f32]) -> crate::Result<FlashAttentionOutput> {
        let d = self.config.head_dim;
        self.validate_config()?;
        let seq_q = q.len() / d;
        let seq_kv = k.len() / d;

        if q.len() != seq_q * d || k.len() != seq_kv * d || v.len() != seq_kv * d {
            return Err(crate::error::CudaRustError::RuntimeError(
                "Flash Attention: tensor dimensions must be divisible by head_dim".into(),
            ));
        }

        let scale = self.config.softmax_scale();
        let bq = self.config.block_size_q;
        let bkv = self.config.block_size_kv;

        // Output accumulator and logsumexp
        let mut output = vec![0.0f32; seq_q * d];
        let mut logsumexp = vec![f32::NEG_INFINITY; seq_q];

        // Running softmax statistics per query row
        let mut m_i = vec![f32::NEG_INFINITY; seq_q]; // row max
        let mut l_i = vec![0.0f32; seq_q];            // row sum of exp

        // This CPU implementation holds one score row, not a Q/K/V SRAM tile.
        // Reuse the row for every query and tile instead of allocating in the loop.
        let mut dots = Vec::with_capacity(bkv.min(seq_kv));
        let peak_sram = dots.capacity().saturating_mul(std::mem::size_of::<f32>());

        // Outer loop: iterate over query tiles
        let num_q_tiles = seq_q.div_ceil(bq);
        let num_kv_tiles = seq_kv.div_ceil(bkv);

        for qi in 0..num_q_tiles {
            let q_start = qi * bq;
            let q_end = q_start.saturating_add(bq).min(seq_q);
            let q_rows = q_end - q_start;

            // Inner loop: iterate over key/value tiles
            let kv_limit = if self.config.causal {
                // For causal masking, only attend to kv positions <= max query position
                let max_q_pos = q_end - 1;
                (max_q_pos / bkv + 1).min(num_kv_tiles)
            } else {
                num_kv_tiles
            };

            for kvi in 0..kv_limit {
                let kv_start = kvi * bkv;
                let kv_end = kv_start.saturating_add(bkv).min(seq_kv);
                let kv_rows = kv_end - kv_start;

                // Compute S_tile = Q_tile · K_tile^T  (q_rows × kv_rows)
                // Then apply softmax scaling and causal mask
                for qi_local in 0..q_rows {
                    let qi_global = q_start + qi_local;

                    // Compute dot products for this query row
                    let mut row_max = m_i[qi_global];

                    // First pass: find new max (for numerical stability)
                    dots.clear();
                    for kvi_local in 0..kv_rows {
                        let kvi_global = kv_start + kvi_local;

                        // Causal mask: skip future positions
                        if self.config.causal && kvi_global > qi_global {
                            dots.push(f32::NEG_INFINITY);
                            continue;
                        }

                        let mut dot = 0.0f32;
                        for dd in 0..d {
                            dot += q[qi_global * d + dd] * k[kvi_global * d + dd];
                        }
                        dot *= scale;
                        dots.push(dot);
                        if dot > row_max {
                            row_max = dot;
                        }
                    }

                    // Online softmax update (Milakov & Gimelshein, 2018)
                    let old_max = m_i[qi_global];
                    let new_max = row_max;

                    // Correction factor to rescale previous accumulator
                    let correction = if old_max == f32::NEG_INFINITY {
                        0.0
                    } else {
                        (old_max - new_max).exp()
                    };

                    // Rescale previous output accumulator BEFORE adding new values
                    for dd in 0..d {
                        output[qi_global * d + dd] *= correction;
                    }

                    // Accumulate new exp(s - new_max) * V
                    let mut new_sum = 0.0f32;
                    for kvi_local in 0..kv_rows {
                        let s = dots[kvi_local];
                        if s == f32::NEG_INFINITY {
                            continue;
                        }
                        let p = (s - new_max).exp();
                        new_sum += p;

                        let kvi_global = kv_start + kvi_local;
                        for dd in 0..d {
                            output[qi_global * d + dd] += p * v[kvi_global * d + dd];
                        }
                    }

                    // Update running statistics
                    l_i[qi_global] = l_i[qi_global] * correction + new_sum;
                    m_i[qi_global] = new_max;
                }
            }
        }

        // Final normalization: output[i] /= l_i[i]
        for qi in 0..seq_q {
            let denom = if l_i[qi] > 0.0 { l_i[qi] } else { 1.0 };
            logsumexp[qi] = m_i[qi] + denom.ln();
            for dd in 0..d {
                output[qi * d + dd] /= denom;
            }
        }

        let flops = 4u64.saturating_mul(seq_q as u64)
            .saturating_mul(seq_kv as u64).saturating_mul(d as u64);

        Ok(FlashAttentionOutput {
            output,
            logsumexp,
            flops,
            peak_sram_bytes: peak_sram,
        })
    }

    /// Multi-head attention forward pass.
    ///
    /// # Arguments
    /// * `q` — (batch, num_heads, seq_len_q, head_dim) flattened.
    /// * `k` — (batch, num_heads, seq_len_kv, head_dim) flattened.
    /// * `v` — (batch, num_heads, seq_len_kv, head_dim) flattened.
    /// * `batch_size` — Number of sequences in batch.
    /// * `seq_len_q` — Query sequence length.
    /// * `seq_len_kv` — Key/value sequence length.
    pub fn forward_multi_head(
        &self,
        q: &[f32], k: &[f32], v: &[f32],
        batch_size: usize, seq_len_q: usize, seq_len_kv: usize,
    ) -> crate::Result<FlashAttentionOutput> {
        let d = self.config.head_dim;
        let h = self.config.num_heads;
        self.validate_config()?;
        let heads = batch_size.checked_mul(h)
            .ok_or_else(|| crate::runtime_error!("Flash Attention MHA: dimension overflow"))?;
        let expected_q = heads.checked_mul(seq_len_q).and_then(|x| x.checked_mul(d))
            .ok_or_else(|| crate::runtime_error!("Flash Attention MHA: dimension overflow"))?;
        let expected_kv = heads.checked_mul(seq_len_kv).and_then(|x| x.checked_mul(d))
            .ok_or_else(|| crate::runtime_error!("Flash Attention MHA: dimension overflow"))?;

        if q.len() != expected_q || k.len() != expected_kv || v.len() != expected_kv {
            return Err(crate::error::CudaRustError::RuntimeError(
                format!("Flash Attention MHA: expected q={}, k=v={}, got q={}, k={}, v={}",
                    expected_q, expected_kv, q.len(), k.len(), v.len()),
            ));
        }

        let head_q_size = seq_len_q * d;
        let head_kv_size = seq_len_kv * d;
        let mut all_output = vec![0.0f32; expected_q];
        let mut all_lse = vec![0.0f32; batch_size * h * seq_len_q];
        let mut total_flops = 0u64;
        let mut peak_sram = 0usize;

        for b in 0..batch_size {
            for head in 0..h {
                let q_offset = (b * h + head) * head_q_size;
                let kv_offset = (b * h + head) * head_kv_size;
                let q_slice = &q[q_offset..q_offset + head_q_size];
                let k_slice = &k[kv_offset..kv_offset + head_kv_size];
                let v_slice = &v[kv_offset..kv_offset + head_kv_size];

                let result = self.forward(q_slice, k_slice, v_slice)?;

                let out_offset = (b * h + head) * head_q_size;
                all_output[out_offset..out_offset + head_q_size]
                    .copy_from_slice(&result.output);

                let lse_offset = (b * h + head) * seq_len_q;
                all_lse[lse_offset..lse_offset + seq_len_q]
                    .copy_from_slice(&result.logsumexp);

                total_flops = total_flops.saturating_add(result.flops);
                if result.peak_sram_bytes > peak_sram {
                    peak_sram = result.peak_sram_bytes;
                }
            }
        }

        Ok(FlashAttentionOutput {
            output: all_output,
            logsumexp: all_lse,
            flops: total_flops,
            peak_sram_bytes: peak_sram,
        })
    }

    fn validate_config(&self) -> crate::Result<()> {
        if self.config.head_dim == 0 || self.config.num_heads == 0 ||
            self.config.block_size_q == 0 || self.config.block_size_kv == 0 {
            return Err(crate::runtime_error!("Flash Attention: head and tile dimensions must be nonzero"));
        }
        if !self.config.softmax_scale().is_finite() {
            return Err(crate::runtime_error!("Flash Attention: scale must be finite"));
        }
        if self.config.dropout_p != 0.0 {
            return Err(crate::error::CudaRustError::NotImplemented(
                "Flash Attention: CPU reference supports inference with dropout_p=0 only".into()));
        }
        Ok(())
    }

    /// Estimate memory savings vs naive attention.
    pub fn memory_savings(&self, seq_len: usize) -> MemorySavings {
        let naive_bytes = seq_len.saturating_mul(seq_len).saturating_mul(4);
        let flash_bytes = self.config.block_size_kv.min(seq_len).saturating_mul(4)
            .saturating_add(seq_len.saturating_mul(4).saturating_mul(3));

        MemorySavings {
            naive_bytes,
            flash_bytes,
            reduction_factor: naive_bytes as f64 / flash_bytes as f64,
            seq_len,
        }
    }
}

/// Memory savings comparison between naive and flash attention.
#[derive(Debug, Clone)]
pub struct MemorySavings {
    pub naive_bytes: usize,
    pub flash_bytes: usize,
    pub reduction_factor: f64,
    pub seq_len: usize,
}

impl fmt::Display for MemorySavings {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "seq_len={}: naive={:.1}MB, flash={:.1}KB, {:.0}x reduction",
            self.seq_len,
            self.naive_bytes as f64 / 1_048_576.0,
            self.flash_bytes as f64 / 1024.0,
            self.reduction_factor)
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn naive_attention(q: &[f32], k: &[f32], v: &[f32], d: usize, scale: f32) -> Vec<f32> {
        let seq_q = q.len() / d;
        let seq_kv = k.len() / d;
        let mut output = vec![0.0f32; seq_q * d];

        for i in 0..seq_q {
            // Compute scores
            let mut scores = vec![0.0f32; seq_kv];
            let mut max_score = f32::NEG_INFINITY;
            for j in 0..seq_kv {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q[i * d + dd] * k[j * d + dd];
                }
                scores[j] = dot * scale;
                if scores[j] > max_score {
                    max_score = scores[j];
                }
            }
            // Softmax
            let mut sum_exp = 0.0f32;
            for j in 0..seq_kv {
                scores[j] = (scores[j] - max_score).exp();
                sum_exp += scores[j];
            }
            for j in 0..seq_kv {
                scores[j] /= sum_exp;
            }
            // Weighted sum of V
            for j in 0..seq_kv {
                for dd in 0..d {
                    output[i * d + dd] += scores[j] * v[j * d + dd];
                }
            }
        }
        output
    }

    #[test]
    fn test_flash_attention_basic() {
        let d = 4;
        let seq = 8;
        let config = FlashAttentionConfig {
            head_dim: d,
            num_heads: 1,
            block_size_q: 4,
            block_size_kv: 4,
            causal: false,
            scale: None,
            dropout_p: 0.0,
        };

        // Simple Q=K=V for testing
        let qkv: Vec<f32> = (0..seq * d).map(|i| (i as f32) * 0.1).collect();
        let fa = FlashAttention::new(config.clone());
        let result = fa.forward(&qkv, &qkv, &qkv).unwrap();

        let scale = 1.0 / (d as f32).sqrt();
        let naive = naive_attention(&qkv, &qkv, &qkv, d, scale);

        // Check output is close to naive (not exact due to online softmax numerics)
        assert_eq!(result.output.len(), naive.len());
        for i in 0..result.output.len() {
            assert!((result.output[i] - naive[i]).abs() < 0.1,
                "Mismatch at {}: flash={}, naive={}", i, result.output[i], naive[i]);
        }
    }

    #[test]
    fn test_flash_attention_causal() {
        let d = 4;
        let seq = 6;
        let config = FlashAttentionConfig::new(d, 1).with_causal(true);
        let fa = FlashAttention::new(config);

        let q: Vec<f32> = (0..seq * d).map(|i| ((i % 7) as f32) * 0.1).collect();
        let k = q.clone();
        let v: Vec<f32> = (0..seq * d).map(|i| ((i % 5) as f32) * 0.2).collect();

        let result = fa.forward(&q, &k, &v).unwrap();
        assert_eq!(result.output.len(), seq * d);
        // First row should only attend to itself
        // Output should be valid (not NaN)
        for val in &result.output {
            assert!(!val.is_nan(), "Output contains NaN");
        }
    }

    #[test]
    fn test_flash_attention_memory_savings() {
        let config = FlashAttentionConfig::new(64, 8);
        let fa = FlashAttention::new(config);

        let savings = fa.memory_savings(2048);
        assert!(savings.reduction_factor > 10.0,
            "Expected >10x reduction, got {:.1}x", savings.reduction_factor);

        let savings_large = fa.memory_savings(8192);
        assert!(savings_large.reduction_factor > savings.reduction_factor,
            "Savings should increase with sequence length");
    }

    #[test]
    fn test_flash_attention_multi_head() {
        let d = 4;
        let h = 2;
        let seq = 4;
        let batch = 1;
        let config = FlashAttentionConfig::new(d, h);
        let fa = FlashAttention::new(config);

        let total = batch * h * seq * d;
        let q: Vec<f32> = (0..total).map(|i| (i as f32) * 0.01).collect();
        let k = q.clone();
        let v = q.clone();

        let result = fa.forward_multi_head(&q, &k, &v, batch, seq, seq).unwrap();
        assert_eq!(result.output.len(), total);
        assert_eq!(result.logsumexp.len(), batch * h * seq);
    }

    #[test]
    fn test_flash_attention_flops() {
        let d = 64;
        let seq = 128;
        let config = FlashAttentionConfig::new(d, 1);
        let fa = FlashAttention::new(config);

        let q = vec![0.1f32; seq * d];
        let k = q.clone();
        let v = q.clone();

        let result = fa.forward(&q, &k, &v).unwrap();
        let expected_flops = 4 * (seq as u64) * (seq as u64) * (d as u64);
        assert_eq!(result.flops, expected_flops);
    }

    #[test]
    fn test_flash_attention_dimension_error() {
        let config = FlashAttentionConfig::new(4, 1);
        let fa = FlashAttention::new(config);

        // Wrong dimension: 7 is not divisible by head_dim=4... actually it makes seq=1 with remainder
        let q = vec![0.1f32; 7];
        let k = vec![0.1f32; 4];
        let v = vec![0.1f32; 4];
        // seq_q = 7/4 = 1, but 1*4 != 7 => this should still work as seq_q=1 with 4 elements...
        // Actually 7/4=1 (integer), 1*4=4 != 7, so error
        let result = fa.forward(&q, &k, &v);
        assert!(result.is_err());
    }

    #[test]
    fn test_flash_attention_single_token() {
        let d = 8;
        let config = FlashAttentionConfig::new(d, 1);
        let fa = FlashAttention::new(config);

        let q = vec![1.0f32; d];
        let k = vec![1.0f32; d];
        let v: Vec<f32> = (0..d).map(|i| i as f32).collect();

        let result = fa.forward(&q, &k, &v).unwrap();
        // With single token, output should equal v (softmax of single element = 1.0)
        for i in 0..d {
            assert!((result.output[i] - v[i]).abs() < 1e-5,
                "Single token: expected {}, got {}", v[i], result.output[i]);
        }
    }
}
