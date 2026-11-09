//! Independent sample lanes; every lane keeps the original connection order.
//! Constructors/extracts avoid unsafe raw vector loads. Multiply and add remain
//! separate operations; no horizontal reduction, fast math or fused operation.

use num_traits::Float;

// Four output rows share each feature load. Every accumulator still starts at
// positive zero and visits features in their original order. Fixed sample blocks
// keep sums local across the feature loop without any additional workspace.
#[inline(always)]
fn dense_scalar<T: Float, const LANES: usize>(
    source: &[T],
    stride: usize,
    weights: [&[T]; 4],
    target: &mut [T],
    start: usize,
    end: usize,
) {
    for sample in (start..end).step_by(LANES) {
        let mut a = [T::zero(); LANES];
        let mut b = a;
        let mut c = a;
        let mut d = a;
        for (feature, (((&wa, &wb), &wc), &wd)) in weights[0]
            .iter()
            .zip(weights[1])
            .zip(weights[2])
            .zip(weights[3])
            .enumerate()
        {
            let values = &source[feature * stride + sample..][..LANES];
            for lane in 0..LANES {
                let value = values[lane];
                a[lane] = a[lane] + value * wa;
                b[lane] = b[lane] + value * wb;
                c[lane] = c[lane] + value * wc;
                d[lane] = d[lane] + value * wd;
            }
        }
        for lane in 0..LANES {
            target[sample + lane] = a[lane];
            target[stride + sample + lane] = b[lane];
            target[2 * stride + sample + lane] = c[lane];
            target[3 * stride + sample + lane] = d[lane];
        }
    }
}

#[inline(always)]
pub(super) fn dense_tile_f64(
    source: &[f64],
    stride: usize,
    weights: [&[f64]; 4],
    target: &mut [f64],
    rows: usize,
) {
    let full_rows = rows / 2 * 2;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use std::arch::wasm32::{f64x2, f64x2_add, f64x2_extract_lane, f64x2_mul, f64x2_splat};
        for sample in (0..full_rows).step_by(2) {
            let mut a = f64x2_splat(0.0);
            let (mut b, mut c, mut d) = (a, a, a);
            for (feature, (((&wa, &wb), &wc), &wd)) in weights[0]
                .iter()
                .zip(weights[1])
                .zip(weights[2])
                .zip(weights[3])
                .enumerate()
            {
                let base = feature * stride + sample;
                let value = f64x2(source[base], source[base + 1]);
                a = f64x2_add(a, f64x2_mul(value, f64x2_splat(wa)));
                b = f64x2_add(b, f64x2_mul(value, f64x2_splat(wb)));
                c = f64x2_add(c, f64x2_mul(value, f64x2_splat(wc)));
                d = f64x2_add(d, f64x2_mul(value, f64x2_splat(wd)));
            }
            for (output, sum) in [a, b, c, d].into_iter().enumerate() {
                target[output * stride + sample] = f64x2_extract_lane::<0>(sum);
                target[output * stride + sample + 1] = f64x2_extract_lane::<1>(sum);
            }
        }
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    dense_scalar::<f64, 2>(source, stride, weights, target, 0, full_rows);
    dense_scalar::<f64, 1>(source, stride, weights, target, full_rows, rows);
}

#[inline(always)]
pub(super) fn dense_tile_f32(
    source: &[f32],
    stride: usize,
    weights: [&[f32]; 4],
    target: &mut [f32],
    rows: usize,
) {
    let full_rows = rows / 4 * 4;
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use std::arch::wasm32::{f32x4, f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat};
        for sample in (0..full_rows).step_by(4) {
            let mut a = f32x4_splat(0.0);
            let (mut b, mut c, mut d) = (a, a, a);
            for (feature, (((&wa, &wb), &wc), &wd)) in weights[0]
                .iter()
                .zip(weights[1])
                .zip(weights[2])
                .zip(weights[3])
                .enumerate()
            {
                let base = feature * stride + sample;
                let value = f32x4(
                    source[base],
                    source[base + 1],
                    source[base + 2],
                    source[base + 3],
                );
                a = f32x4_add(a, f32x4_mul(value, f32x4_splat(wa)));
                b = f32x4_add(b, f32x4_mul(value, f32x4_splat(wb)));
                c = f32x4_add(c, f32x4_mul(value, f32x4_splat(wc)));
                d = f32x4_add(d, f32x4_mul(value, f32x4_splat(wd)));
            }
            for (output, sum) in [a, b, c, d].into_iter().enumerate() {
                let base = output * stride + sample;
                target[base] = f32x4_extract_lane::<0>(sum);
                target[base + 1] = f32x4_extract_lane::<1>(sum);
                target[base + 2] = f32x4_extract_lane::<2>(sum);
                target[base + 3] = f32x4_extract_lane::<3>(sum);
            }
        }
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    dense_scalar::<f32, 4>(source, stride, weights, target, 0, full_rows);
    dense_scalar::<f32, 1>(source, stride, weights, target, full_rows, rows);
}

#[inline(always)]
pub(super) fn accumulate_f64(sums: &mut [f64], values: &[f64], weight: f64) {
    debug_assert_eq!(sums.len(), values.len());
    let consumed = {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            use std::arch::wasm32::{f64x2, f64x2_add, f64x2_extract_lane, f64x2_mul, f64x2_splat};
            let weights = f64x2_splat(weight);
            for (sum, value) in sums.chunks_exact_mut(2).zip(values.chunks_exact(2)) {
                let product = f64x2_mul(f64x2(value[0], value[1]), weights);
                let result = f64x2_add(f64x2(sum[0], sum[1]), product);
                sum[0] = f64x2_extract_lane::<0>(result);
                sum[1] = f64x2_extract_lane::<1>(result);
            }
            sums.len() / 2 * 2
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            0
        }
    };
    for (sum, &value) in sums[consumed..].iter_mut().zip(&values[consumed..]) {
        *sum += value * weight;
    }
}

#[inline(always)]
pub(super) fn accumulate_f32(sums: &mut [f32], values: &[f32], weight: f32) {
    debug_assert_eq!(sums.len(), values.len());
    let consumed = {
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        {
            use std::arch::wasm32::{f32x4, f32x4_add, f32x4_extract_lane, f32x4_mul, f32x4_splat};
            let weights = f32x4_splat(weight);
            for (sum, value) in sums.chunks_exact_mut(4).zip(values.chunks_exact(4)) {
                let product = f32x4_mul(f32x4(value[0], value[1], value[2], value[3]), weights);
                let result = f32x4_add(f32x4(sum[0], sum[1], sum[2], sum[3]), product);
                sum[0] = f32x4_extract_lane::<0>(result);
                sum[1] = f32x4_extract_lane::<1>(result);
                sum[2] = f32x4_extract_lane::<2>(result);
                sum[3] = f32x4_extract_lane::<3>(result);
            }
            sums.len() / 4 * 4
        }
        #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
        {
            0
        }
    };
    for (sum, &value) in sums[consumed..].iter_mut().zip(&values[consumed..]) {
        *sum += value * weight;
    }
}
