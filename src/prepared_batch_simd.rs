//! Independent sample lanes; every lane keeps the original connection order.
//! Constructors/extracts avoid unsafe raw vector loads. Multiply and add remain
//! separate operations; no horizontal reduction, fast math or fused operation.

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
