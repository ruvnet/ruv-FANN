//! Structure-of-arrays (SoA) GEMV fast path for dense feed-forward layers.
//!
//! The per-neuron `Vec<Connection>` layout stays the source of truth; this
//! module maintains a derived, contiguous row-major weight matrix per layer so
//! the forward pass runs as a cache-friendly, autovectorizable GEMV with the
//! activation fused into the output write.
//!
//! Dirty-flag contract: every in-crate mutation of weights, topology, or
//! activation settings must call [`GemvCache::mark_dirty`]; `Network::run`
//! rebuilds the cache lazily before the next forward pass. Code that mutates
//! `Network::layers` through the public fields directly (outside the crate's
//! own APIs) must trigger a rebuild itself, e.g. via `Network::set_weights`.

use crate::layer::Layer;
use crate::neuron::{apply_activation, Neuron};
use crate::ActivationFunction;
use num_traits::Float;

/// Cached dense representation of one layer transition.
#[derive(Debug, Clone)]
pub(crate) struct LayerGemv<T> {
    /// Row-major weights: `out_dim` rows of `in_dim` columns. The previous
    /// layer's bias neuron is an ordinary input (constant 1.0), so its weight
    /// is just another column — no separate bias vector.
    weights: Vec<T>,
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
    func: ActivationFunction,
    steepness: T,
}

/// Per-network cache of [`LayerGemv`] plans, indexed by layer.
#[derive(Debug, Clone)]
pub(crate) struct GemvCache<T> {
    /// `plans[i]` serves `layers[i]`; `None` means the layer is not a standard
    /// dense layer (sparse/cascade/irregular) and takes the per-neuron path.
    plans: Vec<Option<LayerGemv<T>>>,
    dirty: bool,
}

impl<T> Default for GemvCache<T> {
    fn default() -> Self {
        GemvCache {
            plans: Vec::new(),
            dirty: true,
        }
    }
}

impl<T: Float> GemvCache<T> {
    #[inline]
    pub(crate) fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    #[inline]
    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Rebuilds all layer plans from the AoS connection lists.
    pub(crate) fn rebuild(&mut self, layers: &[Layer<T>]) {
        self.plans.clear();
        self.plans.resize_with(layers.len(), || None);
        for i in 1..layers.len() {
            self.plans[i] = build_plan(&layers[i], layers[i - 1].neurons.len());
        }
        self.dirty = false;
    }

    #[inline]
    pub(crate) fn plan(&self, layer_idx: usize) -> Option<&LayerGemv<T>> {
        self.plans.get(layer_idx).and_then(|p| p.as_ref())
    }
}

/// Detects whether `layer` is a standard fully-connected layer and, if so,
/// packs its weights into a row-major matrix.
///
/// Dense criteria (all must hold; true for `NetworkBuilder`-made MLPs):
/// - non-bias neurons form a contiguous prefix, any bias neurons trail;
/// - all non-bias neurons share one activation function and steepness;
/// - every non-bias neuron has exactly `prev_len` connections whose
///   `from_neuron` indices are `0..prev_len` in order.
fn build_plan<T: Float>(layer: &Layer<T>, prev_len: usize) -> Option<LayerGemv<T>> {
    if prev_len == 0 {
        return None;
    }
    let out_dim = layer.neurons.iter().take_while(|n| !n.is_bias).count();
    if out_dim == 0 || layer.neurons[out_dim..].iter().any(|n| !n.is_bias) {
        return None;
    }

    let func = layer.neurons[0].activation_function;
    let steepness = layer.neurons[0].activation_steepness;
    let mut weights = Vec::with_capacity(out_dim * prev_len);
    for neuron in &layer.neurons[..out_dim] {
        if neuron.activation_function != func
            || neuron.activation_steepness != steepness
            || neuron.connections.len() != prev_len
        {
            return None;
        }
        for (j, conn) in neuron.connections.iter().enumerate() {
            if conn.from_neuron != j {
                return None;
            }
            weights.push(conn.weight);
        }
    }

    Some(LayerGemv {
        weights,
        in_dim: prev_len,
        out_dim,
        func,
        steepness,
    })
}

/// Runs one dense layer: GEMV over the cached matrix with the activation
/// fused into the same output-write loop. Writes `sum`/`value` back into the
/// neurons so downstream consumers (training, introspection) see identical
/// state to the per-neuron path.
///
/// The activation variant is resolved once per layer; each arm monomorphizes
/// the row loop so the per-element dispatch folds away (mirrors
/// `layer::calculate_uniform`).
pub(crate) fn run_layer<T: Float>(plan: &LayerGemv<T>, prev: &[T], neurons: &mut [Neuron<T>]) {
    debug_assert_eq!(prev.len(), plan.in_dim);
    use ActivationFunction as AF;
    match plan.func {
        AF::Linear => gemv_rows(plan, prev, neurons, |k, x| apply_activation(AF::Linear, k, x)),
        AF::Sigmoid => gemv_rows(plan, prev, neurons, |k, x| apply_activation(AF::Sigmoid, k, x)),
        AF::ReLU => gemv_rows(plan, prev, neurons, |k, x| apply_activation(AF::ReLU, k, x)),
        AF::ReLULeaky => gemv_rows(plan, prev, neurons, |k, x| {
            apply_activation(AF::ReLULeaky, k, x)
        }),
        AF::Tanh | AF::SigmoidSymmetric => {
            gemv_rows(plan, prev, neurons, |k, x| apply_activation(AF::Tanh, k, x))
        }
        AF::Gaussian => gemv_rows(plan, prev, neurons, |k, x| {
            apply_activation(AF::Gaussian, k, x)
        }),
        other => gemv_rows(plan, prev, neurons, move |k, x| apply_activation(other, k, x)),
    }
}

#[inline]
fn gemv_rows<T: Float>(
    plan: &LayerGemv<T>,
    prev: &[T],
    neurons: &mut [Neuron<T>],
    f: impl Fn(T, T) -> T,
) {
    let rows = plan.weights.chunks_exact(plan.in_dim);
    for (neuron, row) in neurons.iter_mut().zip(rows) {
        let sum = dot4(row, prev);
        neuron.sum = sum;
        neuron.value = f(plan.steepness, sum);
    }
}

/// Dot product with 4 independent accumulators to break the FP dependency
/// chain (the LLVM autovectorization trigger; see docs/research/09).
///
/// Numerics note: splitting into 4 lanes reassociates the float additions
/// relative to the sequential per-connection loop, so results can differ from
/// the AoS path by rounding error (bounded by the usual O(n·eps) dot-product
/// error). The equivalence test below asserts <= 1e-4 relative on a
/// MNIST-sized net.
#[inline]
fn dot4<T: Float>(row: &[T], x: &[T]) -> T {
    let mut s0 = T::zero();
    let mut s1 = T::zero();
    let mut s2 = T::zero();
    let mut s3 = T::zero();
    let mut rc = row.chunks_exact(4);
    let mut xc = x.chunks_exact(4);
    for (r, v) in rc.by_ref().zip(xc.by_ref()) {
        s0 = s0 + r[0] * v[0];
        s1 = s1 + r[1] * v[1];
        s2 = s2 + r[2] * v[2];
        s3 = s3 + r[3] * v[3];
    }
    let mut tail = T::zero();
    for (r, v) in rc.remainder().iter().zip(xc.remainder()) {
        tail = tail + *r * *v;
    }
    (s0 + s1) + (s2 + s3) + tail
}

#[cfg(test)]
mod tests {
    use crate::{Network, NetworkBuilder};

    /// Reference forward pass using the unchanged per-neuron AoS path.
    fn run_reference(net: &mut Network<f32>, input: &[f32]) -> Vec<f32> {
        net.layers[0]
            .set_inputs(input)
            .expect("input size mismatch");
        for i in 1..net.layers.len() {
            let prev: Vec<f32> = net.layers[i - 1].neurons.iter().map(|n| n.value).collect();
            net.layers[i].calculate(&prev);
        }
        net.layers
            .last()
            .unwrap()
            .neurons
            .iter()
            .filter(|n| !n.is_bias)
            .map(|n| n.value)
            .collect()
    }

    #[test]
    fn gemv_fast_path_matches_per_neuron_path() {
        let mut net = Network::<f32>::new(&[784, 128, 64, 10]);
        net.randomize_weights(-0.5, 0.5);
        let input: Vec<f32> = (0..784).map(|i| ((i as f32) * 0.37).sin()).collect();

        let mut reference = net.clone();
        let expected = run_reference(&mut reference, &input);

        let got = net.run(&input).unwrap();

        // The fast path must actually have engaged on a builder-made MLP.
        assert!(!net.gemv_cache.is_dirty());
        for i in 1..net.layers.len() {
            assert!(net.gemv_cache.plan(i).is_some(), "layer {i} not dense");
        }

        assert_eq!(got.len(), expected.len());
        let mut max_rel = 0.0f32;
        for (a, b) in got.iter().zip(&expected) {
            let rel = (a - b).abs() / b.abs().max(1e-6);
            max_rel = max_rel.max(rel);
        }
        eprintln!("gemv equivalence max relative delta: {max_rel:e}");
        assert!(max_rel <= 1e-4, "max relative delta {max_rel} > 1e-4");
    }

    #[test]
    fn gemv_hidden_state_matches_per_neuron_path() {
        // Training reads neuron.sum/value on hidden layers; they must match too.
        let mut net = Network::<f32>::new(&[10, 20, 10]);
        net.randomize_weights(-1.0, 1.0);
        let input: Vec<f32> = (0..10).map(|i| (i as f32) / 10.0 - 0.5).collect();

        let mut reference = net.clone();
        let _ = run_reference(&mut reference, &input);
        let _ = net.run(&input).unwrap();

        for (l, (fast, refr)) in net.layers.iter().zip(&reference.layers).enumerate() {
            for (a, b) in fast.neurons.iter().zip(&refr.neurons) {
                assert!(
                    (a.sum - b.sum).abs() <= 1e-5 && (a.value - b.value).abs() <= 1e-5,
                    "layer {l} neuron state diverged: {} vs {}",
                    a.value,
                    b.value
                );
            }
        }
    }

    #[test]
    fn weight_mutations_invalidate_cache() {
        let mut net = Network::<f32>::new(&[2, 3, 1]);
        let input = [0.25f32, -0.5];
        let before = net.run(&input).unwrap();
        assert!(!net.gemv_cache.is_dirty());

        // set_weights must write through to the next run.
        let weights = vec![0.42f32; net.total_connections()];
        net.set_weights(&weights).unwrap();
        assert!(net.gemv_cache.is_dirty());
        let after = net.run(&input).unwrap();
        assert_ne!(before, after);

        // Reference agreement after mutation.
        let mut reference = net.clone();
        let expected = super::tests::run_reference(&mut reference, &input);
        let got = net.run(&input).unwrap();
        for (a, b) in got.iter().zip(&expected) {
            assert!((a - b).abs() <= 1e-6);
        }

        // Training through Network::train must also invalidate.
        net.train(&[vec![0.0, 1.0]], &[vec![1.0]], 0.5, 1).unwrap();
        assert!(net.gemv_cache.is_dirty());
    }

    #[test]
    fn sparse_network_falls_back_to_per_neuron_path() {
        let mut net = NetworkBuilder::<f32>::new()
            .input_layer(10)
            .hidden_layer(10)
            .output_layer(10)
            .connection_rate(0.5)
            .build();
        let input = vec![0.5f32; 10];
        let out = net.run(&input).unwrap();
        assert_eq!(out.len(), 10);
        // Sparse layers must not get a dense plan.
        assert!(!net.gemv_cache.is_dirty());
        let dense_layers = (1..net.layers.len())
            .filter(|&i| net.gemv_cache.plan(i).is_some())
            .count();
        assert_eq!(dense_layers, 0, "sparse layers wrongly detected as dense");
    }
}
