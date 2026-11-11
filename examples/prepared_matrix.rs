//! Reproducible warm single-inference microbenchmark, without extra dependencies.
//!
//! Run with `cargo run --release --no-default-features --features std --example
//! prepared_matrix`. Output is JSON Lines: one correctness signature and 21 raw
//! timing blocks per case. `--case NAME` selects a case, `--list` lists names,
//! `--blocks N` changes block count, and `--block-ms N` changes the minimum block
//! duration (default 2 ms). Compare identical source, toolchain, flags and CPU
//! affinity across revisions, and alternate revision order over repeated runs.
//! Network construction, preparation, workspace allocation, scalar comparisons
//! and calibration are excluded from reported timing. This measures warm calls,
//! including validation, output copying, input selection and black-box barriers.

use num_traits::Float;
use ruv_fann::{ActivationFunction, Layer, Network};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

trait BenchFloat: Float + std::fmt::Debug {
    const NAME: &'static str;
    fn bits(self) -> u64;
}
impl BenchFloat for f32 {
    const NAME: &'static str = "f32";
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}
impl BenchFloat for f64 {
    const NAME: &'static str = "f64";
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

#[derive(Clone, Copy)]
enum Topology {
    Dense,
    Mixed,
    Sparse,
    Reordered,
}
impl Topology {
    fn name(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Mixed => "mixed",
            Self::Sparse => "sparse",
            Self::Reordered => "reordered",
        }
    }
}
fn random(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    ((*state >> 11) as f64 / ((1u64 << 53) as f64)) * 2.0 - 1.0
}
fn hash(hash: &mut u64, bits: u64) {
    for byte in bits.to_le_bytes() {
        *hash = (*hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
}
fn network<T: BenchFloat>(
    shape: &[usize],
    activation: ActivationFunction,
    topology: Topology,
) -> Network<T> {
    let mut state = 0x7834_a621_b89c_d017;
    // Build directly: no ambient random seed or builder RNG affects the fixture.
    let mut layers: Vec<Layer<T>> = shape
        .iter()
        .enumerate()
        .map(|(index, &width)| {
            if index + 1 == shape.len() {
                Layer::new(width, activation, T::one())
            } else {
                Layer::with_bias(width, activation, T::one())
            }
        })
        .collect();
    for layer in 1..layers.len() {
        let source_width = layers[layer - 1].size();
        for (row, neuron) in layers[layer]
            .neurons
            .iter_mut()
            .filter(|n| !n.is_bias)
            .enumerate()
        {
            for col in 0..source_width {
                let weight = T::from(random(&mut state) / (source_width as f64).sqrt()).unwrap();
                let omit = match topology {
                    Topology::Sparse => col % 3 == row % 3,
                    Topology::Mixed => row % 5 == 1 && col % 3 == 1,
                    _ => false,
                };
                if !omit {
                    neuron.add_connection(col, weight);
                }
            }
            match topology {
                Topology::Reordered => neuron.connections.reverse(),
                Topology::Mixed if row % 5 == 3 => neuron.connections.reverse(),
                _ => (),
            }
        }
    }
    Network {
        layers,
        connection_rate: T::one(),
    }
}
fn inputs<T: BenchFloat>(width: usize) -> Vec<Vec<T>> {
    let mut state = 0x432f_b067_a184_c359;
    (0..8)
        .map(|sample| {
            (0..width)
                .map(|col| {
                    let value = match sample {
                        0 => 0.0,
                        1 => -0.0,
                        2 => {
                            if col % 2 == 0 {
                                1.0
                            } else {
                                -1.0
                            }
                        }
                        3 => {
                            if col % 2 == 0 {
                                1e-10
                            } else {
                                -1e-10
                            }
                        }
                        _ => random(&mut state),
                    };
                    T::from(value).unwrap()
                })
                .collect()
        })
        .collect()
}

struct Options {
    selected: Option<String>,
    list: bool,
    blocks: usize,
    minimum: Duration,
}
fn case<T: BenchFloat>(
    shape: &[usize],
    activation: ActivationFunction,
    topology: Topology,
    options: &Options,
) {
    let shape_name = shape
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join("x");
    let activation_name = match activation {
        ActivationFunction::Linear => "linear",
        _ => "sigmoid",
    };
    let name = format!(
        "{}_{activation_name}_{}_{}",
        T::NAME,
        topology.name(),
        shape_name
    );
    if options.list {
        println!("{name}");
        return;
    }
    if options
        .selected
        .as_deref()
        .is_some_and(|selected| selected != name)
    {
        return;
    }
    let mut scalar = network::<T>(shape, activation, topology);
    let prepared = scalar.prepare().unwrap();
    let mut workspace = prepared.workspace().unwrap();
    let mut output = vec![T::zero(); *shape.last().unwrap()];
    let inputs = inputs::<T>(shape[0]);
    let mut signature = 0xcbf29ce484222325;
    let mut fixture_signature = signature;
    for weight in scalar.get_weights() {
        hash(&mut fixture_signature, weight.bits());
    }
    for input in &inputs {
        for value in input {
            hash(&mut fixture_signature, value.bits());
        }
        let reference = scalar.run(input).unwrap();
        prepared
            .run_into(input, &mut output, &mut workspace)
            .unwrap();
        for (&actual, &expected) in output.iter().zip(&reference) {
            assert_eq!(
                actual.bits(),
                expected.bits(),
                "scalar mismatch in {name}: {actual:?} versus {expected:?}"
            );
            hash(&mut signature, actual.bits());
        }
    }
    println!("{{\"kind\":\"correctness\",\"case\":\"{name}\",\"fixture_signature\":\"{fixture_signature:016x}\",\"output_signature\":\"{signature:016x}\",\"samples\":8}}");
    let mut run = |iterations: usize| {
        let start = Instant::now();
        for iteration in 0..iterations {
            prepared
                .run_into(
                    black_box(&inputs[iteration & 7]),
                    black_box(&mut output),
                    black_box(&mut workspace),
                )
                .unwrap();
            black_box(&output);
        }
        start.elapsed()
    };
    // Calibration and warmup are deliberately unreported. Two minimum durations
    // provide headroom so ordinary fluctuation does not make blocks too short.
    let target = options.minimum.saturating_mul(2);
    let mut iterations = 8usize;
    loop {
        let elapsed = run(iterations);
        if elapsed >= target {
            break;
        }
        let factor = (target.as_secs_f64() / elapsed.as_secs_f64().max(1e-9))
            .ceil()
            .clamp(2.0, 16.0) as usize;
        iterations = iterations.checked_mul(factor).expect("iteration overflow");
    }
    for _ in 0..3 {
        run(iterations);
    }
    for block in 0..options.blocks {
        let mut elapsed = run(iterations);
        // Retry undersized blocks, preserving only measurements meeting the
        // requested duration, with that block's actual iteration count.
        while elapsed < options.minimum {
            iterations = iterations.checked_mul(2).expect("iteration overflow");
            elapsed = run(iterations);
        }
        let ns = elapsed.as_nanos();
        let ns_per_call = ns as f64 / iterations as f64;
        println!("{{\"kind\":\"block\",\"case\":\"{name}\",\"block\":{block},\"iterations\":{iterations},\"elapsed_ns\":{ns},\"ns_per_call\":{ns_per_call:.9}}}");
    }
}
fn matrix<T: BenchFloat>(options: &Options) {
    for activation in [ActivationFunction::Linear, ActivationFunction::Sigmoid] {
        for shape in [
            &[10, 1][..],
            &[10, 4, 1],
            &[5, 7, 3],
            &[33, 17, 5],
            &[63, 17, 5],
            &[64, 17, 5],
            &[65, 17, 5],
            &[100, 50, 25, 10],
            &[784, 128, 64, 10],
        ] {
            case::<T>(shape, activation, Topology::Dense, options);
        }
        for topology in [Topology::Mixed, Topology::Sparse, Topology::Reordered] {
            for shape in [&[33, 17, 5][..], &[65, 17, 5]] {
                case::<T>(shape, activation, topology, options);
            }
        }
    }
}
fn main() {
    let mut options = Options {
        selected: None,
        list: false,
        blocks: 21,
        minimum: Duration::from_millis(2),
    };
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        match argument.as_str() {
            "--list" => options.list = true,
            "--case" => options.selected = Some(args.next().expect("--case requires a name")),
            "--blocks" => {
                options.blocks = args
                    .next()
                    .expect("--blocks requires a count")
                    .parse()
                    .unwrap()
            }
            "--block-ms" => {
                options.minimum = Duration::from_secs_f64(
                    args.next()
                        .expect("--block-ms requires milliseconds")
                        .parse::<f64>()
                        .unwrap()
                        / 1000.0,
                )
            }
            _ => panic!("unknown argument: {argument}"),
        }
    }
    assert!(options.blocks > 0 && options.minimum > Duration::ZERO);
    matrix::<f32>(&options);
    matrix::<f64>(&options);
}
