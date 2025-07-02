#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use num_traits::Float;

/// Activation functions available for neurons
///
/// These functions are based on the FANN library's activation functions
/// and include both common neural network activation functions and
/// some specialized variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Default)]
pub enum ActivationFunction {
    /// Linear activation function: f(x) = x * steepness
    Linear,

    /// Threshold activation function: f(x) = 0 if x < 0, 1 if x >= 0
    /// Note: Cannot be used during training due to zero derivative
    Threshold,

    /// Symmetric threshold: f(x) = -1 if x < 0, 1 if x >= 0
    /// Note: Cannot be used during training due to zero derivative
    ThresholdSymmetric,

    /// Sigmoid activation function: f(x) = 1 / (1 + exp(-2 * steepness * x))
    /// Output range: (0, 1)
    #[default]
    Sigmoid,

    /// Symmetric sigmoid (tanh): f(x) = tanh(steepness * x)
    /// Output range: (-1, 1)
    SigmoidSymmetric,

    /// Hyperbolic tangent: alias for SigmoidSymmetric
    Tanh,

    /// Gaussian activation: f(x) = exp(-x * steepness * x * steepness)
    /// Output range: (0, 1], peaks at x=0
    Gaussian,

    /// Symmetric gaussian: f(x) = exp(-x * steepness * x * steepness) * 2 - 1
    /// Output range: (-1, 1], peaks at x=0
    GaussianSymmetric,

    /// Elliott activation: f(x) = ((x * steepness) / 2) / (1 + |x * steepness|) + 0.5
    /// Fast approximation to sigmoid, output range: (0, 1)
    Elliot,

    /// Symmetric Elliott: f(x) = (x * steepness) / (1 + |x * steepness|)
    /// Fast approximation to tanh, output range: (-1, 1)
    ElliotSymmetric,

    /// Bounded linear: f(x) = max(0, min(1, x * steepness))
    /// Output range: [0, 1]
    LinearPiece,

    /// Symmetric bounded linear: f(x) = max(-1, min(1, x * steepness))
    /// Output range: [-1, 1]
    LinearPieceSymmetric,

    /// Rectified Linear Unit (ReLU): f(x) = max(0, x)
    /// Output range: [0, ∞)
    ReLU,

    /// Leaky ReLU: f(x) = x if x > 0, 0.01 * x if x <= 0
    /// Output range: (-∞, ∞)
    ReLULeaky,

    /// Sine activation: f(x) = sin(x * steepness) / 2 + 0.5
    /// Output range: [0, 1]
    Sin,

    /// Cosine activation: f(x) = cos(x * steepness) / 2 + 0.5
    /// Output range: [0, 1]
    Cos,

    /// Symmetric sine: f(x) = sin(x * steepness)
    /// Output range: [-1, 1]
    SinSymmetric,

    /// Symmetric cosine: f(x) = cos(x * steepness)
    /// Output range: [-1, 1]
    CosSymmetric,
}

impl ActivationFunction {
    /// Applies the activation function to a value with the given steepness parameter
    ///
    /// # Arguments
    /// * `value` - The input value to apply the activation function to
    /// * `steepness` - The steepness parameter for the activation function
    ///
    /// # Example
    /// ```
    /// use ruv_fann::ActivationFunction;
    ///
    /// let sigmoid = ActivationFunction::Sigmoid;
    /// let result = sigmoid.activate(0.0f64, 1.0f64);
    /// assert!((result - 0.5).abs() < 1e-6);
    /// ```
    pub fn activate<T: num_traits::Float>(&self, value: T, steepness: T) -> T {
        match self {
            ActivationFunction::Linear => value * steepness,
            ActivationFunction::Threshold => {
                if value >= T::zero() { T::one() } else { T::zero() }
            },
            ActivationFunction::ThresholdSymmetric => {
                if value >= T::zero() { T::one() } else { -T::one() }
            },
            ActivationFunction::Sigmoid => {
                // Sigmoid: 1 / (1 + exp(-2 * steepness * x))
                let exp_val = (-T::from(2.0).unwrap() * steepness * value).exp();
                T::one() / (T::one() + exp_val)
            },
            ActivationFunction::SigmoidSymmetric | ActivationFunction::Tanh => {
                // Tanh: (exp(steepness * x) - exp(-steepness * x)) / (exp(steepness * x) + exp(-steepness * x))
                let exp_pos = (steepness * value).exp();
                let exp_neg = (-steepness * value).exp();
                (exp_pos - exp_neg) / (exp_pos + exp_neg)
            },
            ActivationFunction::ReLU => {
                if value > T::zero() { value } else { T::zero() }
            },
            ActivationFunction::ReLULeaky => {
                if value > T::zero() { 
                    value 
                } else { 
                    T::from(0.01).unwrap() * value 
                }
            },
            ActivationFunction::Gaussian => {
                // Gaussian: exp(-x * steepness * x * steepness)
                let val = value * steepness;
                (-val * val).exp()
            },
            ActivationFunction::GaussianSymmetric => {
                // Symmetric Gaussian: exp(-x * steepness * x * steepness) * 2 - 1
                let val = value * steepness;
                (-val * val).exp() * T::from(2.0).unwrap() - T::one()
            },
            ActivationFunction::Elliot => {
                // Elliott: ((x * steepness) / 2) / (1 + |x * steepness|) + 0.5
                let val = value * steepness;
                let abs_val = if val >= T::zero() { val } else { -val };
                (val / T::from(2.0).unwrap()) / (T::one() + abs_val) + T::from(0.5).unwrap()
            },
            ActivationFunction::ElliotSymmetric => {
                // Symmetric Elliott: (x * steepness) / (1 + |x * steepness|)
                let val = value * steepness;
                let abs_val = if val >= T::zero() { val } else { -val };
                val / (T::one() + abs_val)
            },
            ActivationFunction::LinearPiece => {
                // Bounded linear: max(0, min(1, x * steepness))
                let val = value * steepness;
                if val < T::zero() {
                    T::zero()
                } else if val > T::one() {
                    T::one()
                } else {
                    val
                }
            },
            ActivationFunction::LinearPieceSymmetric => {
                // Symmetric bounded linear: max(-1, min(1, x * steepness))
                let val = value * steepness;
                let neg_one = -T::one();
                if val < neg_one {
                    neg_one
                } else if val > T::one() {
                    T::one()
                } else {
                    val
                }
            },
            ActivationFunction::Sin => {
                // Sin: sin(x * steepness) / 2 + 0.5
                (value * steepness).sin() / T::from(2.0).unwrap() + T::from(0.5).unwrap()
            },
            ActivationFunction::Cos => {
                // Cos: cos(x * steepness) / 2 + 0.5
                (value * steepness).cos() / T::from(2.0).unwrap() + T::from(0.5).unwrap()
            },
            ActivationFunction::SinSymmetric => {
                // Symmetric Sin: sin(x * steepness)
                (value * steepness).sin()
            },
            ActivationFunction::CosSymmetric => {
                // Symmetric Cos: cos(x * steepness)
                (value * steepness).cos()
            },
        }
    }

    /// Computes the derivative of the activation function for backpropagation
    ///
    /// # Arguments
    /// * `value` - The input value (pre-activation)
    /// * `activated_value` - The output value (post-activation)
    /// * `steepness` - The steepness parameter for the activation function
    ///
    /// # Returns
    /// The derivative of the activation function at the given point
    pub fn derivative<T: num_traits::Float>(&self, value: T, activated_value: T, steepness: T) -> T {
        match self {
            ActivationFunction::Linear => steepness,
            ActivationFunction::Threshold | ActivationFunction::ThresholdSymmetric => {
                // Technically undefined, but we return 0 to indicate non-trainable
                T::zero()
            },
            ActivationFunction::Sigmoid => {
                // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x)) * steepness * 2
                activated_value * (T::one() - activated_value) * steepness * T::from(2.0).unwrap()
            },
            ActivationFunction::SigmoidSymmetric | ActivationFunction::Tanh => {
                // d/dx tanh(x) = (1 - tanh²(x)) * steepness
                (T::one() - activated_value * activated_value) * steepness
            },
            ActivationFunction::ReLU => {
                if value > T::zero() { T::one() } else { T::zero() }
            },
            ActivationFunction::ReLULeaky => {
                if value > T::zero() { T::one() } else { T::from(0.01).unwrap() }
            },
            ActivationFunction::Gaussian => {
                // d/dx exp(-x²s²) = -2xs² * exp(-x²s²)
                let val = value * steepness;
                -T::from(2.0).unwrap() * val * steepness * activated_value
            },
            ActivationFunction::GaussianSymmetric => {
                // Similar to Gaussian but scaled
                let val = value * steepness;
                -T::from(4.0).unwrap() * val * steepness * (activated_value + T::one()) / T::from(2.0).unwrap()
            },
            ActivationFunction::Elliot => {
                // Complex derivative - simplified approximation
                let val = value * steepness;
                let abs_val = if val >= T::zero() { val } else { -val };
                let denom = T::one() + abs_val;
                steepness / (T::from(2.0).unwrap() * denom * denom)
            },
            ActivationFunction::ElliotSymmetric => {
                // Derivative of symmetric Elliott
                let val = value * steepness;
                let abs_val = if val >= T::zero() { val } else { -val };
                let denom = T::one() + abs_val;
                steepness / (denom * denom)
            },
            ActivationFunction::LinearPiece => {
                let val = value * steepness;
                if val > T::zero() && val < T::one() { steepness } else { T::zero() }
            },
            ActivationFunction::LinearPieceSymmetric => {
                let val = value * steepness;
                if val > -T::one() && val < T::one() { steepness } else { T::zero() }
            },
            ActivationFunction::Sin => {
                // d/dx [sin(xs)/2 + 0.5] = cos(xs) * s / 2
                (value * steepness).cos() * steepness / T::from(2.0).unwrap()
            },
            ActivationFunction::Cos => {
                // d/dx [cos(xs)/2 + 0.5] = -sin(xs) * s / 2
                -(value * steepness).sin() * steepness / T::from(2.0).unwrap()
            },
            ActivationFunction::SinSymmetric => {
                // d/dx sin(xs) = cos(xs) * s
                (value * steepness).cos() * steepness
            },
            ActivationFunction::CosSymmetric => {
                // d/dx cos(xs) = -sin(xs) * s
                -(value * steepness).sin() * steepness
            },
        }
    }

    /// Returns the string name of the activation function
    pub fn name(&self) -> &'static str {
        match self {
            ActivationFunction::Linear => "Linear",
            ActivationFunction::Threshold => "Threshold",
            ActivationFunction::ThresholdSymmetric => "ThresholdSymmetric",
            ActivationFunction::Sigmoid => "Sigmoid",
            ActivationFunction::SigmoidSymmetric => "SigmoidSymmetric",
            ActivationFunction::Tanh => "Tanh",
            ActivationFunction::Gaussian => "Gaussian",
            ActivationFunction::GaussianSymmetric => "GaussianSymmetric",
            ActivationFunction::Elliot => "Elliot",
            ActivationFunction::ElliotSymmetric => "ElliotSymmetric",
            ActivationFunction::LinearPiece => "LinearPiece",
            ActivationFunction::LinearPieceSymmetric => "LinearPieceSymmetric",
            ActivationFunction::ReLU => "ReLU",
            ActivationFunction::ReLULeaky => "ReLULeaky",
            ActivationFunction::Sin => "Sin",
            ActivationFunction::Cos => "Cos",
            ActivationFunction::SinSymmetric => "SinSymmetric",
            ActivationFunction::CosSymmetric => "CosSymmetric",
        }
    }

    /// Returns whether this activation function can be used during training
    /// (i.e., has a computable derivative)
    pub fn is_trainable(&self) -> bool {
        !matches!(
            self,
            ActivationFunction::Threshold | ActivationFunction::ThresholdSymmetric
        )
    }

    /// Returns the output range of the activation function
    pub fn output_range(&self) -> (&'static str, &'static str) {
        match self {
            ActivationFunction::Linear => ("-inf", "inf"),
            ActivationFunction::Threshold => ("0", "1"),
            ActivationFunction::ThresholdSymmetric => ("-1", "1"),
            ActivationFunction::Sigmoid => ("0", "1"),
            ActivationFunction::SigmoidSymmetric | ActivationFunction::Tanh => ("-1", "1"),
            ActivationFunction::Gaussian => ("0", "1"),
            ActivationFunction::GaussianSymmetric => ("-1", "1"),
            ActivationFunction::Elliot => ("0", "1"),
            ActivationFunction::ElliotSymmetric => ("-1", "1"),
            ActivationFunction::LinearPiece => ("0", "1"),
            ActivationFunction::LinearPieceSymmetric => ("-1", "1"),
            ActivationFunction::ReLU => ("0", "inf"),
            ActivationFunction::ReLULeaky => ("-inf", "inf"),
            ActivationFunction::Sin | ActivationFunction::Cos => ("0", "1"),
            ActivationFunction::SinSymmetric | ActivationFunction::CosSymmetric => ("-1", "1"),
        }
    }

    /// Converts to GPU backend activation function if supported
    /// Returns None if the activation function requires CPU fallback
    #[cfg(feature = "gpu")]
    pub fn to_gpu_activation(&self) -> Option<crate::webgpu::backend::ActivationFunction> {
        use crate::webgpu::backend::ActivationFunction as GpuActivation;
        
        match self {
            ActivationFunction::Linear => Some(GpuActivation::Linear),
            ActivationFunction::Sigmoid => Some(GpuActivation::Sigmoid),
            ActivationFunction::SigmoidSymmetric | ActivationFunction::Tanh => Some(GpuActivation::Tanh),
            ActivationFunction::ReLU => Some(GpuActivation::ReLU),
            ActivationFunction::ReLULeaky => Some(GpuActivation::LeakyReLU(0.01)),
            // GPU-specific modern activations that we can support
            // These would need WGSL shader implementations
            _ => None, // Fallback to CPU for unsupported functions
        }
    }

    /// Returns whether this activation function has GPU acceleration support
    #[cfg(feature = "gpu")]
    pub fn has_gpu_support(&self) -> bool {
        self.to_gpu_activation().is_some()
    }

    #[cfg(not(feature = "gpu"))]
    pub fn has_gpu_support(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_activation_function_names() {
        assert_eq!(ActivationFunction::Sigmoid.name(), "Sigmoid");
        assert_eq!(ActivationFunction::ReLU.name(), "ReLU");
        assert_eq!(ActivationFunction::Tanh.name(), "Tanh");
    }

    #[test]
    fn test_trainable() {
        assert!(ActivationFunction::Sigmoid.is_trainable());
        assert!(ActivationFunction::ReLU.is_trainable());
        assert!(!ActivationFunction::Threshold.is_trainable());
        assert!(!ActivationFunction::ThresholdSymmetric.is_trainable());
    }

    #[test]
    fn test_output_ranges() {
        assert_eq!(ActivationFunction::Sigmoid.output_range(), ("0", "1"));
        assert_eq!(ActivationFunction::Tanh.output_range(), ("-1", "1"));
        assert_eq!(ActivationFunction::ReLU.output_range(), ("0", "inf"));
        assert_eq!(ActivationFunction::Linear.output_range(), ("-inf", "inf"));
    }

    #[test]
    fn test_activation_functions() {
        // Test Sigmoid at x=0, steepness=1.0 should give 0.5
        let result = ActivationFunction::Sigmoid.activate(0.0_f32, 1.0);
        assert!((result - 0.5).abs() < 1e-6);

        // Test Linear activation
        let result = ActivationFunction::Linear.activate(2.0_f32, 0.5);
        assert_eq!(result, 1.0);

        // Test ReLU
        let result = ActivationFunction::ReLU.activate(-1.0_f32, 1.0);
        assert_eq!(result, 0.0);
        let result = ActivationFunction::ReLU.activate(2.0_f32, 1.0);
        assert_eq!(result, 2.0);

        // Test Tanh at x=0 should give 0.0
        let result = ActivationFunction::Tanh.activate(0.0_f32, 1.0);
        assert!(result.abs() < 1e-6);

        // Test Threshold
        let result = ActivationFunction::Threshold.activate(-1.0_f32, 1.0);
        assert_eq!(result, 0.0);
        let result = ActivationFunction::Threshold.activate(1.0_f32, 1.0);
        assert_eq!(result, 1.0);
    }

    #[test]
    fn test_activation_derivatives() {
        // Test Linear derivative (should be constant steepness)
        let deriv = ActivationFunction::Linear.derivative(0.0_f32, 0.0, 2.0);
        assert_eq!(deriv, 2.0);

        // Test Sigmoid derivative
        let activated = ActivationFunction::Sigmoid.activate(0.0_f32, 1.0);
        let deriv = ActivationFunction::Sigmoid.derivative(0.0_f32, activated, 1.0);
        // At x=0, sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) * steepness * 2
        // = 0.5 * 0.5 * 1.0 * 2 = 0.5
        assert!((deriv - 0.5).abs() < 1e-6);

        // Test ReLU derivative
        let deriv = ActivationFunction::ReLU.derivative(1.0_f32, 1.0, 1.0);
        assert_eq!(deriv, 1.0);
        let deriv = ActivationFunction::ReLU.derivative(-1.0_f32, 0.0, 1.0);
        assert_eq!(deriv, 0.0);

        // Test Tanh derivative
        let activated = ActivationFunction::Tanh.activate(0.0_f32, 1.0);
        let deriv = ActivationFunction::Tanh.derivative(0.0_f32, activated, 1.0);
        // At x=0, tanh'(0) = (1 - tanh²(0)) * steepness = (1 - 0) * 1 = 1
        assert!((deriv - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_numerical_stability() {
        // Test with different numeric types
        let result_f32 = ActivationFunction::Sigmoid.activate(0.0_f32, 1.0_f32);
        let result_f64 = ActivationFunction::Sigmoid.activate(0.0_f64, 1.0_f64);
        assert!((result_f32 as f64 - result_f64).abs() < 1e-6);

        // Test extreme values don't cause overflow/underflow
        let result = ActivationFunction::Sigmoid.activate(100.0_f32, 1.0);
        assert!(result.is_finite() && result > 0.0 && result <= 1.0);
        
        let result = ActivationFunction::Sigmoid.activate(-100.0_f32, 1.0);
        assert!(result.is_finite() && (0.0..1.0).contains(&result));
    }

    #[test]
    fn test_gpu_compatibility() {
        // Test GPU support detection
        assert!(!ActivationFunction::Threshold.has_gpu_support());
        assert!(!ActivationFunction::Gaussian.has_gpu_support());
        assert!(!ActivationFunction::Sin.has_gpu_support());
        
        // These should have GPU support when GPU feature is enabled
        #[cfg(feature = "gpu")]
        {
            assert!(ActivationFunction::Sigmoid.has_gpu_support());
            assert!(ActivationFunction::ReLU.has_gpu_support());
            assert!(ActivationFunction::Tanh.has_gpu_support());
            assert!(ActivationFunction::Linear.has_gpu_support());
            assert!(ActivationFunction::ReLULeaky.has_gpu_support());
        }
        
        #[cfg(not(feature = "gpu"))]
        {
            // When GPU feature is disabled, no activation should have GPU support
            assert!(!ActivationFunction::Sigmoid.has_gpu_support());
            assert!(!ActivationFunction::ReLU.has_gpu_support());
            assert!(!ActivationFunction::Tanh.has_gpu_support());
        }
    }

    #[test]
    fn test_strategy_pattern() {
        // Test that the Strategy Pattern implementation works correctly
        // All activation functions should produce the same results whether called
        // via the new methods or the old neuron-based approach
        
        let test_values = [-2.0_f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
        let steepness = 1.0_f32;
        
        for value in test_values {
            // Test major activation functions
            let sigmoid_result = ActivationFunction::Sigmoid.activate(value, steepness);
            assert!((0.0..=1.0).contains(&sigmoid_result));
            
            let relu_result = ActivationFunction::ReLU.activate(value, steepness);
            assert!(relu_result >= 0.0);
            if value > 0.0 {
                assert_eq!(relu_result, value);
            } else {
                assert_eq!(relu_result, 0.0);
            }
            
            let linear_result = ActivationFunction::Linear.activate(value, steepness);
            assert_eq!(linear_result, value * steepness);
        }
    }
}
