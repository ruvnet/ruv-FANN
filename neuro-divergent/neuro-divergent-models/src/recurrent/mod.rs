//! Recurrent Neural Network Models
//!
//! This module provides implementations of various recurrent neural network architectures
//! for time series forecasting, including basic RNN, LSTM, and GRU models.

mod bptt;
pub mod layers;
pub mod rnn;

// Re-export main model types
pub use rnn::RNN;

// Re-export layer components
pub use layers::{
    BasicRecurrentCell, BidirectionalMergeMode, BidirectionalRecurrent, MultiLayerRecurrent,
    RecurrentLayer,
};

/// Reserved gated forecasting architecture. Construction explicitly fails until
/// gated training is implemented; this type never aliases a vanilla RNN.
#[derive(Debug)]
pub struct LSTM<T>(std::marker::PhantomData<T>);

/// Reserved gated forecasting architecture; see [`LSTM`].
#[derive(Debug)]
pub struct GRU<T>(std::marker::PhantomData<T>);

macro_rules! unsupported_gated_model {
    ($model:ident) => {
        impl<T: num_traits::Float + Send + Sync + std::fmt::Debug + std::iter::Sum + 'static> $model<T> {
            pub fn new<C: crate::foundation::ModelConfig<T>>(_config: C) -> crate::NeuroDivergentResult<Self> {
                Err(crate::NeuroDivergentError::config(concat!(stringify!($model),
                    " forecasting training is unsupported; use RNN for a trained vanilla recurrent model")))
            }
        }
    };
}
unsupported_gated_model!(LSTM);
unsupported_gated_model!(GRU);
