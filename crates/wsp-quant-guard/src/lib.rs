//! Exact, bounded certificates for an explicitly specified integer affine layer.
//!
//! A certificate protects declared integer inputs or constant-output boxes only.
//! It does not authenticate an upstream network, prove generalization, or establish
//! scientific novelty. Quantization is floor division followed by saturation.

mod ledger;
mod model;
mod search;

pub use ledger::{Certificate, Edit, Ledger, Patch, Receipt};
pub use model::{IntBox, Model, Protected};
pub use search::{solve_one_coordinate, Target};

use serde::Serialize;
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error(pub String);

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn fail<T>(message: &str) -> Result<T> {
    Err(Error(message.to_string()))
}

pub(crate) fn hash<T: Serialize>(value: &T) -> Result<String> {
    let bytes = serde_json::to_vec(value).map_err(|e| Error(e.to_string()))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
