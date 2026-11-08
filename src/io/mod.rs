//! I/O and serialization module for rUv-FANN

#[cfg(feature = "binary")]
mod binary;
#[cfg(feature = "compression")]
mod compression;
mod dot_export;
mod error;
mod fann_format;
#[cfg(feature = "serde")]
mod json;
mod limits;
mod streaming;
mod training_data;

// Re-export types
pub use dot_export::DotExporter;
pub use error::{IoError, IoResult};
pub use fann_format::{FannReader, FannWriter};
pub use training_data::{TrainingDataReader, TrainingDataStreamReader, TrainingDataWriter};

#[cfg(feature = "serde")]
pub use json::{read_json, read_json_with_limit, write_json};

#[cfg(feature = "binary")]
pub use binary::{read_binary, read_binary_with_limit, write_binary};

#[cfg(feature = "compression")]
pub use compression::{compress_data, decompress_data, decompress_data_with_limit};

#[cfg(test)]
mod tests;

/// Supported file formats
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileFormat {
    /// Native FANN format (text-based)
    Fann,
    /// JSON format
    Json,
    /// Binary format (using bincode)
    Binary,
    /// DOT format for visualization
    Dot,
    /// Compressed FANN format
    CompressedFann,
    /// Compressed binary format
    CompressedBinary,
}
