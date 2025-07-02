//! Compression support for file formats

use std::io::{Read, Write};
use flate2::{Compression, read::GzDecoder, write::GzEncoder};
use crate::io::error::IoResult;

/// Compress data using gzip
pub fn compress_data<R, W>(reader: &mut R, writer: &mut W) -> IoResult<()>
where
    R: Read,
    W: Write,
{
    let mut encoder = GzEncoder::new(writer, Compression::default());
    std::io::copy(reader, &mut encoder)?;
    encoder.finish()?;
    Ok(())
}

/// Decompress gzip data
pub fn decompress_data<R, W>(reader: &mut R, writer: &mut W) -> IoResult<()>
where
    R: Read,
    W: Write,
{
    let mut decoder = GzDecoder::new(reader);
    std::io::copy(&mut decoder, writer)?;
    Ok(())
}

/// Compress data from bytes to bytes
pub fn compress_bytes(data: &[u8]) -> IoResult<Vec<u8>> {
    let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(data)?;
    let compressed = encoder.finish()?;
    Ok(compressed)
}

/// Decompress data from bytes to bytes
pub fn decompress_bytes(data: &[u8]) -> IoResult<Vec<u8>> {
    let mut decoder = GzDecoder::new(data);
    let mut decompressed = Vec::new();
    decoder.read_to_end(&mut decompressed)?;
    Ok(decompressed)
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    /// Compression level (0-9, where 9 is best compression)
    pub level: u32,
    /// Use faster compression algorithm
    pub fast: bool,
}

impl CompressionConfig {
    /// Create a new compression config with default settings
    pub fn new() -> Self {
        Self {
            level: 6, // Default compression level
            fast: false,
        }
    }
    
    /// Create a config optimized for speed
    pub fn fast() -> Self {
        Self {
            level: 1,
            fast: true,
        }
    }
    
    /// Create a config optimized for compression ratio
    pub fn best() -> Self {
        Self {
            level: 9,
            fast: false,
        }
    }
    
    /// Create a config with custom compression level
    pub fn with_level(level: u32) -> Self {
        Self {
            level: level.min(9),
            fast: false,
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Compression wrapper for readers
pub struct CompressedReader<R> {
    inner: GzDecoder<R>,
}

impl<R: Read> CompressedReader<R> {
    /// Create a new compressed reader
    pub fn new(reader: R) -> Self {
        Self {
            inner: GzDecoder::new(reader),
        }
    }
    
    /// Get a reference to the inner reader
    pub fn get_ref(&self) -> &GzDecoder<R> {
        &self.inner
    }
    
    /// Get a mutable reference to the inner reader
    pub fn get_mut(&mut self) -> &mut GzDecoder<R> {
        &mut self.inner
    }
    
    /// Consume the reader and return the inner reader
    pub fn into_inner(self) -> GzDecoder<R> {
        self.inner
    }
}

impl<R: Read> Read for CompressedReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

/// Compression wrapper for writers
pub struct CompressedWriter<W: Write> {
    inner: GzEncoder<W>,
}

impl<W: Write> CompressedWriter<W> {
    /// Create a new compressed writer with default compression
    pub fn new(writer: W) -> Self {
        Self {
            inner: GzEncoder::new(writer, Compression::default()),
        }
    }
    
    /// Create a new compressed writer with custom compression level
    pub fn with_level(writer: W, level: u32) -> Self {
        Self {
            inner: GzEncoder::new(writer, Compression::new(level)),
        }
    }
    
    /// Create a new compressed writer with config
    pub fn with_config(writer: W, config: CompressionConfig) -> Self {
        let compression = if config.fast {
            Compression::fast()
        } else {
            Compression::new(config.level)
        };
        
        Self {
            inner: GzEncoder::new(writer, compression),
        }
    }
    
    /// Finish compression and return the inner writer
    pub fn finish(self) -> std::io::Result<W> {
        self.inner.finish()
    }
    
    /// Get a reference to the inner writer
    pub fn get_ref(&self) -> &W {
        self.inner.get_ref()
    }
    
    /// Get a mutable reference to the inner writer
    pub fn get_mut(&mut self) -> &mut W {
        self.inner.get_mut()
    }
}

impl<W: Write> Write for CompressedWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.inner.write(buf)
    }
    
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Utilities for compression analysis
pub mod analyze {
    use super::*;
    
    /// Calculate compression ratio
    pub fn compression_ratio(original_size: usize, compressed_size: usize) -> f64 {
        if original_size == 0 {
            0.0
        } else {
            compressed_size as f64 / original_size as f64
        }
    }
    
    /// Calculate space savings percentage
    pub fn space_savings(original_size: usize, compressed_size: usize) -> f64 {
        if original_size == 0 {
            0.0
        } else {
            (1.0 - compression_ratio(original_size, compressed_size)) * 100.0
        }
    }
    
    /// Test compression effectiveness on data
    pub fn test_compression(data: &[u8]) -> IoResult<CompressionStats> {
        let compressed = compress_bytes(data)?;
        
        Ok(CompressionStats {
            original_size: data.len(),
            compressed_size: compressed.len(),
            ratio: compression_ratio(data.len(), compressed.len()),
            savings_percent: space_savings(data.len(), compressed.len()),
        })
    }
    
    /// Compression statistics
    #[derive(Debug, Clone)]
    pub struct CompressionStats {
        pub original_size: usize,
        pub compressed_size: usize,
        pub ratio: f64,
        pub savings_percent: f64,
    }
}