//! Streaming I/O for large datasets

use super::limits;
use crate::io::error::{IoError, IoResult};
use std::io::{BufRead, BufReader, Read};

/// Streaming reader for training data
pub struct TrainingDataStreamReader {
    buffer_size: usize,
}

impl TrainingDataStreamReader {
    /// Create a new streaming reader with default buffer size
    pub fn new() -> Self {
        Self {
            buffer_size: 8192, // 8KB default buffer
        }
    }

    /// Create a new streaming reader with custom buffer size
    pub fn with_buffer_size(buffer_size: usize) -> Self {
        Self { buffer_size }
    }

    /// Read training data with a callback for each sample
    pub fn read_stream<R, F>(&self, reader: &mut R, callback: F) -> IoResult<StreamStats>
    where
        R: BufRead,
        F: FnMut(&[f32], &[f32]) -> IoResult<()>,
    {
        let mut line = String::new();
        let mut samples_processed = 0;
        let mut bytes_read = 0;
        let mut remaining = limits::TRAINING_BYTES;

        // Read header line
        let header_bytes = limits::read_line(reader, &mut line, &mut remaining)?;
        bytes_read += header_bytes;

        let header_parts: Vec<&str> = line.split_whitespace().take(4).collect();

        if header_parts.len() != 3 {
            return Err(IoError::InvalidFileFormat(
                "Header must contain exactly 3 numbers: num_data num_input num_output".to_string(),
            ));
        }

        let num_data: usize = header_parts[0].parse()?;
        let num_input: usize = header_parts[1].parse()?;
        let num_output: usize = header_parts[2].parse()?;

        limits::training_shape(num_data, num_input, num_output)?;

        let mut callback = callback;
        let mut input_values = Vec::new();
        let mut output_values = Vec::new();

        for i in 0..num_data {
            // Read input line
            line.clear();
            let bytes = limits::read_line(reader, &mut line, &mut remaining)?;
            if bytes == 0 {
                return Err(IoError::InvalidTrainingData(format!(
                    "Missing data at sample {i}"
                )));
            }
            bytes_read += bytes;

            parse_values(&line, &mut input_values, num_input, i)?;

            // Read output line
            line.clear();
            let bytes = limits::read_line(reader, &mut line, &mut remaining)?;
            if bytes == 0 {
                return Err(IoError::InvalidTrainingData(format!(
                    "Missing data at sample {i}"
                )));
            }
            bytes_read += bytes;

            parse_values(&line, &mut output_values, num_output, i)?;

            // Call callback with this sample
            callback(&input_values, &output_values)?;
            samples_processed += 1;
        }

        Ok(StreamStats {
            samples_processed,
            bytes_read,
            num_input,
            num_output,
        })
    }

    /// Read training data in batches
    pub fn read_batches<R, F>(
        &self,
        reader: &mut R,
        batch_size: usize,
        mut callback: F,
    ) -> IoResult<StreamStats>
    where
        R: BufRead,
        F: FnMut(&[Vec<f32>], &[Vec<f32>]) -> IoResult<()>,
    {
        if batch_size == 0 {
            return Err(IoError::InvalidTrainingData(
                "batch_size must be positive".into(),
            ));
        }
        // Do not reserve an untrusted caller's full batch size before reading.
        let mut input_batch = Vec::with_capacity(batch_size.min(1024));
        let mut output_batch = Vec::with_capacity(batch_size.min(1024));
        let stats = self.read_stream(reader, |inputs, outputs| {
            input_batch.push(inputs.to_vec());
            output_batch.push(outputs.to_vec());
            if input_batch.len() == batch_size {
                callback(&input_batch, &output_batch)?;
                input_batch.clear();
                output_batch.clear();
            }
            Ok(())
        })?;
        if !input_batch.is_empty() {
            callback(&input_batch, &output_batch)?;
        }
        Ok(stats)
    }
}

fn parse_values(line: &str, values: &mut Vec<f32>, expected: usize, sample: usize) -> IoResult<()> {
    values.clear();
    for token in line.split_whitespace() {
        if values.len() == expected {
            return Err(IoError::InvalidTrainingData(format!(
                "Too many values at sample {sample}"
            )));
        }
        let value: f32 = token.parse()?;
        if !value.is_finite() {
            return Err(IoError::InvalidTrainingData(format!(
                "Non-finite value at sample {sample}"
            )));
        }
        values.push(value);
    }
    if values.len() != expected {
        return Err(IoError::InvalidTrainingData(format!(
            "Expected {expected} values at sample {sample}, got {}",
            values.len()
        )));
    }
    Ok(())
}

impl Default for TrainingDataStreamReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics from streaming operations
#[derive(Debug, Clone)]
pub struct StreamStats {
    pub samples_processed: usize,
    pub bytes_read: usize,
    pub num_input: usize,
    pub num_output: usize,
}

impl StreamStats {
    /// Calculate average bytes per sample
    pub fn avg_bytes_per_sample(&self) -> f64 {
        if self.samples_processed == 0 {
            0.0
        } else {
            self.bytes_read as f64 / self.samples_processed as f64
        }
    }

    /// Calculate total parameters (inputs + outputs) per sample
    pub fn parameters_per_sample(&self) -> usize {
        self.num_input + self.num_output
    }
}

/// Buffered reader wrapper for streaming
pub struct BufferedStreamReader<R> {
    inner: BufReader<R>,
    buffer_size: usize,
}

impl<R: Read> BufferedStreamReader<R> {
    /// Create a new buffered stream reader
    pub fn new(reader: R) -> Self {
        Self {
            inner: BufReader::new(reader),
            buffer_size: 8192,
        }
    }

    /// Create a new buffered stream reader with custom buffer size
    pub fn with_capacity(reader: R, capacity: usize) -> Self {
        Self {
            inner: BufReader::with_capacity(capacity, reader),
            buffer_size: capacity,
        }
    }

    /// Get the buffer size
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Get the number of bytes in the internal buffer
    pub fn buffer_len(&self) -> usize {
        self.inner.buffer().len()
    }
}

impl<R: Read> BufRead for BufferedStreamReader<R> {
    fn fill_buf(&mut self) -> std::io::Result<&[u8]> {
        self.inner.fill_buf()
    }

    fn consume(&mut self, amt: usize) {
        self.inner.consume(amt)
    }
}

impl<R: Read> Read for BufferedStreamReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        self.inner.read(buf)
    }
}

/// Utilities for memory-efficient streaming
pub mod memory {

    /// Estimate memory usage for batch processing
    pub fn estimate_batch_memory(batch_size: usize, num_input: usize, num_output: usize) -> usize {
        // Estimate memory usage in bytes
        let f32_size = std::mem::size_of::<f32>();
        let vec_overhead = std::mem::size_of::<Vec<f32>>();

        let input_memory = batch_size.saturating_mul(
            num_input
                .saturating_mul(f32_size)
                .saturating_add(vec_overhead),
        );
        let output_memory = batch_size.saturating_mul(
            num_output
                .saturating_mul(f32_size)
                .saturating_add(vec_overhead),
        );

        input_memory.saturating_add(output_memory)
    }

    /// Calculate optimal batch size for given memory limit
    pub fn optimal_batch_size(
        memory_limit_bytes: usize,
        num_input: usize,
        num_output: usize,
    ) -> usize {
        let sample_memory = estimate_batch_memory(1, num_input, num_output);
        if sample_memory == 0 {
            1000 // Default fallback
        } else {
            (memory_limit_bytes / sample_memory).max(1)
        }
    }
}
