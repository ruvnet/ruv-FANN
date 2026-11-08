//! Training data file format reader and writer

use crate::io::error::{IoError, IoResult};
use std::io::{BufRead, BufReader, Write};

// Import the mock types for now
use crate::mock_types::MockTrainingData;

/// Training data file format reader
pub struct TrainingDataReader {
    // Configuration options could go here
}

impl TrainingDataReader {
    /// Create a new training data reader
    pub fn new() -> Self {
        Self {}
    }

    /// Read training data from a FANN data format file
    pub fn read_data<R: std::io::Read>(&self, reader: &mut R) -> IoResult<MockTrainingData> {
        let mut buf_reader = BufReader::new(reader);
        let mut inputs = Vec::new();
        let mut outputs = Vec::new();
        let stats = super::streaming::TrainingDataStreamReader::new().read_stream(
            &mut buf_reader,
            |input, output| {
                inputs.push(input.to_vec());
                outputs.push(output.to_vec());
                Ok(())
            },
        )?;
        Ok(MockTrainingData {
            num_data: stats.samples_processed,
            num_input: stats.num_input,
            num_output: stats.num_output,
            inputs,
            outputs,
        })
    }
}

impl Default for TrainingDataReader {
    fn default() -> Self {
        Self::new()
    }
}

/// Training data file format writer
pub struct TrainingDataWriter {
    // Configuration options could go here
}

impl TrainingDataWriter {
    /// Create a new training data writer
    pub fn new() -> Self {
        Self {}
    }

    /// Write training data to FANN data format
    pub fn write_data<W: Write>(&self, data: &MockTrainingData, writer: &mut W) -> IoResult<()> {
        if data.inputs.len() != data.num_data
            || data.outputs.len() != data.num_data
            || data.inputs.iter().any(|row| row.len() != data.num_input)
            || data.outputs.iter().any(|row| row.len() != data.num_output)
        {
            return Err(IoError::InvalidTrainingData(
                "Training shape does not match its header".into(),
            ));
        }
        // Write header
        writeln!(
            writer,
            "{} {} {}",
            data.num_data, data.num_input, data.num_output
        )?;

        // Write data samples
        for i in 0..data.num_data {
            // Write inputs
            for (j, &input) in data.inputs[i].iter().enumerate() {
                if j > 0 {
                    write!(writer, " ")?;
                }
                // Format numbers intelligently (integers as integers, floats as floats)
                if input.fract() == 0.0 && input.abs() < 1e6 {
                    write!(writer, "{}", input as i32)?;
                } else {
                    write!(writer, "{input}")?;
                }
            }
            writeln!(writer)?;

            // Write outputs
            for (j, &output) in data.outputs[i].iter().enumerate() {
                if j > 0 {
                    write!(writer, " ")?;
                }
                // Format numbers intelligently
                if output.fract() == 0.0 && output.abs() < 1e6 {
                    write!(writer, "{}", output as i32)?;
                } else {
                    write!(writer, "{output}")?;
                }
            }
            writeln!(writer)?;
        }

        Ok(())
    }
}

impl Default for TrainingDataWriter {
    fn default() -> Self {
        Self::new()
    }
}

/// Streaming reader for large training datasets
pub struct TrainingDataStreamReader {
    // Configuration options could go here
}

impl TrainingDataStreamReader {
    /// Create a new streaming reader
    pub fn new() -> Self {
        Self {}
    }

    /// Read training data with a callback for each sample
    pub fn read_stream<R, F>(&self, reader: &mut R, mut callback: F) -> IoResult<()>
    where
        R: BufRead,
        F: FnMut(&[f32], &[f32]) -> IoResult<()>,
    {
        super::streaming::TrainingDataStreamReader::new()
            .read_stream(reader, |inputs, outputs| callback(inputs, outputs))?;
        Ok(())
    }
}

impl Default for TrainingDataStreamReader {
    fn default() -> Self {
        Self::new()
    }
}
