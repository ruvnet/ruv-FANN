//! FANN native file format reader and writer

use std::io::{BufRead, BufReader, Write};
use crate::io::error::{IoError, IoResult};
use crate::{Network, NetworkBuilder};
use num_traits::Float;

/// FANN file format reader
pub struct FannReader {
    // Configuration options could go here
}

impl FannReader {
    /// Create a new FANN reader
    pub fn new() -> Self {
        Self {}
    }

    /// Read a neural network from a FANN format file
    pub fn read_network<T: Float + std::str::FromStr, R: std::io::Read>(&self, reader: &mut R) -> IoResult<Network<T>>
    where
        T::Err: std::fmt::Debug,
    {
        let mut buf_reader = BufReader::new(reader);
        let mut line = String::new();
        
        // Read version line
        buf_reader.read_line(&mut line)?;
        if !line.starts_with("FANN_FLO") && !line.starts_with("FANN_FIX") {
            return Err(IoError::InvalidFileFormat(
                "Missing FANN version header".to_string()
            ));
        }
        
        let mut num_layers = 0;
        let mut connection_rate = T::one();
        let mut layer_sizes = Vec::new();
        let mut weights = Vec::new();
        
        // Parse network parameters
        loop {
            line.clear();
            let bytes_read = buf_reader.read_line(&mut line)?;
            if bytes_read == 0 {
                break; // EOF
            }
            
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            
            if let Some((key, value)) = line.split_once('=') {
                match key {
                    "num_layers" => {
                        num_layers = value.parse()
                            .map_err(|e| IoError::ParseError(format!("Invalid num_layers: {:?}", e)))?;
                    }
                    "connection_rate" => {
                        connection_rate = value.parse()
                            .map_err(|e| IoError::ParseError(format!("Invalid connection_rate: {:?}", e)))?;
                    }
                    "layer_sizes" => {
                        layer_sizes = value.split_whitespace()
                            .map(|s| s.parse())
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|e| IoError::ParseError(format!("Invalid layer_sizes: {:?}", e)))?;
                    }
                    "weights" => {
                        weights = value.split_whitespace()
                            .map(|s| s.parse())
                            .collect::<Result<Vec<_>, _>>()
                            .map_err(|e| IoError::ParseError(format!("Invalid weights: {:?}", e)))?;
                    }
                    _ => {
                        // Skip unknown parameters for now
                    }
                }
            }
        }
        
        // Validate network parameters
        if num_layers == 0 {
            return Err(IoError::InvalidNetwork("num_layers must be > 0".to_string()));
        }
        
        if layer_sizes.is_empty() {
            return Err(IoError::InvalidNetwork("layer_sizes must not be empty".to_string()));
        }
        
        if layer_sizes.len() != num_layers {
            return Err(IoError::InvalidNetwork("layer_sizes length must match num_layers".to_string()));
        }
        
        // Build network using NetworkBuilder
        let mut builder = NetworkBuilder::<T>::new();
        
        for (i, &size) in layer_sizes.iter().enumerate() {
            if i == 0 {
                builder = builder.input_layer(size);
            } else if i == layer_sizes.len() - 1 {
                builder = builder.output_layer(size);
            } else {
                builder = builder.hidden_layer(size);
            }
        }
        
        let mut network = builder.connection_rate(connection_rate).build();
        
        // Set weights if provided
        if !weights.is_empty() {
            network.set_weights(&weights)
                .map_err(|e| IoError::InvalidNetwork(format!("Failed to set weights: {}", e)))?;
        }
        
        Ok(network)
    }
}

impl Default for FannReader {
    fn default() -> Self {
        Self::new()
    }
}

/// FANN file format writer
pub struct FannWriter {
    // Configuration options could go here
}

impl FannWriter {
    /// Create a new FANN writer
    pub fn new() -> Self {
        Self {}
    }

    /// Write a neural network to FANN format
    pub fn write_network<T: Float + std::fmt::Display, W: Write>(&self, network: &Network<T>, writer: &mut W) -> IoResult<()> {
        // Write version header
        writeln!(writer, "FANN_FLO:2.1")?;
        
        // Write network parameters
        writeln!(writer, "num_layers={}", network.num_layers())?;
        writeln!(writer, "connection_rate={:.6}", network.connection_rate)?;
        writeln!(writer, "network_type=0")?;
        writeln!(writer, "learning_momentum=0.000000")?;
        writeln!(writer, "training_algorithm=2")?;
        writeln!(writer, "train_error_function=1")?;
        writeln!(writer, "train_stop_function=0")?;
        writeln!(writer, "cascade_output_change_fraction=0.010000")?;
        writeln!(writer, "quickprop_decay=-0.000100")?;
        writeln!(writer, "quickprop_mu=1.750000")?;
        writeln!(writer, "rprop_increase_factor=1.200000")?;
        writeln!(writer, "rprop_decrease_factor=0.500000")?;
        writeln!(writer, "rprop_delta_min=0.000000")?;
        writeln!(writer, "rprop_delta_max=50.000000")?;
        writeln!(writer, "rprop_delta_zero=0.100000")?;
        writeln!(writer, "cascade_output_stagnation_epochs=12")?;
        writeln!(writer, "cascade_candidate_change_fraction=0.010000")?;
        writeln!(writer, "cascade_candidate_stagnation_epochs=12")?;
        writeln!(writer, "cascade_max_out_epochs=150")?;
        writeln!(writer, "cascade_min_out_epochs=50")?;
        writeln!(writer, "cascade_max_cand_epochs=150")?;
        writeln!(writer, "cascade_min_cand_epochs=50")?;
        writeln!(writer, "cascade_num_candidate_groups=2")?;
        writeln!(writer, "bit_fail_limit=0.350000")?;
        writeln!(writer, "cascade_candidate_limit=1000.000000")?;
        writeln!(writer, "cascade_weight_multiplier=0.400000")?;
        writeln!(writer, "cascade_activation_functions_count=10")?;
        writeln!(writer, "cascade_activation_functions=3 5 7 8 10 11 14 15 16 17 ")?;
        writeln!(writer, "cascade_activation_steepnesses_count=4")?;
        writeln!(writer, "cascade_activation_steepnesses=0.250000 0.500000 0.750000 1.000000 ")?;
        
        // Write layer sizes
        write!(writer, "layer_sizes=")?;
        for layer in &network.layers {
            write!(writer, "{} ", layer.num_regular_neurons())?;
        }
        writeln!(writer)?;
        
        // Write weights
        let weights = network.get_weights();
        if !weights.is_empty() {
            write!(writer, "weights=")?;
            for weight in weights {
                write!(writer, "{:.6} ", weight)?;
            }
            writeln!(writer)?;
        }
        
        Ok(())
    }
}

impl Default for FannWriter {
    fn default() -> Self {
        Self::new()
    }
}