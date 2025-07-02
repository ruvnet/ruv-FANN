//! JSON serialization support

use std::io::{Read, Write};
use serde::{Serialize, Deserialize};
use crate::io::error::IoResult;

/// Read JSON data from a reader
pub fn read_json<T, R>(reader: &mut R) -> IoResult<T>
where
    T: for<'de> Deserialize<'de>,
    R: Read,
{
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;
    
    let value = serde_json::from_slice(&buffer)?;
    Ok(value)
}

/// Write JSON data to a writer
pub fn write_json<T, W>(data: &T, writer: &mut W) -> IoResult<()>
where
    T: Serialize,
    W: Write,
{
    let json_string = serde_json::to_string_pretty(data)?;
    writer.write_all(json_string.as_bytes())?;
    Ok(())
}

/// Read JSON data from a reader with custom options
pub fn read_json_with_options<T, R>(reader: &mut R, pretty: bool) -> IoResult<T>
where
    T: for<'de> Deserialize<'de>,
    R: Read,
{
    let mut buffer = Vec::new();
    reader.read_to_end(&mut buffer)?;
    
    if pretty {
        // For pretty reading, we just use the regular deserializer
        let value = serde_json::from_slice(&buffer)?;
        Ok(value)
    } else {
        let value = serde_json::from_slice(&buffer)?;
        Ok(value)
    }
}

/// Write JSON data to a writer with custom options
pub fn write_json_with_options<T, W>(data: &T, writer: &mut W, pretty: bool) -> IoResult<()>
where
    T: Serialize,
    W: Write,
{
    let json_string = if pretty {
        serde_json::to_string_pretty(data)?
    } else {
        serde_json::to_string(data)?
    };
    
    writer.write_all(json_string.as_bytes())?;
    Ok(())
}

/// JSON format configuration
#[derive(Debug, Clone)]
pub struct JsonConfig {
    /// Use pretty printing (indented, human-readable)
    pub pretty: bool,
    /// Include null values in serialization
    pub include_null: bool,
}

impl JsonConfig {
    /// Create a new JSON config with default settings
    pub fn new() -> Self {
        Self {
            pretty: true,
            include_null: false,
        }
    }
    
    /// Create a compact JSON config (no pretty printing)
    pub fn compact() -> Self {
        Self {
            pretty: false,
            include_null: false,
        }
    }
    
    /// Create a pretty JSON config (with indentation)
    pub fn pretty() -> Self {
        Self {
            pretty: true,
            include_null: false,
        }
    }
}

impl Default for JsonConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON reader with configuration
pub struct JsonReader {
    config: JsonConfig,
}

impl JsonReader {
    /// Create a new JSON reader with default config
    pub fn new() -> Self {
        Self {
            config: JsonConfig::new(),
        }
    }
    
    /// Create a new JSON reader with custom config
    pub fn with_config(config: JsonConfig) -> Self {
        Self { config }
    }
    
    /// Read data from a reader
    pub fn read<T, R>(&self, reader: &mut R) -> IoResult<T>
    where
        T: for<'de> Deserialize<'de>,
        R: Read,
    {
        read_json(reader)
    }
}

impl Default for JsonReader {
    fn default() -> Self {
        Self::new()
    }
}

/// JSON writer with configuration
pub struct JsonWriter {
    config: JsonConfig,
}

impl JsonWriter {
    /// Create a new JSON writer with default config
    pub fn new() -> Self {
        Self {
            config: JsonConfig::new(),
        }
    }
    
    /// Create a new JSON writer with custom config
    pub fn with_config(config: JsonConfig) -> Self {
        Self { config }
    }
    
    /// Write data to a writer
    pub fn write<T, W>(&self, data: &T, writer: &mut W) -> IoResult<()>
    where
        T: Serialize,
        W: Write,
    {
        write_json_with_options(data, writer, self.config.pretty)
    }
}

impl Default for JsonWriter {
    fn default() -> Self {
        Self::new()
    }
}