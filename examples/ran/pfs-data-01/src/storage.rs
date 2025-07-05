//! Parquet storage layer for processed data

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, Encoding};
use parquet::file::properties::{WriterProperties, WriterPropertiesBuilder};
use parquet::file::metadata::ParquetMetaData;
use parquet::file::reader::FileReader;
use tracing::{debug, info};

use crate::config::IngestionConfig;
use crate::error::{IngestionError, IngestionResult};

/// Parquet writer for processed data
pub struct ParquetWriter {
    config: Arc<IngestionConfig>,
}

impl ParquetWriter {
    pub fn new(config: Arc<IngestionConfig>) -> Self {
        Self { config }
    }
    
    /// Write record batches to a Parquet file
    pub async fn write_batches(&self, batches: &[RecordBatch], output_path: &Path) -> IngestionResult<ParquetMetadata> {
        if batches.is_empty() {
            return Err(IngestionError::config("no batches to write"));
        }
        
        let schema = batches[0].schema();
        let total_rows: usize = batches.iter().map(|batch| batch.num_rows()).sum();
        
        debug!("Writing {} batches with {} total rows to {:?}", batches.len(), total_rows, output_path);
        
        // Create writer properties
        let props = self.create_writer_properties()?;
        
        // Create output file
        let file = File::create(output_path)?;
        let mut writer = ArrowWriter::try_new(file, schema, Some(props))?;
        
        // Write batches
        for batch in batches {
            writer.write(batch)?;
        }
        
        // Close writer and get metadata
        let _metadata = writer.close()?;
        
        info!("Successfully wrote {} rows to Parquet file: {:?}", total_rows, output_path);
        
        // Read metadata back from file for consistency
        self.read_metadata(output_path).await
    }
    
    /// Write a single record batch to a Parquet file
    pub async fn write_batch(&self, batch: &RecordBatch, output_path: &Path) -> IngestionResult<ParquetMetadata> {
        self.write_batches(&[batch.clone()], output_path).await
    }
    
    /// Create writer properties based on configuration
    fn create_writer_properties(&self) -> IngestionResult<WriterProperties> {
        let mut builder = WriterProperties::builder();
        
        // Set compression
        let compression = match self.config.compression_codec.as_str() {
            "snappy" => Compression::SNAPPY,
            "gzip" => Compression::GZIP(Default::default()),
            "lz4" => Compression::LZ4,
            "brotli" => Compression::BROTLI(Default::default()),
            "zstd" => Compression::ZSTD(Default::default()),
            "uncompressed" => Compression::UNCOMPRESSED,
            _ => return Err(IngestionError::config(
                format!("unsupported compression codec: {}", self.config.compression_codec)
            )),
        };
        builder = builder.set_compression(compression);
        
        // Set row group size
        builder = builder.set_max_row_group_size(self.config.row_group_size);
        
        // Enable/disable statistics
        if self.config.enable_statistics {
            builder = builder.set_statistics_enabled(parquet::file::properties::EnabledStatistics::Chunk);
        } else {
            builder = builder.set_statistics_enabled(parquet::file::properties::EnabledStatistics::None);
        }
        
        // Set dictionary encoding
        if self.config.enable_dictionary_encoding {
            builder = builder.set_dictionary_enabled(true);
            builder = builder.set_encoding(Encoding::RLE_DICTIONARY);
        } else {
            builder = builder.set_dictionary_enabled(false);
            builder = builder.set_encoding(Encoding::PLAIN);
        }
        
        // Set writer version (use V2 for better performance and smaller files)
        builder = builder.set_writer_version(parquet::file::properties::WriterVersion::PARQUET_2_0);
        
        // Set page size (optimize for read performance)
        builder = builder.set_data_page_size_limit(1024 * 1024); // 1MB
        builder = builder.set_dictionary_page_size_limit(1024 * 1024); // 1MB
        
        Ok(builder.build())
    }
    
    /// Read metadata from an existing Parquet file
    pub async fn read_metadata(&self, file_path: &Path) -> IngestionResult<ParquetMetadata> {
        let file = File::open(file_path)?;
        let reader = parquet::file::reader::SerializedFileReader::new(file)?;
        let metadata = reader.metadata().clone();
        
        Ok(ParquetMetadata::from_parquet_metadata(&metadata))
    }
    
    /// Validate Parquet file integrity
    pub async fn validate_file(&self, file_path: &Path) -> IngestionResult<ValidationResult> {
        let metadata = self.read_metadata(file_path).await?;
        
        let mut result = ValidationResult {
            is_valid: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            row_count: metadata.num_rows,
            file_size_bytes: std::fs::metadata(file_path)?.len(),
            compression_ratio: 0.0,
        };
        
        // Check row groups
        if metadata.num_row_groups == 0 {
            result.errors.push("File contains no row groups".to_string());
            result.is_valid = false;
        }
        
        // Check if file is readable
        if let Err(e) = self.try_read_sample(file_path).await {
            result.errors.push(format!("File read error: {}", e));
            result.is_valid = false;
        }
        
        // Calculate approximate compression ratio
        let uncompressed_size = metadata.num_rows * 100; // Rough estimate
        if uncompressed_size > 0 {
            result.compression_ratio = result.file_size_bytes as f64 / uncompressed_size as f64;
        }
        
        Ok(result)
    }
    
    async fn try_read_sample(&self, file_path: &Path) -> IngestionResult<()> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        
        let file = File::open(file_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let mut reader = builder.with_batch_size(100).build()?;
        
        // Try to read first batch
        if let Some(batch) = reader.next() {
            batch?;
        }
        
        Ok(())
    }
    
    /// Merge multiple Parquet files into a single file
    pub async fn merge_files(&self, input_files: &[&Path], output_path: &Path) -> IngestionResult<ParquetMetadata> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        
        if input_files.is_empty() {
            return Err(IngestionError::config("no input files to merge"));
        }
        
        debug!("Merging {} Parquet files into {:?}", input_files.len(), output_path);
        
        let mut all_batches = Vec::new();
        let mut schema = None;
        
        // Read all files
        for file_path in input_files {
            let file = File::open(file_path)?;
            let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
            
            // Check schema compatibility
            if let Some(ref existing_schema) = schema {
                if existing_schema != &builder.schema() {
                    return Err(IngestionError::schema_validation(
                        format!("schema mismatch in file: {:?}", file_path)
                    ));
                }
            } else {
                schema = Some(builder.schema());
            }
            
            let mut reader = builder.build()?;
            while let Some(batch) = reader.next() {
                all_batches.push(batch?);
            }
        }
        
        // Write merged file
        self.write_batches(&all_batches, output_path).await
    }
    
    /// Split a large Parquet file into smaller files
    pub async fn split_file(
        &self,
        input_path: &Path,
        output_dir: &Path,
        max_rows_per_file: usize,
    ) -> IngestionResult<Vec<ParquetMetadata>> {
        use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
        
        let file = File::open(input_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
        let mut reader = builder.build()?;
        
        let mut results = Vec::new();
        let mut file_index = 0;
        let mut current_batches = Vec::new();
        let mut current_rows = 0;
        
        let input_stem = input_path.file_stem()
            .ok_or_else(|| IngestionError::config("invalid input file name"))?;
        
        while let Some(batch) = reader.next() {
            let batch = batch?;
            current_batches.push(batch.clone());
            current_rows += batch.num_rows();
            
            if current_rows >= max_rows_per_file {
                // Write current file
                let output_path = output_dir.join(format!(
                    "{}_part_{:04}.parquet",
                    input_stem.to_string_lossy(),
                    file_index
                ));
                
                let metadata = self.write_batches(&current_batches, &output_path).await?;
                results.push(metadata);
                
                // Reset for next file
                current_batches.clear();
                current_rows = 0;
                file_index += 1;
            }
        }
        
        // Write remaining batches
        if !current_batches.is_empty() {
            let output_path = output_dir.join(format!(
                "{}_part_{:04}.parquet",
                input_stem.to_string_lossy(),
                file_index
            ));
            
            let metadata = self.write_batches(&current_batches, &output_path).await?;
            results.push(metadata);
        }
        
        info!("Split file into {} parts", results.len());
        
        Ok(results)
    }
}

/// Parquet metadata wrapper
#[derive(Debug, Clone)]
pub struct ParquetMetadata {
    pub num_rows: i64,
    pub num_row_groups: i32,
    pub file_size_bytes: Option<i64>,
    pub schema_name: Option<String>,
    pub created_by: Option<String>,
    pub compression_ratios: Vec<f64>,
}

impl ParquetMetadata {
    fn from_parquet_metadata(metadata: &parquet::file::metadata::ParquetMetaData) -> Self {
        let file_metadata = metadata.file_metadata();
        
        let compression_ratios = (0..metadata.num_row_groups())
            .map(|i| {
                let row_group = metadata.row_group(i);
                let compressed_size: i64 = (0..row_group.num_columns())
                    .map(|j| row_group.column(j).compressed_size())
                    .sum();
                let uncompressed_size: i64 = (0..row_group.num_columns())
                    .map(|j| row_group.column(j).uncompressed_size())
                    .sum();
                
                if uncompressed_size > 0 {
                    compressed_size as f64 / uncompressed_size as f64
                } else {
                    1.0
                }
            })
            .collect();
        
        Self {
            num_rows: file_metadata.num_rows(),
            num_row_groups: metadata.num_row_groups() as i32,
            file_size_bytes: None, // Would need to be set separately
            schema_name: None,
            created_by: file_metadata.created_by().map(|s| s.to_string()),
            compression_ratios,
        }
    }
    
    pub fn average_compression_ratio(&self) -> f64 {
        if self.compression_ratios.is_empty() {
            1.0
        } else {
            self.compression_ratios.iter().sum::<f64>() / self.compression_ratios.len() as f64
        }
    }
    
    pub fn estimated_uncompressed_size(&self) -> Option<i64> {
        self.file_size_bytes.map(|size| {
            let avg_ratio = self.average_compression_ratio();
            if avg_ratio > 0.0 {
                (size as f64 / avg_ratio) as i64
            } else {
                size
            }
        })
    }
}

/// File validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub row_count: i64,
    pub file_size_bytes: u64,
    pub compression_ratio: f64,
}

impl ValidationResult {
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }
    
    pub fn has_warnings(&self) -> bool {
        !self.warnings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use tempfile::tempdir;
    
    fn create_test_batch() -> RecordBatch {
        let schema = Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new("name", DataType::Utf8, false),
        ]);
        
        let id_array = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let name_array = StringArray::from(vec!["a", "b", "c", "d", "e"]);
        
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(id_array), Arc::new(name_array)],
        ).unwrap()
    }
    
    #[tokio::test]
    async fn test_parquet_writer_creation() {
        let config = Arc::new(IngestionConfig::default());
        let writer = ParquetWriter::new(config);
        
        assert!(writer.create_writer_properties().is_ok());
    }
    
    #[tokio::test]
    async fn test_write_read_cycle() {
        let config = Arc::new(IngestionConfig::default());
        let writer = ParquetWriter::new(config);
        
        let batch = create_test_batch();
        let temp_dir = tempdir().unwrap();
        let output_path = temp_dir.path().join("test.parquet");
        
        let metadata = writer.write_batch(&batch, &output_path).await.unwrap();
        assert_eq!(metadata.num_rows, 5);
        
        let read_metadata = writer.read_metadata(&output_path).await.unwrap();
        assert_eq!(read_metadata.num_rows, 5);
    }
    
    #[tokio::test]
    async fn test_file_validation() {
        let config = Arc::new(IngestionConfig::default());
        let writer = ParquetWriter::new(config);
        
        let batch = create_test_batch();
        let temp_dir = tempdir().unwrap();
        let output_path = temp_dir.path().join("test.parquet");
        
        writer.write_batch(&batch, &output_path).await.unwrap();
        
        let validation = writer.validate_file(&output_path).await.unwrap();
        assert!(validation.is_valid);
        assert_eq!(validation.row_count, 5);
        assert!(!validation.has_errors());
    }
    
    #[tokio::test]
    async fn test_compression_settings() {
        let mut config = IngestionConfig::default();
        config.compression_codec = "gzip".to_string();
        
        let writer = ParquetWriter::new(Arc::new(config));
        let props = writer.create_writer_properties().unwrap();
        
        // We can't directly test the compression setting, but we can test that it builds without error
        assert!(props.compression(&parquet::basic::ColumnPath::from("test")) == Compression::GZIP);
    }
}