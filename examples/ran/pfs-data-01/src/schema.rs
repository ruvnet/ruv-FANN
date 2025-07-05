//! Schema validation and normalization for RAN data

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};
use arrow::array::*;
use arrow::datatypes::{DataType, Field, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;

use crate::error::{IngestionError, IngestionResult};

/// Standard schema for RAN intelligence data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StandardSchema {
    /// Column mappings
    pub timestamp_column: String,
    pub cell_id_column: String,
    pub kpi_name_column: String,
    pub kpi_value_column: String,
    pub ue_id_column: Option<String>,
    pub sector_id_column: Option<String>,
    
    /// Data type mappings
    pub column_types: HashMap<String, String>,
    
    /// Validation rules
    pub required_columns: Vec<String>,
    pub optional_columns: Vec<String>,
    
    /// Normalization rules
    pub timestamp_format: String,
    pub null_values: Vec<String>,
    pub default_values: HashMap<String, String>,
    
    /// Value constraints
    pub min_timestamp: Option<DateTime<Utc>>,
    pub max_timestamp: Option<DateTime<Utc>>,
    pub allowed_kpi_names: Option<Vec<String>>,
}

impl Default for StandardSchema {
    fn default() -> Self {
        let mut column_types = HashMap::new();
        column_types.insert("timestamp".to_string(), "timestamp".to_string());
        column_types.insert("cell_id".to_string(), "string".to_string());
        column_types.insert("kpi_name".to_string(), "string".to_string());
        column_types.insert("kpi_value".to_string(), "float64".to_string());
        column_types.insert("ue_id".to_string(), "string".to_string());
        column_types.insert("sector_id".to_string(), "string".to_string());
        
        Self {
            timestamp_column: crate::TIMESTAMP_COLUMN.to_string(),
            cell_id_column: crate::CELL_ID_COLUMN.to_string(),
            kpi_name_column: crate::KPI_NAME_COLUMN.to_string(),
            kpi_value_column: crate::KPI_VALUE_COLUMN.to_string(),
            ue_id_column: Some(crate::UE_ID_COLUMN.to_string()),
            sector_id_column: Some(crate::SECTOR_ID_COLUMN.to_string()),
            column_types,
            required_columns: vec![
                "timestamp".to_string(),
                "cell_id".to_string(),
                "kpi_name".to_string(),
                "kpi_value".to_string(),
            ],
            optional_columns: vec![
                "ue_id".to_string(),
                "sector_id".to_string(),
            ],
            timestamp_format: "%Y-%m-%d %H:%M:%S%.3f".to_string(),
            null_values: vec![
                "".to_string(),
                "null".to_string(),
                "NULL".to_string(),
                "nil".to_string(),
                "NIL".to_string(),
                "N/A".to_string(),
                "n/a".to_string(),
                "-".to_string(),
            ],
            default_values: HashMap::new(),
            min_timestamp: None,
            max_timestamp: None,
            allowed_kpi_names: None,
        }
    }
}

impl StandardSchema {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn validate(&self) -> IngestionResult<()> {
        if self.timestamp_column.is_empty() {
            return Err(IngestionError::schema_validation("timestamp_column cannot be empty"));
        }
        
        if self.cell_id_column.is_empty() {
            return Err(IngestionError::schema_validation("cell_id_column cannot be empty"));
        }
        
        if self.kpi_name_column.is_empty() {
            return Err(IngestionError::schema_validation("kpi_name_column cannot be empty"));
        }
        
        if self.kpi_value_column.is_empty() {
            return Err(IngestionError::schema_validation("kpi_value_column cannot be empty"));
        }
        
        // Validate column types
        for (column, data_type) in &self.column_types {
            match data_type.as_str() {
                "string" | "int32" | "int64" | "float32" | "float64" | "boolean" | "timestamp" => {},
                _ => return Err(IngestionError::schema_validation(
                    format!("unsupported data type '{}' for column '{}'", data_type, column)
                )),
            }
        }
        
        // Validate timestamp constraints
        if let (Some(min), Some(max)) = (&self.min_timestamp, &self.max_timestamp) {
            if min >= max {
                return Err(IngestionError::schema_validation(
                    "min_timestamp must be less than max_timestamp"
                ));
            }
        }
        
        Ok(())
    }
    
    /// Create Arrow schema from the standard schema
    pub fn to_arrow_schema(&self) -> IngestionResult<Schema> {
        let mut fields = Vec::new();
        
        // Add required columns
        for column in &self.required_columns {
            let data_type = self.column_types.get(column)
                .ok_or_else(|| IngestionError::schema_validation(
                    format!("missing type for required column '{}'", column)
                ))?;
            
            let field = Field::new(column, self.arrow_data_type(data_type)?, false);
            fields.push(field);
        }
        
        // Add optional columns
        for column in &self.optional_columns {
            if let Some(data_type) = self.column_types.get(column) {
                let field = Field::new(column, self.arrow_data_type(data_type)?, true);
                fields.push(field);
            }
        }
        
        Ok(Schema::new(fields))
    }
    
    /// Convert string data type to Arrow data type
    fn arrow_data_type(&self, data_type: &str) -> IngestionResult<DataType> {
        match data_type {
            "string" => Ok(DataType::Utf8),
            "int32" => Ok(DataType::Int32),
            "int64" => Ok(DataType::Int64),
            "float32" => Ok(DataType::Float32),
            "float64" => Ok(DataType::Float64),
            "boolean" => Ok(DataType::Boolean),
            "timestamp" => Ok(DataType::Timestamp(TimeUnit::Millisecond, Some("UTC".into()))),
            _ => Err(IngestionError::schema_validation(
                format!("unsupported data type: {}", data_type)
            )),
        }
    }
    
    /// Check if a value is considered null
    pub fn is_null_value(&self, value: &str) -> bool {
        self.null_values.contains(&value.to_string())
    }
    
    /// Get default value for a column
    pub fn get_default_value(&self, column: &str) -> Option<&String> {
        self.default_values.get(column)
    }
    
    /// Validate column mapping in data
    pub fn validate_columns(&self, columns: &[String]) -> IngestionResult<()> {
        let column_set: std::collections::HashSet<_> = columns.iter().collect();
        
        // Check required columns
        for required in &self.required_columns {
            if !column_set.contains(required) {
                return Err(IngestionError::schema_validation(
                    format!("missing required column: {}", required)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Map source column names to standard column names
    pub fn map_columns(&self, source_columns: &[String]) -> HashMap<String, String> {
        let mut mapping = HashMap::new();
        
        for source_col in source_columns {
            let normalized = source_col.to_lowercase()
                .replace(" ", "_")
                .replace("-", "_");
            
            // Try to map to standard columns
            if self.matches_column(&normalized, &self.timestamp_column) {
                mapping.insert(source_col.clone(), self.timestamp_column.clone());
            } else if self.matches_column(&normalized, &self.cell_id_column) {
                mapping.insert(source_col.clone(), self.cell_id_column.clone());
            } else if self.matches_column(&normalized, &self.kpi_name_column) {
                mapping.insert(source_col.clone(), self.kpi_name_column.clone());
            } else if self.matches_column(&normalized, &self.kpi_value_column) {
                mapping.insert(source_col.clone(), self.kpi_value_column.clone());
            } else if let Some(ue_col) = &self.ue_id_column {
                if self.matches_column(&normalized, ue_col) {
                    mapping.insert(source_col.clone(), ue_col.clone());
                }
            } else if let Some(sector_col) = &self.sector_id_column {
                if self.matches_column(&normalized, sector_col) {
                    mapping.insert(source_col.clone(), sector_col.clone());
                }
            }
        }
        
        mapping
    }
    
    /// Check if a column name matches a standard column (fuzzy matching)
    fn matches_column(&self, source: &str, target: &str) -> bool {
        if source == target {
            return true;
        }
        
        // Try common variations
        let variations = vec![
            source.to_string(),
            source.replace("_", ""),
            source.replace("_", "-"),
            format!("{}_id", source),
            format!("{}id", source),
        ];
        
        for variation in variations {
            if variation == target || variation.contains(target) || target.contains(&variation) {
                return true;
            }
        }
        
        false
    }
    
    /// Validate KPI name against allowed list
    pub fn validate_kpi_name(&self, kpi_name: &str) -> IngestionResult<()> {
        if let Some(allowed) = &self.allowed_kpi_names {
            if !allowed.contains(&kpi_name.to_string()) {
                return Err(IngestionError::data_validation(
                    format!("KPI name '{}' is not in allowed list", kpi_name)
                ));
            }
        }
        Ok(())
    }
    
    /// Validate timestamp against constraints
    pub fn validate_timestamp(&self, timestamp: &DateTime<Utc>) -> IngestionResult<()> {
        if let Some(min) = &self.min_timestamp {
            if timestamp < min {
                return Err(IngestionError::data_validation(
                    format!("timestamp {} is before minimum {}", timestamp, min)
                ));
            }
        }
        
        if let Some(max) = &self.max_timestamp {
            if timestamp > max {
                return Err(IngestionError::data_validation(
                    format!("timestamp {} is after maximum {}", timestamp, max)
                ));
            }
        }
        
        Ok(())
    }
    
    /// Create a schema for specific KPI type
    pub fn for_kpi_type(kpi_type: &str) -> Self {
        let mut schema = Self::default();
        
        match kpi_type {
            "throughput" => {
                schema.allowed_kpi_names = Some(vec![
                    "throughput_dl".to_string(),
                    "throughput_ul".to_string(),
                    "throughput_total".to_string(),
                ]);
                schema.column_types.insert("kpi_value".to_string(), "float64".to_string());
            }
            "latency" => {
                schema.allowed_kpi_names = Some(vec![
                    "latency_rtt".to_string(),
                    "latency_processing".to_string(),
                    "latency_transmission".to_string(),
                ]);
                schema.column_types.insert("kpi_value".to_string(), "float32".to_string());
            }
            "resource_utilization" => {
                schema.allowed_kpi_names = Some(vec![
                    "prb_utilization_dl".to_string(),
                    "prb_utilization_ul".to_string(),
                    "cpu_utilization".to_string(),
                    "memory_utilization".to_string(),
                ]);
                schema.column_types.insert("kpi_value".to_string(), "float32".to_string());
            }
            _ => {}
        }
        
        schema
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_schema_validation() {
        let schema = StandardSchema::default();
        assert!(schema.validate().is_ok());
    }
    
    #[test]
    fn test_schema_validation_errors() {
        let mut schema = StandardSchema::default();
        
        schema.timestamp_column = "".to_string();
        assert!(schema.validate().is_err());
        
        schema.timestamp_column = "timestamp".to_string();
        schema.column_types.insert("test".to_string(), "invalid_type".to_string());
        assert!(schema.validate().is_err());
    }
    
    #[test]
    fn test_arrow_schema_creation() {
        let schema = StandardSchema::default();
        let arrow_schema = schema.to_arrow_schema().unwrap();
        
        assert!(arrow_schema.field_with_name("timestamp").is_ok());
        assert!(arrow_schema.field_with_name("cell_id").is_ok());
        assert!(arrow_schema.field_with_name("kpi_name").is_ok());
        assert!(arrow_schema.field_with_name("kpi_value").is_ok());
    }
    
    #[test]
    fn test_null_value_detection() {
        let schema = StandardSchema::default();
        assert!(schema.is_null_value(""));
        assert!(schema.is_null_value("null"));
        assert!(schema.is_null_value("NULL"));
        assert!(schema.is_null_value("N/A"));
        assert!(!schema.is_null_value("valid_value"));
    }
    
    #[test]
    fn test_column_mapping() {
        let schema = StandardSchema::default();
        let source_columns = vec![
            "Timestamp".to_string(),
            "Cell-ID".to_string(),
            "KPI_Name".to_string(),
            "KPI Value".to_string(),
        ];
        
        let mapping = schema.map_columns(&source_columns);
        assert!(mapping.len() > 0);
    }
    
    #[test]
    fn test_kpi_specific_schemas() {
        let throughput_schema = StandardSchema::for_kpi_type("throughput");
        assert!(throughput_schema.allowed_kpi_names.is_some());
        assert!(throughput_schema.allowed_kpi_names.unwrap().contains(&"throughput_dl".to_string()));
        
        let latency_schema = StandardSchema::for_kpi_type("latency");
        assert!(latency_schema.allowed_kpi_names.is_some());
        assert!(latency_schema.allowed_kpi_names.unwrap().contains(&"latency_rtt".to_string()));
    }
    
    #[test]
    fn test_timestamp_validation() {
        let mut schema = StandardSchema::default();
        let now = Utc::now();
        let past = now - chrono::Duration::hours(1);
        let future = now + chrono::Duration::hours(1);
        
        schema.min_timestamp = Some(past);
        schema.max_timestamp = Some(future);
        
        assert!(schema.validate_timestamp(&now).is_ok());
        assert!(schema.validate_timestamp(&(past - chrono::Duration::minutes(1))).is_err());
        assert!(schema.validate_timestamp(&(future + chrono::Duration::minutes(1))).is_err());
    }
}