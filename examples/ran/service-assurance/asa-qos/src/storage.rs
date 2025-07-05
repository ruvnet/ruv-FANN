use crate::config::DatabaseConfig;
use crate::error::{Error, Result};
use crate::types::{VoLteMetrics, QualityAlert, ModelMetrics};
use chrono::{DateTime, Utc, Duration};
use sqlx::{PgPool, Row, FromRow};
use tracing::{info, warn, error};

/// Database storage for QoS metrics and forecasting data
pub struct QosStorage {
    pool: PgPool,
}

impl QosStorage {
    pub async fn new(config: &DatabaseConfig) -> Result<Self> {
        let pool = PgPool::connect(&config.url).await?;
        
        let storage = Self { pool };
        
        // Run migrations
        storage.run_migrations().await?;
        
        Ok(storage)
    }
    
    async fn run_migrations(&self) -> Result<()> {
        info!("Running database migrations");
        
        // Create VoLTE metrics table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS volte_metrics (
                id SERIAL PRIMARY KEY,
                cell_id VARCHAR(50) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                prb_utilization_dl REAL NOT NULL,
                active_volte_users INTEGER NOT NULL,
                competing_gbr_traffic_mbps REAL NOT NULL,
                current_jitter_ms REAL NOT NULL,
                packet_loss_rate REAL NOT NULL,
                delay_ms REAL NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        "#)
        .execute(&self.pool)
        .await?;
        
        // Create index for efficient queries
        sqlx::query(r#"
            CREATE INDEX IF NOT EXISTS idx_volte_metrics_cell_timestamp 
            ON volte_metrics (cell_id, timestamp DESC)
        "#)
        .execute(&self.pool)
        .await?;
        
        // Create quality alerts table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS quality_alerts (
                id SERIAL PRIMARY KEY,
                alert_id VARCHAR(36) UNIQUE NOT NULL,
                cell_id VARCHAR(50) NOT NULL,
                alert_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                message TEXT NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                predicted_jitter_ms REAL NOT NULL,
                confidence REAL NOT NULL,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at TIMESTAMPTZ,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        "#)
        .execute(&self.pool)
        .await?;
        
        // Create model metrics table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS model_metrics (
                id SERIAL PRIMARY KEY,
                model_id VARCHAR(50) NOT NULL,
                accuracy_10ms REAL NOT NULL,
                mae REAL NOT NULL,
                rmse REAL NOT NULL,
                mape REAL NOT NULL,
                r2_score REAL NOT NULL,
                training_time_ms BIGINT NOT NULL,
                inference_time_ms BIGINT NOT NULL,
                recorded_at TIMESTAMPTZ DEFAULT NOW()
            )
        "#)
        .execute(&self.pool)
        .await?;
        
        // Create forecasts table
        sqlx::query(r#"
            CREATE TABLE IF NOT EXISTS jitter_forecasts (
                id SERIAL PRIMARY KEY,
                cell_id VARCHAR(50) NOT NULL,
                forecast_timestamp TIMESTAMPTZ NOT NULL,
                predicted_jitter_ms REAL NOT NULL,
                confidence REAL NOT NULL,
                prediction_interval_lower REAL NOT NULL,
                prediction_interval_upper REAL NOT NULL,
                model_used VARCHAR(50) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        "#)
        .execute(&self.pool)
        .await?;
        
        info!("Database migrations completed successfully");
        Ok(())
    }
    
    /// Store VoLTE metrics
    pub async fn store_volte_metrics(&self, metrics: &VoLteMetrics) -> Result<i32> {
        let id = sqlx::query!(
            r#"
            INSERT INTO volte_metrics (
                cell_id, timestamp, prb_utilization_dl, active_volte_users,
                competing_gbr_traffic_mbps, current_jitter_ms, packet_loss_rate, delay_ms
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            "#,
            metrics.cell_id,
            metrics.timestamp,
            metrics.prb_utilization_dl as f32,
            metrics.active_volte_users as i32,
            metrics.competing_gbr_traffic_mbps as f32,
            metrics.current_jitter_ms as f32,
            metrics.packet_loss_rate as f32,
            metrics.delay_ms as f32
        )
        .fetch_one(&self.pool)
        .await?
        .id;
        
        Ok(id)
    }
    
    /// Get recent metrics for a cell
    pub async fn get_recent_metrics(&self, cell_id: &str, limit: i32) -> Result<Vec<VoLteMetrics>> {
        let rows = sqlx::query!(
            r#"
            SELECT cell_id, timestamp, prb_utilization_dl, active_volte_users,
                   competing_gbr_traffic_mbps, current_jitter_ms, packet_loss_rate, delay_ms
            FROM volte_metrics
            WHERE cell_id = $1
            ORDER BY timestamp DESC
            LIMIT $2
            "#,
            cell_id,
            limit
        )
        .fetch_all(&self.pool)
        .await?;
        
        let mut metrics = Vec::new();
        for row in rows {
            metrics.push(VoLteMetrics {
                cell_id: row.cell_id,
                timestamp: row.timestamp,
                prb_utilization_dl: row.prb_utilization_dl as f64,
                active_volte_users: row.active_volte_users as u32,
                competing_gbr_traffic_mbps: row.competing_gbr_traffic_mbps as f64,
                current_jitter_ms: row.current_jitter_ms as f64,
                packet_loss_rate: row.packet_loss_rate as f64,
                delay_ms: row.delay_ms as f64,
            });
        }
        
        // Reverse to get chronological order
        metrics.reverse();
        Ok(metrics)
    }
    
    /// Get historical metrics for training
    pub async fn get_historical_metrics(&self, cell_id: &str, limit: i32) -> Result<Vec<VoLteMetrics>> {
        let query = if cell_id.is_empty() {
            // Get data from all cells if cell_id is empty
            sqlx::query!(
                r#"
                SELECT cell_id, timestamp, prb_utilization_dl, active_volte_users,
                       competing_gbr_traffic_mbps, current_jitter_ms, packet_loss_rate, delay_ms
                FROM volte_metrics
                ORDER BY timestamp ASC
                LIMIT $1
                "#,
                limit
            )
            .fetch_all(&self.pool)
            .await?
        } else {
            sqlx::query!(
                r#"
                SELECT cell_id, timestamp, prb_utilization_dl, active_volte_users,
                       competing_gbr_traffic_mbps, current_jitter_ms, packet_loss_rate, delay_ms
                FROM volte_metrics
                WHERE cell_id = $1
                ORDER BY timestamp ASC
                LIMIT $2
                "#,
                cell_id,
                limit
            )
            .fetch_all(&self.pool)
            .await?
        };
        
        let mut metrics = Vec::new();
        for row in query {
            metrics.push(VoLteMetrics {
                cell_id: row.cell_id,
                timestamp: row.timestamp,
                prb_utilization_dl: row.prb_utilization_dl as f64,
                active_volte_users: row.active_volte_users as u32,
                competing_gbr_traffic_mbps: row.competing_gbr_traffic_mbps as f64,
                current_jitter_ms: row.current_jitter_ms as f64,
                packet_loss_rate: row.packet_loss_rate as f64,
                delay_ms: row.delay_ms as f64,
            });
        }
        
        Ok(metrics)
    }
    
    /// Get metrics within a time window
    pub async fn get_metrics_in_time_window(
        &self,
        cell_id: &str,
        hours: i32,
    ) -> Result<Vec<VoLteMetrics>> {
        let start_time = Utc::now() - Duration::hours(hours as i64);
        
        let rows = sqlx::query!(
            r#"
            SELECT cell_id, timestamp, prb_utilization_dl, active_volte_users,
                   competing_gbr_traffic_mbps, current_jitter_ms, packet_loss_rate, delay_ms
            FROM volte_metrics
            WHERE cell_id = $1 AND timestamp >= $2
            ORDER BY timestamp ASC
            "#,
            cell_id,
            start_time
        )
        .fetch_all(&self.pool)
        .await?;
        
        let mut metrics = Vec::new();
        for row in rows {
            metrics.push(VoLteMetrics {
                cell_id: row.cell_id,
                timestamp: row.timestamp,
                prb_utilization_dl: row.prb_utilization_dl as f64,
                active_volte_users: row.active_volte_users as u32,
                competing_gbr_traffic_mbps: row.competing_gbr_traffic_mbps as f64,
                current_jitter_ms: row.current_jitter_ms as f64,
                packet_loss_rate: row.packet_loss_rate as f64,
                delay_ms: row.delay_ms as f64,
            });
        }
        
        Ok(metrics)
    }
    
    /// Store quality alert
    pub async fn store_alert(&self, alert: &QualityAlert) -> Result<i32> {
        let alert_type = format!("{:?}", alert.alert_type);
        let severity = format!("{:?}", alert.severity);
        
        let id = sqlx::query!(
            r#"
            INSERT INTO quality_alerts (
                alert_id, cell_id, alert_type, severity, message,
                timestamp, predicted_jitter_ms, confidence
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            "#,
            alert.alert_id,
            alert.cell_id,
            alert_type,
            severity,
            alert.message,
            alert.timestamp,
            alert.predicted_jitter_ms as f32,
            alert.confidence as f32
        )
        .fetch_one(&self.pool)
        .await?
        .id;
        
        Ok(id)
    }
    
    /// Get active alerts for a cell
    pub async fn get_active_alerts(&self, cell_id: &str) -> Result<Vec<QualityAlert>> {
        let rows = sqlx::query!(
            r#"
            SELECT alert_id, cell_id, alert_type, severity, message,
                   timestamp, predicted_jitter_ms, confidence
            FROM quality_alerts
            WHERE cell_id = $1 AND resolved = FALSE
            ORDER BY timestamp DESC
            "#,
            cell_id
        )
        .fetch_all(&self.pool)
        .await?;
        
        let mut alerts = Vec::new();
        for row in rows {
            // Note: In a real implementation, you'd need to properly deserialize
            // the alert_type and severity from strings back to enums
            alerts.push(QualityAlert {
                alert_id: row.alert_id,
                cell_id: row.cell_id,
                alert_type: crate::types::AlertType::JitterThresholdExceeded, // Simplified
                severity: crate::types::AlertSeverity::Warning, // Simplified
                message: row.message,
                timestamp: row.timestamp,
                predicted_jitter_ms: row.predicted_jitter_ms as f64,
                confidence: row.confidence as f64,
                recommendations: vec![], // Would need separate table
            });
        }
        
        Ok(alerts)
    }
    
    /// Store model metrics
    pub async fn store_model_metrics(&self, metrics: &ModelMetrics) -> Result<i32> {
        let id = sqlx::query!(
            r#"
            INSERT INTO model_metrics (
                model_id, accuracy_10ms, mae, rmse, mape, r2_score,
                training_time_ms, inference_time_ms
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING id
            "#,
            metrics.model_id,
            metrics.accuracy_10ms as f32,
            metrics.mae as f32,
            metrics.rmse as f32,
            metrics.mape as f32,
            metrics.r2_score as f32,
            metrics.training_time_ms as i64,
            metrics.inference_time_ms as i64
        )
        .fetch_one(&self.pool)
        .await?
        .id;
        
        Ok(id)
    }
    
    /// Store forecast results
    pub async fn store_forecast(
        &self,
        cell_id: &str,
        forecast: &crate::types::JitterForecast,
        model_used: &str,
    ) -> Result<i32> {
        let id = sqlx::query!(
            r#"
            INSERT INTO jitter_forecasts (
                cell_id, forecast_timestamp, predicted_jitter_ms, confidence,
                prediction_interval_lower, prediction_interval_upper, model_used
            ) VALUES ($1, $2, $3, $4, $5, $6, $7)
            RETURNING id
            "#,
            cell_id,
            forecast.timestamp,
            forecast.predicted_jitter_ms as f32,
            forecast.confidence as f32,
            forecast.prediction_interval_lower as f32,
            forecast.prediction_interval_upper as f32,
            model_used
        )
        .fetch_one(&self.pool)
        .await?
        .id;
        
        Ok(id)
    }
    
    /// Get forecast accuracy statistics
    pub async fn get_forecast_accuracy(&self, model_id: &str, days: i32) -> Result<f64> {
        let start_time = Utc::now() - Duration::days(days as i64);
        
        let result = sqlx::query!(
            r#"
            SELECT AVG(
                CASE WHEN ABS(f.predicted_jitter_ms - m.current_jitter_ms) <= 10.0 
                THEN 1.0 ELSE 0.0 END
            ) as accuracy
            FROM jitter_forecasts f
            JOIN volte_metrics m ON f.cell_id = m.cell_id 
                AND ABS(EXTRACT(EPOCH FROM (f.forecast_timestamp - m.timestamp))) < 300
            WHERE f.model_used = $1 AND f.created_at >= $2
            "#,
            model_id,
            start_time
        )
        .fetch_one(&self.pool)
        .await?;
        
        Ok(result.accuracy.unwrap_or(0.0))
    }
    
    /// Clean old data
    pub async fn cleanup_old_data(&self, retention_days: i32) -> Result<()> {
        let cutoff_time = Utc::now() - Duration::days(retention_days as i64);
        
        // Clean old metrics
        let deleted_metrics = sqlx::query!(
            "DELETE FROM volte_metrics WHERE timestamp < $1",
            cutoff_time
        )
        .execute(&self.pool)
        .await?
        .rows_affected();
        
        // Clean old forecasts
        let deleted_forecasts = sqlx::query!(
            "DELETE FROM jitter_forecasts WHERE created_at < $1",
            cutoff_time
        )
        .execute(&self.pool)
        .await?
        .rows_affected();
        
        // Clean resolved alerts
        let deleted_alerts = sqlx::query!(
            "DELETE FROM quality_alerts WHERE resolved = TRUE AND resolved_at < $1",
            cutoff_time
        )
        .execute(&self.pool)
        .await?
        .rows_affected();
        
        info!(
            "Cleanup completed: {} metrics, {} forecasts, {} alerts removed",
            deleted_metrics, deleted_forecasts, deleted_alerts
        );
        
        Ok(())
    }
    
    /// Health check
    pub async fn health_check(&self) -> Result<()> {
        sqlx::query("SELECT 1")
            .execute(&self.pool)
            .await?;
        
        Ok(())
    }
    
    /// Get database statistics
    pub async fn get_statistics(&self) -> Result<DatabaseStatistics> {
        let metrics_count = sqlx::query!("SELECT COUNT(*) as count FROM volte_metrics")
            .fetch_one(&self.pool)
            .await?
            .count
            .unwrap_or(0);
        
        let alerts_count = sqlx::query!("SELECT COUNT(*) as count FROM quality_alerts WHERE resolved = FALSE")
            .fetch_one(&self.pool)
            .await?
            .count
            .unwrap_or(0);
        
        let forecasts_count = sqlx::query!("SELECT COUNT(*) as count FROM jitter_forecasts")
            .fetch_one(&self.pool)
            .await?
            .count
            .unwrap_or(0);
        
        let unique_cells = sqlx::query!("SELECT COUNT(DISTINCT cell_id) as count FROM volte_metrics")
            .fetch_one(&self.pool)
            .await?
            .count
            .unwrap_or(0);
        
        Ok(DatabaseStatistics {
            total_metrics: metrics_count,
            active_alerts: alerts_count,
            total_forecasts: forecasts_count,
            unique_cells,
        })
    }
}

#[derive(Debug)]
pub struct DatabaseStatistics {
    pub total_metrics: i64,
    pub active_alerts: i64,
    pub total_forecasts: i64,
    pub unique_cells: i64,
}