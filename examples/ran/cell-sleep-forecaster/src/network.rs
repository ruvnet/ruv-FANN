//! Network interface module for cellular network management integration

use std::sync::Arc;
use std::collections::HashMap;
use std::time::Duration;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use anyhow::Result;
use reqwest::{Client, header::HeaderMap};
use tokio::sync::RwLock;
use tokio::time::{sleep, Instant};

use crate::{PrbUtilization, SleepWindow, ForecastingError, config::ForecastingConfig};

/// Network management client for cellular infrastructure
pub struct NetworkClient {
    config: Arc<ForecastingConfig>,
    http_client: Client,
    rate_limiter: Arc<RwLock<RateLimiter>>,
    auth_token: Option<String>,
}

/// Rate limiter for API requests
struct RateLimiter {
    requests_per_minute: u32,
    request_timestamps: Vec<Instant>,
}

impl RateLimiter {
    fn new(requests_per_minute: u32) -> Self {
        Self {
            requests_per_minute,
            request_timestamps: Vec::new(),
        }
    }
    
    async fn check_rate_limit(&mut self) -> bool {
        let now = Instant::now();
        let one_minute_ago = now - Duration::from_secs(60);
        
        // Remove old timestamps
        self.request_timestamps.retain(|&timestamp| timestamp > one_minute_ago);
        
        if self.request_timestamps.len() < self.requests_per_minute as usize {
            self.request_timestamps.push(now);
            true
        } else {
            false
        }
    }
    
    async fn wait_for_rate_limit(&mut self) {
        while !self.check_rate_limit().await {
            sleep(Duration::from_millis(100)).await;
        }
    }
}

/// Cell site information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellSite {
    pub cell_id: String,
    pub site_name: String,
    pub latitude: f64,
    pub longitude: f64,
    pub technology: CellTechnology,
    pub frequency_band: String,
    pub max_power_watts: f64,
    pub coverage_radius_km: f64,
    pub neighboring_cells: Vec<String>,
    pub status: CellStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellTechnology {
    #[serde(rename = "4G")]
    FourG,
    #[serde(rename = "5G")]
    FiveG,
    #[serde(rename = "5G_NSA")]
    FiveGNsa,
    #[serde(rename = "5G_SA")]
    FiveGSa,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CellStatus {
    Active,
    Sleeping,
    Maintenance,
    Offline,
}

/// Real-time cell performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellPerformanceData {
    pub cell_id: String,
    pub timestamp: DateTime<Utc>,
    pub prb_utilization: PrbUtilization,
    pub throughput_dl_mbps: f64,
    pub throughput_ul_mbps: f64,
    pub connected_users: u32,
    pub signal_strength_dbm: f64,
    pub interference_level: f64,
    pub energy_consumption_watts: f64,
    pub temperature_celsius: f64,
}

/// Sleep mode command
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepModeCommand {
    pub cell_id: String,
    pub command: SleepCommand,
    pub scheduled_time: Option<DateTime<Utc>>,
    pub duration_minutes: Option<u32>,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SleepCommand {
    Sleep,
    Wake,
    ScheduleSleep,
    CancelSleep,
}

/// Network management API responses
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiResponse<T> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SleepModeResponse {
    pub command_id: String,
    pub cell_id: String,
    pub status: CommandStatus,
    pub message: String,
    pub executed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CommandStatus {
    Pending,
    Executing,
    Completed,
    Failed,
    Cancelled,
}

impl NetworkClient {
    pub async fn new(config: Arc<ForecastingConfig>) -> Result<Self> {
        let mut headers = HeaderMap::new();
        headers.insert("Content-Type", "application/json".parse()?);
        headers.insert("User-Agent", "CellSleepForecaster/1.0".parse()?);
        
        if let Some(token) = &config.network.auth_token {
            headers.insert("Authorization", format!("Bearer {}", token).parse()?);
        }
        
        let http_client = Client::builder()
            .timeout(Duration::from_secs(config.network.timeout_seconds))
            .default_headers(headers)
            .build()?;
        
        let rate_limiter = Arc::new(RwLock::new(RateLimiter::new(
            config.network.rate_limit_requests_per_minute
        )));
        
        Ok(Self {
            config: config.clone(),
            http_client,
            rate_limiter,
            auth_token: config.network.auth_token.clone(),
        })
    }
    
    /// Get current PRB utilization data for a cell
    pub async fn get_prb_utilization(&self, cell_id: &str) -> Result<PrbUtilization> {
        let url = format!("{}/cells/{}/prb-utilization", self.config.network.base_url, cell_id);
        
        self.rate_limiter.write().await.wait_for_rate_limit().await;
        
        let response = self.http_client
            .get(&url)
            .send()
            .await?;
        
        if response.status().is_success() {
            let api_response: ApiResponse<PrbUtilization> = response.json().await?;
            if api_response.success {
                api_response.data.ok_or_else(|| {
                    anyhow::anyhow!("Network request failed")
                })
            } else {
                Err(anyhow::anyhow!("API returned error"))
            }
        } else {
            Err(anyhow::anyhow!("HTTP request failed"))
        }
    }
    
    /// Get historical PRB utilization data for a cell
    pub async fn get_historical_prb_data(
        &self,
        cell_id: &str,
        start_time: DateTime<Utc>,
        end_time: DateTime<Utc>,
    ) -> Result<Vec<PrbUtilization>> {
        let url = format!(
            "{}/cells/{}/prb-utilization/history?start={}&end={}",
            self.config.network.base_url,
            cell_id,
            start_time.format("%Y-%m-%dT%H:%M:%SZ"),
            end_time.format("%Y-%m-%dT%H:%M:%SZ")
        );
        
        self.rate_limiter.write().await.wait_for_rate_limit().await;
        
        let response = self.http_client
            .get(&url)
            .send()
            .await?;
        
        if response.status().is_success() {
            let api_response: ApiResponse<Vec<PrbUtilization>> = response.json().await?;
            if api_response.success {
                Ok(api_response.data.unwrap_or_default())
            } else {
                Err(anyhow::anyhow!("API returned error"))
            }
        } else {
            Err(anyhow::anyhow!("HTTP request failed"))
        }
    }
    
    /// Get cell site information
    pub async fn get_cell_info(&self, cell_id: &str) -> Result<CellSite> {
        let url = format!("{}/cells/{}/info", self.config.network.base_url, cell_id);
        
        self.rate_limiter.write().await.wait_for_rate_limit().await;
        
        let response = self.http_client
            .get(&url)
            .send()
            .await?;
        
        if response.status().is_success() {
            let api_response: ApiResponse<CellSite> = response.json().await?;
            if api_response.success {
                api_response.data.ok_or_else(|| {
                    anyhow::anyhow!("Network request failed")
                })
            } else {
                Err(anyhow::anyhow!("API returned error"))
            }
        } else {
            Err(anyhow::anyhow!("HTTP request failed"))
        }
    }
    
    /// Get real-time performance data for a cell
    pub async fn get_performance_data(&self, cell_id: &str) -> Result<CellPerformanceData> {
        let url = format!("{}/cells/{}/performance", self.config.network.base_url, cell_id);
        
        self.rate_limiter.write().await.wait_for_rate_limit().await;
        
        let response = self.http_client
            .get(&url)
            .send()
            .await?;
        
        if response.status().is_success() {
            let api_response: ApiResponse<CellPerformanceData> = response.json().await?;
            if api_response.success {
                api_response.data.ok_or_else(|| {
                    anyhow::anyhow!("Network request failed")
                })
            } else {
                Err(anyhow::anyhow!("API returned error"))
            }
        } else {
            Err(anyhow::anyhow!("HTTP request failed"))
        }
    }
    
    /// Send sleep mode command to a cell
    pub async fn send_sleep_command(&self, command: SleepModeCommand) -> Result<SleepModeResponse> {
        let url = format!("{}/cells/{}/sleep-mode", self.config.network.base_url, command.cell_id);
        
        self.rate_limiter.write().await.wait_for_rate_limit().await;
        
        let response = self.http_client
            .post(&url)
            .json(&command)
            .send()
            .await?;
        
        if response.status().is_success() {
            let api_response: ApiResponse<SleepModeResponse> = response.json().await?;
            if api_response.success {
                api_response.data.ok_or_else(|| {
                    anyhow::anyhow!("Network request failed")
                })
            } else {
                Err(anyhow::anyhow!("API returned error"))
            }
        } else {
            Err(anyhow::anyhow!("HTTP request failed"))
        }
    }
    
    /// Schedule sleep window for a cell
    pub async fn schedule_sleep_window(&self, sleep_window: &SleepWindow) -> Result<SleepModeResponse> {
        let command = SleepModeCommand {
            cell_id: sleep_window.cell_id.clone(),
            command: SleepCommand::ScheduleSleep,
            scheduled_time: Some(sleep_window.start_time),
            duration_minutes: Some(sleep_window.duration_minutes),
            reason: format!(
                "Automated sleep scheduling - Predicted utilization: {:.1}%, Energy savings: {:.2} kWh",
                sleep_window.predicted_utilization,
                sleep_window.energy_savings_kwh
            ),
        };
        
        self.send_sleep_command(command).await
    }
    
    /// Cancel scheduled sleep for a cell
    pub async fn cancel_sleep_window(&self, cell_id: &str) -> Result<SleepModeResponse> {
        let command = SleepModeCommand {
            cell_id: cell_id.to_string(),
            command: SleepCommand::CancelSleep,
            scheduled_time: None,
            duration_minutes: None,
            reason: "Cancelling scheduled sleep due to forecast update".to_string(),
        };
        
        self.send_sleep_command(command).await
    }
    
    /// Get neighboring cells for load balancing assessment
    pub async fn get_neighboring_cells(&self, cell_id: &str) -> Result<Vec<CellSite>> {
        let url = format!("{}/cells/{}/neighbors", self.config.network.base_url, cell_id);
        
        self.rate_limiter.write().await.wait_for_rate_limit().await;
        
        let response = self.http_client
            .get(&url)
            .send()
            .await?;
        
        if response.status().is_success() {
            let api_response: ApiResponse<Vec<CellSite>> = response.json().await?;
            if api_response.success {
                Ok(api_response.data.unwrap_or_default())
            } else {
                Err(anyhow::anyhow!("API returned error"))
            }
        } else {
            Err(anyhow::anyhow!("HTTP request failed"))
        }
    }
    
    /// Get cell status
    pub async fn get_cell_status(&self, cell_id: &str) -> Result<CellStatus> {
        let url = format!("{}/cells/{}/status", self.config.network.base_url, cell_id);
        
        self.rate_limiter.write().await.wait_for_rate_limit().await;
        
        let response = self.http_client
            .get(&url)
            .send()
            .await?;
        
        if response.status().is_success() {
            let api_response: ApiResponse<CellStatus> = response.json().await?;
            if api_response.success {
                api_response.data.ok_or_else(|| {
                    anyhow::anyhow!("Network request failed")
                })
            } else {
                Err(anyhow::anyhow!("API returned error"))
            }
        } else {
            Err(anyhow::anyhow!("HTTP request failed"))
        }
    }
    
    /// Bulk fetch PRB utilization for multiple cells
    pub async fn get_bulk_prb_utilization(&self, cell_ids: &[String]) -> Result<HashMap<String, PrbUtilization>> {
        let url = format!("{}/cells/bulk/prb-utilization", self.config.network.base_url);
        
        self.rate_limiter.write().await.wait_for_rate_limit().await;
        
        let request_body = serde_json::json!({
            "cell_ids": cell_ids
        });
        
        let response = self.http_client
            .post(&url)
            .json(&request_body)
            .send()
            .await?;
        
        if response.status().is_success() {
            let api_response: ApiResponse<HashMap<String, PrbUtilization>> = response.json().await?;
            if api_response.success {
                Ok(api_response.data.unwrap_or_default())
            } else {
                Err(anyhow::anyhow!("API returned error"))
            }
        } else {
            Err(anyhow::anyhow!("HTTP request failed"))
        }
    }
    
    /// Test network connectivity
    pub async fn test_connectivity(&self) -> Result<bool> {
        let url = format!("{}/health", self.config.network.base_url);
        
        let response = self.http_client
            .get(&url)
            .timeout(Duration::from_secs(5))
            .send()
            .await?;
        
        Ok(response.status().is_success())
    }
}

/// Network interface for mock/testing environments
pub struct MockNetworkClient {
    cell_data: HashMap<String, CellSite>,
    prb_data: HashMap<String, Vec<PrbUtilization>>,
}

impl MockNetworkClient {
    pub fn new() -> Self {
        Self {
            cell_data: HashMap::new(),
            prb_data: HashMap::new(),
        }
    }
    
    pub fn add_mock_cell(&mut self, cell: CellSite) {
        self.cell_data.insert(cell.cell_id.clone(), cell);
    }
    
    pub fn add_mock_prb_data(&mut self, cell_id: String, data: Vec<PrbUtilization>) {
        self.prb_data.insert(cell_id, data);
    }
    
    pub async fn get_prb_utilization(&self, cell_id: &str) -> Result<PrbUtilization> {
        if let Some(data) = self.prb_data.get(cell_id) {
            if let Some(latest) = data.last() {
                Ok(latest.clone())
            } else {
                Err(anyhow::anyhow!("No mock data available"))
            }
        } else {
            // Generate mock data
            Ok(PrbUtilization::new(
                cell_id.to_string(),
                100,
                rand::random::<u32>() % 100,
                rand::random::<f64>() * 200.0,
                rand::random::<u32>() % 50,
                0.8 + rand::random::<f64>() * 0.2,
            ))
        }
    }
    
    pub async fn schedule_sleep_window(&self, sleep_window: &SleepWindow) -> Result<SleepModeResponse> {
        Ok(SleepModeResponse {
            command_id: format!("mock_cmd_{}", Utc::now().timestamp()),
            cell_id: sleep_window.cell_id.clone(),
            status: CommandStatus::Completed,
            message: "Mock sleep command executed successfully".to_string(),
            executed_at: Some(Utc::now()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ForecastingConfig;
    
    #[tokio::test]
    async fn test_network_client_creation() {
        let config = Arc::new(ForecastingConfig::default());
        let client = NetworkClient::new(config).await.unwrap();
        
        // Test basic client properties
        assert!(client.auth_token.is_none());
    }
    
    #[tokio::test]
    async fn test_rate_limiter() {
        let mut rate_limiter = RateLimiter::new(2); // 2 requests per minute
        
        // First two requests should succeed
        assert!(rate_limiter.check_rate_limit().await);
        assert!(rate_limiter.check_rate_limit().await);
        
        // Third request should fail
        assert!(!rate_limiter.check_rate_limit().await);
    }
    
    #[tokio::test]
    async fn test_mock_network_client() {
        let mut mock_client = MockNetworkClient::new();
        
        // Add mock cell
        let cell = CellSite {
            cell_id: "test_cell".to_string(),
            site_name: "Test Site".to_string(),
            latitude: 40.7128,
            longitude: -74.0060,
            technology: CellTechnology::FiveG,
            frequency_band: "n78".to_string(),
            max_power_watts: 1000.0,
            coverage_radius_km: 2.0,
            neighboring_cells: vec!["neighbor1".to_string(), "neighbor2".to_string()],
            status: CellStatus::Active,
        };
        
        mock_client.add_mock_cell(cell);
        
        // Test PRB utilization retrieval
        let prb_data = mock_client.get_prb_utilization("test_cell").await.unwrap();
        assert_eq!(prb_data.cell_id, "test_cell");
        assert!(prb_data.utilization_percentage >= 0.0 && prb_data.utilization_percentage <= 100.0);
    }
    
    #[test]
    fn test_sleep_command_serialization() {
        let command = SleepModeCommand {
            cell_id: "test_cell".to_string(),
            command: SleepCommand::ScheduleSleep,
            scheduled_time: Some(Utc::now()),
            duration_minutes: Some(60),
            reason: "Test sleep command".to_string(),
        };
        
        let json = serde_json::to_string(&command).unwrap();
        let deserialized: SleepModeCommand = serde_json::from_str(&json).unwrap();
        
        assert_eq!(command.cell_id, deserialized.cell_id);
        assert_eq!(command.duration_minutes, deserialized.duration_minutes);
    }
}