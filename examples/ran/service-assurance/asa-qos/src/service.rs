use crate::config::Config;
use crate::error::{Error, Result};
use crate::forecasting::JitterForecaster;
use crate::proto::{
    service_assurance_service_server::ServiceAssuranceService,
    *,
};
use crate::storage::QosStorage;
use crate::types::{VoLteMetrics, Priority, RecommendationType};
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::{Request, Response, Status};
use tracing::{info, warn, error};

/// gRPC service implementation for VoLTE QoS forecasting
pub struct QosService {
    forecaster: Arc<RwLock<JitterForecaster>>,
    storage: Arc<QosStorage>,
    config: Config,
}

impl QosService {
    pub async fn new(config: Config) -> Result<Self> {
        let storage = Arc::new(QosStorage::new(&config.database).await?);
        let mut forecaster = JitterForecaster::new(config.clone());
        forecaster.initialize().await?;
        
        // Load historical data for training if available
        if let Ok(historical_data) = storage.get_historical_metrics("", 1000).await {
            if !historical_data.is_empty() {
                info!("Training models with {} historical data points", historical_data.len());
                forecaster.train(&historical_data).await?;
            }
        }
        
        Ok(Self {
            forecaster: Arc::new(RwLock::new(forecaster)),
            storage,
            config,
        })
    }
    
    async fn convert_proto_to_volte_metrics(&self, proto_metrics: &VoLteMetrics) -> Result<crate::types::VoLteMetrics> {
        let timestamp = if let Some(ts) = &proto_metrics.timestamp {
            DateTime::from_timestamp(ts.seconds, ts.nanos as u32)
                .unwrap_or_else(|| Utc::now())
        } else {
            Utc::now()
        };
        
        Ok(crate::types::VoLteMetrics {
            cell_id: proto_metrics.cell_id.clone(),
            timestamp,
            prb_utilization_dl: proto_metrics.prb_utilization_dl,
            active_volte_users: proto_metrics.active_volte_users,
            competing_gbr_traffic_mbps: proto_metrics.competing_gbr_traffic_mbps,
            current_jitter_ms: proto_metrics.current_jitter_ms,
            packet_loss_rate: proto_metrics.packet_loss_rate,
            delay_ms: proto_metrics.delay_ms,
        })
    }
    
    fn convert_jitter_forecast_to_proto(&self, forecast: &crate::types::JitterForecast) -> JitterForecast {
        JitterForecast {
            timestamp: Some(prost_types::Timestamp {
                seconds: forecast.timestamp.timestamp(),
                nanos: forecast.timestamp.timestamp_subsec_nanos() as i32,
            }),
            predicted_jitter_ms: forecast.predicted_jitter_ms,
            confidence: forecast.confidence,
            prediction_interval_lower: forecast.prediction_interval_lower,
            prediction_interval_upper: forecast.prediction_interval_upper,
        }
    }
    
    fn convert_quality_analysis_to_proto(&self, analysis: &crate::types::QualityAnalysis) -> JitterAnalysis {
        let quality_trend = match analysis.quality_trend {
            crate::types::QualityTrend::Improving => "IMPROVING",
            crate::types::QualityTrend::Stable => "STABLE",
            crate::types::QualityTrend::Degrading => "DEGRADING",
        };
        
        JitterAnalysis {
            cell_id: analysis.cell_id.clone(),
            baseline_jitter_ms: analysis.baseline_jitter_ms,
            peak_jitter_ms: analysis.peak_jitter_ms,
            jitter_variability: analysis.jitter_variability,
            contributing_factors: analysis.contributing_factors.clone(),
            quality_impact_score: analysis.quality_impact_score,
            quality_trend: quality_trend.to_string(),
        }
    }
    
    fn convert_recommendations_to_proto(&self, recommendations: &[crate::types::QualityRecommendation]) -> Vec<QualityRecommendation> {
        recommendations.iter().map(|rec| {
            let recommendation_type = match rec.recommendation_type {
                RecommendationType::TrafficShaping => "TRAFFIC_SHAPING",
                RecommendationType::ResourceAllocation => "RESOURCE_ALLOCATION",
                RecommendationType::PriorityAdjustment => "PRIORITY_ADJUSTMENT",
            };
            
            let priority = match rec.priority {
                Priority::Low => "LOW",
                Priority::Medium => "MEDIUM",
                Priority::High => "HIGH",
                Priority::Critical => "CRITICAL",
            };
            
            QualityRecommendation {
                recommendation_type: recommendation_type.to_string(),
                description: rec.description.clone(),
                expected_improvement_ms: rec.expected_improvement_ms,
                priority: priority.to_string(),
                implementation: rec.implementation.clone(),
            }
        }).collect()
    }
    
    async fn store_forecast_request(&self, cell_id: &str, metrics: &crate::types::VoLteMetrics) -> Result<()> {
        // Store the request metrics for future model training
        if let Err(e) = self.storage.store_volte_metrics(metrics).await {
            warn!("Failed to store VoLTE metrics: {}", e);
        }
        
        Ok(())
    }
}

#[tonic::async_trait]
impl ServiceAssuranceService for QosService {
    async fn forecast_vo_lte_jitter(
        &self,
        request: Request<ForecastVoLteJitterRequest>,
    ) -> std::result::Result<Response<ForecastVoLteJitterResponse>, Status> {
        let req = request.into_inner();
        
        info!("Received jitter forecast request for cell: {}", req.cell_id);
        
        // Validate request
        if req.cell_id.is_empty() {
            return Err(Status::invalid_argument("Cell ID cannot be empty"));
        }
        
        let current_metrics = req.current_metrics
            .ok_or_else(|| Status::invalid_argument("Current metrics are required"))?;
        
        // Convert protobuf metrics to internal format
        let volte_metrics = self.convert_proto_to_volte_metrics(&current_metrics)
            .await
            .map_err(|e| Status::internal(format!("Failed to convert metrics: {}", e)))?;
        
        // Store request for future training
        if let Err(e) = self.store_forecast_request(&req.cell_id, &volte_metrics).await {
            warn!("Failed to store forecast request: {}", e);
        }
        
        // Generate forecast
        let forecaster = self.forecaster.read().await;
        let forecasts = forecaster
            .forecast_jitter(
                &req.cell_id,
                &volte_metrics,
                if req.forecast_horizon_minutes > 0 {
                    Some(req.forecast_horizon_minutes)
                } else {
                    None
                },
            )
            .await
            .map_err(|e| Status::internal(format!("Forecast generation failed: {}", e)))?;
        
        // Generate quality analysis
        let recent_metrics = self.storage
            .get_recent_metrics(&req.cell_id, 100)
            .await
            .unwrap_or_default();
        
        let mut all_metrics = recent_metrics;
        all_metrics.push(volte_metrics.clone());
        
        let analysis = forecaster
            .analyze_quality(&req.cell_id, &all_metrics)
            .await
            .map_err(|e| Status::internal(format!("Quality analysis failed: {}", e)))?;
        
        // Generate recommendations
        let recommendations = forecaster
            .generate_recommendations(&analysis, &volte_metrics)
            .await
            .map_err(|e| Status::internal(format!("Recommendation generation failed: {}", e)))?;
        
        drop(forecaster); // Release read lock
        
        // Convert to protobuf format
        let proto_forecasts: Vec<JitterForecast> = forecasts
            .iter()
            .map(|f| self.convert_jitter_forecast_to_proto(f))
            .collect();
        
        let proto_analysis = self.convert_quality_analysis_to_proto(&analysis);
        let proto_recommendations = self.convert_recommendations_to_proto(&recommendations);
        
        let response = ForecastVoLteJitterResponse {
            cell_id: req.cell_id.clone(),
            forecasts: proto_forecasts,
            analysis: Some(proto_analysis),
            recommendations: proto_recommendations,
            metadata: req.metadata,
        };
        
        info!("Generated {} forecasts for cell: {}", response.forecasts.len(), req.cell_id);
        
        Ok(Response::new(response))
    }
    
    async fn get_qos_analysis(
        &self,
        request: Request<GetQosAnalysisRequest>,
    ) -> std::result::Result<Response<GetQosAnalysisResponse>, Status> {
        let req = request.into_inner();
        
        info!("Received QoS analysis request for {} cells", req.cell_ids.len());
        
        if req.cell_ids.is_empty() {
            return Err(Status::invalid_argument("At least one cell ID is required"));
        }
        
        let time_window_hours = if req.time_window_hours > 0 {
            req.time_window_hours
        } else {
            24 // Default to 24 hours
        };
        
        let mut reports = Vec::new();
        let mut total_cells = 0;
        let mut cells_with_issues = 0;
        let mut overall_quality_sum = 0.0;
        
        for cell_id in &req.cell_ids {
            total_cells += 1;
            
            // Get recent metrics for this cell
            let recent_metrics = self.storage
                .get_metrics_in_time_window(cell_id, time_window_hours)
                .await
                .unwrap_or_default();
            
            if recent_metrics.is_empty() {
                warn!("No metrics found for cell: {}", cell_id);
                continue;
            }
            
            // Generate quality analysis
            let forecaster = self.forecaster.read().await;
            let analysis = forecaster
                .analyze_quality(cell_id, &recent_metrics)
                .await
                .map_err(|e| Status::internal(format!("Quality analysis failed for cell {}: {}", cell_id, e)))?;
            
            // Generate recommendations
            let latest_metrics = recent_metrics.last().unwrap();
            let recommendations = forecaster
                .generate_recommendations(&analysis, latest_metrics)
                .await
                .map_err(|e| Status::internal(format!("Recommendation generation failed for cell {}: {}", cell_id, e)))?;
            
            drop(forecaster);
            
            // Create QoS metrics
            let qos_metrics = QosMetrics {
                throughput_mbps: 0.0, // Would calculate from historical data
                latency_ms: latest_metrics.delay_ms,
                jitter_ms: latest_metrics.current_jitter_ms,
                packet_loss_rate: latest_metrics.packet_loss_rate,
                availability: 0.99, // Would calculate from uptime data
                user_satisfaction_score: (1.0 - analysis.quality_impact_score) * 100.0,
            };
            
            // Determine performance score and risk level
            let performance_score = (1.0 - analysis.quality_impact_score) * 100.0;
            let risk_level = if analysis.quality_impact_score > 0.8 {
                "CRITICAL"
            } else if analysis.quality_impact_score > 0.6 {
                "HIGH"
            } else if analysis.quality_impact_score > 0.3 {
                "MEDIUM"
            } else {
                "LOW"
            };
            
            let quality_issues = analysis.contributing_factors.clone();
            if !quality_issues.is_empty() {
                cells_with_issues += 1;
            }
            
            let qos_analysis = QosAnalysis {
                cell_id: cell_id.clone(),
                service_type: req.service_type.clone(),
                performance_score,
                performance_trend: match analysis.quality_trend {
                    crate::types::QualityTrend::Improving => "IMPROVING",
                    crate::types::QualityTrend::Stable => "STABLE",
                    crate::types::QualityTrend::Degrading => "DEGRADING",
                }.to_string(),
                quality_issues,
                sla_compliance: if performance_score > 95.0 { 1.0 } else { performance_score / 100.0 },
                risk_level: risk_level.to_string(),
            };
            
            let report = QosReport {
                cell_id: cell_id.clone(),
                service_type: req.service_type.clone(),
                metrics: Some(qos_metrics),
                analysis: Some(qos_analysis),
                recommendations: self.convert_recommendations_to_proto(&recommendations),
                last_updated: Some(prost_types::Timestamp {
                    seconds: Utc::now().timestamp(),
                    nanos: 0,
                }),
            };
            
            overall_quality_sum += performance_score;
            reports.push(report);
        }
        
        // Create summary
        let overall_quality = if total_cells > 0 {
            overall_quality_sum / total_cells as f64
        } else {
            0.0
        };
        
        let summary = QosSummary {
            total_cells,
            service_performance: std::collections::HashMap::new(), // Would populate with aggregated metrics
            overall_network_quality: overall_quality,
            sla_compliance_rate: overall_quality / 100.0,
            cells_with_issues,
            top_issues: vec!["High jitter variability".to_string(), "Elevated packet loss".to_string()],
        };
        
        let response = GetQosAnalysisResponse {
            reports,
            summary: Some(summary),
            metadata: req.metadata,
        };
        
        info!("Generated QoS analysis for {} cells", response.reports.len());
        
        Ok(Response::new(response))
    }
    
    async fn health_check(
        &self,
        request: Request<HealthCheckRequest>,
    ) -> std::result::Result<Response<HealthCheckResponse>, Status> {
        let _req = request.into_inner();
        
        // Check forecaster status
        let forecaster_healthy = {
            let forecaster = self.forecaster.read().await;
            // Check if at least one model is trained
            forecaster.get_model_metrics().values().any(|m| m.accuracy_10ms > 0.0)
        };
        
        // Check storage status
        let storage_healthy = self.storage.health_check().await.is_ok();
        
        let status = if forecaster_healthy && storage_healthy {
            "SERVING"
        } else {
            "NOT_SERVING"
        };
        
        let health = ran_common::HealthCheck {
            status: status.to_string(),
            timestamp: Some(prost_types::Timestamp {
                seconds: Utc::now().timestamp(),
                nanos: 0,
            }),
            details: std::collections::HashMap::new(),
        };
        
        let response = HealthCheckResponse {
            health: Some(health),
        };
        
        Ok(Response::new(response))
    }
    
    // Placeholder implementations for other service methods
    
    async fn classify_ul_interference(
        &self,
        _request: Request<ClassifyUlInterferenceRequest>,
    ) -> std::result::Result<Response<ClassifyUlInterferenceResponse>, Status> {
        Err(Status::unimplemented("UL interference classification not implemented in QoS service"))
    }
    
    async fn get_interference_analysis(
        &self,
        _request: Request<GetInterferenceAnalysisRequest>,
    ) -> std::result::Result<Response<GetInterferenceAnalysisResponse>, Status> {
        Err(Status::unimplemented("Interference analysis not implemented in QoS service"))
    }
    
    async fn predict_endc_failure(
        &self,
        _request: Request<PredictEndcFailureRequest>,
    ) -> std::result::Result<Response<PredictEndcFailureResponse>, Status> {
        Err(Status::unimplemented("ENDC failure prediction not implemented in QoS service"))
    }
    
    async fn get5_g_service_health(
        &self,
        _request: Request<Get5GServiceHealthRequest>,
    ) -> std::result::Result<Response<Get5GServiceHealthResponse>, Status> {
        Err(Status::unimplemented("5G service health not implemented in QoS service"))
    }
}