//! Dynamic resource management for autonomous agents

#[cfg(feature = "async")]
use async_trait::async_trait;

use serde::{Deserialize, Serialize};

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::{boxed::Box, string::String, vec::Vec};

use crate::{traits::ResourceAllocation, DAAResult as Result};

/// Dynamic resource management capabilities
#[cfg_attr(feature = "async", async_trait)]
pub trait DynamicResourceManager: Send + Sync {
    /// Resource type managed
    type Resource: Send + Sync;

    /// Allocate resources dynamically
    #[cfg(feature = "async")]
    async fn allocate_resources(&mut self, request: &ResourceRequest)
        -> Result<ResourceAllocation>;

    /// Allocate resources (sync version)
    #[cfg(not(feature = "async"))]
    fn allocate_resources(&mut self, request: &ResourceRequest) -> Result<ResourceAllocation>;

    /// Deallocate resources
    #[cfg(feature = "async")]
    async fn deallocate_resources(&mut self, allocation_id: &str) -> Result<()>;

    /// Deallocate resources (sync version)
    #[cfg(not(feature = "async"))]
    fn deallocate_resources(&mut self, allocation_id: &str) -> Result<()>;

    /// Optimize resource usage
    #[cfg(feature = "async")]
    async fn optimize_usage(&mut self) -> Result<OptimizationResult>;

    /// Optimize usage (sync version)
    #[cfg(not(feature = "async"))]
    fn optimize_usage(&mut self) -> Result<OptimizationResult>;

    /// Monitor resource utilization
    fn monitor_utilization(&self) -> ResourceUtilization;

    /// Predict resource needs
    fn predict_needs(&self, horizon: u32) -> Result<ResourcePrediction>;

    /// Check resource availability
    fn check_availability(&self, request: &ResourceRequest) -> ResourceAvailability;

    /// Get current resource status
    fn resource_status(&self) -> ResourceStatus;
}

/// Resource monitoring capabilities
#[cfg_attr(feature = "async", async_trait)]
pub trait ResourceMonitor: Send + Sync {
    /// Start monitoring resources
    #[cfg(feature = "async")]
    async fn start_monitoring(&mut self) -> Result<()>;

    /// Start monitoring (sync version)
    #[cfg(not(feature = "async"))]
    fn start_monitoring(&mut self) -> Result<()>;

    /// Stop monitoring
    #[cfg(feature = "async")]
    async fn stop_monitoring(&mut self) -> Result<()>;

    /// Stop monitoring (sync version)
    #[cfg(not(feature = "async"))]
    fn stop_monitoring(&mut self) -> Result<()>;

    /// Get current metrics
    fn get_metrics(&self) -> ResourceMetrics;

    /// Get historical data
    fn get_history(&self, duration: u64) -> Vec<ResourceSnapshot>;

    /// Set monitoring parameters
    fn set_parameters(&mut self, parameters: MonitoringParameters);

    /// Check for resource alerts
    fn check_alerts(&self) -> Vec<ResourceAlert>;

    /// Generate resource report
    fn generate_report(&self) -> ResourceReport;
}

/// Performance optimization for resource usage
#[cfg_attr(feature = "async", async_trait)]
pub trait PerformanceOptimizer: Send + Sync {
    /// Optimize performance
    #[cfg(feature = "async")]
    async fn optimize_performance(&mut self) -> Result<PerformanceResult>;

    /// Optimize performance (sync version)
    #[cfg(not(feature = "async"))]
    fn optimize_performance(&mut self) -> Result<PerformanceResult>;

    /// Analyze performance bottlenecks
    #[cfg(feature = "async")]
    async fn analyze_bottlenecks(&self) -> Result<BottleneckAnalysis>;

    /// Analyze bottlenecks (sync version)
    #[cfg(not(feature = "async"))]
    fn analyze_bottlenecks(&self) -> Result<BottleneckAnalysis>;

    /// Apply performance tuning
    #[cfg(feature = "async")]
    async fn tune_performance(
        &mut self,
        recommendations: &[TuningRecommendation],
    ) -> Result<TuningResult>;

    /// Tune performance (sync version)
    #[cfg(not(feature = "async"))]
    fn tune_performance(
        &mut self,
        recommendations: &[TuningRecommendation],
    ) -> Result<TuningResult>;

    /// Measure performance impact
    fn measure_impact(&self, baseline: &PerformanceBaseline) -> PerformanceImpact;

    /// Get optimization history
    fn optimization_history(&self) -> Vec<OptimizationRecord>;
}

/// Memory management for autonomous agents
#[cfg_attr(feature = "async", async_trait)]
pub trait MemoryManager: Send + Sync {
    /// Allocate memory
    #[cfg(feature = "async")]
    async fn allocate_memory(
        &mut self,
        size: usize,
        priority: MemoryPriority,
    ) -> Result<MemoryAllocation>;

    /// Allocate memory (sync version)
    #[cfg(not(feature = "async"))]
    fn allocate_memory(
        &mut self,
        size: usize,
        priority: MemoryPriority,
    ) -> Result<MemoryAllocation>;

    /// Deallocate memory
    #[cfg(feature = "async")]
    async fn deallocate_memory(&mut self, allocation: &MemoryAllocation) -> Result<()>;

    /// Deallocate memory (sync version)
    #[cfg(not(feature = "async"))]
    fn deallocate_memory(&mut self, allocation: &MemoryAllocation) -> Result<()>;

    /// Garbage collection
    #[cfg(feature = "async")]
    async fn garbage_collect(&mut self) -> Result<GcResult>;

    /// Garbage collection (sync version)
    #[cfg(not(feature = "async"))]
    fn garbage_collect(&mut self) -> Result<GcResult>;

    /// Memory compaction
    #[cfg(feature = "async")]
    async fn compact_memory(&mut self) -> Result<CompactionResult>;

    /// Memory compaction (sync version)
    #[cfg(not(feature = "async"))]
    fn compact_memory(&mut self) -> Result<CompactionResult>;

    /// Get memory statistics
    fn memory_statistics(&self) -> MemoryStatistics;

    /// Check memory pressure
    fn memory_pressure(&self) -> MemoryPressure;

    /// Set memory limits
    fn set_memory_limits(&mut self, limits: MemoryLimits);
}

/// Resource request specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceRequest {
    /// Request identifier
    pub id: String,

    /// Requested resource type
    pub resource_type: ResourceType,

    /// Amount requested
    pub amount: f64,

    /// Priority level
    pub priority: ResourcePriority,

    /// Duration needed
    pub duration: Option<u64>,

    /// Quality of service requirements
    pub qos_requirements: QosRequirements,

    /// Constraints
    pub constraints: Vec<ResourceConstraint>,
}

/// Types of resources
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceType {
    /// CPU processing power
    Cpu,
    /// Memory
    Memory,
    /// Network bandwidth
    Network,
    /// Storage space
    Storage,
    /// GPU processing power
    Gpu,
    /// Energy/power
    Energy,
    /// Custom resource type
    Custom,
}

/// Resource priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ResourcePriority {
    /// Low priority
    Low,
    /// Normal priority
    Normal,
    /// High priority
    High,
    /// Critical priority
    Critical,
    /// Emergency priority
    Emergency,
}

/// Quality of service requirements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QosRequirements {
    /// Maximum latency (milliseconds)
    pub max_latency: Option<u64>,

    /// Minimum throughput
    pub min_throughput: Option<f64>,

    /// Reliability requirement (0.0 to 1.0)
    pub reliability: f64,

    /// Availability requirement (0.0 to 1.0)
    pub availability: f64,

    /// Performance consistency required
    pub consistency: bool,
}

impl Default for QosRequirements {
    fn default() -> Self {
        QosRequirements {
            max_latency: None,
            min_throughput: None,
            reliability: 0.95,
            availability: 0.99,
            consistency: false,
        }
    }
}

/// Resource constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResourceConstraint {
    /// Maximum cost allowed
    MaxCost(f64),
    /// Specific location requirement
    Location(String),
    /// Minimum performance level
    MinPerformance(f64),
    /// Maximum energy consumption
    MaxEnergy(f64),
    /// Compatibility requirement
    Compatibility(String),
    /// Security level requirement
    SecurityLevel(SecurityLevel),
}

/// Security levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecurityLevel {
    /// No security requirements
    None,
    /// Basic security
    Basic,
    /// Standard security
    Standard,
    /// High security
    High,
    /// Maximum security
    Maximum,
}

/// Resource utilization information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUtilization {
    /// CPU utilization (0.0 to 1.0)
    pub cpu_utilization: f64,

    /// Memory utilization (0.0 to 1.0)
    pub memory_utilization: f64,

    /// Network utilization (0.0 to 1.0)
    pub network_utilization: f64,

    /// Storage utilization (0.0 to 1.0)
    pub storage_utilization: f64,

    /// Energy consumption
    pub energy_consumption: f64,

    /// Resource efficiency score (0.0 to 1.0)
    pub efficiency_score: f64,

    /// Timestamp
    pub timestamp: u64,
}

impl ResourceUtilization {
    /// Check if any resource is over-utilized
    pub fn is_over_utilized(&self, threshold: f64) -> bool {
        self.cpu_utilization > threshold
            || self.memory_utilization > threshold
            || self.network_utilization > threshold
            || self.storage_utilization > threshold
    }

    /// Get overall utilization score
    pub fn overall_utilization(&self) -> f64 {
        (self.cpu_utilization
            + self.memory_utilization
            + self.network_utilization
            + self.storage_utilization)
            / 4.0
    }
}

/// Resource prediction information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePrediction {
    /// Predicted CPU needs
    pub cpu_needs: f64,

    /// Predicted memory needs
    pub memory_needs: f64,

    /// Predicted network needs
    pub network_needs: f64,

    /// Predicted storage needs
    pub storage_needs: f64,

    /// Prediction confidence (0.0 to 1.0)
    pub confidence: f64,

    /// Time horizon for prediction
    pub time_horizon: u64,

    /// Prediction methodology used
    pub methodology: PredictionMethodology,
}

/// Prediction methodologies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PredictionMethodology {
    /// Simple linear extrapolation
    Linear,
    /// Moving average
    MovingAverage,
    /// Exponential smoothing
    ExponentialSmoothing,
    /// Machine learning model
    MachineLearning,
    /// Hybrid approach
    Hybrid,
}

/// Resource availability information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAvailability {
    /// Whether resources are available
    pub available: bool,

    /// Available amounts by resource type
    pub available_amounts: Vec<(ResourceType, f64)>,

    /// Estimated wait time if not available
    pub estimated_wait_time: Option<u64>,

    /// Alternative resources available
    pub alternatives: Vec<ResourceAlternative>,

    /// Availability confidence
    pub confidence: f64,
}

/// Alternative resource options
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAlternative {
    /// Resource type
    pub resource_type: ResourceType,

    /// Available amount
    pub amount: f64,

    /// Quality difference from requested (-1.0 to 1.0)
    pub quality_difference: f64,

    /// Cost difference
    pub cost_difference: f64,

    /// Availability time
    pub availability_time: u64,
}

/// Current resource status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceStatus {
    /// Total capacity by resource type
    pub total_capacity: Vec<(ResourceType, f64)>,

    /// Currently allocated amounts
    pub allocated: Vec<(ResourceType, f64)>,

    /// Currently available amounts
    pub available: Vec<(ResourceType, f64)>,

    /// Reserved amounts
    pub reserved: Vec<(ResourceType, f64)>,

    /// Resource health status
    pub health_status: ResourceHealth,

    /// Active allocations count
    pub active_allocations: usize,
}

/// Resource health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ResourceHealth {
    /// All resources healthy
    Healthy,
    /// Some resources degraded
    Degraded,
    /// Resource issues detected
    Unhealthy,
    /// Critical resource problems
    Critical,
    /// Resources unavailable
    Unavailable,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    /// Optimization success
    pub success: bool,

    /// Efficiency improvement achieved
    pub efficiency_improvement: f64,

    /// Cost reduction achieved
    pub cost_reduction: f64,

    /// Optimizations applied
    pub optimizations_applied: Vec<String>,

    /// Optimization time
    pub optimization_time: u64,

    /// Performance impact
    pub performance_impact: f64,
}

/// Resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    /// Current utilization
    pub utilization: ResourceUtilization,

    /// Performance metrics
    pub performance: PerformanceMetrics,

    /// Cost metrics
    pub cost: CostMetrics,

    /// Availability metrics
    pub availability: AvailabilityMetrics,

    /// Quality metrics
    pub quality: QualityMetrics,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average response time
    pub avg_response_time: f64,

    /// Throughput rate
    pub throughput: f64,

    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,

    /// Performance stability
    pub stability: f64,

    /// Performance trend
    pub trend: PerformanceTrend,
}

/// Performance trends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PerformanceTrend {
    /// Performance improving
    Improving,
    /// Performance stable
    Stable,
    /// Performance declining
    Declining,
    /// Performance fluctuating
    Fluctuating,
    /// Insufficient data
    Unknown,
}

/// Cost metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostMetrics {
    /// Total cost incurred
    pub total_cost: f64,

    /// Cost per unit time
    pub cost_per_time: f64,

    /// Cost efficiency
    pub cost_efficiency: f64,

    /// Budget utilization (0.0 to 1.0)
    pub budget_utilization: f64,

    /// Cost trend
    pub cost_trend: CostTrend,
}

/// Cost trends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CostTrend {
    /// Costs decreasing
    Decreasing,
    /// Costs stable
    Stable,
    /// Costs increasing
    Increasing,
    /// Costs fluctuating
    Fluctuating,
    /// Insufficient data
    Unknown,
}

/// Availability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvailabilityMetrics {
    /// Uptime percentage (0.0 to 1.0)
    pub uptime: f64,

    /// Mean time between failures
    pub mtbf: f64,

    /// Mean time to recovery
    pub mttr: f64,

    /// Service level agreement compliance
    pub sla_compliance: f64,

    /// Availability trend
    pub availability_trend: AvailabilityTrend,
}

/// Availability trends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AvailabilityTrend {
    /// Availability improving
    Improving,
    /// Availability stable
    Stable,
    /// Availability declining
    Declining,
    /// Availability fluctuating
    Fluctuating,
    /// Insufficient data
    Unknown,
}

/// Quality metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Service quality score (0.0 to 1.0)
    pub quality_score: f64,

    /// Reliability measure (0.0 to 1.0)
    pub reliability: f64,

    /// Consistency measure (0.0 to 1.0)
    pub consistency: f64,

    /// Customer satisfaction (0.0 to 1.0)
    pub satisfaction: f64,

    /// Quality trend
    pub quality_trend: QualityTrend,
}

/// Quality trends
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityTrend {
    /// Quality improving
    Improving,
    /// Quality stable
    Stable,
    /// Quality declining
    Declining,
    /// Quality fluctuating
    Fluctuating,
    /// Insufficient data
    Unknown,
}

/// Resource snapshot for historical data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    /// Snapshot timestamp
    pub timestamp: u64,

    /// Resource utilization at time
    pub utilization: ResourceUtilization,

    /// Active allocations count
    pub active_allocations: usize,

    /// System load at time
    pub system_load: f64,
}

/// Monitoring parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringParameters {
    /// Monitoring frequency (milliseconds)
    pub frequency: u64,

    /// Data retention period (seconds)
    pub retention_period: u64,

    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,

    /// Metrics to collect
    pub metrics_to_collect: Vec<MetricType>,

    /// Monitoring enabled
    pub enabled: bool,
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// CPU utilization threshold
    pub cpu_threshold: f64,

    /// Memory utilization threshold
    pub memory_threshold: f64,

    /// Network utilization threshold
    pub network_threshold: f64,

    /// Response time threshold
    pub response_time_threshold: f64,

    /// Error rate threshold
    pub error_rate_threshold: f64,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        AlertThresholds {
            cpu_threshold: 0.8,
            memory_threshold: 0.85,
            network_threshold: 0.9,
            response_time_threshold: 1000.0, // 1 second
            error_rate_threshold: 0.05,      // 5%
        }
    }
}

/// Types of metrics to collect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    /// CPU metrics
    Cpu,
    /// Memory metrics
    Memory,
    /// Network metrics
    Network,
    /// Storage metrics
    Storage,
    /// Performance metrics
    Performance,
    /// Cost metrics
    Cost,
    /// Availability metrics
    Availability,
    /// Quality metrics
    Quality,
}

/// Resource alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAlert {
    /// Alert identifier
    pub id: String,

    /// Alert type
    pub alert_type: AlertType,

    /// Alert severity
    pub severity: AlertSeverity,

    /// Alert message
    pub message: String,

    /// Resource affected
    pub resource_type: ResourceType,

    /// Current value
    pub current_value: f64,

    /// Threshold exceeded
    pub threshold: f64,

    /// Alert timestamp
    pub timestamp: u64,
}

/// Types of alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertType {
    /// High utilization alert
    HighUtilization,
    /// Performance degradation
    PerformanceDegradation,
    /// Resource exhaustion
    ResourceExhaustion,
    /// Availability issue
    AvailabilityIssue,
    /// Cost overrun
    CostOverrun,
    /// Quality degradation
    QualityDegradation,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning level
    Warning,
    /// Error level
    Error,
    /// Critical level
    Critical,
    /// Emergency level
    Emergency,
}

/// Resource report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceReport {
    /// Report identifier
    pub id: String,

    /// Report period
    pub period_start: u64,
    pub period_end: u64,

    /// Summary metrics
    pub summary: ResourceMetrics,

    /// Trends analysis
    pub trends: TrendsAnalysis,

    /// Recommendations
    pub recommendations: Vec<String>,

    /// Alerts generated
    pub alerts_count: usize,

    /// Report generation time
    pub generated_at: u64,
}

/// Trends analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendsAnalysis {
    /// Utilization trend
    pub utilization_trend: PerformanceTrend,

    /// Cost trend
    pub cost_trend: CostTrend,

    /// Performance trend
    pub performance_trend: PerformanceTrend,

    /// Quality trend
    pub quality_trend: QualityTrend,

    /// Predicted issues
    pub predicted_issues: Vec<String>,
}

// Performance optimization types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceResult {
    /// Optimization success
    pub success: bool,

    /// Performance improvement
    pub improvement: f64,

    /// Optimizations applied
    pub optimizations: Vec<String>,

    /// Time taken
    pub optimization_time: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BottleneckAnalysis {
    /// Identified bottlenecks
    pub bottlenecks: Vec<Bottleneck>,

    /// Analysis confidence
    pub confidence: f64,

    /// Analysis time
    pub analysis_time: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Bottleneck type
    pub bottleneck_type: BottleneckType,

    /// Severity (0.0 to 1.0)
    pub severity: f64,

    /// Description
    pub description: String,

    /// Suggested remedies
    pub remedies: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    /// CPU bottleneck
    Cpu,
    /// Memory bottleneck
    Memory,
    /// I/O bottleneck
    Io,
    /// Network bottleneck
    Network,
    /// Algorithm bottleneck
    Algorithm,
    /// Concurrency bottleneck
    Concurrency,
}

// Memory management types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPriority {
    /// Low priority memory
    Low,
    /// Normal priority memory
    Normal,
    /// High priority memory
    High,
    /// Critical priority memory
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryAllocation {
    /// Allocation identifier
    pub id: String,

    /// Allocated size
    pub size: usize,

    /// Memory address (if applicable)
    pub address: Option<usize>,

    /// Priority level
    pub priority: MemoryPriority,

    /// Allocation timestamp
    pub allocated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryStatistics {
    /// Total memory available
    pub total_memory: usize,

    /// Currently allocated
    pub allocated_memory: usize,

    /// Free memory
    pub free_memory: usize,

    /// Memory fragmentation (0.0 to 1.0)
    pub fragmentation: f64,

    /// Number of allocations
    pub allocation_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryPressure {
    /// No memory pressure
    None,
    /// Low memory pressure
    Low,
    /// Medium memory pressure
    Medium,
    /// High memory pressure
    High,
    /// Critical memory pressure
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryLimits {
    /// Maximum memory usage
    pub max_memory: usize,

    /// Memory warning threshold
    pub warning_threshold: usize,

    /// Emergency threshold
    pub emergency_threshold: usize,

    /// Allocation size limit
    pub max_allocation_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GcResult {
    /// Memory freed
    pub memory_freed: usize,

    /// Objects collected
    pub objects_collected: usize,

    /// GC time
    pub gc_time: u64,

    /// GC efficiency
    pub efficiency: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionResult {
    /// Memory compacted
    pub memory_compacted: usize,

    /// Fragmentation reduced
    pub fragmentation_reduced: f64,

    /// Compaction time
    pub compaction_time: u64,
}

// Additional helper types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningRecommendation {
    /// Parameter to tune
    pub parameter: String,

    /// Recommended value
    pub recommended_value: f64,

    /// Expected improvement
    pub expected_improvement: f64,

    /// Confidence in recommendation
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TuningResult {
    /// Tuning success
    pub success: bool,

    /// Parameters tuned
    pub parameters_tuned: Vec<String>,

    /// Performance improvement
    pub improvement: f64,

    /// Tuning time
    pub tuning_time: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Baseline metrics
    pub metrics: PerformanceMetrics,

    /// Baseline timestamp
    pub timestamp: u64,

    /// Test conditions
    pub conditions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceImpact {
    /// Performance change
    pub performance_change: f64,

    /// Response time change
    pub response_time_change: f64,

    /// Throughput change
    pub throughput_change: f64,

    /// Overall impact score
    pub impact_score: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationRecord {
    /// Optimization timestamp
    pub timestamp: u64,

    /// Optimization type
    pub optimization_type: String,

    /// Parameters changed
    pub parameters_changed: Vec<String>,

    /// Improvement achieved
    pub improvement: f64,

    /// Duration
    pub duration: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resource_utilization() {
        let utilization = ResourceUtilization {
            cpu_utilization: 0.8,
            memory_utilization: 0.7,
            network_utilization: 0.6,
            storage_utilization: 0.5,
            energy_consumption: 100.0,
            efficiency_score: 0.8,
            timestamp: 1000,
        };

        assert!(utilization.is_over_utilized(0.75));
        assert!(!utilization.is_over_utilized(0.85));
        assert_eq!(utilization.overall_utilization(), 0.65);
    }

    #[test]
    fn test_qos_requirements_default() {
        let qos = QosRequirements::default();
        assert_eq!(qos.reliability, 0.95);
        assert_eq!(qos.availability, 0.99);
        assert!(!qos.consistency);
    }

    #[test]
    fn test_memory_statistics() {
        let stats = MemoryStatistics {
            total_memory: 1000,
            allocated_memory: 600,
            free_memory: 400,
            fragmentation: 0.2,
            allocation_count: 50,
        };

        assert_eq!(
            stats.allocated_memory + stats.free_memory,
            stats.total_memory
        );
    }
}
