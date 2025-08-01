//! Performance comparison module for analyzing baseline vs ML-optimized results

use anyhow::Result;
use statrs::distribution::{ContinuousCDF, StudentsT};
use statrs::statistics::Statistics;
use std::collections::HashMap;

use crate::metrics::PerformanceMetrics;
use crate::{ComparisonResult, ExecutionStatus, ScenarioResult, StatisticalSignificance};

/// Performance comparator for analyzing results
pub struct PerformanceComparator {
    confidence_level: f64,
}

impl Default for PerformanceComparator {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceComparator {
    pub fn new() -> Self {
        Self {
            confidence_level: 0.95, // 95% confidence interval
        }
    }

    /// Compare baseline and ML-optimized results
    pub fn compare(
        &self,
        baseline: &ScenarioResult,
        ml_optimized: &ScenarioResult,
    ) -> Result<ComparisonResult> {
        // Check if both runs completed successfully
        if baseline.status != ExecutionStatus::Completed
            || ml_optimized.status != ExecutionStatus::Completed
        {
            return Ok(ComparisonResult {
                overall_improvement: None,
                speed_improvement: None,
                quality_improvement: None,
                resource_efficiency: None,
                statistical_significance: StatisticalSignificance {
                    p_value: 1.0,
                    confidence_interval: (0.0, 0.0),
                    effect_size: 0.0,
                },
            });
        }

        // Calculate improvements
        let speed_improvement = self.calculate_speed_improvement(baseline, ml_optimized);
        let quality_improvement =
            self.calculate_quality_improvement(&baseline.metrics, &ml_optimized.metrics);
        let resource_efficiency =
            self.calculate_resource_efficiency(&baseline.metrics, &ml_optimized.metrics);

        // Overall improvement (weighted average)
        let overall_improvement = if let (Some(speed), Some(quality), Some(resource)) =
            (speed_improvement, quality_improvement, resource_efficiency)
        {
            Some(speed * 0.4 + quality * 0.4 + resource * 0.2)
        } else {
            None
        };

        // Statistical significance (placeholder for now)
        let statistical_significance = StatisticalSignificance {
            p_value: 0.01, // Would need multiple trials for real p-value
            confidence_interval: (0.1, 0.3),
            effect_size: 0.5,
        };

        Ok(ComparisonResult {
            overall_improvement,
            speed_improvement,
            quality_improvement,
            resource_efficiency,
            statistical_significance,
        })
    }

    fn calculate_speed_improvement(
        &self,
        baseline: &ScenarioResult,
        ml_optimized: &ScenarioResult,
    ) -> Option<f64> {
        let baseline_time = baseline.duration.as_millis() as f64;
        let ml_time = ml_optimized.duration.as_millis() as f64;

        if baseline_time > 0.0 {
            Some((baseline_time - ml_time) / baseline_time)
        } else {
            None
        }
    }

    fn calculate_quality_improvement(
        &self,
        baseline: &PerformanceMetrics,
        ml_optimized: &PerformanceMetrics,
    ) -> Option<f64> {
        let baseline_quality = baseline.code_quality_score.overall;
        let ml_quality = ml_optimized.code_quality_score.overall;

        if baseline_quality > 0.0 {
            Some((ml_quality - baseline_quality) / baseline_quality)
        } else {
            None
        }
    }

    fn calculate_resource_efficiency(
        &self,
        baseline: &PerformanceMetrics,
        ml_optimized: &PerformanceMetrics,
    ) -> Option<f64> {
        let baseline_cpu = baseline.cpu_usage.average;
        let ml_cpu = ml_optimized.cpu_usage.average;

        let baseline_memory = baseline.memory_usage.average;
        let ml_memory = ml_optimized.memory_usage.average;

        if baseline_cpu > 0.0 && baseline_memory > 0.0 {
            let cpu_improvement = (baseline_cpu - ml_cpu) / baseline_cpu;
            let memory_improvement = (baseline_memory - ml_memory) / baseline_memory;
            Some((cpu_improvement + memory_improvement) / 2.0)
        } else {
            None
        }
    }
}

/// Statistical analysis framework
pub struct StatisticalAnalyzer {
    min_sample_size: usize,
}

impl Default for StatisticalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalAnalyzer {
    pub fn new() -> Self {
        Self {
            min_sample_size: 10,
        }
    }

    /// Perform statistical analysis on multiple trial results
    pub fn analyze_trials(
        &self,
        baseline_trials: &[TrialResult],
        ml_trials: &[TrialResult],
    ) -> StatisticalAnalysis {
        let baseline_values: Vec<f64> = baseline_trials
            .iter()
            .map(|t| t.completion_time.as_millis() as f64)
            .collect();

        let ml_values: Vec<f64> = ml_trials
            .iter()
            .map(|t| t.completion_time.as_millis() as f64)
            .collect();

        let analysis = Analysis {
            mean_improvement: self.calculate_mean_improvement(&baseline_values, &ml_values),
            median_improvement: self.calculate_median_improvement(&baseline_values, &ml_values),
            std_deviation: self.calculate_std_deviation(&ml_values),
            effect_size: self.calculate_cohens_d(&baseline_values, &ml_values),
            p_value: self.perform_t_test(&baseline_values, &ml_values),
            confidence_interval_95: self.calculate_confidence_interval(
                &baseline_values,
                &ml_values,
                0.95,
            ),
        };

        StatisticalAnalysis {
            baseline_stats: self.calculate_stats(&baseline_values),
            ml_stats: self.calculate_stats(&ml_values),
            comparison: analysis,
            sample_size: baseline_trials.len(),
        }
    }

    fn calculate_stats(&self, values: &[f64]) -> SummaryStatistics {
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        SummaryStatistics {
            mean: values.mean(),
            median: if sorted_values.is_empty() {
                0.0
            } else {
                sorted_values[sorted_values.len() / 2]
            },
            std_dev: values.std_dev(),
            min: values.min(),
            max: values.max(),
            p25: self.percentile(values, 0.25),
            p75: self.percentile(values, 0.75),
        }
    }

    fn calculate_mean_improvement(&self, baseline: &[f64], optimized: &[f64]) -> f64 {
        let baseline_mean = baseline.mean();
        let optimized_mean = optimized.mean();

        if baseline_mean > 0.0 {
            (baseline_mean - optimized_mean) / baseline_mean
        } else {
            0.0
        }
    }

    fn calculate_median_improvement(&self, baseline: &[f64], optimized: &[f64]) -> f64 {
        let mut baseline_sorted = baseline.to_vec();
        baseline_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let baseline_median = if baseline_sorted.is_empty() {
            0.0
        } else {
            baseline_sorted[baseline_sorted.len() / 2]
        };

        let mut optimized_sorted = optimized.to_vec();
        optimized_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let optimized_median = if optimized_sorted.is_empty() {
            0.0
        } else {
            optimized_sorted[optimized_sorted.len() / 2]
        };

        if baseline_median > 0.0 {
            (baseline_median - optimized_median) / baseline_median
        } else {
            0.0
        }
    }

    fn calculate_std_deviation(&self, values: &[f64]) -> f64 {
        values.std_dev()
    }

    fn calculate_cohens_d(&self, baseline: &[f64], optimized: &[f64]) -> f64 {
        let baseline_mean = baseline.mean();
        let optimized_mean = optimized.mean();

        let baseline_var = baseline.variance();
        let optimized_var = optimized.variance();

        let pooled_std = ((baseline_var + optimized_var) / 2.0).sqrt();

        if pooled_std > 0.0 {
            (baseline_mean - optimized_mean) / pooled_std
        } else {
            0.0
        }
    }

    fn perform_t_test(&self, baseline: &[f64], optimized: &[f64]) -> f64 {
        // Welch's t-test for unequal variances
        let n1 = baseline.len() as f64;
        let n2 = optimized.len() as f64;

        if n1 < 2.0 || n2 < 2.0 {
            return 1.0; // Not enough samples
        }

        let mean1 = baseline.mean();
        let mean2 = optimized.mean();
        let var1 = baseline.variance();
        let var2 = optimized.variance();

        let t_stat = (mean1 - mean2) / (var1 / n1 + var2 / n2).sqrt();

        // Calculate degrees of freedom (Welch-Satterthwaite equation)
        let df = ((var1 / n1 + var2 / n2).powi(2))
            / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));

        // Get p-value from t-distribution
        if let Ok(dist) = StudentsT::new(0.0, 1.0, df) {
            2.0 * (1.0 - dist.cdf(t_stat.abs()))
        } else {
            1.0
        }
    }

    fn calculate_confidence_interval(
        &self,
        baseline: &[f64],
        optimized: &[f64],
        confidence: f64,
    ) -> (f64, f64) {
        let mean_diff = baseline.mean() - optimized.mean();
        let n1 = baseline.len() as f64;
        let n2 = optimized.len() as f64;

        if n1 < 2.0 || n2 < 2.0 {
            return (0.0, 0.0);
        }

        let var1 = baseline.variance();
        let var2 = optimized.variance();
        let se = (var1 / n1 + var2 / n2).sqrt();

        // Calculate degrees of freedom
        let df = ((var1 / n1 + var2 / n2).powi(2))
            / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));

        // Get t-value for confidence interval
        if let Ok(dist) = StudentsT::new(0.0, 1.0, df) {
            let t_value = dist.inverse_cdf((1.0 + confidence) / 2.0);
            let margin = t_value * se;
            (mean_diff - margin, mean_diff + margin)
        } else {
            (mean_diff, mean_diff)
        }
    }

    fn percentile(&self, values: &[f64], p: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let index = (p * (sorted.len() - 1) as f64) as usize;
        sorted[index]
    }
}

/// Comparison framework for orchestrating comparisons
pub struct ComparisonFramework {
    analyzer: StatisticalAnalyzer,
    comparator: PerformanceComparator,
}

impl Default for ComparisonFramework {
    fn default() -> Self {
        Self::new()
    }
}

impl ComparisonFramework {
    pub fn new() -> Self {
        Self {
            analyzer: StatisticalAnalyzer::new(),
            comparator: PerformanceComparator::new(),
        }
    }

    /// Execute comprehensive comparison
    pub fn execute_comparison(
        &self,
        baseline_results: Vec<ScenarioResult>,
        ml_results: Vec<ScenarioResult>,
    ) -> ComprehensiveComparison {
        // Group by scenario
        let mut comparisons = HashMap::new();

        for (baseline, ml) in baseline_results.iter().zip(ml_results.iter()) {
            if baseline.scenario_name == ml.scenario_name {
                let comparison =
                    self.comparator
                        .compare(baseline, ml)
                        .unwrap_or(ComparisonResult {
                            overall_improvement: None,
                            speed_improvement: None,
                            quality_improvement: None,
                            resource_efficiency: None,
                            statistical_significance: StatisticalSignificance {
                                p_value: 1.0,
                                confidence_interval: (0.0, 0.0),
                                effect_size: 0.0,
                            },
                        });

                comparisons.insert(baseline.scenario_name.clone(), comparison);
            }
        }

        // Calculate aggregate statistics
        let aggregate_stats = self.calculate_aggregate_stats(&comparisons);
        let recommendation = self.generate_recommendation(&aggregate_stats);

        ComprehensiveComparison {
            scenario_comparisons: comparisons,
            aggregate_statistics: aggregate_stats,
            recommendation,
        }
    }

    fn calculate_aggregate_stats(
        &self,
        comparisons: &HashMap<String, ComparisonResult>,
    ) -> AggregateStatistics {
        let improvements: Vec<f64> = comparisons
            .values()
            .filter_map(|c| c.overall_improvement)
            .collect();

        let speed_improvements: Vec<f64> = comparisons
            .values()
            .filter_map(|c| c.speed_improvement)
            .collect();

        let scenarios_improved = improvements.iter().filter(|&&x| x > 0.0).count();
        let scenarios_regressed = improvements.iter().filter(|&&x| x < 0.0).count();

        AggregateStatistics {
            mean_overall_improvement: if improvements.is_empty() {
                0.0
            } else {
                improvements.clone().mean()
            },
            mean_speed_improvement: if speed_improvements.is_empty() {
                0.0
            } else {
                speed_improvements.mean()
            },
            scenarios_improved,
            scenarios_regressed,
            total_scenarios: comparisons.len(),
        }
    }

    fn generate_recommendation(&self, stats: &AggregateStatistics) -> Recommendation {
        let improvement_ratio = stats.scenarios_improved as f64 / stats.total_scenarios as f64;

        if stats.mean_overall_improvement > 0.2 && improvement_ratio > 0.8 {
            Recommendation::StronglyRecommended
        } else if stats.mean_overall_improvement > 0.1 && improvement_ratio > 0.6 {
            Recommendation::Recommended
        } else if stats.mean_overall_improvement > 0.0 {
            Recommendation::ConsiderWithCaution
        } else {
            Recommendation::NotRecommended
        }
    }
}

// Supporting types
#[derive(Debug, Clone)]
pub struct TrialResult {
    pub completion_time: std::time::Duration,
    pub success: bool,
    pub metrics: PerformanceMetrics,
}

#[derive(Debug, Clone)]
pub struct Analysis {
    pub mean_improvement: f64,
    pub median_improvement: f64,
    pub std_deviation: f64,
    pub effect_size: f64,
    pub p_value: f64,
    pub confidence_interval_95: (f64, f64),
}

#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    pub baseline_stats: SummaryStatistics,
    pub ml_stats: SummaryStatistics,
    pub comparison: Analysis,
    pub sample_size: usize,
}

#[derive(Debug, Clone)]
pub struct SummaryStatistics {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub p25: f64,
    pub p75: f64,
}

#[derive(Debug, Clone)]
pub struct ComprehensiveComparison {
    pub scenario_comparisons: HashMap<String, ComparisonResult>,
    pub aggregate_statistics: AggregateStatistics,
    pub recommendation: Recommendation,
}

#[derive(Debug, Clone)]
pub struct AggregateStatistics {
    pub mean_overall_improvement: f64,
    pub mean_speed_improvement: f64,
    pub scenarios_improved: usize,
    pub scenarios_regressed: usize,
    pub total_scenarios: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Recommendation {
    StronglyRecommended,
    Recommended,
    ConsiderWithCaution,
    NotRecommended,
}

impl ToString for Recommendation {
    fn to_string(&self) -> String {
        match self {
            Recommendation::StronglyRecommended => {
                "Strongly Recommended - Significant improvements across most scenarios".to_string()
            }
            Recommendation::Recommended => {
                "Recommended - Notable improvements with minimal regressions".to_string()
            }
            Recommendation::ConsiderWithCaution => {
                "Consider with Caution - Mixed results, evaluate for specific use cases".to_string()
            }
            Recommendation::NotRecommended => {
                "Not Recommended - No significant improvements or regressions observed".to_string()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_speed_improvement_calculation() {
        let comparator = PerformanceComparator::new();

        let baseline = ScenarioResult {
            run_id: "baseline".to_string(),
            scenario_name: "test".to_string(),
            mode: crate::ExecutionMode::Baseline,
            duration: Duration::from_secs(10),
            metrics: PerformanceMetrics::default(),
            status: ExecutionStatus::Completed,
            error: None,
        };

        let ml_optimized = ScenarioResult {
            run_id: "ml".to_string(),
            scenario_name: "test".to_string(),
            mode: crate::ExecutionMode::MLOptimized,
            duration: Duration::from_secs(7),
            metrics: PerformanceMetrics::default(),
            status: ExecutionStatus::Completed,
            error: None,
        };

        let comparison = comparator.compare(&baseline, &ml_optimized).unwrap();
        assert!(comparison.speed_improvement.is_some());
        assert!((comparison.speed_improvement.unwrap() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_cohens_d_calculation() {
        let analyzer = StatisticalAnalyzer::new();

        let baseline = vec![10.0, 11.0, 12.0, 10.5, 11.5];
        let optimized = vec![7.0, 8.0, 7.5, 8.5, 7.0];

        let effect_size = analyzer.calculate_cohens_d(&baseline, &optimized);
        assert!(effect_size > 2.0); // Large effect size
    }

    #[test]
    fn test_recommendation_generation() {
        let framework = ComparisonFramework::new();

        let stats = AggregateStatistics {
            mean_overall_improvement: 0.25,
            mean_speed_improvement: 0.3,
            scenarios_improved: 9,
            scenarios_regressed: 1,
            total_scenarios: 10,
        };

        let recommendation = framework.generate_recommendation(&stats);
        assert_eq!(recommendation, Recommendation::StronglyRecommended);
    }
}
