//! SWE-Bench instance loader with difficulty categorization

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::path::{Path, PathBuf};
use tokio::fs;
use tracing::{debug, info, warn};

/// Difficulty levels for SWE-Bench instances
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DifficultyLevel {
    Easy,
    Medium,
    Hard,
    Expert,
}

impl DifficultyLevel {
    /// Get priority level for task scheduling
    pub fn priority(&self) -> u8 {
        match self {
            Self::Easy => 1,
            Self::Medium => 2,
            Self::Hard => 3,
            Self::Expert => 4,
        }
    }

    /// Categorize difficulty based on instance metrics
    pub fn from_metrics(metrics: &InstanceMetrics) -> Self {
        let score = metrics.calculate_difficulty_score();

        match score {
            0..=25 => Self::Easy,
            26..=50 => Self::Medium,
            51..=75 => Self::Hard,
            _ => Self::Expert,
        }
    }
}

impl fmt::Display for DifficultyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Easy => write!(f, "Easy"),
            Self::Medium => write!(f, "Medium"),
            Self::Hard => write!(f, "Hard"),
            Self::Expert => write!(f, "Expert"),
        }
    }
}

/// SWE-Bench instance representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SWEBenchInstance {
    pub instance_id: String,
    pub repo: String,
    pub version: String,
    pub issue_title: String,
    pub issue_description: String,
    pub hints: Vec<String>,
    pub test_patch: String,
    pub test_directives: Vec<String>,
    pub difficulty: DifficultyLevel,
    pub metrics: InstanceMetrics,
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Metrics for difficulty categorization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstanceMetrics {
    pub files_changed: usize,
    pub lines_changed: usize,
    pub test_count: usize,
    pub dependencies: usize,
    pub complexity_score: f64,
    pub domain_specific: bool,
    pub requires_external_api: bool,
}

impl InstanceMetrics {
    /// Calculate difficulty score (0-100)
    pub fn calculate_difficulty_score(&self) -> u32 {
        let mut score = 0u32;

        // File changes contribution (0-25)
        score += match self.files_changed {
            1 => 5,
            2..=3 => 10,
            4..=5 => 15,
            6..=10 => 20,
            _ => 25,
        };

        // Lines changed contribution (0-25)
        score += match self.lines_changed {
            1..=20 => 5,
            21..=50 => 10,
            51..=100 => 15,
            101..=200 => 20,
            _ => 25,
        };

        // Test complexity contribution (0-20)
        score += match self.test_count {
            1..=3 => 5,
            4..=6 => 10,
            7..=10 => 15,
            _ => 20,
        };

        // Complexity score contribution (0-15)
        score += (self.complexity_score * 15.0) as u32;

        // Special factors (0-15)
        if self.domain_specific {
            score += 5;
        }
        if self.requires_external_api {
            score += 5;
        }
        if self.dependencies > 5 {
            score += 5;
        }

        score.min(100)
    }
}

/// Instance loader for SWE-Bench dataset
pub struct InstanceLoader {
    instances_dir: PathBuf,
    cache: HashMap<String, SWEBenchInstance>,
}

impl InstanceLoader {
    /// Create a new instance loader
    pub fn new(instances_dir: impl AsRef<Path>) -> Result<Self> {
        let instances_dir = instances_dir.as_ref().to_path_buf();

        if !instances_dir.exists() {
            std::fs::create_dir_all(&instances_dir)?;
        }

        Ok(Self {
            instances_dir,
            cache: HashMap::new(),
        })
    }

    /// Load a specific instance by ID
    pub async fn load_instance(&mut self, instance_id: &str) -> Result<SWEBenchInstance> {
        // Check cache first
        if let Some(instance) = self.cache.get(instance_id) {
            debug!("Loaded instance {} from cache", instance_id);
            return Ok(instance.clone());
        }

        // Load from disk
        let instance_path = self.instances_dir.join(format!("{}.json", instance_id));

        if !instance_path.exists() {
            // Try to download from remote repository
            self.download_instance(instance_id)
                .await
                .context("Failed to download instance")?;
        }

        let content = fs::read_to_string(&instance_path)
            .await
            .context("Failed to read instance file")?;

        let mut instance: SWEBenchInstance =
            serde_json::from_str(&content).context("Failed to parse instance JSON")?;

        // Calculate difficulty if not set
        if instance.metrics.calculate_difficulty_score() == 0 {
            instance.metrics = self.analyze_instance_metrics(&instance).await?;
            instance.difficulty = DifficultyLevel::from_metrics(&instance.metrics);
        }

        // Cache the instance
        self.cache.insert(instance_id.to_string(), instance.clone());

        info!(
            "Loaded instance {} with difficulty: {}",
            instance_id, instance.difficulty
        );
        Ok(instance)
    }

    /// Load all instances matching criteria
    pub async fn load_instances(
        &mut self,
        filter: Option<InstanceFilter>,
    ) -> Result<Vec<SWEBenchInstance>> {
        let mut instances = Vec::new();

        let mut entries = fs::read_dir(&self.instances_dir).await?;

        while let Some(entry) = entries.next_entry().await? {
            let path = entry.path();

            if path.extension().is_some_and(|ext| ext == "json") {
                if let Some(stem) = path.file_stem() {
                    let instance_id = stem.to_string_lossy().to_string();

                    match self.load_instance(&instance_id).await {
                        Ok(instance) => {
                            if filter.as_ref().is_none_or(|f| f.matches(&instance)) {
                                instances.push(instance);
                            }
                        }
                        Err(e) => {
                            warn!("Failed to load instance {}: {}", instance_id, e);
                        }
                    }
                }
            }
        }

        Ok(instances)
    }

    /// Download instance from remote repository
    async fn download_instance(&self, instance_id: &str) -> Result<()> {
        // This would connect to the actual SWE-Bench repository
        // For now, create a mock instance
        info!("Downloading instance: {}", instance_id);

        let mock_instance = SWEBenchInstance {
            instance_id: instance_id.to_string(),
            repo: "mock/repo".to_string(),
            version: "1.0.0".to_string(),
            issue_title: format!("Mock issue for {}", instance_id),
            issue_description: "This is a mock instance for testing".to_string(),
            hints: vec!["Check the main function".to_string()],
            test_patch: "diff --git a/test.py b/test.py\n+test content".to_string(),
            test_directives: vec!["pytest test.py".to_string()],
            difficulty: DifficultyLevel::Medium,
            metrics: InstanceMetrics {
                files_changed: 2,
                lines_changed: 50,
                test_count: 5,
                dependencies: 3,
                complexity_score: 0.5,
                domain_specific: false,
                requires_external_api: false,
            },
            metadata: HashMap::new(),
        };

        let instance_path = self.instances_dir.join(format!("{}.json", instance_id));
        let content = serde_json::to_string_pretty(&mock_instance)?;
        fs::write(instance_path, content).await?;

        Ok(())
    }

    /// Analyze instance to determine metrics
    async fn analyze_instance_metrics(
        &self,
        instance: &SWEBenchInstance,
    ) -> Result<InstanceMetrics> {
        // Parse test patch to count changes
        let lines: Vec<&str> = instance.test_patch.lines().collect();
        let files_changed = lines
            .iter()
            .filter(|line| line.starts_with("diff --git"))
            .count();

        let lines_changed = lines
            .iter()
            .filter(|line| line.starts_with('+') || line.starts_with('-'))
            .filter(|line| !line.starts_with("+++") && !line.starts_with("---"))
            .count();

        // Analyze complexity based on issue description
        let complexity_score = self.calculate_complexity_score(&instance.issue_description);

        // Check for domain-specific keywords
        let domain_specific = self.is_domain_specific(&instance.issue_description);

        // Check for external API requirements
        let requires_external_api = instance.issue_description.contains("API")
            || instance.issue_description.contains("external service");

        Ok(InstanceMetrics {
            files_changed,
            lines_changed,
            test_count: instance.test_directives.len(),
            dependencies: instance
                .metadata
                .get("dependencies")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize,
            complexity_score,
            domain_specific,
            requires_external_api,
        })
    }

    fn calculate_complexity_score(&self, description: &str) -> f64 {
        let complexity_keywords = [
            "complex",
            "intricate",
            "advanced",
            "sophisticated",
            "multi-threaded",
            "concurrent",
            "parallel",
            "distributed",
            "optimization",
            "performance",
            "algorithm",
            "architecture",
        ];

        let word_count = description.split_whitespace().count() as f64;
        let complexity_count = complexity_keywords
            .iter()
            .filter(|kw| description.to_lowercase().contains(*kw))
            .count() as f64;

        (complexity_count / word_count.max(1.0)).min(1.0)
    }

    fn is_domain_specific(&self, description: &str) -> bool {
        let domain_keywords = [
            "machine learning",
            "neural network",
            "deep learning",
            "cryptography",
            "blockchain",
            "quantum",
            "bioinformatics",
            "genomics",
            "medical",
            "finance",
            "trading",
            "banking",
        ];

        domain_keywords
            .iter()
            .any(|kw| description.to_lowercase().contains(kw))
    }

    /// Get statistics about loaded instances
    pub fn get_statistics(&self) -> LoaderStatistics {
        let total = self.cache.len();
        let by_difficulty = self
            .cache
            .values()
            .fold(HashMap::new(), |mut acc, instance| {
                *acc.entry(instance.difficulty).or_insert(0) += 1;
                acc
            });

        LoaderStatistics {
            total_loaded: total,
            by_difficulty,
            cache_size: self.cache.len(),
        }
    }
}

/// Filter for loading instances
#[derive(Debug, Clone)]
pub struct InstanceFilter {
    pub difficulty: Option<DifficultyLevel>,
    pub repo: Option<String>,
    pub min_files: Option<usize>,
    pub max_files: Option<usize>,
    pub keywords: Vec<String>,
}

impl InstanceFilter {
    /// Check if instance matches filter criteria
    pub fn matches(&self, instance: &SWEBenchInstance) -> bool {
        if let Some(diff) = self.difficulty {
            if instance.difficulty != diff {
                return false;
            }
        }

        if let Some(repo) = &self.repo {
            if !instance.repo.contains(repo) {
                return false;
            }
        }

        if let Some(min) = self.min_files {
            if instance.metrics.files_changed < min {
                return false;
            }
        }

        if let Some(max) = self.max_files {
            if instance.metrics.files_changed > max {
                return false;
            }
        }

        if !self.keywords.is_empty() {
            let text = format!("{} {}", instance.issue_title, instance.issue_description);
            if !self.keywords.iter().any(|kw| text.contains(kw)) {
                return false;
            }
        }

        true
    }
}

/// Loader statistics
#[derive(Debug, Clone)]
pub struct LoaderStatistics {
    pub total_loaded: usize,
    pub by_difficulty: HashMap<DifficultyLevel, usize>,
    pub cache_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_difficulty_level_priority() {
        assert_eq!(DifficultyLevel::Easy.priority(), 1);
        assert_eq!(DifficultyLevel::Medium.priority(), 2);
        assert_eq!(DifficultyLevel::Hard.priority(), 3);
        assert_eq!(DifficultyLevel::Expert.priority(), 4);
    }

    #[test]
    fn test_difficulty_score_calculation() {
        let metrics = InstanceMetrics {
            files_changed: 3,
            lines_changed: 75,
            test_count: 5,
            dependencies: 2,
            complexity_score: 0.6,
            domain_specific: true,
            requires_external_api: false,
        };

        let score = metrics.calculate_difficulty_score();
        assert!(score > 30 && score < 70);
        assert_eq!(
            DifficultyLevel::from_metrics(&metrics),
            DifficultyLevel::Medium
        );
    }

    #[test]
    fn test_instance_filter() {
        let instance = SWEBenchInstance {
            instance_id: "test-001".to_string(),
            repo: "python/cpython".to_string(),
            version: "3.9".to_string(),
            issue_title: "Fix memory leak".to_string(),
            issue_description: "Memory leak in dict implementation".to_string(),
            hints: vec![],
            test_patch: String::new(),
            test_directives: vec![],
            difficulty: DifficultyLevel::Hard,
            metrics: InstanceMetrics {
                files_changed: 3,
                lines_changed: 100,
                test_count: 5,
                dependencies: 0,
                complexity_score: 0.7,
                domain_specific: false,
                requires_external_api: false,
            },
            metadata: HashMap::new(),
        };

        let filter = InstanceFilter {
            difficulty: Some(DifficultyLevel::Hard),
            repo: Some("python".to_string()),
            min_files: Some(2),
            max_files: Some(5),
            keywords: vec!["memory".to_string()],
        };

        assert!(filter.matches(&instance));
    }
}
