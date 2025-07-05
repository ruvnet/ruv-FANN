//! Directory watching for automatic file discovery and processing

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::channel::mpsc;
use futures::sink::SinkExt;
use futures::stream::StreamExt;
use notify::{Event, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use tokio::sync::RwLock;
use tokio::time::{interval, sleep};
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::config::IngestionConfig;
use crate::error::{IngestionError, IngestionResult};
use crate::ingestion::IngestionEngine;

/// Directory watcher for automatic file processing
pub struct DirectoryWatcher {
    config: Arc<IngestionConfig>,
    engine: Arc<IngestionEngine>,
    watched_directories: Arc<RwLock<HashMap<String, WatchedDirectory>>>,
    active_jobs: Arc<RwLock<HashMap<String, WatchJob>>>,
}

impl DirectoryWatcher {
    pub fn new(config: Arc<IngestionConfig>, engine: Arc<IngestionEngine>) -> Self {
        Self {
            config,
            engine,
            watched_directories: Arc::new(RwLock::new(HashMap::new())),
            active_jobs: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Start watching a directory for new files
    pub async fn start_watching(
        &self,
        input_dir: PathBuf,
        output_dir: PathBuf,
        watch_config: WatchConfig,
    ) -> IngestionResult<String> {
        let watch_id = Uuid::new_v4().to_string();
        
        info!("Starting to watch directory: {:?} -> {:?}", input_dir, output_dir);
        
        // Validate directories
        if !input_dir.exists() {
            return Err(IngestionError::config(
                format!("input directory does not exist: {:?}", input_dir)
            ));
        }
        
        tokio::fs::create_dir_all(&output_dir).await?;
        
        // Create file system watcher
        let (tx, mut rx) = mpsc::channel(1000);
        let mut watcher = RecommendedWatcher::new(
            move |res| {
                let _ = futures::executor::block_on(async {
                    match res {
                        Ok(event) => tx.clone().send(WatchEvent::FileSystem(event)).await,
                        Err(e) => tx.clone().send(WatchEvent::Error(e.to_string())).await,
                    }
                });
            },
            notify::Config::default(),
        )?;
        
        let recursive_mode = if self.config.recursive {
            RecursiveMode::Recursive
        } else {
            RecursiveMode::NonRecursive
        };
        
        watcher.watch(&input_dir, recursive_mode)?;
        
        // Create watched directory entry
        let watched_dir = WatchedDirectory {
            watch_id: watch_id.clone(),
            input_dir: input_dir.clone(),
            output_dir: output_dir.clone(),
            config: watch_config.clone(),
            watcher: Some(watcher),
            stats: WatchStats::new(),
            last_activity: Instant::now(),
        };
        
        self.watched_directories.write().await.insert(watch_id.clone(), watched_dir);
        
        // Start event processing task
        let engine = Arc::clone(&self.engine);
        let config = Arc::clone(&self.config);
        let active_jobs = Arc::clone(&self.active_jobs);
        let watched_directories = Arc::clone(&self.watched_directories);
        let watch_id_clone = watch_id.clone();
        
        tokio::spawn(async move {
            Self::process_watch_events(
                watch_id_clone,
                engine,
                config,
                active_jobs,
                watched_directories,
                rx,
                input_dir,
                output_dir,
                watch_config,
            ).await;
        });
        
        // Process existing files if requested
        if watch_config.process_existing_files {
            self.process_existing_files(&watch_id, &input_dir, &output_dir).await?;
        }
        
        info!("Started watching directory with ID: {}", watch_id);
        Ok(watch_id)
    }
    
    /// Stop watching a directory
    pub async fn stop_watching(&self, watch_id: &str) -> IngestionResult<()> {
        let mut watched_dirs = self.watched_directories.write().await;
        
        if let Some(mut watched_dir) = watched_dirs.remove(watch_id) {
            // Drop the watcher to stop file system monitoring
            watched_dir.watcher = None;
            
            info!("Stopped watching directory: {:?}", watched_dir.input_dir);
            Ok(())
        } else {
            Err(IngestionError::config(format!("watch ID not found: {}", watch_id)))
        }
    }
    
    /// Get watch status
    pub async fn get_watch_status(&self, watch_id: &str) -> Option<WatchStatus> {
        let watched_dirs = self.watched_directories.read().await;
        let active_jobs = self.active_jobs.read().await;
        
        if let Some(watched_dir) = watched_dirs.get(watch_id) {
            let active_job_count = active_jobs.values()
                .filter(|job| job.watch_id == watch_id && !job.completed)
                .count();
            
            Some(WatchStatus {
                watch_id: watch_id.to_string(),
                input_directory: watched_dir.input_dir.clone(),
                output_directory: watched_dir.output_dir.clone(),
                is_active: watched_dir.watcher.is_some(),
                stats: watched_dir.stats.clone(),
                active_jobs: active_job_count,
                last_activity: watched_dir.last_activity,
            })
        } else {
            None
        }
    }
    
    /// List all watched directories
    pub async fn list_watches(&self) -> Vec<WatchStatus> {
        let watched_dirs = self.watched_directories.read().await;
        let active_jobs = self.active_jobs.read().await;
        
        let mut statuses = Vec::new();
        for (watch_id, watched_dir) in watched_dirs.iter() {
            let active_job_count = active_jobs.values()
                .filter(|job| job.watch_id == *watch_id && !job.completed)
                .count();
            
            statuses.push(WatchStatus {
                watch_id: watch_id.clone(),
                input_directory: watched_dir.input_dir.clone(),
                output_directory: watched_dir.output_dir.clone(),
                is_active: watched_dir.watcher.is_some(),
                stats: watched_dir.stats.clone(),
                active_jobs: active_job_count,
                last_activity: watched_dir.last_activity,
            });
        }
        
        statuses
    }
    
    /// Get job status
    pub async fn get_job_status(&self, job_id: &str) -> Option<WatchJob> {
        let active_jobs = self.active_jobs.read().await;
        active_jobs.get(job_id).cloned()
    }
    
    /// Process existing files in a directory
    async fn process_existing_files(
        &self,
        watch_id: &str,
        input_dir: &Path,
        output_dir: &Path,
    ) -> IngestionResult<()> {
        info!("Processing existing files in directory: {:?}", input_dir);
        
        let job_id = Uuid::new_v4().to_string();
        let job = WatchJob {
            job_id: job_id.clone(),
            watch_id: watch_id.to_string(),
            trigger: WatchTrigger::ExistingFiles,
            files: Vec::new(),
            started_at: Instant::now(),
            completed: false,
            success: false,
        };
        
        self.active_jobs.write().await.insert(job_id.clone(), job);
        
        // Process directory
        let result = self.engine.process_directory(input_dir, output_dir).await;
        
        // Update job status
        let mut active_jobs = self.active_jobs.write().await;
        if let Some(job) = active_jobs.get_mut(&job_id) {
            job.completed = true;
            job.success = result.is_ok();
        }
        
        match result {
            Ok(processing_result) => {
                self.update_watch_stats(watch_id, &processing_result).await;
                info!("Completed processing existing files: {:?}", processing_result);
                Ok(())
            }
            Err(e) => {
                error!("Failed to process existing files: {}", e);
                Err(e)
            }
        }
    }
    
    /// Update watch statistics
    async fn update_watch_stats(
        &self,
        watch_id: &str,
        processing_result: &crate::ingestion::ProcessingResult,
    ) {
        let mut watched_dirs = self.watched_directories.write().await;
        if let Some(watched_dir) = watched_dirs.get_mut(watch_id) {
            watched_dir.stats.files_processed += processing_result.files_processed;
            watched_dir.stats.files_failed += processing_result.files_failed;
            watched_dir.stats.bytes_processed += processing_result.input_size_bytes;
            watched_dir.stats.total_processing_time += processing_result.processing_time;
            watched_dir.last_activity = Instant::now();
        }
    }
    
    /// Process watch events (runs in background task)
    async fn process_watch_events(
        watch_id: String,
        engine: Arc<IngestionEngine>,
        config: Arc<IngestionConfig>,
        active_jobs: Arc<RwLock<HashMap<String, WatchJob>>>,
        watched_directories: Arc<RwLock<HashMap<String, WatchedDirectory>>>,
        mut event_receiver: mpsc::Receiver<WatchEvent>,
        input_dir: PathBuf,
        output_dir: PathBuf,
        watch_config: WatchConfig,
    ) {
        let mut pending_files: HashMap<PathBuf, Instant> = HashMap::new();
        let mut cleanup_interval = interval(Duration::from_secs(60));
        
        loop {
            tokio::select! {
                // Process file system events
                event = event_receiver.next() => {
                    match event {
                        Some(WatchEvent::FileSystem(fs_event)) => {
                            if let Err(e) = Self::handle_file_system_event(
                                &watch_id,
                                &fs_event,
                                &engine,
                                &config,
                                &active_jobs,
                                &watched_directories,
                                &mut pending_files,
                                &input_dir,
                                &output_dir,
                                &watch_config,
                            ).await {
                                error!("Error handling file system event: {}", e);
                            }
                        }
                        Some(WatchEvent::Error(error)) => {
                            error!("File system watcher error: {}", error);
                        }
                        None => {
                            debug!("Watch event stream ended for {}", watch_id);
                            break;
                        }
                    }
                }
                
                // Cleanup pending files periodically
                _ = cleanup_interval.tick() => {
                    Self::cleanup_pending_files(&mut pending_files, &watch_config);
                }
            }
        }
        
        info!("Watch event processing stopped for {}", watch_id);
    }
    
    async fn handle_file_system_event(
        watch_id: &str,
        event: &Event,
        engine: &Arc<IngestionEngine>,
        config: &Arc<IngestionConfig>,
        active_jobs: &Arc<RwLock<HashMap<String, WatchJob>>>,
        watched_directories: &Arc<RwLock<HashMap<String, WatchedDirectory>>>,
        pending_files: &mut HashMap<PathBuf, Instant>,
        input_dir: &Path,
        output_dir: &Path,
        watch_config: &WatchConfig,
    ) -> IngestionResult<()> {
        match event.kind {
            EventKind::Create(_) | EventKind::Modify(_) => {
                for path in &event.paths {
                    if path.is_file() && config.matches_pattern(path) {
                        debug!("File event detected: {:?}", path);
                        
                        // Add to pending files with current timestamp
                        pending_files.insert(path.clone(), Instant::now());
                        
                        // Process stable files
                        Self::process_stable_files(
                            watch_id,
                            engine,
                            active_jobs,
                            watched_directories,
                            pending_files,
                            output_dir,
                            watch_config,
                        ).await?;
                    }
                }
            }
            EventKind::Remove(_) => {
                for path in &event.paths {
                    pending_files.remove(path);
                    debug!("File removed from pending: {:?}", path);
                }
            }
            _ => {
                // Ignore other event types
            }
        }
        
        Ok(())
    }
    
    async fn process_stable_files(
        watch_id: &str,
        engine: &Arc<IngestionEngine>,
        active_jobs: &Arc<RwLock<HashMap<String, WatchJob>>>,
        watched_directories: &Arc<RwLock<HashMap<String, WatchedDirectory>>>,
        pending_files: &mut HashMap<PathBuf, Instant>,
        output_dir: &Path,
        watch_config: &WatchConfig,
    ) -> IngestionResult<()> {
        let now = Instant::now();
        let stability_threshold = Duration::from_secs(watch_config.file_stability_seconds);
        
        let mut stable_files = Vec::new();
        let mut to_remove = Vec::new();
        
        for (path, timestamp) in pending_files.iter() {
            if now.duration_since(*timestamp) >= stability_threshold {
                if path.exists() {
                    stable_files.push(path.clone());
                }
                to_remove.push(path.clone());
            }
        }
        
        // Remove processed files from pending
        for path in to_remove {
            pending_files.remove(&path);
        }
        
        // Process stable files
        if !stable_files.is_empty() {
            let job_id = Uuid::new_v4().to_string();
            let job = WatchJob {
                job_id: job_id.clone(),
                watch_id: watch_id.to_string(),
                trigger: WatchTrigger::NewFiles,
                files: stable_files.clone(),
                started_at: now,
                completed: false,
                success: false,
            };
            
            active_jobs.write().await.insert(job_id.clone(), job);
            
            info!("Processing {} stable files", stable_files.len());
            
            // Process files
            let engine_clone = Arc::clone(engine);
            let active_jobs_clone = Arc::clone(active_jobs);
            let watched_directories_clone = Arc::clone(watched_directories);
            let watch_id_clone = watch_id.to_string();
            let output_dir_clone = output_dir.to_path_buf();
            
            tokio::spawn(async move {
                let mut total_success = true;
                
                for file_path in stable_files {
                    match engine_clone.process_file(&file_path, &output_dir_clone).await {
                        Ok(result) => {
                            debug!("Successfully processed file: {:?}", file_path);
                            
                            // Update statistics
                            let mut watched_dirs = watched_directories_clone.write().await;
                            if let Some(watched_dir) = watched_dirs.get_mut(&watch_id_clone) {
                                watched_dir.stats.files_processed += result.files_processed;
                                watched_dir.stats.bytes_processed += result.input_size_bytes;
                                watched_dir.stats.total_processing_time += result.processing_time;
                                watched_dir.last_activity = Instant::now();
                            }
                        }
                        Err(e) => {
                            error!("Failed to process file {:?}: {}", file_path, e);
                            total_success = false;
                            
                            // Update failure statistics
                            let mut watched_dirs = watched_directories_clone.write().await;
                            if let Some(watched_dir) = watched_dirs.get_mut(&watch_id_clone) {
                                watched_dir.stats.files_failed += 1;
                                watched_dir.last_activity = Instant::now();
                            }
                        }
                    }
                }
                
                // Update job completion
                let mut active_jobs = active_jobs_clone.write().await;
                if let Some(job) = active_jobs.get_mut(&job_id) {
                    job.completed = true;
                    job.success = total_success;
                }
            });
        }
        
        Ok(())
    }
    
    fn cleanup_pending_files(pending_files: &mut HashMap<PathBuf, Instant>, watch_config: &WatchConfig) {
        let now = Instant::now();
        let max_age = Duration::from_secs(watch_config.max_pending_age_seconds);
        
        pending_files.retain(|path, timestamp| {
            let age = now.duration_since(*timestamp);
            if age > max_age {
                debug!("Removing stale pending file: {:?} (age: {:?})", path, age);
                false
            } else {
                true
            }
        });
    }
}

/// Watch configuration
#[derive(Debug, Clone)]
pub struct WatchConfig {
    pub process_existing_files: bool,
    pub file_stability_seconds: u64,
    pub max_pending_age_seconds: u64,
    pub batch_processing: bool,
    pub batch_size: usize,
    pub batch_timeout_seconds: u64,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            process_existing_files: true,
            file_stability_seconds: 5,
            max_pending_age_seconds: 3600, // 1 hour
            batch_processing: true,
            batch_size: 10,
            batch_timeout_seconds: 60,
        }
    }
}

/// Watched directory information
struct WatchedDirectory {
    watch_id: String,
    input_dir: PathBuf,
    output_dir: PathBuf,
    config: WatchConfig,
    watcher: Option<RecommendedWatcher>,
    stats: WatchStats,
    last_activity: Instant,
}

/// Watch statistics
#[derive(Debug, Clone)]
pub struct WatchStats {
    pub files_processed: u64,
    pub files_failed: u64,
    pub bytes_processed: u64,
    pub total_processing_time: Duration,
    pub started_at: Instant,
}

impl WatchStats {
    fn new() -> Self {
        Self {
            files_processed: 0,
            files_failed: 0,
            bytes_processed: 0,
            total_processing_time: Duration::from_secs(0),
            started_at: Instant::now(),
        }
    }
    
    pub fn average_processing_time(&self) -> Duration {
        if self.files_processed == 0 {
            Duration::from_secs(0)
        } else {
            self.total_processing_time / self.files_processed as u32
        }
    }
    
    pub fn throughput_mb_per_second(&self) -> f64 {
        let total_seconds = self.started_at.elapsed().as_secs_f64();
        if total_seconds == 0.0 {
            0.0
        } else {
            (self.bytes_processed as f64 / 1024.0 / 1024.0) / total_seconds
        }
    }
    
    pub fn success_rate(&self) -> f64 {
        let total_files = self.files_processed + self.files_failed;
        if total_files == 0 {
            1.0
        } else {
            self.files_processed as f64 / total_files as f64
        }
    }
}

/// Watch status
#[derive(Debug, Clone)]
pub struct WatchStatus {
    pub watch_id: String,
    pub input_directory: PathBuf,
    pub output_directory: PathBuf,
    pub is_active: bool,
    pub stats: WatchStats,
    pub active_jobs: usize,
    pub last_activity: Instant,
}

/// Watch job information
#[derive(Debug, Clone)]
pub struct WatchJob {
    pub job_id: String,
    pub watch_id: String,
    pub trigger: WatchTrigger,
    pub files: Vec<PathBuf>,
    pub started_at: Instant,
    pub completed: bool,
    pub success: bool,
}

/// Watch trigger type
#[derive(Debug, Clone)]
pub enum WatchTrigger {
    ExistingFiles,
    NewFiles,
    Manual,
}

/// Watch events
enum WatchEvent {
    FileSystem(Event),
    Error(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[tokio::test]
    async fn test_directory_watcher_creation() {
        let config = Arc::new(IngestionConfig::default());
        let engine = Arc::new(IngestionEngine::new((*config).clone()));
        let watcher = DirectoryWatcher::new(config, engine);
        
        let watches = watcher.list_watches().await;
        assert!(watches.is_empty());
    }
    
    #[test]
    fn test_watch_config_default() {
        let config = WatchConfig::default();
        assert!(config.process_existing_files);
        assert_eq!(config.file_stability_seconds, 5);
        assert_eq!(config.max_pending_age_seconds, 3600);
    }
    
    #[test]
    fn test_watch_stats() {
        let mut stats = WatchStats::new();
        stats.files_processed = 100;
        stats.files_failed = 5;
        stats.bytes_processed = 1024 * 1024 * 100; // 100 MB
        
        assert_eq!(stats.success_rate(), 100.0 / 105.0);
        assert!(stats.throughput_mb_per_second() >= 0.0);
    }
}