//! SCell Manager gRPC Client

use clap::{Arg, Command, SubCommand};
use log::{error, info};
use scell_manager::proto::s_cell_manager_service_client::SCellManagerServiceClient;
use scell_manager::proto::*;
use scell_manager::types::UEMetrics;
use chrono::Utc;
use std::time::Duration;
use tonic::transport::Channel;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();
    
    // Parse command line arguments
    let matches = Command::new("scell_manager_client")
        .version(env!("CARGO_PKG_VERSION"))
        .about("SCell Manager gRPC Client - Test and interact with SCell Manager")
        .arg(
            Arg::new("server")
                .short('s')
                .long("server")
                .value_name("ADDRESS")
                .help("Server address")
                .default_value("http://127.0.0.1:50051")
        )
        .arg(
            Arg::new("timeout")
                .short('t')
                .long("timeout")
                .value_name("SECONDS")
                .help("Request timeout in seconds")
                .default_value("30")
        )
        .subcommand(
            Command::new("predict")
                .about("Make a prediction for a UE")
                .arg(
                    Arg::new("ue-id")
                        .short('u')
                        .long("ue-id")
                        .value_name("ID")
                        .help("UE identifier")
                        .required(true)
                )
                .arg(
                    Arg::new("throughput")
                        .long("throughput")
                        .value_name("MBPS")
                        .help("Current throughput in Mbps")
                        .default_value("50.0")
                )
                .arg(
                    Arg::new("buffer")
                        .long("buffer")
                        .value_name("BYTES")
                        .help("Buffer status report in bytes")
                        .default_value("100000")
                )
                .arg(
                    Arg::new("cqi")
                        .long("cqi")
                        .value_name("VALUE")
                        .help("Channel Quality Indicator (0-15)")
                        .default_value("10.0")
                )
                .arg(
                    Arg::new("rsrp")
                        .long("rsrp")
                        .value_name("DBM")
                        .help("Reference Signal Received Power in dBm")
                        .default_value("-80.0")
                )
                .arg(
                    Arg::new("sinr")
                        .long("sinr")
                        .value_name("DB")
                        .help("Signal to Interference plus Noise Ratio in dB")
                        .default_value("15.0")
                )
                .arg(
                    Arg::new("bearers")
                        .long("bearers")
                        .value_name("COUNT")
                        .help("Number of active bearers")
                        .default_value("2")
                )
                .arg(
                    Arg::new("data-rate")
                        .long("data-rate")
                        .value_name("MBPS")
                        .help("Data rate requirement in Mbps")
                        .default_value("100.0")
                )
                .arg(
                    Arg::new("horizon")
                        .long("horizon")
                        .value_name("SECONDS")
                        .help("Prediction horizon in seconds")
                        .default_value("30")
                )
        )
        .subcommand(
            Command::new("status")
                .about("Get system status")
        )
        .subcommand(
            Command::new("metrics")
                .about("Get model metrics")
                .arg(
                    Arg::new("model-id")
                        .short('m')
                        .long("model-id")
                        .value_name("ID")
                        .help("Model identifier")
                        .default_value("scell_predictor_v1")
                )
        )
        .subcommand(
            Command::new("train")
                .about("Train model with synthetic data")
                .arg(
                    Arg::new("model-id")
                        .short('m')
                        .long("model-id")
                        .value_name("ID")
                        .help("Model identifier")
                        .default_value("scell_predictor_v1")
                )
                .arg(
                    Arg::new("samples")
                        .short('n')
                        .long("samples")
                        .value_name("COUNT")
                        .help("Number of training samples to generate")
                        .default_value("1000")
                )
        )
        .subcommand(
            Command::new("stream")
                .about("Stream predictions for multiple UEs")
                .arg(
                    Arg::new("ue-ids")
                        .short('u')
                        .long("ue-ids")
                        .value_name("IDS")
                        .help("Comma-separated list of UE IDs")
                        .default_value("ue_001,ue_002,ue_003")
                )
                .arg(
                    Arg::new("interval")
                        .short('i')
                        .long("interval")
                        .value_name("SECONDS")
                        .help("Update interval in seconds")
                        .default_value("5")
                )
                .arg(
                    Arg::new("duration")
                        .short('d')
                        .long("duration")
                        .value_name("SECONDS")
                        .help("Stream duration in seconds")
                        .default_value("60")
                )
        )
        .subcommand(
            Command::new("benchmark")
                .about("Run performance benchmark")
                .arg(
                    Arg::new("requests")
                        .short('n')
                        .long("requests")
                        .value_name("COUNT")
                        .help("Number of requests to send")
                        .default_value("1000")
                )
                .arg(
                    Arg::new("concurrent")
                        .short('c')
                        .long("concurrent")
                        .value_name("COUNT")
                        .help("Number of concurrent requests")
                        .default_value("10")
                )
        )
        .get_matches();
    
    let server_address = matches.get_one::<String>("server").unwrap();
    let timeout_seconds: u64 = matches.get_one::<String>("timeout").unwrap().parse()?;
    
    // Connect to server
    info!("Connecting to SCell Manager at: {}", server_address);
    let channel = Channel::from_shared(server_address.clone())?
        .timeout(Duration::from_secs(timeout_seconds))
        .connect()
        .await?;
    
    let mut client = SCellManagerServiceClient::new(channel);
    info!("Connected successfully");
    
    // Handle subcommands
    match matches.subcommand() {
        Some(("predict", sub_matches)) => {
            handle_predict_command(&mut client, sub_matches).await?;
        }
        Some(("status", _)) => {
            handle_status_command(&mut client).await?;
        }
        Some(("metrics", sub_matches)) => {
            handle_metrics_command(&mut client, sub_matches).await?;
        }
        Some(("train", sub_matches)) => {
            handle_train_command(&mut client, sub_matches).await?;
        }
        Some(("stream", sub_matches)) => {
            handle_stream_command(&mut client, sub_matches).await?;
        }
        Some(("benchmark", sub_matches)) => {
            handle_benchmark_command(&mut client, sub_matches).await?;
        }
        _ => {
            println!("No subcommand specified. Use --help for usage information.");
        }
    }
    
    Ok(())
}

async fn handle_predict_command(
    client: &mut SCellManagerServiceClient<Channel>,
    matches: &clap::ArgMatches,
) -> Result<(), Box<dyn std::error::Error>> {
    let ue_id = matches.get_one::<String>("ue-id").unwrap();
    let throughput: f32 = matches.get_one::<String>("throughput").unwrap().parse()?;
    let buffer: i64 = matches.get_one::<String>("buffer").unwrap().parse()?;
    let cqi: f32 = matches.get_one::<String>("cqi").unwrap().parse()?;
    let rsrp: f32 = matches.get_one::<String>("rsrp").unwrap().parse()?;
    let sinr: f32 = matches.get_one::<String>("sinr").unwrap().parse()?;
    let bearers: i32 = matches.get_one::<String>("bearers").unwrap().parse()?;
    let data_rate: f32 = matches.get_one::<String>("data-rate").unwrap().parse()?;
    let horizon: i32 = matches.get_one::<String>("horizon").unwrap().parse()?;
    
    let request = PredictScellNeedRequest {
        ue_id: ue_id.clone(),
        current_metrics: Some(UeMetrics {
            ue_id: ue_id.clone(),
            pcell_throughput_mbps: throughput,
            buffer_status_report_bytes: buffer,
            pcell_cqi: cqi,
            pcell_rsrp: rsrp,
            pcell_sinr: sinr,
            active_bearers: bearers,
            data_rate_req_mbps: data_rate,
            timestamp_utc: Utc::now().timestamp(),
        }),
        historical_metrics: vec![], // No historical data for this example
        prediction_horizon_seconds: horizon,
    };
    
    info!("Making prediction for UE: {}", ue_id);
    let start_time = std::time::Instant::now();
    
    match client.predict_scell_need(request).await {
        Ok(response) => {
            let prediction = response.into_inner();
            let elapsed = start_time.elapsed();
            
            println!("\n📊 SCell Prediction Results");
            println!("═══════════════════════════");
            println!("UE ID: {}", prediction.ue_id);
            println!("SCell Activation Recommended: {}", 
                if prediction.scell_activation_recommended { "✅ YES" } else { "❌ NO" });
            println!("Confidence Score: {:.3}", prediction.confidence_score);
            println!("Predicted Throughput Demand: {:.1} Mbps", prediction.predicted_throughput_demand);
            println!("Reasoning: {}", prediction.reasoning);
            println!("Prediction Time: {:.2}ms", elapsed.as_millis());
            println!("Timestamp: {}", chrono::DateTime::from_timestamp(prediction.timestamp_utc, 0)
                .unwrap_or_else(|| Utc::now()).format("%Y-%m-%d %H:%M:%S UTC"));
        }
        Err(e) => {
            error!("Prediction failed: {}", e);
            return Err(Box::new(e));
        }
    }
    
    Ok(())
}

async fn handle_status_command(
    client: &mut SCellManagerServiceClient<Channel>,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("Getting system status...");
    
    match client.get_system_status(GetSystemStatusRequest {}).await {
        Ok(response) => {
            let status = response.into_inner();
            
            println!("\n🔍 System Status");
            println!("═══════════════");
            println!("Health: {}", if status.healthy { "✅ Healthy" } else { "❌ Unhealthy" });
            println!("Version: {}", status.version);
            println!("Active Models: {}", status.active_models);
            println!("Total Predictions: {}", status.total_predictions);
            println!("Average Prediction Time: {:.2}ms", status.average_prediction_time_ms);
            println!("Uptime: {}s ({:.1}h)", status.uptime_seconds, status.uptime_seconds as f64 / 3600.0);
            
            if !status.system_info.is_empty() {
                println!("\nSystem Info:");
                for (key, value) in &status.system_info {
                    println!("  {}: {}", key, value);
                }
            }
        }
        Err(e) => {
            error!("Failed to get system status: {}", e);
            return Err(Box::new(e));
        }
    }
    
    Ok(())
}

async fn handle_metrics_command(
    client: &mut SCellManagerServiceClient<Channel>,
    matches: &clap::ArgMatches,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_id = matches.get_one::<String>("model-id").unwrap();
    
    info!("Getting metrics for model: {}", model_id);
    
    match client.get_model_metrics(GetModelMetricsRequest {
        model_id: model_id.clone(),
    }).await {
        Ok(response) => {
            let response = response.into_inner();
            
            if let Some(metrics) = response.metrics {
                println!("\n📈 Model Metrics: {}", response.model_id);
                println!("═══════════════════════════");
                println!("Accuracy: {:.4} ({:.1}%)", metrics.accuracy, metrics.accuracy * 100.0);
                println!("Precision: {:.4} ({:.1}%)", metrics.precision, metrics.precision * 100.0);
                println!("Recall: {:.4} ({:.1}%)", metrics.recall, metrics.recall * 100.0);
                println!("F1 Score: {:.4}", metrics.f1_score);
                println!("AUC-ROC: {:.4}", metrics.auc_roc);
                println!("Mean Absolute Error: {:.4}", metrics.mean_absolute_error);
                println!("Total Predictions: {}", metrics.total_predictions);
                
                println!("\nConfusion Matrix:");
                println!("  True Positives: {}", metrics.true_positives);
                println!("  False Positives: {}", metrics.false_positives);
                println!("  True Negatives: {}", metrics.true_negatives);
                println!("  False Negatives: {}", metrics.false_negatives);
                
                println!("Last Updated: {}", chrono::DateTime::from_timestamp(response.last_updated_utc, 0)
                    .unwrap_or_else(|| Utc::now()).format("%Y-%m-%d %H:%M:%S UTC"));
            } else {
                println!("No metrics available for model: {}", model_id);
            }
        }
        Err(e) => {
            error!("Failed to get model metrics: {}", e);
            return Err(Box::new(e));
        }
    }
    
    Ok(())
}

async fn handle_train_command(
    client: &mut SCellManagerServiceClient<Channel>,
    matches: &clap::ArgMatches,
) -> Result<(), Box<dyn std::error::Error>> {
    let model_id = matches.get_one::<String>("model-id").unwrap();
    let samples: usize = matches.get_one::<String>("samples").unwrap().parse()?;
    
    info!("Generating {} synthetic training samples...", samples);
    
    // Generate synthetic training data
    let mut training_data = Vec::new();
    let mut rng = fastrand::Rng::new();
    
    for i in 0..samples {
        let ue_id = format!("synthetic_ue_{:06}", i);
        
        // Generate realistic metrics
        let throughput = rng.f32() * 200.0;
        let buffer = rng.i64() % 1000000;
        let cqi = rng.f32() * 15.0;
        let rsrp = -120.0 + rng.f32() * 50.0;
        let sinr = -10.0 + rng.f32() * 30.0;
        let bearers = 1 + (rng.u32() % 4) as i32;
        let data_rate = throughput * (0.8 + rng.f32() * 0.4);
        
        // Determine label based on heuristics
        let high_throughput = throughput > 80.0;
        let high_demand = data_rate > 100.0;
        let good_signal = cqi > 8.0 && sinr > 5.0;
        let large_buffer = buffer > 100000;
        
        let score = (high_throughput as u32) + (high_demand as u32) + 
                   (good_signal as u32) + (large_buffer as u32);
        let scell_needed = score >= 2;
        
        let actual_throughput_demand = if scell_needed {
            data_rate * (1.2 + rng.f32() * 0.5)
        } else {
            data_rate * (0.7 + rng.f32() * 0.4)
        };
        
        let example = TrainingExample {
            input_metrics: Some(UeMetrics {
                ue_id: ue_id.clone(),
                pcell_throughput_mbps: throughput,
                buffer_status_report_bytes: buffer,
                pcell_cqi: cqi,
                pcell_rsrp: rsrp,
                pcell_sinr: sinr,
                active_bearers: bearers,
                data_rate_req_mbps: data_rate,
                timestamp_utc: Utc::now().timestamp(),
            }),
            historical_sequence: vec![], // No historical data for synthetic examples
            actual_scell_needed: scell_needed,
            actual_throughput_demand,
        };
        
        training_data.push(example);
    }
    
    let request = TrainModelRequest {
        model_id: model_id.clone(),
        training_data,
        config: Some(TrainingConfig {
            epochs: 100,
            learning_rate: 0.001,
            batch_size: 32,
            validation_split: 0.2,
            sequence_length: 10,
        }),
    };
    
    info!("Starting training for model: {}", model_id);
    let start_time = std::time::Instant::now();
    
    match client.train_model(request).await {
        Ok(response) => {
            let result = response.into_inner();
            let elapsed = start_time.elapsed();
            
            if result.success {
                println!("\n🎯 Training Results");
                println!("═══════════════════");
                println!("Model ID: {}", result.model_id);
                println!("Training Status: ✅ Success");
                println!("Training Time: {:.2}s", elapsed.as_secs_f64());
                
                if let Some(metrics) = result.metrics {
                    println!("\nFinal Metrics:");
                    println!("  Accuracy: {:.4} ({:.1}%)", metrics.accuracy, metrics.accuracy * 100.0);
                    println!("  Precision: {:.4} ({:.1}%)", metrics.precision, metrics.precision * 100.0);
                    println!("  Recall: {:.4} ({:.1}%)", metrics.recall, metrics.recall * 100.0);
                    println!("  F1 Score: {:.4}", metrics.f1_score);
                }
            } else {
                println!("❌ Training failed: {}", result.error_message);
            }
        }
        Err(e) => {
            error!("Training request failed: {}", e);
            return Err(Box::new(e));
        }
    }
    
    Ok(())
}

async fn handle_stream_command(
    client: &mut SCellManagerServiceClient<Channel>,
    matches: &clap::ArgMatches,
) -> Result<(), Box<dyn std::error::Error>> {
    let ue_ids_str = matches.get_one::<String>("ue-ids").unwrap();
    let interval: i32 = matches.get_one::<String>("interval").unwrap().parse()?;
    let duration: u64 = matches.get_one::<String>("duration").unwrap().parse()?;
    
    let ue_ids: Vec<String> = ue_ids_str.split(',')
        .map(|s| s.trim().to_string())
        .collect();
    
    info!("Starting prediction stream for {} UEs, interval: {}s, duration: {}s", 
          ue_ids.len(), interval, duration);
    
    let request = StreamPredictionsRequest {
        ue_ids: ue_ids.clone(),
        update_interval_seconds: interval,
    };
    
    let mut stream = client.stream_predictions(request).await?.into_inner();
    
    println!("\n📡 Streaming Predictions");
    println!("═══════════════════════");
    println!("UEs: {:?}", ue_ids);
    println!("Update Interval: {}s", interval);
    println!("Duration: {}s", duration);
    println!("\nPress Ctrl+C to stop...\n");
    
    let start_time = std::time::Instant::now();
    let mut prediction_count = 0;
    
    while let Some(update) = stream.message().await? {
        if start_time.elapsed().as_secs() >= duration {
            break;
        }
        
        prediction_count += 1;
        
        if let Some(prediction) = update.prediction {
            let timestamp = chrono::DateTime::from_timestamp(update.timestamp_utc, 0)
                .unwrap_or_else(|| Utc::now());
            
            println!("[{}] UE: {} | SCell: {} | Confidence: {:.3} | Throughput: {:.1} Mbps",
                timestamp.format("%H:%M:%S"),
                update.ue_id,
                if prediction.scell_activation_recommended { "✅" } else { "❌" },
                prediction.confidence_score,
                prediction.predicted_throughput_demand
            );
        }
    }
    
    println!("\n✅ Stream completed. Received {} predictions in {:.1}s",
             prediction_count, start_time.elapsed().as_secs_f64());
    
    Ok(())
}

async fn handle_benchmark_command(
    client: &mut SCellManagerServiceClient<Channel>,
    matches: &clap::ArgMatches,
) -> Result<(), Box<dyn std::error::Error>> {
    let requests: usize = matches.get_one::<String>("requests").unwrap().parse()?;
    let concurrent: usize = matches.get_one::<String>("concurrent").unwrap().parse()?;
    
    info!("Running benchmark: {} requests with {} concurrent connections", requests, concurrent);
    
    println!("\n⚡ Performance Benchmark");
    println!("═══════════════════════");
    println!("Total Requests: {}", requests);
    println!("Concurrent: {}", concurrent);
    println!("Starting benchmark...\n");
    
    let start_time = std::time::Instant::now();
    let mut handles = Vec::new();
    let requests_per_task = requests / concurrent;
    let remaining_requests = requests % concurrent;
    
    // Create concurrent tasks
    for i in 0..concurrent {
        let mut task_client = client.clone();
        let task_requests = if i < remaining_requests { 
            requests_per_task + 1 
        } else { 
            requests_per_task 
        };
        
        let handle = tokio::spawn(async move {
            let mut successful = 0;
            let mut failed = 0;
            let mut total_time = Duration::default();
            
            for j in 0..task_requests {
                let ue_id = format!("benchmark_ue_{}_{}", i, j);
                
                let request = PredictScellNeedRequest {
                    ue_id: ue_id.clone(),
                    current_metrics: Some(UeMetrics {
                        ue_id,
                        pcell_throughput_mbps: 50.0 + (j as f32 * 10.0) % 200.0,
                        buffer_status_report_bytes: 100000 + (j as i64 * 1000) % 900000,
                        pcell_cqi: 5.0 + (j as f32) % 10.0,
                        pcell_rsrp: -100.0 - (j as f32) % 20.0,
                        pcell_sinr: 10.0 + (j as f32) % 15.0,
                        active_bearers: 1 + (j as i32) % 4,
                        data_rate_req_mbps: 80.0 + (j as f32 * 5.0) % 100.0,
                        timestamp_utc: Utc::now().timestamp(),
                    }),
                    historical_metrics: vec![],
                    prediction_horizon_seconds: 30,
                };
                
                let req_start = std::time::Instant::now();
                
                match task_client.predict_scell_need(request).await {
                    Ok(_) => {
                        successful += 1;
                        total_time += req_start.elapsed();
                    }
                    Err(_) => {
                        failed += 1;
                    }
                }
            }
            
            (successful, failed, total_time)
        });
        
        handles.push(handle);
    }
    
    // Wait for all tasks to complete
    let mut total_successful = 0;
    let mut total_failed = 0;
    let mut total_request_time = Duration::default();
    
    for handle in handles {
        let (successful, failed, request_time) = handle.await?;
        total_successful += successful;
        total_failed += failed;
        total_request_time += request_time;
    }
    
    let total_time = start_time.elapsed();
    let rps = total_successful as f64 / total_time.as_secs_f64();
    let avg_request_time = if total_successful > 0 {
        total_request_time.as_millis() as f64 / total_successful as f64
    } else {
        0.0
    };
    
    println!("📊 Benchmark Results");
    println!("═══════════════════");
    println!("Total Time: {:.2}s", total_time.as_secs_f64());
    println!("Successful Requests: {}", total_successful);
    println!("Failed Requests: {}", total_failed);
    println!("Success Rate: {:.2}%", 
             (total_successful as f64 / requests as f64) * 100.0);
    println!("Requests per Second: {:.2}", rps);
    println!("Average Request Time: {:.2}ms", avg_request_time);
    
    if rps > 100.0 {
        println!("🚀 Excellent performance!");
    } else if rps > 50.0 {
        println!("✅ Good performance");
    } else {
        println!("⚠️  Performance could be improved");
    }
    
    Ok(())
}