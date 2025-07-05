//! Demonstration of the ASA-INT-01 Interference Classifier
//! 
//! This example shows how to use the interference classifier programmatically.

use uplink_interference_classifier::{
    models::InterferenceClassifierModel,
    features::FeatureExtractor,
    InterferenceClass, ModelConfig, TrainingExample,
    NoiseFloorMeasurement, CellParameters,
};
use chrono::Utc;
use rand::Rng;
use std::collections::HashMap;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    env_logger::init();
    
    println!("🤖 ASA-INT-01 Interference Classifier Demo");
    println!("==========================================");
    
    // Step 1: Create and configure the model
    println!("\n📋 Step 1: Creating Model Configuration");
    let config = ModelConfig {
        hidden_layers: vec![32, 16],
        learning_rate: 0.01,
        max_epochs: 100, // Reduced for demo
        target_accuracy: 0.90, // Slightly lower for demo
        activation_function: "relu".to_string(),
        dropout_rate: 0.1,
    };
    
    println!("   Architecture: {:?}", config.hidden_layers);
    println!("   Learning Rate: {}", config.learning_rate);
    println!("   Target Accuracy: {:.1}%", config.target_accuracy * 100.0);
    
    // Step 2: Generate synthetic training data
    println!("\n🔧 Step 2: Generating Training Data");
    let training_examples = generate_demo_training_data(500);
    println!("   Generated {} training examples", training_examples.len());
    
    // Count examples per class
    let mut class_counts = HashMap::new();
    for example in &training_examples {
        *class_counts.entry(example.true_interference_class.clone()).or_insert(0) += 1;
    }
    
    println!("   Class Distribution:");
    for (class, count) in &class_counts {
        println!("     {}: {} examples", class.as_str(), count);
    }
    
    // Step 3: Create and train the model
    println!("\n🧠 Step 3: Training the Model");
    let mut model = InterferenceClassifierModel::new(config)?;
    
    println!("   Starting training...");
    let start_time = std::time::Instant::now();
    let metrics = model.train(&training_examples)?;
    let training_time = start_time.elapsed();
    
    println!("   Training completed in {:.2} seconds", training_time.as_secs_f64());
    println!("   Final Accuracy: {:.2}%", metrics.accuracy * 100.0);
    println!("   Precision: {:.4}", metrics.precision);
    println!("   Recall: {:.4}", metrics.recall);
    println!("   F1-Score: {:.4}", metrics.f1_score);
    
    // Step 4: Test classification on new data
    println!("\n🔍 Step 4: Testing Classification");
    
    // Create test scenarios for different interference types
    let test_scenarios = vec![
        ("Thermal Noise", create_thermal_noise_scenario()),
        ("External Jammer", create_external_jammer_scenario()),
        ("PIM", create_pim_scenario()),
        ("Co-Channel", create_co_channel_scenario()),
    ];
    
    let feature_extractor = FeatureExtractor::new();
    
    for (scenario_name, (measurements, cell_params)) in test_scenarios {
        println!("\n   Testing scenario: {}", scenario_name);
        
        // Extract features
        let mut features = feature_extractor.extract_features(&measurements, &cell_params)?;
        feature_extractor.normalize_features(&mut features)?;
        
        // Classify
        let result = model.classify(&features)?;
        
        println!("     Predicted Class: {}", result.interference_class.as_str());
        println!("     Confidence: {:.2}%", result.confidence * 100.0);
        
        // Get probabilities for all classes
        let probabilities = model.get_class_probabilities(&features)?;
        println!("     All Class Probabilities:");
        for (class, prob) in &probabilities {
            println!("       {}: {:.4}", class.as_str(), prob);
        }
    }
    
    // Step 5: Demonstrate model saving and loading
    println!("\n💾 Step 5: Model Persistence");
    let model_path = "demo_model.fann";
    
    println!("   Saving model to: {}", model_path);
    model.save_model(model_path)?;
    
    println!("   Loading model from file...");
    let loaded_model = InterferenceClassifierModel::load_model(model_path)?;
    let model_info = loaded_model.get_model_info();
    
    println!("   Loaded model info:");
    println!("     ID: {}", model_info.model_id);
    println!("     Version: {}", model_info.model_version);
    println!("     Created: {}", model_info.created_at);
    println!("     Features: {}", model_info.feature_vector_size);
    println!("     Classes: {}", model_info.num_classes);
    
    // Step 6: Performance analysis
    println!("\n📊 Step 6: Performance Analysis");
    let test_data = generate_demo_training_data(100);
    let test_metrics = model.evaluate_model(&test_data)?;
    
    println!("   Test Set Performance:");
    println!("     Accuracy: {:.2}%", test_metrics.accuracy * 100.0);
    println!("     Precision: {:.4}", test_metrics.precision);
    println!("     Recall: {:.4}", test_metrics.recall);
    println!("     F1-Score: {:.4}", test_metrics.f1_score);
    
    // Display confusion matrix
    println!("\n   Confusion Matrix:");
    println!("     Predicted →");
    print!("   True ↓");
    for i in 0..InterferenceClass::num_classes() {
        print!(" {:>8}", InterferenceClass::from_index(i).as_str());
    }
    println!();
    
    for (i, row) in test_metrics.confusion_matrix.iter().enumerate() {
        let class_name = InterferenceClass::from_index(i).as_str();
        print!("   {:>12}", class_name);
        for &value in row {
            print!(" {:>8}", value);
        }
        println!();
    }
    
    println!("\n✅ Demo completed successfully!");
    println!("🎯 Model achieves {:.2}% accuracy on test set", test_metrics.accuracy * 100.0);
    
    // Clean up
    std::fs::remove_file(model_path).ok();
    std::fs::remove_file(format!("{}.json", model_path)).ok();
    
    Ok(())
}

fn generate_demo_training_data(num_samples: usize) -> Vec<TrainingExample> {
    let mut rng = rand::thread_rng();
    let mut examples = Vec::new();
    
    let interference_classes = [
        InterferenceClass::ThermalNoise,
        InterferenceClass::CoChannelInterference,
        InterferenceClass::AdjacentChannelInterference,
        InterferenceClass::PassiveIntermodulation,
        InterferenceClass::ExternalJammer,
        InterferenceClass::SpuriousEmissions,
    ];
    
    for i in 0..num_samples {
        let class = &interference_classes[i % interference_classes.len()];
        let (measurements, cell_params) = create_scenario_for_class(class, &mut rng, i);
        
        let example = TrainingExample {
            measurements,
            cell_params,
            true_interference_class: class.clone(),
        };
        
        examples.push(example);
    }
    
    examples
}

fn create_scenario_for_class(
    class: &InterferenceClass,
    rng: &mut impl Rng,
    id: usize,
) -> (Vec<NoiseFloorMeasurement>, CellParameters) {
    let num_measurements = rng.gen_range(15..=30);
    let mut measurements = Vec::new();
    
    for j in 0..num_measurements {
        let (base_pusch, base_pucch) = match class {
            InterferenceClass::ThermalNoise => (-110.0, -112.0),
            InterferenceClass::CoChannelInterference => (-105.0, -107.0),
            InterferenceClass::AdjacentChannelInterference => (-108.0, -110.0),
            InterferenceClass::PassiveIntermodulation => (-95.0, -97.0),
            InterferenceClass::ExternalJammer => (-85.0, -87.0),
            InterferenceClass::SpuriousEmissions => (-100.0, -102.0),
            _ => (-110.0, -112.0),
        };
        
        let noise_var = rng.gen_range(-5.0..5.0);
        let pusch_noise = base_pusch + noise_var;
        let pucch_noise = base_pucch + noise_var;
        
        let measurement = NoiseFloorMeasurement {
            timestamp: Utc::now() - chrono::Duration::minutes(j as i64),
            noise_floor_pusch: pusch_noise,
            noise_floor_pucch: pucch_noise,
            cell_ret: rng.gen_range(0.01..0.15),
            rsrp: rng.gen_range(-120.0..-60.0),
            sinr: rng.gen_range(0.0..30.0),
            active_users: rng.gen_range(10..200),
            prb_utilization: rng.gen_range(0.1..0.9),
        };
        
        measurements.push(measurement);
    }
    
    let cell_params = CellParameters {
        cell_id: format!("demo_cell_{:06}", id),
        frequency_band: ["B1", "B3", "B7", "B20"][rng.gen_range(0..4)].to_string(),
        tx_power: rng.gen_range(30.0..50.0),
        antenna_count: [2, 4, 8][rng.gen_range(0..3)],
        bandwidth_mhz: [10.0, 15.0, 20.0][rng.gen_range(0..3)],
        technology: ["LTE", "NR"][rng.gen_range(0..2)].to_string(),
    };
    
    (measurements, cell_params)
}

fn create_thermal_noise_scenario() -> (Vec<NoiseFloorMeasurement>, CellParameters) {
    let mut measurements = Vec::new();
    
    for i in 0..20 {
        let measurement = NoiseFloorMeasurement {
            timestamp: Utc::now() - chrono::Duration::minutes(i),
            noise_floor_pusch: -110.0 + (i as f64 * 0.1), // Stable thermal noise
            noise_floor_pucch: -112.0 + (i as f64 * 0.1),
            cell_ret: 0.02,
            rsrp: -85.0,
            sinr: 20.0,
            active_users: 50,
            prb_utilization: 0.4,
        };
        measurements.push(measurement);
    }
    
    let cell_params = CellParameters {
        cell_id: "thermal_test_cell".to_string(),
        frequency_band: "B1".to_string(),
        tx_power: 43.0,
        antenna_count: 4,
        bandwidth_mhz: 20.0,
        technology: "LTE".to_string(),
    };
    
    (measurements, cell_params)
}

fn create_external_jammer_scenario() -> (Vec<NoiseFloorMeasurement>, CellParameters) {
    let mut measurements = Vec::new();
    
    for i in 0..20 {
        let jammer_effect = if i > 10 { -85.0 } else { -110.0 }; // Jammer appears suddenly
        
        let measurement = NoiseFloorMeasurement {
            timestamp: Utc::now() - chrono::Duration::minutes(i),
            noise_floor_pusch: jammer_effect,
            noise_floor_pucch: jammer_effect - 2.0,
            cell_ret: if i > 10 { 0.25 } else { 0.02 },
            rsrp: -90.0,
            sinr: if i > 10 { 5.0 } else { 20.0 },
            active_users: 100,
            prb_utilization: 0.8,
        };
        measurements.push(measurement);
    }
    
    let cell_params = CellParameters {
        cell_id: "jammer_test_cell".to_string(),
        frequency_band: "B1".to_string(),
        tx_power: 43.0,
        antenna_count: 4,
        bandwidth_mhz: 20.0,
        technology: "LTE".to_string(),
    };
    
    (measurements, cell_params)
}

fn create_pim_scenario() -> (Vec<NoiseFloorMeasurement>, CellParameters) {
    let mut measurements = Vec::new();
    
    for i in 0..20 {
        // PIM shows characteristic pattern with load
        let load_factor = (i as f64 / 20.0) * 0.8 + 0.2;
        let pim_noise = -110.0 + (load_factor * 15.0); // Increases with load
        
        let measurement = NoiseFloorMeasurement {
            timestamp: Utc::now() - chrono::Duration::minutes(i),
            noise_floor_pusch: pim_noise,
            noise_floor_pucch: pim_noise - 1.0,
            cell_ret: 0.15,
            rsrp: -80.0,
            sinr: 15.0,
            active_users: (load_factor * 200.0) as u32,
            prb_utilization: load_factor,
        };
        measurements.push(measurement);
    }
    
    let cell_params = CellParameters {
        cell_id: "pim_test_cell".to_string(),
        frequency_band: "B1".to_string(),
        tx_power: 46.0, // Higher power increases PIM
        antenna_count: 8,
        bandwidth_mhz: 20.0,
        technology: "LTE".to_string(),
    };
    
    (measurements, cell_params)
}

fn create_co_channel_scenario() -> (Vec<NoiseFloorMeasurement>, CellParameters) {
    let mut measurements = Vec::new();
    
    for i in 0..20 {
        // Co-channel interference varies with time of day pattern
        let time_factor = ((i as f64 / 20.0) * 2.0 * std::f64::consts::PI).sin().abs();
        let co_channel_noise = -110.0 + (time_factor * 8.0);
        
        let measurement = NoiseFloorMeasurement {
            timestamp: Utc::now() - chrono::Duration::minutes(i),
            noise_floor_pusch: co_channel_noise,
            noise_floor_pucch: co_channel_noise - 2.0,
            cell_ret: 0.08,
            rsrp: -85.0,
            sinr: 12.0,
            active_users: 75,
            prb_utilization: 0.6,
        };
        measurements.push(measurement);
    }
    
    let cell_params = CellParameters {
        cell_id: "co_channel_test_cell".to_string(),
        frequency_band: "B1".to_string(),
        tx_power: 43.0,
        antenna_count: 4,
        bandwidth_mhz: 20.0,
        technology: "LTE".to_string(),
    };
    
    (measurements, cell_params)
}