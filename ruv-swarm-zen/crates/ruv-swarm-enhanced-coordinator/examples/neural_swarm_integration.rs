// neural_swarm_integration.rs - Example of neural networks integrated with swarm agents

use ruv_swarm_core::{Agent, AgentType, Swarm, SwarmTopology, Task};
use ruv_fann::{Network, NetworkBuilder, ActivationFunction, TrainingData};
use ruv_fann::training::{TrainingAlgorithm, Rprop};
use std::collections::HashMap;

/// Agent with integrated neural network for decision making
struct NeuralAgent {
    agent: Agent,
    network: Network<f32>,
    cognitive_pattern: CognitivePattern,
}

#[derive(Debug, Clone)]
enum CognitivePattern {
    Convergent,  // Analytical, focused
    Divergent,   // Creative, exploratory
    Lateral,     // Associative
    Systems,     // Holistic
    Critical,    // Evaluative
    Abstract,    // Conceptual
}

impl NeuralAgent {
    fn new(agent_type: AgentType, pattern: CognitivePattern) -> Self {
        let agent = Agent::new(agent_type.clone());
        
        // Build neural network based on cognitive pattern
        let network = match pattern {
            CognitivePattern::Convergent => {
                // Narrowing architecture for focused analysis
                NetworkBuilder::<f32>::new()
                    .input_layer(10)
                    .hidden_layer_with_activation(128, ActivationFunction::ReLU, 1.0)
                    .hidden_layer_with_activation(64, ActivationFunction::ReLU, 1.0)
                    .hidden_layer_with_activation(32, ActivationFunction::ReLU, 1.0)
                    .output_layer_with_activation(5, ActivationFunction::Sigmoid, 1.0)
                    .build()
            },
            CognitivePattern::Divergent => {
                // Expanding then contracting for exploration
                NetworkBuilder::<f32>::new()
                    .input_layer(10)
                    .hidden_layer_with_activation(256, ActivationFunction::Sigmoid, 1.0)
                    .hidden_layer_with_activation(128, ActivationFunction::Tanh, 1.0)
                    .hidden_layer_with_activation(64, ActivationFunction::Sigmoid, 1.0)
                    .hidden_layer_with_activation(32, ActivationFunction::ReLU, 1.0)
                    .output_layer_with_activation(5, ActivationFunction::SigmoidSymmetric, 1.0)
                    .build()
            },
            CognitivePattern::Lateral => {
                // Balanced architecture for associations
                NetworkBuilder::<f32>::new()
                    .input_layer(10)
                    .hidden_layer_with_activation(200, ActivationFunction::Elliot, 1.0)
                    .hidden_layer_with_activation(100, ActivationFunction::ElliotSymmetric, 1.0)
                    .hidden_layer_with_activation(50, ActivationFunction::Sigmoid, 1.0)
                    .output_layer_with_activation(5, ActivationFunction::Sigmoid, 1.0)
                    .build()
            },
            CognitivePattern::Systems => {
                // Deep architecture for holistic processing
                NetworkBuilder::<f32>::new()
                    .input_layer(10)
                    .hidden_layer_with_activation(300, ActivationFunction::ReLU, 1.0)
                    .hidden_layer_with_activation(150, ActivationFunction::ReLU, 1.0)
                    .hidden_layer_with_activation(75, ActivationFunction::Tanh, 1.0)
                    .hidden_layer_with_activation(40, ActivationFunction::Sigmoid, 1.0)
                    .output_layer_with_activation(5, ActivationFunction::Linear, 1.0)
                    .build()
            },
            CognitivePattern::Critical => {
                // Moderate architecture for evaluation
                NetworkBuilder::<f32>::new()
                    .input_layer(10)
                    .hidden_layer_with_activation(150, ActivationFunction::Sigmoid, 2.0)
                    .hidden_layer_with_activation(75, ActivationFunction::Sigmoid, 2.0)
                    .hidden_layer_with_activation(40, ActivationFunction::ReLU, 1.0)
                    .output_layer_with_activation(5, ActivationFunction::Threshold, 1.0)
                    .build()
            },
            CognitivePattern::Abstract => {
                // Very deep architecture for conceptual processing
                NetworkBuilder::<f32>::new()
                    .input_layer(10)
                    .hidden_layer_with_activation(400, ActivationFunction::Gaussian, 0.5)
                    .hidden_layer_with_activation(200, ActivationFunction::GaussianSymmetric, 0.5)
                    .hidden_layer_with_activation(100, ActivationFunction::Tanh, 1.0)
                    .hidden_layer_with_activation(50, ActivationFunction::Sigmoid, 1.0)
                    .output_layer_with_activation(5, ActivationFunction::SigmoidSymmetric, 1.0)
                    .build()
            },
        };
        
        NeuralAgent {
            agent,
            network,
            cognitive_pattern: pattern,
        }
    }
    
    /// Process task through neural network
    fn evaluate_task(&mut self, task: &Task) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Convert task to neural input
        let input = self.task_to_input(task);
        
        // Run through network
        let output = self.network.run(&input)?;
        
        Ok(output)
    }
    
    /// Convert task to neural network input
    fn task_to_input(&self, task: &Task) -> Vec<f32> {
        let mut input = vec![0.0; 10];
        
        // Encode task priority
        input[0] = match task.priority {
            ruv_swarm_core::Priority::Low => 0.0,
            ruv_swarm_core::Priority::Medium => 0.5,
            ruv_swarm_core::Priority::High => 1.0,
        };
        
        // Encode task type (simplified)
        input[1] = task.task_type.len() as f32 / 100.0;
        
        // Encode payload size
        input[2] = (task.payload.len() as f32).ln() / 10.0;
        
        // Add some randomness for diversity
        for i in 3..10 {
            input[i] = rand::random::<f32>();
        }
        
        input
    }
    
    /// Train the neural network on task outcomes
    fn train_on_experience(&mut self, experiences: &[(Vec<f32>, Vec<f32>)]) -> Result<f32, Box<dyn std::error::Error>> {
        let training_data = TrainingData {
            inputs: experiences.iter().map(|(i, _)| i.clone()).collect(),
            outputs: experiences.iter().map(|(_, o)| o.clone()).collect(),
        };
        
        let mut trainer = Rprop::new();
        let error = trainer.train_epoch(&mut self.network, &training_data)?;
        
        Ok(error)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Neural Swarm Integration Example ===\n");
    
    // Create swarm with hierarchical topology
    let mut swarm = Swarm::builder()
        .topology(SwarmTopology::Hierarchical)
        .max_agents(10)
        .build()?;
    
    // Create agents with different cognitive patterns
    let mut neural_agents = vec![
        NeuralAgent::new(AgentType::Researcher, CognitivePattern::Divergent),
        NeuralAgent::new(AgentType::Coder, CognitivePattern::Convergent),
        NeuralAgent::new(AgentType::Analyst, CognitivePattern::Critical),
        NeuralAgent::new(AgentType::Optimizer, CognitivePattern::Systems),
        NeuralAgent::new(AgentType::Coordinator, CognitivePattern::Lateral),
    ];
    
    // Add agents to swarm
    for neural_agent in &neural_agents {
        swarm.add_agent(neural_agent.agent.clone())?;
    }
    
    println!("Created {} agents with neural networks:", neural_agents.len());
    for (i, agent) in neural_agents.iter().enumerate() {
        println!("  Agent {}: {:?} with {:?} pattern", 
            i, agent.agent.agent_type(), agent.cognitive_pattern);
    }
    
    // Create test tasks
    let tasks = vec![
        Task::new("analyze_data", serde_json::json!({"data": "test"}), ruv_swarm_core::Priority::High),
        Task::new("optimize_code", serde_json::json!({"code": "fn main() {}"}), ruv_swarm_core::Priority::Medium),
        Task::new("research_topic", serde_json::json!({"topic": "AI"}), ruv_swarm_core::Priority::Low),
    ];
    
    println!("\nEvaluating tasks with neural networks:");
    
    // Each agent evaluates each task
    let mut evaluations = HashMap::new();
    
    for (agent_idx, neural_agent) in neural_agents.iter_mut().enumerate() {
        for (task_idx, task) in tasks.iter().enumerate() {
            let evaluation = neural_agent.evaluate_task(task)?;
            
            println!("  Agent {} evaluating Task {}: {:?}", 
                agent_idx, task_idx, evaluation.iter().map(|v| format!("{:.3}", v)).collect::<Vec<_>>());
            
            evaluations.insert((agent_idx, task_idx), evaluation);
        }
    }
    
    // Simulate training on outcomes
    println!("\nTraining neural networks on simulated outcomes:");
    
    for (agent_idx, neural_agent) in neural_agents.iter_mut().enumerate() {
        // Create synthetic training data
        let experiences: Vec<(Vec<f32>, Vec<f32>)> = (0..10)
            .map(|_| {
                let input = (0..10).map(|_| rand::random::<f32>()).collect();
                let output = (0..5).map(|_| rand::random::<f32>()).collect();
                (input, output)
            })
            .collect();
        
        let error = neural_agent.train_on_experience(&experiences)?;
        println!("  Agent {} training error: {:.6}", agent_idx, error);
    }
    
    // Demonstrate cognitive diversity in action
    println!("\nCognitive diversity demonstration:");
    
    let complex_task = Task::new(
        "complex_problem",
        serde_json::json!({
            "problem": "Design a distributed system",
            "constraints": ["scalability", "fault-tolerance", "performance"]
        }),
        ruv_swarm_core::Priority::High
    );
    
    println!("Complex task evaluation by different cognitive patterns:");
    for (i, neural_agent) in neural_agents.iter_mut().enumerate() {
        let evaluation = neural_agent.evaluate_task(&complex_task)?;
        let interpretation = interpret_output(&evaluation, &neural_agent.cognitive_pattern);
        
        println!("  {:?} agent: {}", neural_agent.cognitive_pattern, interpretation);
    }
    
    println!("\n✅ Neural swarm integration example complete!");
    
    Ok(())
}

/// Interpret neural network output based on cognitive pattern
fn interpret_output(output: &[f32], pattern: &CognitivePattern) -> String {
    match pattern {
        CognitivePattern::Convergent => {
            format!("Focused solution confidence: {:.1}%", output[0] * 100.0)
        },
        CognitivePattern::Divergent => {
            format!("Creative alternatives found: {}", output.iter().filter(|&&v| v > 0.5).count())
        },
        CognitivePattern::Lateral => {
            format!("Cross-domain connections: {:.0}", output.iter().sum::<f32>() * 10.0)
        },
        CognitivePattern::Systems => {
            format!("System complexity score: {:.2}", output.iter().map(|v| v * v).sum::<f32>())
        },
        CognitivePattern::Critical => {
            format!("Risk assessment: {}", if output[0] > 0.7 { "HIGH" } else if output[0] > 0.3 { "MEDIUM" } else { "LOW" })
        },
        CognitivePattern::Abstract => {
            format!("Conceptual abstraction level: {:.1}", output.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() * 10.0)
        },
    }
}