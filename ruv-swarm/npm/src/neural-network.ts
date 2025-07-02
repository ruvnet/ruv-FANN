// neural-network.ts - TypeScript wrapper for WASM neural network functionality

export interface NetworkConfig {
  inputSize: number;
  hiddenLayers: LayerConfig[];
  outputSize: number;
  outputActivation: string;
  connectionRate?: number;
  randomSeed?: number;
}

export interface LayerConfig {
  size: number;
  activation: string;
  steepness?: number;
}

export interface TrainingDataConfig {
  inputs: number[][];
  outputs: number[][];
}

export interface TrainingConfig {
  algorithm: 'incremental_backprop' | 'batch_backprop' | 'rprop' | 'quickprop' | 'sarprop';
  learningRate?: number;
  momentum?: number;
  maxEpochs: number;
  targetError: number;
  validationSplit?: number;
  earlyStopping?: boolean;
}

export interface AgentNetworkConfig {
  agentId: string;
  agentType: string;
  cognitivePattern: 'convergent' | 'divergent' | 'lateral' | 'systems' | 'critical' | 'abstract';
  inputSize: number;
  outputSize: number;
  taskSpecialization?: string[];
}

export interface CascadeConfig {
  maxHiddenNeurons: number;
  numCandidates: number;
  outputMaxEpochs: number;
  candidateMaxEpochs: number;
  outputLearningRate: number;
  candidateLearningRate: number;
  outputTargetError: number;
  candidateTargetCorrelation: number;
  minCorrelationImprovement: number;
  candidateWeightMin: number;
  candidateWeightMax: number;
  candidateActivations: string[];
  verbose: boolean;
}

export interface NetworkInfo {
  numLayers: number;
  numInputs: number;
  numOutputs: number;
  totalNeurons: number;
  totalConnections: number;
  metrics: {
    trainingError: number;
    validationError: number;
    epochsTrained: number;
    totalConnections: number;
    memoryUsage: number;
  };
}

export interface TrainingResult {
  converged: boolean;
  finalError: number;
  epochs: number;
  targetError: number;
}

export interface CognitiveState {
  agentId: string;
  cognitivePattern: any;
  neuralArchitecture: {
    layers: number;
    neurons: number;
    connections: number;
  };
  trainingProgress: {
    epochsTrained: number;
    currentLoss: number;
    bestLoss: number;
    isTraining: boolean;
  };
  performance: any;
  adaptationHistoryLength: number;
}

let wasmModule: any = null;

export async function initializeNeuralWasm() {
  if (wasmModule) return wasmModule;
  
  try {
    // Dynamic import of WASM module
    const { default: init, ...exports } = await import('../wasm/ruv_swarm_wasm');
    await init();
    wasmModule = exports;
    return wasmModule;
  } catch (error) {
    throw new Error(`Failed to initialize WASM neural module: ${error}`);
  }
}

export class NeuralNetwork {
  private network: any;
  
  constructor(private wasm: any, config: NetworkConfig) {
    this.network = new wasm.WasmNeuralNetwork(config);
  }
  
  async run(inputs: number[]): Promise<number[]> {
    return this.network.run(new Float32Array(inputs));
  }
  
  getWeights(): Float32Array {
    return this.network.get_weights();
  }
  
  setWeights(weights: Float32Array): void {
    this.network.set_weights(weights);
  }
  
  getInfo(): NetworkInfo {
    return this.network.get_network_info();
  }
  
  setTrainingData(data: TrainingDataConfig): void {
    this.network.set_training_data(data);
  }
}

export class NeuralTrainer {
  private trainer: any;
  
  constructor(private wasm: any, config: TrainingConfig) {
    this.trainer = new wasm.WasmTrainer(config);
  }
  
  async trainEpoch(network: NeuralNetwork, data: TrainingDataConfig): Promise<number> {
    return this.trainer.train_epoch(network.network, data);
  }
  
  async trainUntilTarget(
    network: NeuralNetwork,
    data: TrainingDataConfig,
    targetError: number,
    maxEpochs: number,
  ): Promise<TrainingResult> {
    return this.trainer.train_until_target(network.network, data, targetError, maxEpochs);
  }
  
  getTrainingHistory(): any[] {
    return this.trainer.get_training_history();
  }
  
  getAlgorithmInfo(): any {
    return this.trainer.get_algorithm_info();
  }
}

export class AgentNeuralManager {
  private manager: any;
  
  constructor(private wasm: any) {
    this.manager = new wasm.AgentNeuralNetworkManager();
  }
  
  async createAgentNetwork(config: AgentNetworkConfig): Promise<string> {
    return this.manager.create_agent_network(config);
  }
  
  async trainAgentNetwork(agentId: string, data: TrainingDataConfig): Promise<any> {
    return this.manager.train_agent_network(agentId, data);
  }
  
  async getAgentInference(agentId: string, inputs: number[]): Promise<number[]> {
    return this.manager.get_agent_inference(agentId, new Float32Array(inputs));
  }
  
  async getAgentCognitiveState(agentId: string): Promise<CognitiveState> {
    return this.manager.get_agent_cognitive_state(agentId);
  }
  
  async fineTuneDuringExecution(agentId: string, experienceData: any): Promise<any> {
    return this.manager.fine_tune_during_execution(agentId, experienceData);
  }
}

export class ActivationFunctions {
  static async getAll(wasm: any): Promise<[string, string][]> {
    return wasm.ActivationFunctionManager.get_all_functions();
  }
  
  static async test(wasm: any, name: string, input: number, steepness: number = 1.0): Promise<number> {
    return wasm.ActivationFunctionManager.test_activation_function(name, input, steepness);
  }
  
  static async compare(wasm: any, input: number): Promise<Record<string, number>> {
    return wasm.ActivationFunctionManager.compare_functions(input);
  }
  
  static async getProperties(wasm: any, name: string): Promise<any> {
    return wasm.ActivationFunctionManager.get_function_properties(name);
  }
}

export class CascadeTrainer {
  private trainer: any;
  
  constructor(private wasm: any, config: CascadeConfig | null, network: NeuralNetwork, data: TrainingDataConfig) {
    this.trainer = new wasm.WasmCascadeTrainer(config || this.getDefaultConfig(), network.network, data);
  }
  
  async train(): Promise<any> {
    return this.trainer.train();
  }
  
  getConfig(): any {
    return this.trainer.get_config();
  }
  
  static getDefaultConfig(wasm: any): CascadeConfig {
    return wasm.WasmCascadeTrainer.create_default_config();
  }
  
  private getDefaultConfig(): CascadeConfig {
    return CascadeTrainer.getDefaultConfig(this.wasm);
  }
}

// High-level helper functions
export async function createNeuralNetwork(config: NetworkConfig): Promise<NeuralNetwork> {
  const wasm = await initializeNeuralWasm();
  return new NeuralNetwork(wasm, config);
}

export async function createTrainer(config: TrainingConfig): Promise<NeuralTrainer> {
  const wasm = await initializeNeuralWasm();
  return new NeuralTrainer(wasm, config);
}

export async function createAgentNeuralManager(): Promise<AgentNeuralManager> {
  const wasm = await initializeNeuralWasm();
  return new AgentNeuralManager(wasm);
}

// Export activation function names for convenience
export const ACTIVATION_FUNCTIONS = {
  LINEAR: 'linear',
  SIGMOID: 'sigmoid',
  SIGMOID_SYMMETRIC: 'sigmoid_symmetric',
  TANH: 'tanh',
  GAUSSIAN: 'gaussian',
  GAUSSIAN_SYMMETRIC: 'gaussian_symmetric',
  ELLIOT: 'elliot',
  ELLIOT_SYMMETRIC: 'elliot_symmetric',
  RELU: 'relu',
  RELU_LEAKY: 'relu_leaky',
  COS: 'cos',
  COS_SYMMETRIC: 'cos_symmetric',
  SIN: 'sin',
  SIN_SYMMETRIC: 'sin_symmetric',
  THRESHOLD: 'threshold',
  THRESHOLD_SYMMETRIC: 'threshold_symmetric',
  LINEAR_PIECE: 'linear_piece',
  LINEAR_PIECE_SYMMETRIC: 'linear_piece_symmetric',
} as const;

// Export training algorithm names
export const TRAINING_ALGORITHMS = {
  INCREMENTAL_BACKPROP: 'incremental_backprop',
  BATCH_BACKPROP: 'batch_backprop',
  RPROP: 'rprop',
  QUICKPROP: 'quickprop',
  SARPROP: 'sarprop',
} as const;

// Export cognitive patterns
export const COGNITIVE_PATTERNS = {
  CONVERGENT: 'convergent',
  DIVERGENT: 'divergent',
  LATERAL: 'lateral',
  SYSTEMS: 'systems',
  CRITICAL: 'critical',
  ABSTRACT: 'abstract',
} as const;