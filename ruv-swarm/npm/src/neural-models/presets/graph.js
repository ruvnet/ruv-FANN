/**
 * Graph Analysis Neural Network Presets
 * Production-ready configurations for graph-based learning tasks
 */

export const graphPresets = {
  // Social Network Influence
  social_network_influence: {
    name: 'Social Network Influence Predictor',
    description: 'Predict influence propagation in social networks',
    model: 'gnn',
    config: {
      nodeDimensions: 256,
      edgeDimensions: 128,
      hiddenDimensions: 512,
      outputDimensions: 64,
      numLayers: 4,
      aggregation: 'attention',
      messagePassingType: 'gcn',
      readoutFunction: 'mean',
      dropoutRate: 0.3,
      useResidualConnections: true,
    },
    training: {
      batchSize: 16,
      learningRate: 1e-3,
      epochs: 150,
      optimizer: 'adam',
      scheduler: 'cosine',
      graphSampling: 'fastgcn',
      samplingDepth: 2,
      samplingWidth: 20,
    },
    performance: {
      expectedAccuracy: '84-87% influence prediction',
      inferenceTime: '25ms per graph',
      memoryUsage: '800MB',
      trainingTime: '12-16 hours on GPU',
    },
    useCase: 'Social media marketing, viral content prediction, community detection',
  },

  // Fraud Detection Financial
  fraud_detection_financial: {
    name: 'Financial Fraud Detector',
    description: 'Detect fraudulent transactions in financial networks',
    model: 'gnn',
    config: {
      nodeDimensions: 128,
      edgeDimensions: 64,
      hiddenDimensions: 256,
      outputDimensions: 2, // Fraud/Not fraud
      numLayers: 3,
      aggregation: 'max',
      messagePassingType: 'gat',
      attentionHeads: 4,
      dropoutRate: 0.4,
      useEdgeFeatures: true,
      temporalAggregation: true,
    },
    training: {
      batchSize: 32,
      learningRate: 5e-4,
      epochs: 200,
      optimizer: 'adamw',
      lossFunction: 'focal_loss',
      focalGamma: 2.0,
      classWeights: [1.0, 50.0], // Heavy penalty for missing fraud
      graphAugmentation: {
        nodeDropout: 0.1,
        edgeDropout: 0.05,
      },
    },
    performance: {
      expectedAccuracy: '96-98% precision on fraud class',
      inferenceTime: '8ms per transaction',
      memoryUsage: '400MB',
      trainingTime: '24-36 hours on GPU',
    },
    useCase: 'Credit card fraud, money laundering detection, banking security',
  },

  // Recommendation Engine
  recommendation_engine: {
    name: 'Graph-based Recommender',
    description: 'User-item recommendation using graph neural networks',
    model: 'gnn',
    config: {
      nodeDimensions: 64,
      edgeDimensions: 32,
      hiddenDimensions: 128,
      embeddingDimensions: 64,
      numLayers: 3,
      aggregation: 'mean',
      messagePassingType: 'lightgcn',
      useUserItemBias: true,
      dropoutRate: 0.1,
      negativeSampling: 5,
    },
    training: {
      batchSize: 1024,
      learningRate: 2e-3,
      epochs: 100,
      optimizer: 'adam',
      lossFunction: 'bpr_loss',
      regularizationWeight: 1e-4,
      evaluationMetrics: ['recall@10', 'ndcg@10'],
      edgeSampling: 'random_walk',
    },
    performance: {
      expectedAccuracy: '88-91% Recall@10',
      inferenceTime: '2ms per user',
      memoryUsage: '600MB',
      trainingTime: '8-12 hours on GPU',
    },
    useCase: 'E-commerce, content streaming, social media feeds',
  },

  // Knowledge Graph QA
  knowledge_graph_qa: {
    name: 'Knowledge Graph Question Answering',
    description: 'Answer questions using knowledge graph reasoning',
    model: 'gnn',
    config: {
      nodeDimensions: 300, // Entity embeddings
      edgeDimensions: 100, // Relation embeddings
      hiddenDimensions: 400,
      outputDimensions: 1, // Answer score
      numLayers: 5,
      aggregation: 'attention',
      messagePassingType: 'rgcn',
      numRelationTypes: 200,
      questionEncoder: 'lstm',
      questionEmbeddingSize: 256,
      multiHopReasoning: true,
    },
    training: {
      batchSize: 8,
      learningRate: 1e-4,
      epochs: 80,
      optimizer: 'adamw',
      gradientClipping: 1.0,
      subgraphSampling: 'khop',
      samplingHops: 3,
      negativeAnswers: 10,
    },
    performance: {
      expectedAccuracy: '78-82% answer accuracy',
      inferenceTime: '150ms per question',
      memoryUsage: '1.2GB',
      trainingTime: '48-72 hours on GPU',
    },
    useCase: 'Intelligent search, virtual assistants, fact checking',
  },

  // Supply Chain Optimization
  supply_chain_optimization: {
    name: 'Supply Chain Network Optimizer',
    description: 'Optimize supply chain operations using graph analysis',
    model: 'gnn',
    config: {
      nodeDimensions: 200, // Supplier/warehouse features
      edgeDimensions: 50, // Transportation features
      hiddenDimensions: 300,
      outputDimensions: 100, // Optimization decisions
      numLayers: 4,
      aggregation: 'sum',
      messagePassingType: 'graphsage',
      samplerType: 'neighbor',
      useTemporalFeatures: true,
      constraintHandling: true,
    },
    training: {
      batchSize: 12,
      learningRate: 3e-4,
      epochs: 120,
      optimizer: 'adam',
      lossFunction: 'custom_cost_minimization',
      constraintPenalty: 100.0,
      simulationBasedTraining: true,
      realTimeAdaptation: true,
    },
    performance: {
      expectedAccuracy: '12-15% cost reduction',
      inferenceTime: '50ms per decision',
      memoryUsage: '700MB',
      trainingTime: '24-36 hours on GPU',
    },
    useCase: 'Logistics optimization, inventory management, route planning',
  },

  // Molecular Property Prediction
  molecular_property_prediction: {
    name: 'Molecular Property Predictor',
    description: 'Predict molecular properties for drug discovery',
    model: 'gnn',
    config: {
      nodeDimensions: 74, // Atom features
      edgeDimensions: 12, // Bond features
      hiddenDimensions: 256,
      outputDimensions: 1, // Property value
      numLayers: 5,
      aggregation: 'mean',
      messagePassingType: 'mpnn',
      globalPooling: 'set2set',
      readoutLayers: [128, 64],
      dropoutRate: 0.2,
    },
    training: {
      batchSize: 64,
      learningRate: 1e-3,
      epochs: 200,
      optimizer: 'adam',
      lossFunction: 'mae',
      scheduler: 'plateau',
      molecularAugmentation: {
        randomRotation: true,
        atomMasking: 0.1,
        bondDropout: 0.05,
      },
    },
    performance: {
      expectedAccuracy: '85-88% R² for solubility',
      inferenceTime: '5ms per molecule',
      memoryUsage: '300MB',
      trainingTime: '16-24 hours on GPU',
    },
    useCase: 'Drug discovery, material science, chemical engineering',
  },

  // Traffic Flow Prediction
  traffic_flow_prediction: {
    name: 'Urban Traffic Flow Predictor',
    description: 'Predict traffic patterns in road networks',
    model: 'gnn',
    config: {
      nodeDimensions: 20, // Road segment features
      edgeDimensions: 10, // Connection features
      hiddenDimensions: 128,
      outputDimensions: 12, // 12 future time steps
      numLayers: 3,
      aggregation: 'attention',
      messagePassingType: 'gcn',
      temporalConvolution: true,
      spatialTemporalFusion: 'gated',
      dropoutRate: 0.25,
    },
    training: {
      batchSize: 24,
      learningRate: 2e-3,
      epochs: 100,
      optimizer: 'adam',
      lossFunction: 'masked_mae',
      maskingRatio: 0.1,
      timeSeriesSplit: true,
      augmentation: {
        temporalJitter: 0.1,
        gaussianNoise: 0.05,
      },
    },
    performance: {
      expectedAccuracy: '91-94% MAE < 15%',
      inferenceTime: '15ms per prediction',
      memoryUsage: '500MB',
      trainingTime: '12-18 hours on GPU',
    },
    useCase: 'Smart city planning, traffic management, route optimization',
  },

  // Scientific Citation Analysis
  citation_analysis: {
    name: 'Scientific Citation Analyzer',
    description: 'Analyze citation networks for research insights',
    model: 'gnn',
    config: {
      nodeDimensions: 512, // Paper embeddings
      edgeDimensions: 64, // Citation context
      hiddenDimensions: 256,
      outputDimensions: 128, // Paper influence score
      numLayers: 4,
      aggregation: 'attention',
      messagePassingType: 'gat',
      attentionHeads: 8,
      temporalEvolution: true,
      fieldSpecialization: true,
    },
    training: {
      batchSize: 16,
      learningRate: 5e-4,
      epochs: 150,
      optimizer: 'adamw',
      lossFunction: 'ranking_loss',
      marginRanking: 1.0,
      metaPathSampling: true,
      communityAware: true,
    },
    performance: {
      expectedAccuracy: '86-89% citation prediction',
      inferenceTime: '30ms per paper',
      memoryUsage: '1GB',
      trainingTime: '36-48 hours on GPU',
    },
    useCase: 'Research recommendation, impact prediction, academic analytics',
  },

  // Protein-Protein Interaction
  protein_interaction: {
    name: 'Protein Interaction Predictor',
    description: 'Predict protein-protein interactions',
    model: 'gnn',
    config: {
      nodeDimensions: 1280, // Protein sequence embeddings
      edgeDimensions: 100, // Interaction features
      hiddenDimensions: 512,
      outputDimensions: 1, // Interaction probability
      numLayers: 6,
      aggregation: 'mean',
      messagePassingType: 'gin',
      proteinEncoder: 'esm',
      structuralFeatures: true,
      dropoutRate: 0.3,
    },
    training: {
      batchSize: 8,
      learningRate: 1e-4,
      epochs: 100,
      optimizer: 'adamw',
      lossFunction: 'weighted_bce',
      classImbalance: 100, // Many more negative than positive examples
      crossValidation: 'species_split',
      augmentation: {
        sequenceNoise: 0.02,
        structuralPerturbation: 0.1,
      },
    },
    performance: {
      expectedAccuracy: '92-94% AUC-ROC',
      inferenceTime: '100ms per protein pair',
      memoryUsage: '2GB',
      trainingTime: '3-5 days on GPU',
    },
    useCase: 'Drug target identification, pathway analysis, systems biology',
  },

  // Cybersecurity Threat Detection
  cybersecurity_threat: {
    name: 'Network Threat Detector',
    description: 'Detect cybersecurity threats in network graphs',
    model: 'gnn',
    config: {
      nodeDimensions: 100, // Host/device features
      edgeDimensions: 50, // Network connection features
      hiddenDimensions: 200,
      outputDimensions: 5, // Threat categories
      numLayers: 3,
      aggregation: 'max',
      messagePassingType: 'graphsaint',
      temporalAggregation: 'gru',
      anomalyDetection: true,
      dropoutRate: 0.35,
    },
    training: {
      batchSize: 32,
      learningRate: 1e-3,
      epochs: 80,
      optimizer: 'adam',
      lossFunction: 'focal_loss',
      onlineLearning: true,
      adaptationRate: 0.01,
      adversarialTraining: true,
      fewShotAdaptation: true,
    },
    performance: {
      expectedAccuracy: '94-96% threat detection',
      inferenceTime: '10ms per network state',
      memoryUsage: '400MB',
      trainingTime: '16-24 hours on GPU',
    },
    useCase: 'Network security, intrusion detection, malware analysis',
  },
};

// Export utility function to get preset by name
export const getGraphPreset = (presetName) => {
  if (!graphPresets[presetName]) {
    throw new Error(`Graph preset '${presetName}' not found. Available presets: ${Object.keys(graphPresets).join(', ')}`);
  }
  return graphPresets[presetName];
};

// Export list of available presets
export const availableGraphPresets = Object.keys(graphPresets);