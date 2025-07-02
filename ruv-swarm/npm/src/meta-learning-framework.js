/**
 * Meta-Learning Framework
 * Enables learning how to learn and domain adaptation
 */

class MetaLearningFramework {
  constructor() {
    this.agentExperiences = new Map();
    this.domainAdaptations = new Map();
    this.transferLearning = new Map();
    this.metaStrategies = new Map();
    this.learningMetrics = new Map();

    // Initialize meta-learning strategies
    this.initializeMetaStrategies();
  }

  /**
   * Initialize meta-learning strategies
   */
  initializeMetaStrategies() {
    // Model-Agnostic Meta-Learning (MAML)
    this.metaStrategies.set('maml', {
      name: 'Model-Agnostic Meta-Learning',
      description: 'Learn good parameter initializations for quick adaptation',
      type: 'gradient_based',
      parameters: {
        innerLearningRate: 0.01,
        outerLearningRate: 0.001,
        innerSteps: 5,
        metaBatchSize: 4,
      },
      applicability: {
        fewShotLearning: 0.9,
        domainTransfer: 0.8,
        taskAdaptation: 0.9,
        continualLearning: 0.6,
      },
    });

    // Prototypical Networks
    this.metaStrategies.set('prototypical', {
      name: 'Prototypical Networks',
      description: 'Learn metric space for few-shot classification',
      type: 'metric_based',
      parameters: {
        embeddingDim: 64,
        distanceMetric: 'euclidean',
        temperatureScale: 1.0,
      },
      applicability: {
        fewShotLearning: 0.95,
        domainTransfer: 0.7,
        taskAdaptation: 0.8,
        continualLearning: 0.5,
      },
    });

    // Memory-Augmented Networks
    this.metaStrategies.set('memory_augmented', {
      name: 'Memory-Augmented Networks',
      description: 'Use external memory for rapid learning',
      type: 'memory_based',
      parameters: {
        memorySize: 128,
        keySize: 64,
        valueSize: 64,
        readHeads: 1,
        writeHeads: 1,
      },
      applicability: {
        fewShotLearning: 0.8,
        domainTransfer: 0.6,
        taskAdaptation: 0.7,
        continualLearning: 0.9,
      },
    });

    // Reptile Meta-Learning
    this.metaStrategies.set('reptile', {
      name: 'Reptile',
      description: 'Simple meta-learning algorithm for good initialization',
      type: 'gradient_based',
      parameters: {
        innerLearningRate: 0.02,
        outerLearningRate: 1.0,
        innerSteps: 10,
        metaBatchSize: 5,
      },
      applicability: {
        fewShotLearning: 0.85,
        domainTransfer: 0.75,
        taskAdaptation: 0.8,
        continualLearning: 0.7,
      },
    });

    // Learning to Optimize
    this.metaStrategies.set('learning_to_optimize', {
      name: 'Learning to Optimize',
      description: 'Learn optimization strategies for different tasks',
      type: 'optimization_based',
      parameters: {
        optimizerType: 'lstm',
        optimizerHiddenSize: 20,
        learningRate: 0.001,
        coordinatewise: true,
      },
      applicability: {
        fewShotLearning: 0.7,
        domainTransfer: 0.8,
        taskAdaptation: 0.9,
        continualLearning: 0.8,
      },
    });

    // Meta-Learning for Domain Adaptation
    this.metaStrategies.set('domain_adaptation', {
      name: 'Meta-Domain Adaptation',
      description: 'Learn domain-invariant representations',
      type: 'domain_based',
      parameters: {
        domainDiscriminatorStrength: 0.1,
        gradientReversalLambda: 1.0,
        alignmentLoss: 'coral',
        adaptationSteps: 20,
      },
      applicability: {
        fewShotLearning: 0.6,
        domainTransfer: 0.95,
        taskAdaptation: 0.7,
        continualLearning: 0.6,
      },
    });

    // Continual Meta-Learning
    this.metaStrategies.set('continual_meta', {
      name: 'Continual Meta-Learning',
      description: 'Meta-learning while avoiding catastrophic forgetting',
      type: 'continual_based',
      parameters: {
        regularizationStrength: 0.01,
        memoryReplayRatio: 0.2,
        plasticity: 0.8,
        stability: 0.7,
      },
      applicability: {
        fewShotLearning: 0.7,
        domainTransfer: 0.7,
        taskAdaptation: 0.8,
        continualLearning: 0.95,
      },
    });

    // Multi-Task Meta-Learning
    this.metaStrategies.set('multi_task_meta', {
      name: 'Multi-Task Meta-Learning',
      description: 'Learn shared representations across multiple tasks',
      type: 'multi_task_based',
      parameters: {
        sharedLayers: 3,
        taskSpecificLayers: 2,
        taskWeighting: 'equal',
        gradientNormalization: true,
      },
      applicability: {
        fewShotLearning: 0.8,
        domainTransfer: 0.8,
        taskAdaptation: 0.9,
        continualLearning: 0.8,
      },
    });
  }

  /**
   * Adapt configuration for agent based on meta-learning
   * @param {string} agentId - Agent identifier
   * @param {Object} config - Initial configuration
   */
  async adaptConfiguration(agentId, config) {
    // Get agent's learning history
    const experiences = this.agentExperiences.get(agentId) || [];

    if (experiences.length === 0) {
      // No prior experience, return base config
      return this.applyDefaultMetaLearning(config);
    }

    // Analyze learning patterns
    const learningPatterns = this.analyzeLearningPatterns(experiences);

    // Select appropriate meta-learning strategy
    const strategy = this.selectMetaLearningStrategy(learningPatterns, config);

    // Adapt configuration based on strategy
    const adaptedConfig = await this.applyMetaLearningStrategy(config, strategy, learningPatterns);

    console.log(`Applied meta-learning strategy '${strategy.name}' for agent ${agentId}`);

    return adaptedConfig;
  }

  /**
   * Apply default meta-learning configuration for new agents
   * @param {Object} config - Base configuration
   */
  applyDefaultMetaLearning(config) {
    // Apply conservative meta-learning defaults
    return {
      ...config,
      metaLearning: {
        enabled: true,
        strategy: 'maml',
        adaptiveRate: 0.01,
        experienceBuffer: 100,
        transferThreshold: 0.7,
      },
    };
  }

  /**
   * Analyze learning patterns from agent experiences
   * @param {Array} experiences - Agent's learning experiences
   */
  analyzeLearningPatterns(experiences) {
    const patterns = {
      learningSpeed: this.calculateLearningSpeed(experiences),
      convergenceStability: this.calculateConvergenceStability(experiences),
      domainVariability: this.calculateDomainVariability(experiences),
      taskComplexity: this.calculateAverageTaskComplexity(experiences),
      adaptationSuccess: this.calculateAdaptationSuccess(experiences),
      forgettingRate: this.calculateForgettingRate(experiences),
      transferEfficiency: this.calculateTransferEfficiency(experiences),
    };

    return patterns;
  }

  /**
   * Calculate learning speed from experiences
   * @param {Array} experiences - Learning experiences
   */
  calculateLearningSpeed(experiences) {
    if (experiences.length === 0) {
      return 0.5;
    }

    let totalSpeed = 0;
    let validExperiences = 0;

    for (const exp of experiences) {
      if (exp.metrics && exp.metrics.convergenceEpochs) {
        // Faster convergence = higher speed
        const speed = 1 / (1 + exp.metrics.convergenceEpochs / 10);
        totalSpeed += speed;
        validExperiences++;
      }
    }

    return validExperiences > 0 ? totalSpeed / validExperiences : 0.5;
  }

  /**
   * Calculate convergence stability
   * @param {Array} experiences - Learning experiences
   */
  calculateConvergenceStability(experiences) {
    if (experiences.length === 0) {
      return 0.5;
    }

    let totalStability = 0;
    let validExperiences = 0;

    for (const exp of experiences) {
      if (exp.metrics && exp.metrics.lossVariance !== undefined) {
        // Lower variance = higher stability
        const stability = 1 / (1 + exp.metrics.lossVariance);
        totalStability += stability;
        validExperiences++;
      }
    }

    return validExperiences > 0 ? totalStability / validExperiences : 0.5;
  }

  /**
   * Calculate domain variability across experiences
   * @param {Array} experiences - Learning experiences
   */
  calculateDomainVariability(experiences) {
    if (experiences.length === 0) {
      return 0.5;
    }

    const domains = new Set();

    for (const exp of experiences) {
      if (exp.domain) {
        domains.add(exp.domain);
      }
    }

    // Normalize by maximum expected domains
    return Math.min(1, domains.size / 10);
  }

  /**
   * Calculate average task complexity
   * @param {Array} experiences - Learning experiences
   */
  calculateAverageTaskComplexity(experiences) {
    if (experiences.length === 0) {
      return 0.5;
    }

    let totalComplexity = 0;
    let validExperiences = 0;

    for (const exp of experiences) {
      if (exp.taskComplexity !== undefined) {
        totalComplexity += exp.taskComplexity;
        validExperiences++;
      }
    }

    return validExperiences > 0 ? totalComplexity / validExperiences : 0.5;
  }

  /**
   * Calculate adaptation success rate
   * @param {Array} experiences - Learning experiences
   */
  calculateAdaptationSuccess(experiences) {
    if (experiences.length === 0) {
      return 0.5;
    }

    const successfulAdaptations = experiences.filter(exp =>
      exp.adaptationResult && exp.adaptationResult.success,
    ).length;

    return successfulAdaptations / experiences.length;
  }

  /**
   * Calculate forgetting rate
   * @param {Array} experiences - Learning experiences
   */
  calculateForgettingRate(experiences) {
    if (experiences.length < 2) {
      return 0.5;
    }

    let totalForgetting = 0;
    let validComparisons = 0;

    for (let i = 1; i < experiences.length; i++) {
      const prev = experiences[i - 1];
      const curr = experiences[i];

      if (prev.metrics && curr.metrics && prev.metrics.accuracy && curr.metrics.accuracy) {
        // If accuracy drops significantly when learning new task, high forgetting
        const forgetting = Math.max(0, prev.metrics.accuracy - curr.metrics.accuracy);
        totalForgetting += forgetting;
        validComparisons++;
      }
    }

    return validComparisons > 0 ? totalForgetting / validComparisons : 0.5;
  }

  /**
   * Calculate transfer learning efficiency
   * @param {Array} experiences - Learning experiences
   */
  calculateTransferEfficiency(experiences) {
    if (experiences.length === 0) {
      return 0.5;
    }

    const transferExperiences = experiences.filter(exp => exp.transferLearning);
    if (transferExperiences.length === 0) {
      return 0.5;
    }

    let totalEfficiency = 0;

    for (const exp of transferExperiences) {
      if (exp.transferLearning.efficiencyGain !== undefined) {
        totalEfficiency += exp.transferLearning.efficiencyGain;
      }
    }

    return transferExperiences.length > 0 ? totalEfficiency / transferExperiences.length : 0.5;
  }

  /**
   * Select appropriate meta-learning strategy
   * @param {Object} patterns - Learning patterns
   * @param {Object} config - Configuration
   */
  selectMetaLearningStrategy(patterns, config) {
    let bestStrategy = null;
    let bestScore = 0;

    // Define task characteristics
    const taskCharacteristics = this.inferTaskCharacteristics(patterns, config);

    for (const [strategyName, strategy] of this.metaStrategies.entries()) {
      let score = 0;

      // Score based on applicability to current task characteristics
      if (taskCharacteristics.fewShot) {
        score += strategy.applicability.fewShotLearning * 0.3;
      }

      if (taskCharacteristics.domainTransfer) {
        score += strategy.applicability.domainTransfer * 0.3;
      }

      if (taskCharacteristics.taskAdaptation) {
        score += strategy.applicability.taskAdaptation * 0.2;
      }

      if (taskCharacteristics.continualLearning) {
        score += strategy.applicability.continualLearning * 0.2;
      }

      // Adjust score based on learning patterns
      if (patterns.learningSpeed < 0.3 && strategy.type === 'gradient_based') {
        score += 0.1; // Boost gradient-based methods for slow learners
      }

      if (patterns.forgettingRate > 0.7 && strategy.type === 'memory_based') {
        score += 0.2; // Boost memory-based methods for high forgetting
      }

      if (patterns.domainVariability > 0.6 && strategy.type === 'domain_based') {
        score += 0.15; // Boost domain adaptation for high variability
      }

      if (score > bestScore) {
        bestScore = score;
        bestStrategy = strategy;
      }
    }

    return bestStrategy || this.metaStrategies.get('maml');
  }

  /**
   * Infer task characteristics from patterns and config
   * @param {Object} patterns - Learning patterns
   * @param {Object} config - Configuration
   */
  inferTaskCharacteristics(patterns, config) {
    return {
      fewShot: patterns.learningSpeed < 0.4 || config.dataSize < 1000,
      domainTransfer: patterns.domainVariability > 0.5,
      taskAdaptation: patterns.adaptationSuccess < 0.6,
      continualLearning: patterns.forgettingRate > 0.5,
    };
  }

  /**
   * Apply meta-learning strategy to configuration
   * @param {Object} config - Base configuration
   * @param {Object} strategy - Selected strategy
   * @param {Object} patterns - Learning patterns
   */
  async applyMetaLearningStrategy(config, strategy, patterns) {
    const adaptedConfig = { ...config };

    // Apply strategy-specific adaptations
    switch (strategy.type) {
    case 'gradient_based':
      adaptedConfig.metaLearning = this.applyGradientBasedMeta(strategy, patterns);
      break;

    case 'metric_based':
      adaptedConfig.metaLearning = this.applyMetricBasedMeta(strategy, patterns);
      break;

    case 'memory_based':
      adaptedConfig.metaLearning = this.applyMemoryBasedMeta(strategy, patterns);
      break;

    case 'optimization_based':
      adaptedConfig.metaLearning = this.applyOptimizationBasedMeta(strategy, patterns);
      break;

    case 'domain_based':
      adaptedConfig.metaLearning = this.applyDomainBasedMeta(strategy, patterns);
      break;

    case 'continual_based':
      adaptedConfig.metaLearning = this.applyContinualBasedMeta(strategy, patterns);
      break;

    case 'multi_task_based':
      adaptedConfig.metaLearning = this.applyMultiTaskBasedMeta(strategy, patterns);
      break;
    }

    // Add common meta-learning properties
    adaptedConfig.metaLearning.strategyName = strategy.name;
    adaptedConfig.metaLearning.enabled = true;
    adaptedConfig.metaLearning.adaptiveThreshold = this.calculateAdaptiveThreshold(patterns);

    return adaptedConfig;
  }

  /**
   * Apply gradient-based meta-learning configuration
   * @param {Object} strategy - Strategy configuration
   * @param {Object} patterns - Learning patterns
   */
  applyGradientBasedMeta(strategy, patterns) {
    const config = { ...strategy.parameters };

    // Adapt inner learning rate based on learning speed
    if (patterns.learningSpeed < 0.3) {
      config.innerLearningRate *= 1.5; // Increase for slow learners
    } else if (patterns.learningSpeed > 0.7) {
      config.innerLearningRate *= 0.7; // Decrease for fast learners
    }

    // Adapt inner steps based on convergence stability
    if (patterns.convergenceStability < 0.4) {
      config.innerSteps = Math.max(3, config.innerSteps - 2);
    } else if (patterns.convergenceStability > 0.8) {
      config.innerSteps = Math.min(10, config.innerSteps + 3);
    }

    return { type: 'gradient_based', ...config };
  }

  /**
   * Apply metric-based meta-learning configuration
   * @param {Object} strategy - Strategy configuration
   * @param {Object} patterns - Learning patterns
   */
  applyMetricBasedMeta(strategy, patterns) {
    const config = { ...strategy.parameters };

    // Adapt embedding dimension based on task complexity
    if (patterns.taskComplexity > 0.7) {
      config.embeddingDim = Math.min(128, config.embeddingDim * 1.5);
    } else if (patterns.taskComplexity < 0.3) {
      config.embeddingDim = Math.max(32, config.embeddingDim * 0.7);
    }

    // Adapt temperature based on convergence stability
    if (patterns.convergenceStability < 0.5) {
      config.temperatureScale = Math.max(0.5, config.temperatureScale - 0.2);
    }

    return { type: 'metric_based', ...config };
  }

  /**
   * Apply memory-based meta-learning configuration
   * @param {Object} strategy - Strategy configuration
   * @param {Object} patterns - Learning patterns
   */
  applyMemoryBasedMeta(strategy, patterns) {
    const config = { ...strategy.parameters };

    // Increase memory size for high forgetting rate
    if (patterns.forgettingRate > 0.6) {
      config.memorySize = Math.min(256, config.memorySize * 1.5);
    }

    // Adjust read/write heads based on domain variability
    if (patterns.domainVariability > 0.5) {
      config.readHeads = Math.min(4, config.readHeads + 1);
      config.writeHeads = Math.min(2, config.writeHeads + 1);
    }

    return { type: 'memory_based', ...config };
  }

  /**
   * Apply optimization-based meta-learning configuration
   * @param {Object} strategy - Strategy configuration
   * @param {Object} patterns - Learning patterns
   */
  applyOptimizationBasedMeta(strategy, patterns) {
    const config = { ...strategy.parameters };

    // Adapt optimizer based on learning speed
    if (patterns.learningSpeed < 0.4) {
      config.optimizerHiddenSize = Math.min(40, config.optimizerHiddenSize * 1.3);
    }

    // Enable coordinate-wise optimization for complex tasks
    if (patterns.taskComplexity > 0.6) {
      config.coordinatewise = true;
    }

    return { type: 'optimization_based', ...config };
  }

  /**
   * Apply domain-based meta-learning configuration
   * @param {Object} strategy - Strategy configuration
   * @param {Object} patterns - Learning patterns
   */
  applyDomainBasedMeta(strategy, patterns) {
    const config = { ...strategy.parameters };

    // Strengthen domain discriminator for high domain variability
    if (patterns.domainVariability > 0.7) {
      config.domainDiscriminatorStrength *= 1.3;
      config.gradientReversalLambda *= 1.2;
    }

    // Increase adaptation steps for low transfer efficiency
    if (patterns.transferEfficiency < 0.4) {
      config.adaptationSteps = Math.min(50, config.adaptationSteps * 1.5);
    }

    return { type: 'domain_based', ...config };
  }

  /**
   * Apply continual-based meta-learning configuration
   * @param {Object} strategy - Strategy configuration
   * @param {Object} patterns - Learning patterns
   */
  applyContinualBasedMeta(strategy, patterns) {
    const config = { ...strategy.parameters };

    // Increase regularization for high forgetting
    if (patterns.forgettingRate > 0.6) {
      config.regularizationStrength *= 1.4;
      config.stability = Math.min(0.9, config.stability + 0.1);
    }

    // Increase memory replay for domain variability
    if (patterns.domainVariability > 0.5) {
      config.memoryReplayRatio = Math.min(0.4, config.memoryReplayRatio + 0.1);
    }

    return { type: 'continual_based', ...config };
  }

  /**
   * Apply multi-task based meta-learning configuration
   * @param {Object} strategy - Strategy configuration
   * @param {Object} patterns - Learning patterns
   */
  applyMultiTaskBasedMeta(strategy, patterns) {
    const config = { ...strategy.parameters };

    // Adjust shared layers based on transfer efficiency
    if (patterns.transferEfficiency > 0.7) {
      config.sharedLayers = Math.min(5, config.sharedLayers + 1);
    } else if (patterns.transferEfficiency < 0.3) {
      config.taskSpecificLayers = Math.min(4, config.taskSpecificLayers + 1);
    }

    // Enable gradient normalization for stability
    if (patterns.convergenceStability < 0.5) {
      config.gradientNormalization = true;
    }

    return { type: 'multi_task_based', ...config };
  }

  /**
   * Calculate adaptive threshold based on patterns
   * @param {Object} patterns - Learning patterns
   */
  calculateAdaptiveThreshold(patterns) {
    // Base threshold adjusted by learning characteristics
    let threshold = 0.7;

    if (patterns.learningSpeed < 0.3) {
      threshold -= 0.1;
    } // Lower threshold for slow learners
    if (patterns.adaptationSuccess < 0.5) {
      threshold -= 0.05;
    } // Lower threshold for poor adapters
    if (patterns.forgettingRate > 0.6) {
      threshold += 0.1;
    } // Higher threshold if prone to forgetting

    return Math.max(0.3, Math.min(0.9, threshold));
  }

  /**
   * Optimize training parameters using meta-learning
   * @param {string} agentId - Agent identifier
   * @param {Object} options - Training options
   */
  async optimizeTraining(agentId, options) {
    const experiences = this.agentExperiences.get(agentId) || [];

    if (experiences.length === 0) {
      return options; // No optimization without experience
    }

    const patterns = this.analyzeLearningPatterns(experiences);
    const optimizedOptions = { ...options };

    // Optimize learning rate
    optimizedOptions.learningRate = this.optimizeLearningRate(patterns, options.learningRate);

    // Optimize batch size
    optimizedOptions.batchSize = this.optimizeBatchSize(patterns, options.batchSize);

    // Optimize epochs
    optimizedOptions.epochs = this.optimizeEpochs(patterns, options.epochs);

    // Add meta-learning specific optimizations
    optimizedOptions.metaOptimizations = {
      warmupEpochs: this.calculateWarmupEpochs(patterns),
      schedulerType: this.selectSchedulerType(patterns),
      regularizationStrength: this.optimizeRegularization(patterns),
      earlyStoppingPatience: this.optimizeEarlyStopping(patterns),
    };

    console.log(`Optimized training parameters for agent ${agentId} based on meta-learning`);

    return optimizedOptions;
  }

  /**
   * Optimize learning rate based on patterns
   * @param {Object} patterns - Learning patterns
   * @param {number} baseLR - Base learning rate
   */
  optimizeLearningRate(patterns, baseLR) {
    let multiplier = 1.0;

    // Adjust based on learning speed
    if (patterns.learningSpeed < 0.3) {
      multiplier *= 1.3; // Increase LR for slow learners
    } else if (patterns.learningSpeed > 0.7) {
      multiplier *= 0.8; // Decrease LR for fast learners
    }

    // Adjust based on convergence stability
    if (patterns.convergenceStability < 0.4) {
      multiplier *= 0.7; // Lower LR for unstable convergence
    }

    return baseLR * multiplier;
  }

  /**
   * Optimize batch size based on patterns
   * @param {Object} patterns - Learning patterns
   * @param {number} baseBatchSize - Base batch size
   */
  optimizeBatchSize(patterns, baseBatchSize) {
    let multiplier = 1.0;

    // Adjust based on convergence stability
    if (patterns.convergenceStability < 0.4) {
      multiplier *= 1.5; // Larger batches for stability
    } else if (patterns.convergenceStability > 0.8) {
      multiplier *= 0.8; // Smaller batches for exploration
    }

    // Adjust based on task complexity
    if (patterns.taskComplexity > 0.7) {
      multiplier *= 0.7; // Smaller batches for complex tasks
    }

    const optimizedSize = Math.round(baseBatchSize * multiplier);
    return Math.max(1, Math.min(256, optimizedSize)); // Clamp to reasonable range
  }

  /**
   * Optimize number of epochs based on patterns
   * @param {Object} patterns - Learning patterns
   * @param {number} baseEpochs - Base number of epochs
   */
  optimizeEpochs(patterns, baseEpochs) {
    let multiplier = 1.0;

    // Adjust based on learning speed
    if (patterns.learningSpeed < 0.3) {
      multiplier *= 1.5; // More epochs for slow learners
    } else if (patterns.learningSpeed > 0.7) {
      multiplier *= 0.7; // Fewer epochs for fast learners
    }

    // Adjust based on forgetting rate
    if (patterns.forgettingRate > 0.6) {
      multiplier *= 0.8; // Fewer epochs to avoid overfitting
    }

    const optimizedEpochs = Math.round(baseEpochs * multiplier);
    return Math.max(1, Math.min(200, optimizedEpochs)); // Clamp to reasonable range
  }

  /**
   * Calculate optimal warmup epochs
   * @param {Object} patterns - Learning patterns
   */
  calculateWarmupEpochs(patterns) {
    let warmupEpochs = 0;

    // Use warmup for unstable convergence
    if (patterns.convergenceStability < 0.5) {
      warmupEpochs = Math.ceil(5 * (1 - patterns.convergenceStability));
    }

    return Math.max(0, Math.min(10, warmupEpochs));
  }

  /**
   * Select learning rate scheduler type
   * @param {Object} patterns - Learning patterns
   */
  selectSchedulerType(patterns) {
    if (patterns.convergenceStability < 0.4) {
      return 'cosine_annealing'; // Smooth schedule for unstable training
    } else if (patterns.learningSpeed < 0.3) {
      return 'exponential_decay'; // Gradual reduction for slow learners
    } else if (patterns.taskComplexity > 0.7) {
      return 'step_decay'; // Stepwise reduction for complex tasks
    }
    return 'constant'; // Keep constant for stable cases

  }

  /**
   * Optimize regularization strength
   * @param {Object} patterns - Learning patterns
   */
  optimizeRegularization(patterns) {
    let baseStrength = 0.01;

    // Increase regularization for high task complexity
    if (patterns.taskComplexity > 0.6) {
      baseStrength *= 1.5;
    }

    // Increase regularization for low convergence stability
    if (patterns.convergenceStability < 0.5) {
      baseStrength *= 1.3;
    }

    // Decrease regularization for high forgetting rate (may be overregularized)
    if (patterns.forgettingRate > 0.7) {
      baseStrength *= 0.7;
    }

    return Math.max(0.001, Math.min(0.1, baseStrength));
  }

  /**
   * Optimize early stopping patience
   * @param {Object} patterns - Learning patterns
   */
  optimizeEarlyStopping(patterns) {
    let basePatienceEpochs = 10;

    // Increase patience for slow learners
    if (patterns.learningSpeed < 0.3) {
      basePatienceEpochs *= 1.5;
    }

    // Decrease patience for fast learners
    if (patterns.learningSpeed > 0.7) {
      basePatienceEpochs *= 0.7;
    }

    // Increase patience for unstable convergence
    if (patterns.convergenceStability < 0.4) {
      basePatienceEpochs *= 1.3;
    }

    return Math.max(3, Math.min(25, Math.round(basePatienceEpochs)));
  }

  /**
   * Extract experiences from agent for meta-learning
   * @param {string} agentId - Agent identifier
   */
  async extractExperiences(agentId) {
    return this.agentExperiences.get(agentId) || [];
  }

  /**
   * Record learning experience for meta-learning
   * @param {string} agentId - Agent identifier
   * @param {Object} experience - Learning experience
   */
  recordExperience(agentId, experience) {
    if (!this.agentExperiences.has(agentId)) {
      this.agentExperiences.set(agentId, []);
    }

    const experiences = this.agentExperiences.get(agentId);

    // Add timestamp and unique ID
    const enrichedExperience = {
      ...experience,
      timestamp: Date.now(),
      id: `exp_${agentId}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
    };

    experiences.push(enrichedExperience);

    // Keep only recent experiences (last 100)
    if (experiences.length > 100) {
      experiences.splice(0, experiences.length - 100);
    }

    // Update learning metrics
    this.updateLearningMetrics(agentId, enrichedExperience);
  }

  /**
   * Update learning metrics for agent
   * @param {string} agentId - Agent identifier
   * @param {Object} experience - Learning experience
   */
  updateLearningMetrics(agentId, experience) {
    if (!this.learningMetrics.has(agentId)) {
      this.learningMetrics.set(agentId, {
        totalExperiences: 0,
        averageLearningTime: 0,
        averageAccuracy: 0,
        adaptationSuccessRate: 0,
        domainTransferCount: 0,
        lastUpdate: Date.now(),
      });
    }

    const metrics = this.learningMetrics.get(agentId);

    metrics.totalExperiences++;
    metrics.lastUpdate = Date.now();

    // Update running averages
    if (experience.metrics) {
      if (experience.metrics.trainingTime) {
        metrics.averageLearningTime = this.updateRunningAverage(
          metrics.averageLearningTime,
          experience.metrics.trainingTime,
          metrics.totalExperiences,
        );
      }

      if (experience.metrics.accuracy) {
        metrics.averageAccuracy = this.updateRunningAverage(
          metrics.averageAccuracy,
          experience.metrics.accuracy,
          metrics.totalExperiences,
        );
      }
    }

    // Update success rate
    if (experience.adaptationResult) {
      const successCount = metrics.adaptationSuccessRate * (metrics.totalExperiences - 1);
      const newSuccess = experience.adaptationResult.success ? 1 : 0;
      metrics.adaptationSuccessRate = (successCount + newSuccess) / metrics.totalExperiences;
    }

    // Count domain transfers
    if (experience.transferLearning) {
      metrics.domainTransferCount++;
    }
  }

  /**
   * Update running average
   * @param {number} currentAvg - Current average
   * @param {number} newValue - New value
   * @param {number} count - Total count
   */
  updateRunningAverage(currentAvg, newValue, count) {
    return currentAvg + (newValue - currentAvg) / count;
  }

  /**
   * Perform domain adaptation using meta-learning
   * @param {string} agentId - Agent identifier
   * @param {Object} sourceData - Source domain data
   * @param {Object} targetData - Target domain data
   */
  async performDomainAdaptation(agentId, sourceData, targetData) {
    // Analyze domain shift
    const domainShift = this.analyzeDomainShift(sourceData, targetData);

    // Select adaptation strategy
    const adaptationStrategy = this.selectAdaptationStrategy(domainShift);

    // Apply domain adaptation
    const adaptationResult = await this.applyDomainAdaptation(
      agentId,
      adaptationStrategy,
      sourceData,
      targetData,
    );

    // Record domain adaptation experience
    this.recordExperience(agentId, {
      type: 'domain_adaptation',
      sourceData: this.summarizeData(sourceData),
      targetData: this.summarizeData(targetData),
      domainShift,
      adaptationStrategy,
      adaptationResult,
      transferLearning: {
        enabled: true,
        efficiencyGain: adaptationResult.efficiencyGain || 0,
      },
    });

    return adaptationResult;
  }

  /**
   * Analyze domain shift between source and target
   * @param {Object} sourceData - Source domain data
   * @param {Object} targetData - Target domain data
   */
  analyzeDomainShift(sourceData, targetData) {
    return {
      distributionShift: this.calculateDistributionShift(sourceData, targetData),
      featureShift: this.calculateFeatureShift(sourceData, targetData),
      labelShift: this.calculateLabelShift(sourceData, targetData),
      marginalShift: this.calculateMarginalShift(sourceData, targetData),
      conditionalShift: this.calculateConditionalShift(sourceData, targetData),
    };
  }

  /**
   * Calculate distribution shift between domains
   * @param {Object} sourceData - Source domain data
   * @param {Object} targetData - Target domain data
   */
  calculateDistributionShift(sourceData, targetData) {
    // Simplified distribution shift calculation
    if (!sourceData.samples || !targetData.samples) {
      return 0.5;
    }

    // Calculate basic statistics for both domains
    const sourceStats = this.calculateDataStatistics(sourceData.samples);
    const targetStats = this.calculateDataStatistics(targetData.samples);

    // Calculate shift as difference in statistics
    const meanShift = Math.abs(sourceStats.mean - targetStats.mean);
    const varianceShift = Math.abs(sourceStats.variance - targetStats.variance);

    return Math.min(1, (meanShift + varianceShift) / 2);
  }

  /**
   * Calculate basic data statistics
   * @param {Array} samples - Data samples
   */
  calculateDataStatistics(samples) {
    if (samples.length === 0) {
      return { mean: 0, variance: 0 };
    }

    // Flatten samples to get all numeric values
    const values = samples.flat().filter(v => typeof v === 'number');

    if (values.length === 0) {
      return { mean: 0, variance: 0 };
    }

    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;

    return { mean, variance };
  }

  /**
   * Calculate feature shift (simplified)
   * @param {Object} sourceData - Source domain data
   * @param {Object} targetData - Target domain data
   */
  calculateFeatureShift(sourceData, targetData) {
    // Simplified feature shift - compare feature dimensions
    const sourceDim = this.getFeatureDimensions(sourceData);
    const targetDim = this.getFeatureDimensions(targetData);

    if (sourceDim === 0 || targetDim === 0) {
      return 0.5;
    }

    return Math.abs(sourceDim - targetDim) / Math.max(sourceDim, targetDim);
  }

  /**
   * Get feature dimensions from data
   * @param {Object} data - Data object
   */
  getFeatureDimensions(data) {
    if (!data.samples || data.samples.length === 0) {
      return 0;
    }

    const sample = data.samples[0];
    if (Array.isArray(sample)) {
      return sample.length;
    }
    if (typeof sample === 'object' && sample.input) {
      return Array.isArray(sample.input) ? sample.input.length : 1;
    }

    return 1;
  }

  /**
   * Calculate label shift (simplified)
   * @param {Object} sourceData - Source domain data
   * @param {Object} targetData - Target domain data
   */
  calculateLabelShift(sourceData, targetData) {
    // Compare label distributions
    const sourceLabels = this.extractLabels(sourceData);
    const targetLabels = this.extractLabels(targetData);

    if (sourceLabels.size === 0 || targetLabels.size === 0) {
      return 0.5;
    }

    const intersection = new Set([...sourceLabels].filter(x => targetLabels.has(x)));
    const union = new Set([...sourceLabels, ...targetLabels]);

    return 1 - (intersection.size / union.size); // Jaccard distance
  }

  /**
   * Extract unique labels from data
   * @param {Object} data - Data object
   */
  extractLabels(data) {
    const labels = new Set();

    if (data.samples) {
      data.samples.forEach(sample => {
        if (sample.label !== undefined) {
          labels.add(sample.label);
        }
        if (sample.target !== undefined) {
          labels.add(sample.target);
        }
      });
    }

    return labels;
  }

  /**
   * Calculate marginal shift (simplified)
   * @param {Object} sourceData - Source domain data
   * @param {Object} targetData - Target domain data
   */
  calculateMarginalShift(sourceData, targetData) {
    // Simplified marginal shift calculation
    return this.calculateDistributionShift(sourceData, targetData);
  }

  /**
   * Calculate conditional shift (simplified)
   * @param {Object} sourceData - Source domain data
   * @param {Object} targetData - Target domain data
   */
  calculateConditionalShift(sourceData, targetData) {
    // Simplified conditional shift calculation
    const featureShift = this.calculateFeatureShift(sourceData, targetData);
    const labelShift = this.calculateLabelShift(sourceData, targetData);

    return (featureShift + labelShift) / 2;
  }

  /**
   * Select appropriate domain adaptation strategy
   * @param {Object} domainShift - Domain shift analysis
   */
  selectAdaptationStrategy(domainShift) {
    const { distributionShift, featureShift, labelShift } = domainShift;

    if (distributionShift > 0.7) {
      return 'adversarial_adaptation';
    } else if (featureShift > 0.6) {
      return 'feature_alignment';
    } else if (labelShift > 0.5) {
      return 'label_adaptation';
    }
    return 'fine_tuning';

  }

  /**
   * Apply domain adaptation strategy
   * @param {string} agentId - Agent identifier
   * @param {string} strategy - Adaptation strategy
   * @param {Object} sourceData - Source domain data
   * @param {Object} targetData - Target domain data
   */
  async applyDomainAdaptation(agentId, strategy, sourceData, targetData) {
    console.log(`Applying domain adaptation strategy '${strategy}' for agent ${agentId}`);

    // Simulate domain adaptation (in practice, would involve actual training)
    const adaptationResult = {
      strategy,
      success: Math.random() > 0.3, // 70% success rate simulation
      efficiencyGain: Math.random() * 0.4 + 0.1, // 10-50% efficiency gain
      accuracyImprovement: Math.random() * 0.2 + 0.05, // 5-25% accuracy improvement
      adaptationTime: Math.random() * 100 + 50, // 50-150 time units
      transferredKnowledge: this.calculateTransferredKnowledge(sourceData, targetData),
    };

    // Store adaptation in transfer learning map
    if (!this.transferLearning.has(agentId)) {
      this.transferLearning.set(agentId, []);
    }

    this.transferLearning.get(agentId).push({
      timestamp: Date.now(),
      strategy,
      result: adaptationResult,
      sourceDataSummary: this.summarizeData(sourceData),
      targetDataSummary: this.summarizeData(targetData),
    });

    return adaptationResult;
  }

  /**
   * Calculate amount of knowledge transferred
   * @param {Object} sourceData - Source domain data
   * @param {Object} targetData - Target domain data
   */
  calculateTransferredKnowledge(sourceData, targetData) {
    // Simplified calculation based on data similarity
    const similarity = 1 - this.calculateDistributionShift(sourceData, targetData);
    return Math.max(0.1, similarity * 0.8); // 10-80% knowledge transfer
  }

  /**
   * Summarize data for storage
   * @param {Object} data - Data to summarize
   */
  summarizeData(data) {
    return {
      sampleCount: data.samples ? data.samples.length : 0,
      featureDimensions: this.getFeatureDimensions(data),
      uniqueLabels: this.extractLabels(data).size,
      dataType: this.inferDataType(data),
    };
  }

  /**
   * Infer data type from samples
   * @param {Object} data - Data object
   */
  inferDataType(data) {
    if (!data.samples || data.samples.length === 0) {
      return 'unknown';
    }

    const sample = data.samples[0];

    if (Array.isArray(sample)) {
      return sample.length > 100 ? 'image' : 'vector';
    }

    if (typeof sample === 'object') {
      if (sample.sequence) {
        return 'sequence';
      }
      if (sample.text) {
        return 'text';
      }
      if (sample.image) {
        return 'image';
      }
    }

    return 'scalar';
  }

  /**
   * Get meta-learning statistics
   */
  getStatistics() {
    const totalAgents = this.agentExperiences.size;
    let totalExperiences = 0;
    let totalAdaptations = 0;
    let avgSuccessRate = 0;

    for (const [agentId, experiences] of this.agentExperiences.entries()) {
      totalExperiences += experiences.length;

      const adaptations = experiences.filter(exp => exp.type === 'domain_adaptation');
      totalAdaptations += adaptations.length;

      const metrics = this.learningMetrics.get(agentId);
      if (metrics) {
        avgSuccessRate += metrics.adaptationSuccessRate;
      }
    }

    return {
      totalAgents,
      totalExperiences,
      totalAdaptations,
      avgExperiencesPerAgent: totalAgents > 0 ? totalExperiences / totalAgents : 0,
      avgSuccessRate: totalAgents > 0 ? avgSuccessRate / totalAgents : 0,
      availableStrategies: this.metaStrategies.size,
      transferLearningInstances: this.transferLearning.size,
    };
  }

  /**
   * Preserve meta-learning state for agent
   * @param {string} agentId - Agent identifier
   */
  async preserveState(agentId) {
    return {
      experiences: this.agentExperiences.get(agentId) || [],
      domainAdaptations: this.domainAdaptations.get(agentId) || [],
      transferLearning: this.transferLearning.get(agentId) || [],
      learningMetrics: this.learningMetrics.get(agentId) || null,
    };
  }

  /**
   * Restore meta-learning state for agent
   * @param {string} agentId - Agent identifier
   * @param {Object} state - Preserved state
   */
  async restoreState(agentId, state) {
    if (state.experiences) {
      this.agentExperiences.set(agentId, state.experiences);
    }

    if (state.domainAdaptations) {
      this.domainAdaptations.set(agentId, state.domainAdaptations);
    }

    if (state.transferLearning) {
      this.transferLearning.set(agentId, state.transferLearning);
    }

    if (state.learningMetrics) {
      this.learningMetrics.set(agentId, state.learningMetrics);
    }
  }
}

export { MetaLearningFramework };