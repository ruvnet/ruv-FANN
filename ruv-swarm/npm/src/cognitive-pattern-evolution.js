/**
 * Cognitive Pattern Evolution System
 * Enables autonomous learning and adaptation of cognitive patterns
 */

class CognitivePatternEvolution {
  constructor() {
    this.agentPatterns = new Map();
    this.evolutionHistory = new Map();
    this.patternTemplates = new Map();
    this.crossAgentPatterns = new Map();
    this.evolutionMetrics = new Map();

    // Initialize base cognitive pattern templates
    this.initializePatternTemplates();
  }

  /**
   * Initialize base cognitive pattern templates
   */
  initializePatternTemplates() {
    // Convergent thinking patterns
    this.patternTemplates.set('convergent', {
      name: 'Convergent Thinking',
      description: 'Focus on single optimal solutions',
      characteristics: {
        searchStrategy: 'directed',
        explorationRate: 0.1,
        exploitationRate: 0.9,
        decisionMaking: 'decisive',
        patternRecognition: 'exact_match',
      },
      adaptationRules: {
        increasePrecision: (context) => context.accuracy > 0.8,
        reduceExploration: (context) => context.confidence > 0.7,
        focusAttention: (context) => context.taskComplexity < 0.5,
      },
    });

    // Divergent thinking patterns
    this.patternTemplates.set('divergent', {
      name: 'Divergent Thinking',
      description: 'Explore multiple creative solutions',
      characteristics: {
        searchStrategy: 'random',
        explorationRate: 0.8,
        exploitationRate: 0.2,
        decisionMaking: 'exploratory',
        patternRecognition: 'flexible_match',
      },
      adaptationRules: {
        increaseCreativity: (context) => context.noveltyScore > 0.6,
        expandSearch: (context) => context.solutionDiversity < 0.5,
        encourageRisk: (context) => context.safetyMargin > 0.8,
      },
    });

    // Lateral thinking patterns
    this.patternTemplates.set('lateral', {
      name: 'Lateral Thinking',
      description: 'Approach problems from unexpected angles',
      characteristics: {
        searchStrategy: 'lateral',
        explorationRate: 0.6,
        exploitationRate: 0.4,
        decisionMaking: 'innovative',
        patternRecognition: 'analogical',
      },
      adaptationRules: {
        seekAlternatives: (context) => context.standardSolutionFailed,
        useAnalogies: (context) => context.domainKnowledge > 0.5,
        breakAssumptions: (context) => context.progressStalled,
      },
    });

    // Systems thinking patterns
    this.patternTemplates.set('systems', {
      name: 'Systems Thinking',
      description: 'Consider holistic interconnections and emergent properties',
      characteristics: {
        searchStrategy: 'holistic',
        explorationRate: 0.4,
        exploitationRate: 0.6,
        decisionMaking: 'systemic',
        patternRecognition: 'pattern_networks',
      },
      adaptationRules: {
        mapConnections: (context) => context.systemComplexity > 0.7,
        identifyFeedback: (context) => context.iterationCount > 5,
        emergentProperties: (context) => context.componentInteractions > 0.6,
      },
    });

    // Critical thinking patterns
    this.patternTemplates.set('critical', {
      name: 'Critical Thinking',
      description: 'Systematic evaluation and logical analysis',
      characteristics: {
        searchStrategy: 'systematic',
        explorationRate: 0.3,
        exploitationRate: 0.7,
        decisionMaking: 'analytical',
        patternRecognition: 'evidence_based',
      },
      adaptationRules: {
        validateEvidence: (context) => context.informationQuality < 0.8,
        checkBias: (context) => context.subjectivity > 0.5,
        logicalConsistency: (context) => context.contradictions > 0.2,
      },
    });

    // Abstract thinking patterns
    this.patternTemplates.set('abstract', {
      name: 'Abstract Thinking',
      description: 'Work with concepts, principles, and generalizations',
      characteristics: {
        searchStrategy: 'conceptual',
        explorationRate: 0.5,
        exploitationRate: 0.5,
        decisionMaking: 'principled',
        patternRecognition: 'abstraction_layers',
      },
      adaptationRules: {
        generalizePatterns: (context) => context.specificExamples > 3,
        identifyPrinciples: (context) => context.abstraction_level < 0.6,
        conceptualMapping: (context) => context.domainTransfer > 0.4,
      },
    });
  }

  /**
   * Initialize agent with cognitive patterns
   * @param {string} agentId - Agent identifier
   * @param {Object} config - Agent configuration
   */
  async initializeAgent(agentId, config) {
    const initialPatterns = this.selectInitialPatterns(config);

    this.agentPatterns.set(agentId, {
      activePatterns: initialPatterns,
      dominantPattern: initialPatterns[0] || 'convergent',
      adaptationHistory: [],
      evolutionScore: 0,
      lastEvolution: Date.now(),
      crossAgentLearning: new Map(),
      specializations: new Set(),
    });

    this.evolutionHistory.set(agentId, []);
    this.evolutionMetrics.set(agentId, {
      totalEvolutions: 0,
      successfulAdaptations: 0,
      patternSwitches: 0,
      crossAgentTransfers: 0,
      emergentPatterns: 0,
    });

    console.log(`Initialized cognitive patterns for agent ${agentId}: ${initialPatterns.join(', ')}`);
  }

  /**
   * Select initial cognitive patterns based on configuration
   * @param {Object} config - Agent configuration
   */
  selectInitialPatterns(config) {
    const patterns = [];

    // Select patterns based on model type and use case
    if (config.modelType) {
      switch (config.modelType) {
      case 'transformer':
      case 'lstm':
      case 'gru':
        patterns.push('convergent', 'systems');
        break;
      case 'cnn':
      case 'resnet':
        patterns.push('critical', 'abstract');
        break;
      case 'gnn':
      case 'gat':
        patterns.push('systems', 'lateral');
        break;
      case 'vae':
      case 'autoencoder':
        patterns.push('divergent', 'abstract');
        break;
      case 'diffusion_model':
      case 'neural_ode':
        patterns.push('divergent', 'lateral');
        break;
      default:
        patterns.push('convergent', 'critical');
      }
    }

    // Add patterns based on task characteristics
    if (config.template) {
      if (config.template.includes('analyzer')) {
        patterns.push('critical');
      }
      if (config.template.includes('generator')) {
        patterns.push('divergent');
      }
      if (config.template.includes('processor')) {
        patterns.push('systems');
      }
      if (config.template.includes('learner')) {
        patterns.push('abstract');
      }
    }

    // Ensure at least one pattern
    if (patterns.length === 0) {
      patterns.push('convergent');
    }

    return [...new Set(patterns)]; // Remove duplicates
  }

  /**
   * Evolve cognitive patterns based on training data and performance
   * @param {string} agentId - Agent identifier
   * @param {Object} trainingData - Training data context
   */
  async evolvePatterns(agentId, trainingData) {
    const agentData = this.agentPatterns.get(agentId);
    if (!agentData) {
      return;
    }

    const context = this.analyzeTrainingContext(trainingData);
    const currentPatterns = agentData.activePatterns;

    // Evaluate current pattern effectiveness
    const patternEffectiveness = await this.evaluatePatternEffectiveness(agentId, context);

    // Determine if evolution is needed
    const evolutionNeed = this.assessEvolutionNeed(patternEffectiveness, context);

    if (evolutionNeed.required) {
      const evolutionStrategy = this.selectEvolutionStrategy(evolutionNeed, context);
      const newPatterns = await this.applyEvolution(agentId, evolutionStrategy, context);

      // Record evolution
      this.recordEvolution(agentId, {
        timestamp: Date.now(),
        trigger: evolutionNeed.reason,
        strategy: evolutionStrategy,
        oldPatterns: [...currentPatterns],
        newPatterns,
        context,
        effectiveness: patternEffectiveness,
      });

      console.log(`Evolved cognitive patterns for agent ${agentId}: ${evolutionStrategy.type}`);
    }
  }

  /**
   * Analyze training context to understand cognitive requirements
   * @param {Object} trainingData - Training data
   */
  analyzeTrainingContext(trainingData) {
    const context = {
      dataComplexity: this.calculateDataComplexity(trainingData),
      taskType: this.inferTaskType(trainingData),
      noiseLevel: this.estimateNoiseLevel(trainingData),
      patternRegularity: this.assessPatternRegularity(trainingData),
      dimensionality: this.calculateDimensionality(trainingData),
      temporalDependency: this.assessTemporalDependency(trainingData),
      abstractionLevel: this.estimateAbstractionLevel(trainingData),
      creativity_required: this.assessCreativityRequirement(trainingData),
    };

    return context;
  }

  /**
   * Calculate data complexity score
   * @param {Object} trainingData - Training data
   */
  calculateDataComplexity(trainingData) {
    if (!trainingData.samples || trainingData.samples.length === 0) {
      return 0.5;
    }

    const sampleSize = trainingData.samples.length;
    const featureVariance = this.calculateFeatureVariance(trainingData.samples);
    const labelDistribution = this.calculateLabelDistribution(trainingData.samples);

    // Combine metrics for overall complexity
    const sizeComplexity = Math.min(1, sampleSize / 10000);
    const varianceComplexity = Math.min(1, featureVariance);
    const distributionComplexity = labelDistribution;

    return (sizeComplexity + varianceComplexity + distributionComplexity) / 3;
  }

  /**
   * Calculate feature variance across samples
   * @param {Array} samples - Training samples
   */
  calculateFeatureVariance(samples) {
    if (samples.length < 2) {
      return 0;
    }

    const firstSample = Array.isArray(samples[0]) ? samples[0] : [samples[0]];
    const numFeatures = firstSample.length;

    let totalVariance = 0;

    for (let f = 0; f < numFeatures; f++) {
      const values = samples.map(s => Array.isArray(s) ? s[f] : s).filter(v => typeof v === 'number');
      if (values.length < 2) {
        continue;
      }

      const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
      const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
      totalVariance += variance;
    }

    return totalVariance / numFeatures;
  }

  /**
   * Calculate label distribution entropy
   * @param {Array} samples - Training samples
   */
  calculateLabelDistribution(samples) {
    const labelCounts = new Map();

    samples.forEach(sample => {
      const label = sample.label || sample.target || 'unknown';
      labelCounts.set(label, (labelCounts.get(label) || 0) + 1);
    });

    const totalSamples = samples.length;
    let entropy = 0;

    for (const count of labelCounts.values()) {
      const probability = count / totalSamples;
      entropy -= probability * Math.log2(probability);
    }

    // Normalize entropy (max entropy for uniform distribution)
    const maxEntropy = Math.log2(labelCounts.size);
    return maxEntropy > 0 ? entropy / maxEntropy : 0;
  }

  /**
   * Infer task type from training data characteristics
   * @param {Object} trainingData - Training data
   */
  inferTaskType(trainingData) {
    if (!trainingData.samples) {
      return 'unknown';
    }

    const sample = trainingData.samples[0];
    if (!sample) {
      return 'unknown';
    }

    // Check for common task patterns
    if (sample.target && Array.isArray(sample.target)) {
      return sample.target.length > 1 ? 'multi_classification' : 'regression';
    }

    if (sample.label !== undefined) {
      return 'classification';
    }

    if (sample.sequence || Array.isArray(sample.input)) {
      return 'sequence';
    }

    return 'regression';
  }

  /**
   * Estimate noise level in training data
   * @param {Object} trainingData - Training data
   */
  estimateNoiseLevel(trainingData) {
    if (!trainingData.samples || trainingData.samples.length < 10) {
      return 0.5;
    }

    // Simple heuristic: calculate coefficient of variation
    const values = trainingData.samples.map(s => {
      if (typeof s === 'number') {
        return s;
      }
      if (Array.isArray(s)) {
        return s.reduce((sum, v) => sum + v, 0) / s.length;
      }
      return 0;
    });

    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);

    return mean !== 0 ? Math.min(1, stdDev / Math.abs(mean)) : 0.5;
  }

  /**
   * Assess pattern regularity in data
   * @param {Object} trainingData - Training data
   */
  assessPatternRegularity(trainingData) {
    // Simplified regularity assessment
    if (!trainingData.samples || trainingData.samples.length < 5) {
      return 0.5;
    }

    // Check for periodic patterns or consistent structures
    const labelSequence = trainingData.samples.map(s => s.label || s.target || 0);
    const uniqueLabels = new Set(labelSequence);

    // More unique labels = less regular
    const regularity = 1 - (uniqueLabels.size / labelSequence.length);
    return Math.max(0, Math.min(1, regularity));
  }

  /**
   * Calculate effective dimensionality
   * @param {Object} trainingData - Training data
   */
  calculateDimensionality(trainingData) {
    if (!trainingData.samples || trainingData.samples.length === 0) {
      return 0;
    }

    const sample = trainingData.samples[0];
    if (Array.isArray(sample)) {
      return Math.min(1, sample.length / 1000); // Normalize to 0-1
    }

    return 0.1; // Low dimensionality for non-array data
  }

  /**
   * Assess temporal dependency in data
   * @param {Object} trainingData - Training data
   */
  assessTemporalDependency(trainingData) {
    // Check if data has temporal structure
    const hasTimestamps = trainingData.samples?.some(s => s.timestamp || s.time);
    const hasSequence = trainingData.samples?.some(s => s.sequence || Array.isArray(s.input));

    if (hasTimestamps) {
      return 0.8;
    }
    if (hasSequence) {
      return 0.6;
    }
    return 0.2;
  }

  /**
   * Estimate required abstraction level
   * @param {Object} trainingData - Training data
   */
  estimateAbstractionLevel(trainingData) {
    // Higher abstraction for complex, structured data
    const complexity = this.calculateDataComplexity(trainingData);
    const dimensionality = this.calculateDimensionality(trainingData);

    return (complexity + dimensionality) / 2;
  }

  /**
   * Assess creativity requirement from data
   * @param {Object} trainingData - Training data
   */
  assessCreativityRequirement(trainingData) {
    // Check for generation tasks or high variability
    const taskType = this.inferTaskType(trainingData);
    const noiseLevel = this.estimateNoiseLevel(trainingData);

    if (taskType.includes('generation')) {
      return 0.8;
    }
    if (noiseLevel > 0.7) {
      return 0.6;
    }
    return 0.3;
  }

  /**
   * Evaluate effectiveness of current cognitive patterns
   * @param {string} agentId - Agent identifier
   * @param {Object} context - Training context
   */
  async evaluatePatternEffectiveness(agentId, context) {
    const agentData = this.agentPatterns.get(agentId);
    if (!agentData) {
      return {};
    }

    const effectiveness = {};

    for (const patternType of agentData.activePatterns) {
      const template = this.patternTemplates.get(patternType);
      if (!template) {
        continue;
      }

      // Evaluate how well this pattern matches the context
      const contextMatch = this.calculateContextMatch(template, context);
      const historicalPerformance = this.getHistoricalPerformance(agentId, patternType);
      const adaptationSuccess = this.getAdaptationSuccess(agentId, patternType);

      effectiveness[patternType] = {
        contextMatch,
        historicalPerformance,
        adaptationSuccess,
        overall: (contextMatch + historicalPerformance + adaptationSuccess) / 3,
      };
    }

    return effectiveness;
  }

  /**
   * Calculate how well a pattern template matches the current context
   * @param {Object} template - Pattern template
   * @param {Object} context - Current context
   */
  calculateContextMatch(template, context) {
    const { characteristics } = template;
    let totalMatch = 0;
    let weightSum = 0;

    // Match exploration vs exploitation preference
    const explorationNeed = context.creativity_required + context.noiseLevel;
    const explorationMatch = Math.abs(characteristics.explorationRate - explorationNeed);
    totalMatch += (1 - explorationMatch) * 0.3;
    weightSum += 0.3;

    // Match decision making style
    const systematicNeed = context.dataComplexity + context.patternRegularity;
    const systematicMatch = this.matchDecisionStyle(characteristics.decisionMaking, systematicNeed);
    totalMatch += systematicMatch * 0.25;
    weightSum += 0.25;

    // Match pattern recognition approach
    const abstractionMatch = this.matchPatternRecognition(characteristics.patternRecognition, context);
    totalMatch += abstractionMatch * 0.25;
    weightSum += 0.25;

    // Match search strategy
    const searchMatch = this.matchSearchStrategy(characteristics.searchStrategy, context);
    totalMatch += searchMatch * 0.2;
    weightSum += 0.2;

    return weightSum > 0 ? totalMatch / weightSum : 0;
  }

  /**
   * Match decision making style to context needs
   * @param {string} style - Decision making style
   * @param {number} systematicNeed - Need for systematic approach (0-1)
   */
  matchDecisionStyle(style, systematicNeed) {
    const styleScores = {
      'decisive': 0.9,
      'analytical': 0.8,
      'systematic': 0.8,
      'principled': 0.7,
      'exploratory': 0.3,
      'innovative': 0.2,
    };

    const styleScore = styleScores[style] || 0.5;
    return 1 - Math.abs(styleScore - systematicNeed);
  }

  /**
   * Match pattern recognition approach to context
   * @param {string} approach - Pattern recognition approach
   * @param {Object} context - Context object
   */
  matchPatternRecognition(approach, context) {
    const approachScores = {
      'exact_match': context.patternRegularity,
      'flexible_match': 1 - context.patternRegularity,
      'analogical': context.abstractionLevel,
      'pattern_networks': context.dataComplexity,
      'evidence_based': 1 - context.noiseLevel,
      'abstraction_layers': context.abstractionLevel,
    };

    return approachScores[approach] || 0.5;
  }

  /**
   * Match search strategy to context
   * @param {string} strategy - Search strategy
   * @param {Object} context - Context object
   */
  matchSearchStrategy(strategy, context) {
    const strategyScores = {
      'directed': 1 - context.creativity_required,
      'random': context.creativity_required,
      'lateral': context.noiseLevel + context.creativity_required,
      'holistic': context.dataComplexity,
      'systematic': context.patternRegularity,
      'conceptual': context.abstractionLevel,
    };

    return Math.min(1, strategyScores[strategy] || 0.5);
  }

  /**
   * Get historical performance of a pattern for an agent
   * @param {string} agentId - Agent identifier
   * @param {string} patternType - Pattern type
   */
  getHistoricalPerformance(agentId, patternType) {
    const history = this.evolutionHistory.get(agentId) || [];
    const patternHistory = history.filter(h => h.oldPatterns.includes(patternType) || h.newPatterns.includes(patternType));

    if (patternHistory.length === 0) {
      return 0.5;
    } // Default neutral score

    // Calculate average effectiveness from historical data
    const totalEffectiveness = patternHistory.reduce((sum, h) => {
      const effectiveness = h.effectiveness?.[patternType]?.overall || 0.5;
      return sum + effectiveness;
    }, 0);

    return totalEffectiveness / patternHistory.length;
  }

  /**
   * Get adaptation success rate for a pattern
   * @param {string} agentId - Agent identifier
   * @param {string} patternType - Pattern type
   */
  getAdaptationSuccess(agentId, patternType) {
    const agentData = this.agentPatterns.get(agentId);
    if (!agentData) {
      return 0.5;
    }

    const adaptations = agentData.adaptationHistory.filter(a => a.patternType === patternType);
    if (adaptations.length === 0) {
      return 0.5;
    }

    const successfulAdaptations = adaptations.filter(a => a.success).length;
    return successfulAdaptations / adaptations.length;
  }

  /**
   * Assess if cognitive evolution is needed
   * @param {Object} effectiveness - Pattern effectiveness scores
   * @param {Object} context - Current context
   */
  assessEvolutionNeed(effectiveness, context) {
    const avgEffectiveness = Object.values(effectiveness).reduce((sum, e) => sum + e.overall, 0) / Object.keys(effectiveness).length;

    // Evolution needed if effectiveness is low
    if (avgEffectiveness < 0.4) {
      return { required: true, reason: 'low_effectiveness', urgency: 'high' };
    }

    // Evolution needed if context has changed significantly
    if (context.dataComplexity > 0.8 && avgEffectiveness < 0.6) {
      return { required: true, reason: 'high_complexity', urgency: 'medium' };
    }

    // Evolution for exploration if effectiveness is moderate
    if (avgEffectiveness < 0.7 && context.creativity_required > 0.6) {
      return { required: true, reason: 'creativity_required', urgency: 'low' };
    }

    return { required: false, reason: 'stable', urgency: 'none' };
  }

  /**
   * Select evolution strategy based on need and context
   * @param {Object} evolutionNeed - Evolution need assessment
   * @param {Object} context - Current context
   */
  selectEvolutionStrategy(evolutionNeed, context) {
    const strategies = {
      'pattern_addition': {
        type: 'pattern_addition',
        description: 'Add new cognitive patterns',
        priority: context.creativity_required > 0.6 ? 0.8 : 0.4,
      },
      'pattern_removal': {
        type: 'pattern_removal',
        description: 'Remove ineffective patterns',
        priority: evolutionNeed.urgency === 'high' ? 0.9 : 0.3,
      },
      'pattern_modification': {
        type: 'pattern_modification',
        description: 'Modify existing patterns',
        priority: 0.6,
      },
      'pattern_rebalancing': {
        type: 'pattern_rebalancing',
        description: 'Rebalance pattern weights',
        priority: evolutionNeed.urgency === 'medium' ? 0.7 : 0.5,
      },
      'pattern_hybridization': {
        type: 'pattern_hybridization',
        description: 'Create hybrid patterns',
        priority: context.dataComplexity > 0.7 ? 0.8 : 0.3,
      },
    };

    // Select strategy with highest priority
    const selectedStrategy = Object.values(strategies).reduce((best, current) =>
      current.priority > best.priority ? current : best,
    );

    return selectedStrategy;
  }

  /**
   * Apply evolution strategy to agent patterns
   * @param {string} agentId - Agent identifier
   * @param {Object} strategy - Evolution strategy
   * @param {Object} context - Current context
   */
  async applyEvolution(agentId, strategy, context) {
    const agentData = this.agentPatterns.get(agentId);
    if (!agentData) {
      return [];
    }

    let newPatterns = [...agentData.activePatterns];

    switch (strategy.type) {
    case 'pattern_addition':
      newPatterns = await this.addPatterns(agentId, newPatterns, context);
      break;

    case 'pattern_removal':
      newPatterns = await this.removePatterns(agentId, newPatterns, context);
      break;

    case 'pattern_modification':
      newPatterns = await this.modifyPatterns(agentId, newPatterns, context);
      break;

    case 'pattern_rebalancing':
      newPatterns = await this.rebalancePatterns(agentId, newPatterns, context);
      break;

    case 'pattern_hybridization':
      newPatterns = await this.hybridizePatterns(agentId, newPatterns, context);
      break;
    }

    // Update agent patterns
    agentData.activePatterns = newPatterns;
    agentData.dominantPattern = this.selectDominantPattern(newPatterns, context);
    agentData.lastEvolution = Date.now();
    agentData.evolutionScore += 1;

    // Update metrics
    const metrics = this.evolutionMetrics.get(agentId);
    if (metrics) {
      metrics.totalEvolutions++;
      if (strategy.type === 'pattern_addition') {
        metrics.patternSwitches++;
      }
    }

    return newPatterns;
  }

  /**
   * Add new cognitive patterns
   * @param {string} agentId - Agent identifier
   * @param {Array} currentPatterns - Current patterns
   * @param {Object} context - Current context
   */
  async addPatterns(agentId, currentPatterns, context) {
    const availablePatterns = Array.from(this.patternTemplates.keys());
    const unusedPatterns = availablePatterns.filter(p => !currentPatterns.includes(p));

    if (unusedPatterns.length === 0) {
      return currentPatterns;
    }

    // Select best pattern to add based on context
    let bestPattern = null;
    let bestScore = 0;

    for (const pattern of unusedPatterns) {
      const template = this.patternTemplates.get(pattern);
      const score = this.calculateContextMatch(template, context);

      if (score > bestScore) {
        bestScore = score;
        bestPattern = pattern;
      }
    }

    if (bestPattern && bestScore > 0.6) {
      return [...currentPatterns, bestPattern];
    }

    return currentPatterns;
  }

  /**
   * Remove ineffective cognitive patterns
   * @param {string} agentId - Agent identifier
   * @param {Array} currentPatterns - Current patterns
   * @param {Object} context - Current context
   */
  async removePatterns(agentId, currentPatterns, context) {
    if (currentPatterns.length <= 1) {
      return currentPatterns;
    } // Keep at least one pattern

    // Find least effective pattern
    let worstPattern = null;
    let worstScore = 1;

    for (const pattern of currentPatterns) {
      const template = this.patternTemplates.get(pattern);
      const score = this.calculateContextMatch(template, context);

      if (score < worstScore) {
        worstScore = score;
        worstPattern = pattern;
      }
    }

    if (worstPattern && worstScore < 0.3) {
      return currentPatterns.filter(p => p !== worstPattern);
    }

    return currentPatterns;
  }

  /**
   * Modify existing patterns (create adaptive variants)
   * @param {string} agentId - Agent identifier
   * @param {Array} currentPatterns - Current patterns
   * @param {Object} context - Current context
   */
  async modifyPatterns(agentId, currentPatterns, context) {
    // Create modified versions of existing patterns
    const modifiedPatterns = [];

    for (const pattern of currentPatterns) {
      const template = this.patternTemplates.get(pattern);
      if (!template) {
        continue;
      }

      // Create adaptive modification
      const modifiedPattern = `${pattern}_adaptive_${Date.now()}`;
      const modifiedTemplate = this.createAdaptiveVariant(template, context);

      this.patternTemplates.set(modifiedPattern, modifiedTemplate);
      modifiedPatterns.push(modifiedPattern);
    }

    return modifiedPatterns.length > 0 ? modifiedPatterns : currentPatterns;
  }

  /**
   * Create adaptive variant of a pattern template
   * @param {Object} template - Original template
   * @param {Object} context - Current context
   */
  createAdaptiveVariant(template, context) {
    const adaptiveTemplate = JSON.parse(JSON.stringify(template)); // Deep copy

    // Adapt characteristics based on context
    if (context.creativity_required > 0.7) {
      adaptiveTemplate.characteristics.explorationRate = Math.min(1, adaptiveTemplate.characteristics.explorationRate + 0.2);
      adaptiveTemplate.characteristics.exploitationRate = Math.max(0, adaptiveTemplate.characteristics.exploitationRate - 0.2);
    }

    if (context.dataComplexity > 0.8) {
      adaptiveTemplate.characteristics.patternRecognition = 'pattern_networks';
      adaptiveTemplate.characteristics.searchStrategy = 'systematic';
    }

    if (context.noiseLevel > 0.6) {
      adaptiveTemplate.characteristics.decisionMaking = 'exploratory';
    }

    adaptiveTemplate.name += ' (Adaptive)';
    adaptiveTemplate.description += ' - Adapted for current context';

    return adaptiveTemplate;
  }

  /**
   * Rebalance pattern priorities and weights
   * @param {string} agentId - Agent identifier
   * @param {Array} currentPatterns - Current patterns
   * @param {Object} context - Current context
   */
  async rebalancePatterns(agentId, currentPatterns, context) {
    // Rebalancing keeps the same patterns but changes their relative importance
    // This would typically involve adjusting weights in the neural network
    // For now, we reorder patterns by effectiveness

    const patternScores = [];

    for (const pattern of currentPatterns) {
      const template = this.patternTemplates.get(pattern);
      const score = this.calculateContextMatch(template, context);
      patternScores.push({ pattern, score });
    }

    // Sort by score (descending)
    patternScores.sort((a, b) => b.score - a.score);

    return patternScores.map(ps => ps.pattern);
  }

  /**
   * Create hybrid patterns by combining existing ones
   * @param {string} agentId - Agent identifier
   * @param {Array} currentPatterns - Current patterns
   * @param {Object} context - Current context
   */
  async hybridizePatterns(agentId, currentPatterns, context) {
    if (currentPatterns.length < 2) {
      return currentPatterns;
    }

    // Create hybrid of two best patterns
    const hybridPattern = `hybrid_${currentPatterns[0]}_${currentPatterns[1]}_${Date.now()}`;
    const template1 = this.patternTemplates.get(currentPatterns[0]);
    const template2 = this.patternTemplates.get(currentPatterns[1]);

    if (!template1 || !template2) {
      return currentPatterns;
    }

    const hybridTemplate = this.createHybridTemplate(template1, template2, context);
    this.patternTemplates.set(hybridPattern, hybridTemplate);

    return [hybridPattern, ...currentPatterns.slice(2)];
  }

  /**
   * Create hybrid template from two parent templates
   * @param {Object} template1 - First parent template
   * @param {Object} template2 - Second parent template
   * @param {Object} context - Current context
   */
  createHybridTemplate(template1, template2, context) {
    const hybrid = {
      name: `Hybrid: ${template1.name} + ${template2.name}`,
      description: `Combination of ${template1.name.toLowerCase()} and ${template2.name.toLowerCase()}`,
      characteristics: {},
      adaptationRules: {},
    };

    // Blend characteristics
    const chars1 = template1.characteristics;
    const chars2 = template2.characteristics;

    hybrid.characteristics = {
      searchStrategy: context.creativity_required > 0.5 ? chars2.searchStrategy : chars1.searchStrategy,
      explorationRate: (chars1.explorationRate + chars2.explorationRate) / 2,
      exploitationRate: (chars1.exploitationRate + chars2.exploitationRate) / 2,
      decisionMaking: context.dataComplexity > 0.6 ? chars1.decisionMaking : chars2.decisionMaking,
      patternRecognition: chars1.patternRecognition, // Use first template's approach
    };

    // Combine adaptation rules
    hybrid.adaptationRules = {
      ...template1.adaptationRules,
      ...template2.adaptationRules,
    };

    return hybrid;
  }

  /**
   * Select dominant pattern from active patterns
   * @param {Array} patterns - Active patterns
   * @param {Object} context - Current context
   */
  selectDominantPattern(patterns, context) {
    if (patterns.length === 0) {
      return 'convergent';
    }
    if (patterns.length === 1) {
      return patterns[0];
    }

    // Select pattern that best matches current context
    let bestPattern = patterns[0];
    let bestScore = 0;

    for (const pattern of patterns) {
      const template = this.patternTemplates.get(pattern);
      if (!template) {
        continue;
      }

      const score = this.calculateContextMatch(template, context);
      if (score > bestScore) {
        bestScore = score;
        bestPattern = pattern;
      }
    }

    return bestPattern;
  }

  /**
   * Record evolution event
   * @param {string} agentId - Agent identifier
   * @param {Object} evolution - Evolution details
   */
  recordEvolution(agentId, evolution) {
    const history = this.evolutionHistory.get(agentId) || [];
    history.push(evolution);

    // Keep only recent evolution history (last 50 events)
    if (history.length > 50) {
      history.splice(0, history.length - 50);
    }

    this.evolutionHistory.set(agentId, history);
  }

  /**
   * Enable cross-agent pattern evolution
   * @param {Array} agentIds - List of agent IDs
   * @param {Object} session - Collaborative session
   */
  async enableCrossAgentEvolution(agentIds, session) {
    // Create cross-agent pattern exchange matrix
    const exchangeMatrix = {};

    for (const agentId of agentIds) {
      exchangeMatrix[agentId] = new Map();

      // Initialize exchange relationships
      for (const otherAgentId of agentIds) {
        if (agentId !== otherAgentId) {
          exchangeMatrix[agentId].set(otherAgentId, {
            lastExchange: 0,
            exchangeCount: 0,
            successRate: 0.5,
            patternCompatibility: 0.5,
          });
        }
      }
    }

    this.crossAgentPatterns.set(session.id, exchangeMatrix);

    console.log(`Cross-agent pattern evolution enabled for ${agentIds.length} agents`);
  }

  /**
   * Transfer patterns between agents
   * @param {string} targetAgentId - Target agent ID
   * @param {Array} patterns - Patterns to transfer
   */
  async transferPatterns(targetAgentId, patterns) {
    const targetData = this.agentPatterns.get(targetAgentId);
    if (!targetData) {
      return;
    }

    // Evaluate pattern compatibility
    const compatiblePatterns = [];

    for (const pattern of patterns) {
      const compatibility = await this.evaluatePatternCompatibility(targetAgentId, pattern);

      if (compatibility > 0.6) {
        compatiblePatterns.push(pattern);
      }
    }

    // Transfer compatible patterns
    if (compatiblePatterns.length > 0) {
      targetData.activePatterns = [...new Set([...targetData.activePatterns, ...compatiblePatterns])];

      // Update metrics
      const metrics = this.evolutionMetrics.get(targetAgentId);
      if (metrics) {
        metrics.crossAgentTransfers += compatiblePatterns.length;
      }

      console.log(`Transferred ${compatiblePatterns.length} patterns to agent ${targetAgentId}`);
    }
  }

  /**
   * Evaluate pattern compatibility with target agent
   * @param {string} agentId - Target agent ID
   * @param {Object} pattern - Pattern to evaluate
   */
  async evaluatePatternCompatibility(agentId, pattern) {
    const agentData = this.agentPatterns.get(agentId);
    if (!agentData) {
      return 0;
    }

    // Check if pattern type is already present
    if (agentData.activePatterns.includes(pattern.type)) {
      return 0.3; // Low compatibility if already present
    }

    // Evaluate based on agent's current pattern mix
    const currentPatternTypes = agentData.activePatterns.map(p => p.split('_')[0]);
    const patternType = pattern.type?.split('_')[0] || 'unknown';

    // Check for complementary patterns
    const complementaryPatterns = {
      'convergent': ['divergent', 'lateral'],
      'divergent': ['convergent', 'critical'],
      'lateral': ['systems', 'convergent'],
      'systems': ['lateral', 'abstract'],
      'critical': ['divergent', 'abstract'],
      'abstract': ['critical', 'systems'],
    };

    const complements = complementaryPatterns[patternType] || [];
    const hasComplement = currentPatternTypes.some(ct => complements.includes(ct));

    return hasComplement ? 0.8 : 0.5;
  }

  /**
   * Extract patterns from agent for sharing
   * @param {string} agentId - Agent identifier
   */
  async extractPatterns(agentId) {
    const agentData = this.agentPatterns.get(agentId);
    if (!agentData) {
      return [];
    }

    const extractedPatterns = [];

    for (const patternType of agentData.activePatterns) {
      const template = this.patternTemplates.get(patternType);
      if (!template) {
        continue;
      }

      extractedPatterns.push({
        type: patternType,
        template,
        effectiveness: this.getHistoricalPerformance(agentId, patternType),
        adaptationHistory: agentData.adaptationHistory.filter(a => a.patternType === patternType),
        dominance: patternType === agentData.dominantPattern ? 1.0 : 0.5,
      });
    }

    return extractedPatterns;
  }

  /**
   * Apply pattern updates from coordination
   * @param {string} agentId - Agent identifier
   * @param {Array} patternUpdates - Pattern updates
   */
  async applyPatternUpdates(agentId, patternUpdates) {
    const agentData = this.agentPatterns.get(agentId);
    if (!agentData) {
      return;
    }

    for (const update of patternUpdates) {
      if (update.type === 'add_pattern') {
        if (!agentData.activePatterns.includes(update.pattern)) {
          agentData.activePatterns.push(update.pattern);
        }
      } else if (update.type === 'remove_pattern') {
        agentData.activePatterns = agentData.activePatterns.filter(p => p !== update.pattern);
      } else if (update.type === 'modify_pattern') {
        // Apply modifications to pattern template
        const template = this.patternTemplates.get(update.pattern);
        if (template && update.modifications) {
          Object.assign(template.characteristics, update.modifications);
        }
      } else if (update.type === 'set_dominant') {
        agentData.dominantPattern = update.pattern;
      }
    }

    // Ensure at least one pattern remains active
    if (agentData.activePatterns.length === 0) {
      agentData.activePatterns.push('convergent');
      agentData.dominantPattern = 'convergent';
    }
  }

  /**
   * Calculate aggregation weights for gradient coordination
   * @param {Array} gradients - Array of gradient sets
   */
  calculateAggregationWeights(gradients) {
    // Weight gradients based on cognitive pattern effectiveness
    const weights = new Array(gradients.length).fill(1 / gradients.length);

    // This would typically incorporate pattern effectiveness scores
    // For now, return uniform weights
    return weights;
  }

  /**
   * Assess cognitive growth for an agent
   * @param {string} agentId - Agent identifier
   */
  async assessGrowth(agentId) {
    const agentData = this.agentPatterns.get(agentId);
    const metrics = this.evolutionMetrics.get(agentId);

    if (!agentData || !metrics) {
      return 0;
    }

    const growth = {
      patternDiversity: agentData.activePatterns.length / 6, // Normalize by max patterns
      evolutionFrequency: metrics.totalEvolutions / Math.max(1, (Date.now() - agentData.lastEvolution) / (24 * 60 * 60 * 1000)),
      adaptationSuccess: metrics.successfulAdaptations / Math.max(1, metrics.totalEvolutions),
      crossAgentLearning: metrics.crossAgentTransfers / Math.max(1, metrics.totalEvolutions),
      emergentPatterns: metrics.emergentPatterns / Math.max(1, metrics.totalEvolutions),
    };

    // Calculate overall growth score
    const overallGrowth = (
      growth.patternDiversity * 0.2 +
      growth.evolutionFrequency * 0.2 +
      growth.adaptationSuccess * 0.3 +
      growth.crossAgentLearning * 0.15 +
      growth.emergentPatterns * 0.15
    );

    return Math.min(1, overallGrowth);
  }

  /**
   * Get statistics for the cognitive evolution system
   */
  getStatistics() {
    const totalAgents = this.agentPatterns.size;
    let totalEvolutions = 0;
    let totalPatterns = 0;
    const avgGrowthScore = 0;

    for (const [agentId, metrics] of this.evolutionMetrics.entries()) {
      totalEvolutions += metrics.totalEvolutions;
      const agentData = this.agentPatterns.get(agentId);
      if (agentData) {
        totalPatterns += agentData.activePatterns.length;
      }
    }

    return {
      totalAgents,
      totalEvolutions,
      avgPatternsPerAgent: totalAgents > 0 ? totalPatterns / totalAgents : 0,
      availablePatternTypes: this.patternTemplates.size,
      crossAgentSessions: this.crossAgentPatterns.size,
    };
  }

  /**
   * Preserve cognitive evolution history before agent reset
   * @param {string} agentId - Agent identifier
   */
  async preserveHistory(agentId) {
    const agentData = this.agentPatterns.get(agentId);
    const history = this.evolutionHistory.get(agentId);
    const metrics = this.evolutionMetrics.get(agentId);

    return {
      patterns: agentData ? { ...agentData } : null,
      history: history ? [...history] : [],
      metrics: metrics ? { ...metrics } : null,
    };
  }

  /**
   * Restore cognitive evolution history after agent reset
   * @param {string} agentId - Agent identifier
   * @param {Object} preservedHistory - Preserved history
   */
  async restoreHistory(agentId, preservedHistory) {
    if (preservedHistory.patterns) {
      this.agentPatterns.set(agentId, preservedHistory.patterns);
    }

    if (preservedHistory.history) {
      this.evolutionHistory.set(agentId, preservedHistory.history);
    }

    if (preservedHistory.metrics) {
      this.evolutionMetrics.set(agentId, preservedHistory.metrics);
    }
  }
}

export { CognitivePatternEvolution };