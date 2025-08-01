/**
 * DAA Cognition Module
 * Decentralized Autonomous Agent Cognitive Integration
 */

export class DAACognition {
  constructor() {
    this.cognitiveAgents = new Map();
    this.distributedMemory = new Map();
    this.consensusProtocol = new Map();
    this.autonomyLevels = new Map();
    this.emergentBehaviors = new Map();

    // Initialize DAA-specific cognitive patterns
    this.initializeDAAPatterns();
  }

  /**
   * Initialize DAA-specific cognitive patterns
   */
  initializeDAAPatterns() {
    this.daaPatterns = {
      autonomous_decision: {
        name: 'Autonomous Decision Making',
        description: 'Independent decision-making without central control',
        characteristics: {
          autonomyLevel: 0.9,
          consensusRequirement: 0.3,
          decisionSpeed: 0.8,
          riskTolerance: 0.6,
        },
      },
      distributed_reasoning: {
        name: 'Distributed Reasoning',
        description: 'Collective reasoning across multiple agents',
        characteristics: {
          collaborationLevel: 0.9,
          informationSharing: 0.8,
          consensusBuilding: 0.7,
          knowledgeAggregation: 0.8,
        },
      },
      emergent_intelligence: {
        name: 'Emergent Intelligence',
        description: 'Intelligence emerging from agent interactions',
        characteristics: {
          emergenceThreshold: 0.7,
          collectiveIQ: 0.8,
          adaptiveCapacity: 0.9,
          selfOrganization: 0.85,
        },
      },
      swarm_cognition: {
        name: 'Swarm Cognition',
        description: 'Collective cognitive processing as a swarm',
        characteristics: {
          swarmCoherence: 0.8,
          localInteractions: 0.9,
          globalOptimization: 0.7,
          scalability: 0.95,
        },
      },
      decentralized_learning: {
        name: 'Decentralized Learning',
        description: 'Learning without centralized coordination',
        characteristics: {
          peerLearning: 0.85,
          knowledgePropagation: 0.8,
          adaptationRate: 0.75,
          robustness: 0.9,
        },
      },
    };
  }

  /**
   * Initialize DAA cognitive agent
   * @param {string} agentId - Agent identifier
   * @param {Object} config - Agent configuration
   */
  async initializeDAAAgent(agentId, config) {
    const daaAgent = {
      id: agentId,
      autonomyLevel: config.autonomyLevel || 0.7,
      cognitivePattern: this.selectDAAPattern(config),
      localMemory: new Map(),
      peerConnections: new Set(),
      consensusState: {
        proposals: new Map(),
        votes: new Map(),
        decisions: [],
      },
      emergentTraits: new Set(),
      learningState: {
        localKnowledge: new Map(),
        sharedKnowledge: new Map(),
        propagationQueue: [],
      },
    };

    this.cognitiveAgents.set(agentId, daaAgent);

    // Initialize in distributed memory
    this.initializeDistributedMemory(agentId);

    console.log(`Initialized DAA cognitive agent ${agentId} with autonomy level ${daaAgent.autonomyLevel}`);

    return daaAgent;
  }

  /**
   * Select appropriate DAA cognitive pattern
   * @param {Object} config - Agent configuration
   */
  selectDAAPattern(config) {
    // Select based on agent type and requirements
    if (config.requiresAutonomy) {
      return this.daaPatterns.autonomous_decision;
    } else if (config.requiresCollaboration) {
      return this.daaPatterns.distributed_reasoning;
    } else if (config.enableEmergence) {
      return this.daaPatterns.emergent_intelligence;
    } else if (config.swarmMode) {
      return this.daaPatterns.swarm_cognition;
    }
    return this.daaPatterns.decentralized_learning;

  }

  /**
   * Initialize distributed memory for agent
   * @param {string} agentId - Agent identifier
   */
  initializeDistributedMemory(agentId) {
    this.distributedMemory.set(agentId, {
      localSegment: new Map(),
      sharedSegments: new Map(),
      replicationFactor: 3,
      consistencyLevel: 'eventual',
      lastSync: Date.now(),
    });
  }

  /**
   * Enable autonomous decision making
   * @param {string} agentId - Agent identifier
   * @param {Object} decision - Decision context
   */
  async makeAutonomousDecision(agentId, decision) {
    const agent = this.cognitiveAgents.get(agentId);
    if (!agent) {
      return null;
    }

    // Evaluate decision based on local knowledge
    const localEvaluation = this.evaluateLocally(agent, decision);

    // Check if consensus is needed based on autonomy level
    if (agent.autonomyLevel < decision.consensusThreshold) {
      return this.seekConsensus(agentId, decision, localEvaluation);
    }

    // Make autonomous decision
    const autonomousDecision = {
      agentId,
      decision: localEvaluation.recommendation,
      confidence: localEvaluation.confidence,
      reasoning: localEvaluation.reasoning,
      timestamp: Date.now(),
      autonomous: true,
    };

    // Record decision
    agent.consensusState.decisions.push(autonomousDecision);

    // Propagate decision to peers
    await this.propagateDecision(agentId, autonomousDecision);

    return autonomousDecision;
  }

  /**
   * Evaluate decision locally
   * @param {Object} agent - DAA agent
   * @param {Object} decision - Decision context
   */
  evaluateLocally(agent, decision) {
    const evaluation = {
      recommendation: null,
      confidence: 0,
      reasoning: [],
    };

    // Use local knowledge for evaluation
    const _relevantKnowledge = this.retrieveRelevantKnowledge(agent, decision);

    // Apply cognitive pattern
    const pattern = agent.cognitivePattern;
    if (pattern.characteristics.autonomyLevel > 0.5) {
      evaluation.confidence += 0.3;
      evaluation.reasoning.push('High autonomy pattern supports independent decision');
    }

    // Analyze based on past decisions
    const similarDecisions = this.findSimilarDecisions(agent, decision);
    if (similarDecisions.length > 0) {
      const avgOutcome = this.calculateAverageOutcome(similarDecisions);
      evaluation.confidence += avgOutcome * 0.4;
      evaluation.reasoning.push(`Historical success rate: ${(avgOutcome * 100).toFixed(1)}%`);
    }

    // Make recommendation
    evaluation.recommendation = evaluation.confidence > 0.6 ? 'approve' : 'reject';

    return evaluation;
  }

  /**
   * Retrieve relevant knowledge for decision
   * @param {Object} agent - DAA agent
   * @param {Object} decision - Decision context
   */
  retrieveRelevantKnowledge(agent, decision) {
    const relevant = [];

    // Check local memory
    for (const [key, value] of agent.localMemory) {
      if (this.isRelevantToDecision(key, value, decision)) {
        relevant.push({ source: 'local', key, value });
      }
    }

    // Check shared knowledge
    for (const [key, value] of agent.learningState.sharedKnowledge) {
      if (this.isRelevantToDecision(key, value, decision)) {
        relevant.push({ source: 'shared', key, value });
      }
    }

    return relevant;
  }

  /**
   * Check if knowledge is relevant to decision
   * @param {string} key - Knowledge key
   * @param {*} value - Knowledge value
   * @param {Object} decision - Decision context
   */
  isRelevantToDecision(key, value, decision) {
    // Simple relevance check based on keywords
    const decisionKeywords = decision.context?.keywords || [];
    return decisionKeywords.some(keyword =>
      key.includes(keyword) ||
      (typeof value === 'string' && value.includes(keyword)),
    );
  }

  /**
   * Find similar past decisions
   * @param {Object} agent - DAA agent
   * @param {Object} decision - Current decision
   */
  findSimilarDecisions(agent, decision) {
    return agent.consensusState.decisions.filter(pastDecision => {
      // Simple similarity based on decision type
      return pastDecision.decision === decision.type;
    });
  }

  /**
   * Calculate average outcome of decisions
   * @param {Array} decisions - Past decisions
   */
  calculateAverageOutcome(decisions) {
    if (decisions.length === 0) {
      return 0.5;
    }

    const successfulDecisions = decisions.filter(d => d.outcome === 'success').length;
    return successfulDecisions / decisions.length;
  }

  /**
   * Seek consensus from peer agents
   * @param {string} agentId - Agent identifier
   * @param {Object} decision - Decision context
   * @param {Object} localEvaluation - Local evaluation
   */
  async seekConsensus(agentId, decision, localEvaluation) {
    const agent = this.cognitiveAgents.get(agentId);
    if (!agent) {
      return null;
    }

    // Create consensus proposal
    const proposal = {
      id: `proposal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      proposer: agentId,
      decision,
      localEvaluation,
      timestamp: Date.now(),
      votes: new Map(),
      status: 'pending',
    };

    agent.consensusState.proposals.set(proposal.id, proposal);

    // Request votes from peers
    const votePromises = [];
    for (const peerId of agent.peerConnections) {
      votePromises.push(this.requestVote(peerId, proposal));
    }

    // Collect votes
    const votes = await Promise.all(votePromises);

    // Tally results
    const consensusResult = this.tallyVotes(votes, proposal);

    // Update proposal status
    proposal.status = consensusResult.approved ? 'approved' : 'rejected';
    proposal.consensusLevel = consensusResult.consensusLevel;

    // Create consensus decision
    const consensusDecision = {
      agentId,
      decision: consensusResult.approved ? 'approve' : 'reject',
      confidence: consensusResult.consensusLevel,
      reasoning: [...localEvaluation.reasoning, `Consensus level: ${(consensusResult.consensusLevel * 100).toFixed(1)}%`],
      timestamp: Date.now(),
      autonomous: false,
      proposalId: proposal.id,
    };

    agent.consensusState.decisions.push(consensusDecision);

    return consensusDecision;
  }

  /**
   * Request vote from peer agent
   * @param {string} peerId - Peer agent ID
   * @param {Object} proposal - Consensus proposal
   */
  async requestVote(peerId, proposal) {
    const peerAgent = this.cognitiveAgents.get(peerId);
    if (!peerAgent) {
      return { agentId: peerId, vote: 'abstain', reason: 'Agent not found' };
    }

    // Peer evaluates proposal
    const peerEvaluation = this.evaluateLocally(peerAgent, proposal.decision);

    // Cast vote based on evaluation
    const vote = {
      agentId: peerId,
      vote: peerEvaluation.confidence > 0.5 ? 'approve' : 'reject',
      confidence: peerEvaluation.confidence,
      reason: peerEvaluation.reasoning[0] || 'No specific reason',
    };

    return vote;
  }

  /**
   * Tally votes for consensus
   * @param {Array} votes - Vote results
   * @param {Object} proposal - Consensus proposal
   */
  tallyVotes(votes, proposal) {
    let approveCount = 0;
    let totalWeight = 0;

    for (const vote of votes) {
      const weight = vote.confidence || 0.5;
      totalWeight += weight;

      if (vote.vote === 'approve') {
        approveCount += weight;
      }

      // Store vote in proposal
      proposal.votes.set(vote.agentId, vote);
    }

    const consensusLevel = totalWeight > 0 ? approveCount / totalWeight : 0;
    const approved = consensusLevel > 0.5;

    return { approved, consensusLevel, totalVotes: votes.length };
  }

  /**
   * Propagate decision to peer agents
   * @param {string} agentId - Agent identifier
   * @param {Object} decision - Decision to propagate
   */
  async propagateDecision(agentId, decision) {
    const agent = this.cognitiveAgents.get(agentId);
    if (!agent) {
      return;
    }

    // Add to propagation queue
    agent.learningState.propagationQueue.push({
      type: 'decision',
      content: decision,
      timestamp: Date.now(),
    });

    // Propagate to connected peers
    for (const peerId of agent.peerConnections) {
      await this.sendToPeer(peerId, {
        type: 'decision_update',
        from: agentId,
        decision,
      });
    }
  }

  /**
   * Send message to peer agent
   * @param {string} peerId - Peer agent ID
   * @param {Object} message - Message to send
   */
  async sendToPeer(peerId, message) {
    const peerAgent = this.cognitiveAgents.get(peerId);
    if (!peerAgent) {
      return;
    }

    // Process message based on type
    switch (message.type) {
      case 'decision_update':
        this.processDecisionUpdate(peerId, message);
        break;
      case 'knowledge_share':
        this.processKnowledgeShare(peerId, message);
        break;
      case 'emergent_behavior':
        this.processEmergentBehavior(peerId, message);
        break;
    }
  }

  /**
   * Process decision update from peer
   * @param {string} agentId - Receiving agent ID
   * @param {Object} message - Update message
   */
  processDecisionUpdate(agentId, message) {
    const agent = this.cognitiveAgents.get(agentId);
    if (!agent) {
      return;
    }

    // Store peer decision for learning
    const peerDecision = {
      ...message.decision,
      receivedFrom: message.from,
      receivedAt: Date.now(),
    };

    agent.learningState.sharedKnowledge.set(
      `peer_decision_${message.decision.timestamp}`,
      peerDecision,
    );
  }

  /**
   * Enable distributed learning
   * @param {string} agentId - Agent identifier
   * @param {Object} learningData - Data to learn from
   */
  async performDistributedLearning(agentId, learningData) {
    const agent = this.cognitiveAgents.get(agentId);
    if (!agent) {
      return null;
    }

    // Local learning phase
    const localLearning = await this.performLocalLearning(agent, learningData);

    // Share learning with peers
    const sharedLearning = await this.shareLearning(agentId, localLearning);

    // Aggregate peer learning
    const aggregatedLearning = await this.aggregatePeerLearning(agentId, sharedLearning);

    // Update agent's knowledge
    this.updateAgentKnowledge(agent, aggregatedLearning);

    return {
      localLearning,
      sharedLearning,
      aggregatedLearning,
      knowledgeGrowth: this.calculateKnowledgeGrowth(agent),
    };
  }

  /**
   * Perform local learning
   * @param {Object} agent - DAA agent
   * @param {Object} learningData - Learning data
   */
  async performLocalLearning(agent, learningData) {
    const learning = {
      patterns: [],
      insights: [],
      confidence: 0,
    };

    // Extract patterns from data
    if (learningData.samples) {
      const patterns = this.extractPatterns(learningData.samples);
      learning.patterns = patterns;
      learning.confidence = patterns.length > 0 ? 0.7 : 0.3;
    }

    // Generate insights
    if (learning.patterns.length > 0) {
      learning.insights = this.generateInsights(learning.patterns);
    }

    // Store in local memory
    learning.patterns.forEach((pattern, idx) => {
      agent.localMemory.set(`pattern_${Date.now()}_${idx}`, pattern);
    });

    return learning;
  }

  /**
   * Extract patterns from data samples
   * @param {Array} samples - Data samples
   */
  extractPatterns(samples) {
    const patterns = [];

    // Simple pattern extraction (placeholder for more sophisticated methods)
    if (samples.length > 10) {
      patterns.push({
        type: 'frequency',
        description: 'High sample frequency detected',
        confidence: 0.8,
      });
    }

    // Look for sequences
    const isSequential = samples.every((sample, idx) =>
      idx === 0 || this.isSequentialWith(samples[idx - 1], sample),
    );

    if (isSequential) {
      patterns.push({
        type: 'sequential',
        description: 'Sequential pattern detected',
        confidence: 0.9,
      });
    }

    return patterns;
  }

  /**
   * Check if samples are sequential
   * @param {*} prev - Previous sample
   * @param {*} current - Current sample
   */
  isSequentialWith(prev, current) {
    // Simple check - can be made more sophisticated
    if (typeof prev === 'number' && typeof current === 'number') {
      return Math.abs(current - prev) < 10;
    }
    return false;
  }

  /**
   * Generate insights from patterns
   * @param {Array} patterns - Detected patterns
   */
  generateInsights(patterns) {
    const insights = [];

    // Generate insights based on pattern combinations
    const hasSequential = patterns.some(p => p.type === 'sequential');
    const hasFrequency = patterns.some(p => p.type === 'frequency');

    if (hasSequential && hasFrequency) {
      insights.push({
        type: 'combined',
        description: 'High-frequency sequential data detected',
        actionable: 'Consider time-series optimization',
      });
    }

    return insights;
  }

  /**
   * Share learning with peer agents
   * @param {string} agentId - Agent identifier
   * @param {Object} localLearning - Local learning results
   */
  async shareLearning(agentId, localLearning) {
    const agent = this.cognitiveAgents.get(agentId);
    if (!agent) {
      return [];
    }

    const sharingResults = [];

    // Share with each peer
    for (const peerId of agent.peerConnections) {
      const shareResult = await this.shareWithPeer(agentId, peerId, localLearning);
      sharingResults.push(shareResult);
    }

    return sharingResults;
  }

  /**
   * Share learning with specific peer
   * @param {string} agentId - Sharing agent ID
   * @param {string} peerId - Peer agent ID
   * @param {Object} learning - Learning to share
   */
  async shareWithPeer(agentId, peerId, learning) {
    await this.sendToPeer(peerId, {
      type: 'knowledge_share',
      from: agentId,
      learning: {
        patterns: learning.patterns,
        insights: learning.insights,
        confidence: learning.confidence,
        timestamp: Date.now(),
      },
    });

    return {
      peer: peerId,
      shared: true,
      timestamp: Date.now(),
    };
  }

  /**
   * Process knowledge share from peer
   * @param {string} agentId - Receiving agent ID
   * @param {Object} message - Knowledge share message
   */
  processKnowledgeShare(agentId, message) {
    const agent = this.cognitiveAgents.get(agentId);
    if (!agent) {
      return;
    }

    // Store shared knowledge
    const sharedKnowledge = {
      ...message.learning,
      source: message.from,
      receivedAt: Date.now(),
    };

    agent.learningState.sharedKnowledge.set(
      `shared_${message.from}_${message.learning.timestamp}`,
      sharedKnowledge,
    );

    // Check for emergent patterns
    this.checkForEmergentPatterns(agentId);
  }

  /**
   * Aggregate learning from peers
   * @param {string} agentId - Agent identifier
   * @param {Array} sharingResults - Results of sharing
   */
  async aggregatePeerLearning(agentId, _sharingResults) {
    const agent = this.cognitiveAgents.get(agentId);
    if (!agent) {
      return null;
    }

    const aggregated = {
      patterns: new Map(),
      insights: [],
      consensusLevel: 0,
    };

    // Collect all shared knowledge
    for (const [_key, knowledge] of agent.learningState.sharedKnowledge) {
      if (knowledge.patterns) {
        knowledge.patterns.forEach(pattern => {
          const patternKey = `${pattern.type}_${pattern.description}`;
          if (!aggregated.patterns.has(patternKey)) {
            aggregated.patterns.set(patternKey, {
              ...pattern,
              sources: [],
            });
          }
          aggregated.patterns.get(patternKey).sources.push(knowledge.source);
        });
      }

      if (knowledge.insights) {
        aggregated.insights.push(...knowledge.insights);
      }
    }

    // Calculate consensus level
    const totalPeers = agent.peerConnections.size;
    if (totalPeers > 0) {
      aggregated.patterns.forEach(pattern => {
        pattern.consensusLevel = pattern.sources.length / totalPeers;
      });
    }

    return aggregated;
  }

  /**
   * Update agent knowledge with aggregated learning
   * @param {Object} agent - DAA agent
   * @param {Object} aggregatedLearning - Aggregated learning
   */
  updateAgentKnowledge(agent, aggregatedLearning) {
    if (!aggregatedLearning) {
      return;
    }

    // Update local knowledge with high-consensus patterns
    aggregatedLearning.patterns.forEach((pattern, key) => {
      if (pattern.consensusLevel > 0.6) {
        agent.localMemory.set(`consensus_${key}`, pattern);
      }
    });

    // Store unique insights
    const uniqueInsights = this.deduplicateInsights(aggregatedLearning.insights);
    uniqueInsights.forEach((insight, idx) => {
      agent.localMemory.set(`insight_${Date.now()}_${idx}`, insight);
    });
  }

  /**
   * Deduplicate insights
   * @param {Array} insights - Array of insights
   */
  deduplicateInsights(insights) {
    const seen = new Set();
    return insights.filter(insight => {
      const key = `${insight.type}_${insight.description}`;
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }

  /**
   * Calculate knowledge growth for agent
   * @param {Object} agent - DAA agent
   */
  calculateKnowledgeGrowth(agent) {
    const localSize = agent.localMemory.size;
    const sharedSize = agent.learningState.sharedKnowledge.size;

    return {
      localKnowledge: localSize,
      sharedKnowledge: sharedSize,
      totalKnowledge: localSize + sharedSize,
      knowledgeDensity: (localSize + sharedSize) / (agent.peerConnections.size + 1),
    };
  }

  /**
   * Check for emergent patterns across agents
   * @param {string} agentId - Agent identifier
   */
  checkForEmergentPatterns(agentId) {
    const agent = this.cognitiveAgents.get(agentId);
    if (!agent) {
      return;
    }

    // Analyze collective patterns
    const collectivePatterns = this.analyzeCollectivePatterns();

    // Check for emergence criteria
    collectivePatterns.forEach(pattern => {
      if (pattern.occurrence > 0.7 && pattern.diversity > 0.5) {
        const emergentBehavior = {
          type: 'pattern_emergence',
          pattern: pattern.type,
          strength: pattern.occurrence,
          diversity: pattern.diversity,
          timestamp: Date.now(),
        };

        agent.emergentTraits.add(emergentBehavior.type);

        // Notify peers of emergent behavior
        this.notifyEmergentBehavior(agentId, emergentBehavior);
      }
    });
  }

  /**
   * Analyze patterns across all agents
   */
  analyzeCollectivePatterns() {
    const patternCounts = new Map();
    const patternAgents = new Map();

    // Count patterns across all agents
    for (const [agentId, agent] of this.cognitiveAgents) {
      for (const [key, value] of agent.localMemory) {
        if (key.startsWith('pattern_') || key.startsWith('consensus_')) {
          const patternType = value.type || 'unknown';

          if (!patternCounts.has(patternType)) {
            patternCounts.set(patternType, 0);
            patternAgents.set(patternType, new Set());
          }

          patternCounts.set(patternType, patternCounts.get(patternType) + 1);
          patternAgents.get(patternType).add(agentId);
        }
      }
    }

    // Calculate pattern statistics
    const totalAgents = this.cognitiveAgents.size;
    const patterns = [];

    for (const [patternType, count] of patternCounts) {
      const agentSet = patternAgents.get(patternType);
      patterns.push({
        type: patternType,
        count,
        occurrence: agentSet.size / totalAgents,
        diversity: agentSet.size / count, // How spread out the pattern is
      });
    }

    return patterns;
  }

  /**
   * Notify peers of emergent behavior
   * @param {string} agentId - Agent identifier
   * @param {Object} emergentBehavior - Emergent behavior detected
   */
  notifyEmergentBehavior(agentId, emergentBehavior) {
    const agent = this.cognitiveAgents.get(agentId);
    if (!agent) {
      return;
    }

    // Record in emergent behaviors
    if (!this.emergentBehaviors.has(emergentBehavior.type)) {
      this.emergentBehaviors.set(emergentBehavior.type, []);
    }
    this.emergentBehaviors.get(emergentBehavior.type).push({
      ...emergentBehavior,
      discoveredBy: agentId,
    });

    // Notify all peers
    for (const peerId of agent.peerConnections) {
      this.sendToPeer(peerId, {
        type: 'emergent_behavior',
        from: agentId,
        behavior: emergentBehavior,
      });
    }
  }

  /**
   * Process emergent behavior notification
   * @param {string} agentId - Receiving agent ID
   * @param {Object} message - Emergent behavior message
   */
  processEmergentBehavior(agentId, message) {
    const agent = this.cognitiveAgents.get(agentId);
    if (!agent) {
      return;
    }

    // Add to agent's emergent traits
    agent.emergentTraits.add(message.behavior.type);

    // Store in local memory for future reference
    agent.localMemory.set(
      `emergent_${message.behavior.type}_${Date.now()}`,
      {
        ...message.behavior,
        reportedBy: message.from,
      },
    );
  }

  /**
   * Get DAA statistics
   */
  getStatistics() {
    const stats = {
      totalAgents: this.cognitiveAgents.size,
      autonomyLevels: {},
      emergentBehaviors: this.emergentBehaviors.size,
      distributedKnowledge: 0,
      consensusDecisions: 0,
      autonomousDecisions: 0,
    };

    // Calculate detailed statistics
    for (const [_agentId, agent] of this.cognitiveAgents) {
      // Autonomy distribution
      const level = Math.floor(agent.autonomyLevel * 10) / 10;
      stats.autonomyLevels[level] = (stats.autonomyLevels[level] || 0) + 1;

      // Knowledge statistics
      stats.distributedKnowledge += agent.localMemory.size + agent.learningState.sharedKnowledge.size;

      // Decision statistics
      agent.consensusState.decisions.forEach(decision => {
        if (decision.autonomous) {
          stats.autonomousDecisions++;
        } else {
          stats.consensusDecisions++;
        }
      });
    }

    // Average metrics
    stats.avgKnowledgePerAgent = stats.totalAgents > 0 ?
      stats.distributedKnowledge / stats.totalAgents : 0;

    stats.autonomyRate = (stats.autonomousDecisions + stats.consensusDecisions) > 0 ?
      stats.autonomousDecisions / (stats.autonomousDecisions + stats.consensusDecisions) : 0;

    return stats;
  }

  /**
   * Connect two agents as peers
   * @param {string} agentId1 - First agent
   * @param {string} agentId2 - Second agent
   */
  connectAgents(agentId1, agentId2) {
    const agent1 = this.cognitiveAgents.get(agentId1);
    const agent2 = this.cognitiveAgents.get(agentId2);

    if (agent1 && agent2) {
      agent1.peerConnections.add(agentId2);
      agent2.peerConnections.add(agentId1);

      console.log(`Connected DAA agents ${agentId1} and ${agentId2}`);
    }
  }

  /**
   * Create mesh network of agents
   * @param {Array} agentIds - List of agent IDs
   */
  createMeshNetwork(agentIds) {
    // Connect every agent to every other agent
    for (let i = 0; i < agentIds.length; i++) {
      for (let j = i + 1; j < agentIds.length; j++) {
        this.connectAgents(agentIds[i], agentIds[j]);
      }
    }

    console.log(`Created mesh network with ${agentIds.length} agents`);
  }
}
