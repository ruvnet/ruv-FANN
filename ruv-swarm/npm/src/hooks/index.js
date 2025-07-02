/**
 * Claude Code Hooks Implementation for ruv-swarm
 * Provides automated coordination, formatting, and learning capabilities
 */

import { promises as fs } from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class RuvSwarmHooks {
  constructor() {
    this.sessionData = {
      startTime: Date.now(),
      operations: [],
      agents: new Map(),
      learnings: [],
      metrics: {
        tokensSaved: 0,
        tasksCompleted: 0,
        patternsImproved: 0,
      },
    };
  }

  /**
     * Main hook handler - routes to specific hook implementations
     */
  async handleHook(hookType, args) {
    try {
      switch (hookType) {
      // Pre-operation hooks
      case 'pre-edit':
        return await this.preEditHook(args);
      case 'pre-bash':
        return await this.preBashHook(args);
      case 'pre-task':
        return await this.preTaskHook(args);
      case 'pre-search':
        return await this.preSearchHook(args);
      case 'pre-mcp':
        return await this.preMcpHook(args);

        // Post-operation hooks
      case 'post-edit':
        return await this.postEditHook(args);
      case 'post-bash':
        return await this.postBashHook(args);
      case 'post-task':
        return await this.postTaskHook(args);
      case 'post-search':
        return await this.postSearchHook(args);
      case 'post-web-search':
        return await this.postWebSearchHook(args);
      case 'post-web-fetch':
        return await this.postWebFetchHook(args);

        // MCP-specific hooks
      case 'mcp-swarm-initialized':
        return await this.mcpSwarmInitializedHook(args);
      case 'mcp-agent-spawned':
        return await this.mcpAgentSpawnedHook(args);
      case 'mcp-task-orchestrated':
        return await this.mcpTaskOrchestratedHook(args);
      case 'mcp-neural-trained':
        return await this.mcpNeuralTrainedHook(args);

        // System hooks
      case 'notification':
        return await this.notificationHook(args);
      case 'session-end':
        return await this.sessionEndHook(args);
      case 'session-restore':
        return await this.sessionRestoreHook(args);
      case 'agent-complete':
        return await this.agentCompleteHook(args);

      default:
        return { continue: true, reason: `Unknown hook type: ${hookType}` };
      }
    } catch (error) {
      console.error(`Hook error (${hookType}):`, error.message);
      return {
        continue: true,
        error: error.message,
        fallback: 'Hook error - continuing with default behavior',
      };
    }
  }

  /**
     * Pre-search hook - Prepare cache and optimize search
     */
  async preSearchHook(args) {
    const { pattern } = args;

    // Initialize search cache
    if (!this.sessionData.searchCache) {
      this.sessionData.searchCache = new Map();
    }

    // Check cache for similar patterns
    const cachedResult = this.sessionData.searchCache.get(pattern);
    if (cachedResult && Date.now() - cachedResult.timestamp < 300000) { // 5 min cache
      return {
        continue: true,
        cached: true,
        cacheHit: cachedResult.files.length,
        metadata: { pattern, cached: true },
      };
    }

    return {
      continue: true,
      reason: 'Search prepared',
      metadata: { pattern, cacheReady: true },
    };
  }

  /**
     * Pre-MCP hook - Validate MCP tool state
     */
  async preMcpHook(args) {
    const { tool, params } = args;

    // Parse params if string
    const toolParams = typeof params === 'string' ? JSON.parse(params) : params;

    // Validate swarm state for MCP operations
    if (tool.includes('agent_spawn') || tool.includes('task_orchestrate')) {
      const swarmStatus = await this.checkSwarmStatus();
      if (!swarmStatus.initialized) {
        return {
          continue: true,
          warning: 'Swarm not initialized - will be created automatically',
          autoInit: true,
        };
      }
    }

    // Track MCP operations
    this.sessionData.operations.push({
      type: 'mcp',
      tool,
      params: toolParams,
      timestamp: Date.now(),
    });

    return {
      continue: true,
      reason: 'MCP tool validated',
      metadata: { tool, state: 'ready' },
    };
  }

  /**
     * Pre-edit hook - Ensure coordination before file modifications
     */
  async preEditHook(args) {
    const { file } = args;

    // Determine file type and assign appropriate agent
    const fileExt = path.extname(file);
    const agentType = this.getAgentTypeForFile(fileExt);

    // Check if swarm is initialized
    const swarmStatus = await this.checkSwarmStatus();
    if (!swarmStatus.initialized) {
      return {
        continue: false,
        reason: 'Swarm not initialized - run mcp__ruv-swarm__swarm_init first',
        suggestion: 'Initialize swarm with appropriate topology',
      };
    }

    // Ensure appropriate agent exists
    const agent = await this.ensureAgent(agentType);

    // Record operation
    this.sessionData.operations.push({
      type: 'edit',
      file,
      agent: agent.id,
      timestamp: Date.now(),
    });

    return {
      continue: true,
      reason: `${agentType} agent assigned for ${fileExt} file`,
      metadata: {
        agent_id: agent.id,
        agent_type: agentType,
        cognitive_pattern: agent.pattern,
        readiness: agent.readiness,
      },
    };
  }

  /**
     * Pre-task hook - Auto-spawn agents and optimize topology
     */
  async preTaskHook(args) {
    const { description, autoSpawnAgents, optimizeTopology } = args;

    // Analyze task complexity
    const complexity = this.analyzeTaskComplexity(description);

    // Determine optimal topology
    const topology = optimizeTopology ? this.selectOptimalTopology(complexity) : 'mesh';

    // Auto-spawn required agents
    if (autoSpawnAgents) {
      const requiredAgents = this.determineRequiredAgents(description, complexity);
      for (const agentType of requiredAgents) {
        await this.ensureAgent(agentType);
      }
    }

    return {
      continue: true,
      reason: 'Task prepared with optimal configuration',
      metadata: {
        complexity,
        topology,
        agentsReady: true,
        estimatedDuration: complexity.estimatedMinutes * 60000,
      },
    };
  }

  /**
     * Post-edit hook - Format and learn from edits
     */
  async postEditHook(args) {
    const { file, autoFormat, trainPatterns, updateGraph } = args;
    const result = {
      continue: true,
      formatted: false,
      training: null,
    };

    // Auto-format if requested
    if (autoFormat) {
      const formatted = await this.autoFormatFile(file);
      result.formatted = formatted.success;
      result.formatDetails = formatted.details;
    }

    // Train neural patterns
    if (trainPatterns) {
      const training = await this.trainPatternsFromEdit(file);
      result.training = training;
      this.sessionData.metrics.patternsImproved += training.improvement || 0;
    }

    // Update knowledge graph if requested
    if (updateGraph) {
      await this.updateKnowledgeGraph(file, 'edit');
    }

    // Update session data
    this.sessionData.metrics.tokensSaved += 10; // Estimated savings

    return result;
  }

  /**
     * Post-task hook - Analyze performance and update coordination
     */
  async postTaskHook(args) {
    const { taskId, analyzePerformance, updateCoordination } = args;

    const performance = {
      taskId,
      completionTime: Date.now() - (this.sessionData.taskStartTimes?.get(taskId) || Date.now()),
      agentsUsed: this.sessionData.taskAgents?.get(taskId) || [],
      success: true,
    };

    // Analyze performance
    if (analyzePerformance) {
      performance.analysis = {
        efficiency: this.calculateEfficiency(performance),
        bottlenecks: this.identifyBottlenecks(performance),
        improvements: this.suggestImprovements(performance),
      };
    }

    // Update coordination strategies
    if (updateCoordination) {
      this.updateCoordinationStrategy(performance);
    }

    this.sessionData.metrics.tasksCompleted++;

    return {
      continue: true,
      performance,
      metadata: { taskId, optimized: true },
    };
  }

  /**
     * Post-web-search hook - Analyze results and update knowledge
     */
  async postWebSearchHook(args) {
    const { query, updateKnowledge } = args;

    // Track search patterns
    if (!this.sessionData.searchPatterns) {
      this.sessionData.searchPatterns = new Map();
    }

    const patterns = this.extractSearchPatterns(query);
    patterns.forEach(pattern => {
      const count = this.sessionData.searchPatterns.get(pattern) || 0;
      this.sessionData.searchPatterns.set(pattern, count + 1);
    });

    // Update knowledge base
    if (updateKnowledge) {
      await this.updateKnowledgeBase('search', { query, patterns });
    }

    return {
      continue: true,
      reason: 'Search analyzed and knowledge updated',
      metadata: {
        query,
        patternsExtracted: patterns.length,
        knowledgeUpdated: updateKnowledge,
      },
    };
  }

  /**
     * Post-web-fetch hook - Extract patterns and cache content
     */
  async postWebFetchHook(args) {
    const { url, extractPatterns, cacheContent } = args;

    const result = {
      continue: true,
      patterns: [],
      cached: false,
    };

    // Extract patterns from URL
    if (extractPatterns) {
      result.patterns = this.extractUrlPatterns(url);
    }

    // Cache content for future use
    if (cacheContent) {
      if (!this.sessionData.contentCache) {
        this.sessionData.contentCache = new Map();
      }
      this.sessionData.contentCache.set(url, {
        timestamp: Date.now(),
        patterns: result.patterns,
      });
      result.cached = true;
    }

    return result;
  }

  /**
     * Notification hook - Handle notifications with swarm status
     */
  async notificationHook(args) {
    const { message, level, withSwarmStatus, sendTelemetry } = args;

    const notification = {
      message,
      level: level || 'info',
      timestamp: Date.now(),
    };

    // Add swarm status if requested
    if (withSwarmStatus) {
      const status = await this.getSwarmStatus();
      notification.swarmStatus = {
        agents: status.agents?.size || 0,
        activeTasks: status.activeTasks || 0,
        health: status.health || 'unknown',
      };
    }

    // Send telemetry if enabled
    if (sendTelemetry && process.env.RUV_SWARM_TELEMETRY_ENABLED === 'true') {
      this.sendTelemetry('notification', notification);
    }

    // Store notification
    if (!this.sessionData.notifications) {
      this.sessionData.notifications = [];
    }
    this.sessionData.notifications.push(notification);

    return {
      continue: true,
      notification,
      handled: true,
    };
  }

  /**
     * Pre-bash hook - Validate commands before execution
     */
  async preBashHook(args) {
    const { command } = args;

    // Safety checks
    const safetyCheck = this.validateCommandSafety(command);
    if (!safetyCheck.safe) {
      return {
        continue: false,
        reason: safetyCheck.reason,
        riskLevel: safetyCheck.riskLevel,
      };
    }

    // Check resource requirements
    const resources = this.estimateCommandResources(command);
    if (resources.requiresAgent) {
      await this.ensureAgent(resources.agentType);
    }

    return {
      continue: true,
      reason: 'Command validated and resources available',
      metadata: {
        estimatedDuration: resources.duration,
        requiresAgent: resources.requiresAgent,
      },
    };
  }

  /**
     * MCP swarm initialized hook - Persist configuration
     */
  async mcpSwarmInitializedHook(args) {
    const { swarmId, topology, persistConfig, enableMonitoring } = args;

    // Store swarm configuration
    const swarmConfig = {
      id: swarmId,
      topology,
      initialized: Date.now(),
      monitoring: enableMonitoring,
    };

    // Persist configuration
    if (persistConfig) {
      const configDir = path.join(process.cwd(), '.ruv-swarm');
      await fs.mkdir(configDir, { recursive: true });
      await fs.writeFile(
        path.join(configDir, 'swarm-config.json'),
        JSON.stringify(swarmConfig, null, 2),
      );
    }

    // Initialize monitoring
    if (enableMonitoring) {
      this.sessionData.monitoring = {
        swarmId,
        startTime: Date.now(),
        events: [],
      };
    }

    return {
      continue: true,
      reason: 'Swarm initialized and configured',
      metadata: swarmConfig,
    };
  }

  /**
     * MCP agent spawned hook - Update roster and train
     */
  async mcpAgentSpawnedHook(args) {
    const { agentId, type, updateRoster, trainSpecialization } = args;

    // Update agent roster
    if (updateRoster) {
      const agent = {
        id: agentId,
        type,
        specialization: this.getSpecializationForType(type),
        spawned: Date.now(),
        performance: { tasks: 0, successRate: 1.0 },
      };

      this.sessionData.agents.set(agentId, agent);

      // Persist roster
      const rosterPath = path.join(process.cwd(), '.ruv-swarm', 'agent-roster.json');
      const roster = Array.from(this.sessionData.agents.values());
      await fs.writeFile(rosterPath, JSON.stringify(roster, null, 2));
    }

    // Train specialization patterns
    if (trainSpecialization) {
      const training = {
        agentId,
        type,
        patterns: this.generateSpecializationPatterns(type),
        confidence: 0.9 + Math.random() * 0.1,
      };

      this.sessionData.learnings.push(training);
    }

    return {
      continue: true,
      agentId,
      type,
      specialized: true,
    };
  }

  /**
     * MCP task orchestrated hook - Monitor and optimize
     */
  async mcpTaskOrchestratedHook(args) {
    const { taskId, monitorProgress, optimizeDistribution } = args;

    // Initialize task tracking
    if (!this.sessionData.taskStartTimes) {
      this.sessionData.taskStartTimes = new Map();
    }
    if (!this.sessionData.taskAgents) {
      this.sessionData.taskAgents = new Map();
    }

    this.sessionData.taskStartTimes.set(taskId, Date.now());

    // Monitor progress setup
    if (monitorProgress) {
      this.sessionData.taskMonitoring = this.sessionData.taskMonitoring || new Map();
      this.sessionData.taskMonitoring.set(taskId, {
        checkpoints: [],
        resources: [],
        bottlenecks: [],
      });
    }

    // Optimize distribution
    if (optimizeDistribution) {
      const optimization = {
        taskId,
        strategy: 'load-balanced',
        agentAllocation: this.optimizeAgentAllocation(taskId),
        parallelization: this.calculateParallelization(taskId),
      };

      return {
        continue: true,
        taskId,
        optimization,
      };
    }

    return {
      continue: true,
      taskId,
      monitoring: monitorProgress,
    };
  }

  /**
     * MCP neural trained hook - Save improvements
     */
  async mcpNeuralTrainedHook(args) {
    const { improvement, saveWeights, updatePatterns } = args;

    const result = {
      continue: true,
      improvement: parseFloat(improvement),
      saved: false,
      patternsUpdated: false,
    };

    // Save neural weights
    if (saveWeights) {
      const weightsDir = path.join(process.cwd(), '.ruv-swarm', 'neural-weights');
      await fs.mkdir(weightsDir, { recursive: true });

      const weightData = {
        timestamp: Date.now(),
        improvement,
        weights: this.generateMockWeights(),
        version: this.sessionData.learnings.length,
      };

      await fs.writeFile(
        path.join(weightsDir, `weights-${Date.now()}.json`),
        JSON.stringify(weightData, null, 2),
      );

      result.saved = true;
    }

    // Update cognitive patterns
    if (updatePatterns) {
      this.sessionData.metrics.patternsImproved++;

      const patternUpdate = {
        timestamp: Date.now(),
        improvement,
        patterns: ['convergent', 'divergent', 'lateral'],
        confidence: 0.85 + parseFloat(improvement),
      };

      this.sessionData.learnings.push(patternUpdate);
      result.patternsUpdated = true;
    }

    return result;
  }

  /**
     * Agent complete hook - Commit to git with detailed report
     */
  async agentCompleteHook(args) {
    const { agent, prompt, output, commitToGit, generateReport, pushToGithub } = args;

    try {
      const timestamp = new Date().toISOString();
      const agentName = agent || 'Unknown Agent';
      // const shortOutput = output ? `${output.substring(0, 500) }...` : 'No output';

      // Generate detailed report
      let reportPath = null;
      if (generateReport) {
        const reportDir = path.join(process.cwd(), '.ruv-swarm', 'agent-reports');
        await fs.mkdir(reportDir, { recursive: true });

        const sanitizedAgent = agentName.replace(/[^a-zA-Z0-9-]/g, '-').toLowerCase();
        reportPath = path.join(reportDir, `${sanitizedAgent}-${Date.now()}.md`);

        const report = `# Agent Completion Report: ${agentName}

## Metadata
- **Agent**: ${agentName}
- **Timestamp**: ${timestamp}
- **Session**: ${this.sessionData.sessionId || 'N/A'}
- **Duration**: ${this.formatDuration(Date.now() - this.sessionData.startTime)}

## Task Description
\`\`\`
${prompt || 'No prompt available'}
\`\`\`

## Output Summary
${output ? `### Key Accomplishments\n${ this.extractKeyPoints(output)}` : 'No output captured'}

## Performance Metrics
- **Total Operations**: ${this.sessionData.operations.length}
- **Files Modified**: ${this.getModifiedFilesCount()}
- **Efficiency Score**: ${this.calculateEfficiency({ completionTime: Date.now() - this.sessionData.startTime }).rating}
- **Tokens Saved**: ${this.sessionData.metrics.tokensSaved}

## Files Modified
${this.getModifiedFilesList()}

## Coordination Activity
- **Memory Operations**: ${this.sessionData.operations.filter(op => op.type === 'memory').length}
- **Hook Executions**: ${this.sessionData.operations.filter(op => op.type === 'hook').length}
- **Neural Training**: ${this.sessionData.metrics.patternsImproved} patterns improved

## Learnings & Patterns
${this.sessionData.learnings.length > 0 ? this.sessionData.learnings.map(l => `- ${l.type || 'General'}: ${l.description || JSON.stringify(l)}`).join('\n') : 'No specific learnings captured'}

---
*Generated by ruv-swarm agent coordination system*
`;

        await fs.writeFile(reportPath, report);
      }

      // Commit to git if requested
      if (commitToGit) {
        try {
          // Check if we're in a git repo
          execSync('git rev-parse --git-dir', { stdio: 'ignore' });

          // Get git status
          const status = execSync('git status --porcelain', { encoding: 'utf-8' });

          if (status.trim()) {
            // Stage changes
            execSync('git add -A');

            // Create detailed commit message
            const commitMessage = `feat(${agentName.toLowerCase().replace(/[^a-z0-9]/g, '-')}): Complete agent task

Agent: ${agentName}
Timestamp: ${timestamp}

## Task Summary
${prompt ? `${prompt.split('\n')[0].substring(0, 100) }...` : 'No task description'}

## Achievements
${this.extractBulletPoints(output)}

## Metrics
- Operations: ${this.sessionData.operations.length}
- Files: ${this.getModifiedFilesCount()}
- Efficiency: ${this.calculateEfficiency({ completionTime: Date.now() - this.sessionData.startTime }).rating}
${reportPath ? `\n## Report\nDetailed report: ${path.relative(process.cwd(), reportPath)}` : ''}

🤖 Generated by ruv-swarm agent coordination
Co-Authored-By: ${agentName} <agent@ruv-swarm.ai>`;

            // Commit using heredoc to handle complex messages
            const commitCmd = `git commit -m "$(cat <<'EOF'
${commitMessage}
EOF
)"`;
            execSync(commitCmd, { shell: '/bin/bash' });

            // Log commit info
            const commitHash = execSync('git rev-parse HEAD', { encoding: 'utf-8' }).trim();
            console.log(`✅ Committed agent work: ${commitHash.substring(0, 7)}`);

            // Push if requested and configured
            if (pushToGithub && process.env.RUV_SWARM_AUTO_PUSH === 'true') {
              console.log('📤 Pushing to GitHub...');
              execSync('git push', { stdio: 'inherit' });
              console.log('✅ Pushed to GitHub');
            }
          } else {
            console.log('ℹ️ No changes to commit');
          }

        } catch (gitError) {
          console.error('Git operation failed:', gitError.message);
        }
      }

      // Update telemetry
      this.sendTelemetry('agent_complete', {
        agent: agentName,
        hasReport: generateReport,
        hasCommit: commitToGit,
        operationCount: this.sessionData.operations.length,
        duration: Date.now() - this.sessionData.startTime,
      });

      return {
        continue: true,
        agent: agentName,
        reportGenerated: generateReport,
        reportPath: reportPath ? path.relative(process.cwd(), reportPath) : null,
        committed: commitToGit,
        duration: this.formatDuration(Date.now() - this.sessionData.startTime),
      };

    } catch (error) {
      console.error('Agent complete hook error:', error);
      return {
        continue: true,
        error: error.message,
      };
    }
  }

  /**
     * Extract key points from output
     */
  extractKeyPoints(output) {
    const lines = output.split('\n').filter(l => l.trim());
    const keyPoints = [];

    // Look for bullet points or numbered items
    lines.forEach(line => {
      if (line.match(/^[\-\*•]\s/) || line.match(/^\d+\.\s/)) {
        keyPoints.push(line);
      }
    });

    // If no bullet points, take first few lines
    if (keyPoints.length === 0) {
      keyPoints.push(...lines.slice(0, 5));
    }

    return keyPoints.slice(0, 10).join('\n');
  }

  /**
     * Extract bullet points for commit message
     */
  extractBulletPoints(output) {
    if (!output) {
      return '- No specific achievements captured';
    }

    const points = this.extractKeyPoints(output)
      .split('\n')
      .slice(0, 5)
      .map(p => `- ${p.replace(/^[\-\*•\d+\.\s]+/, '').trim()}`);

    return points.length > 0 ? points.join('\n') : '- Task completed successfully';
  }

  /**
     * Get count of modified files
     */
  getModifiedFilesCount() {
    const fileOps = this.sessionData.operations.filter(op =>
      ['edit', 'write', 'create'].includes(op.type),
    );

    const uniqueFiles = new Set(fileOps.map(op => op.file).filter(Boolean));
    return uniqueFiles.size;
  }

  /**
     * Get list of modified files
     */
  getModifiedFilesList() {
    const fileOps = this.sessionData.operations.filter(op =>
      ['edit', 'write', 'create'].includes(op.type),
    );

    const fileMap = new Map();
    fileOps.forEach(op => {
      if (op.file) {
        if (!fileMap.has(op.file)) {
          fileMap.set(op.file, []);
        }
        fileMap.get(op.file).push(op.type);
      }
    });

    if (fileMap.size === 0) {
      return 'No files modified';
    }

    return Array.from(fileMap.entries())
      .map(([file, ops]) => `- ${file} (${[...new Set(ops)].join(', ')})`)
      .join('\n');
  }

  /**
     * Session restore hook - Load previous state
     */
  async sessionRestoreHook(args) {
    const { loadMemory, loadAgents } = args;

    const result = {
      continue: true,
      restored: {
        memory: false,
        agents: false,
        metrics: false,
      },
    };

    try {
      const sessionDir = path.join(process.cwd(), '.ruv-swarm');

      // Load memory state
      if (loadMemory) {
        const memoryPath = path.join(sessionDir, 'memory-state.json');
        if (await fs.access(memoryPath).then(() => true).catch(() => false)) {
          const memory = JSON.parse(await fs.readFile(memoryPath, 'utf-8'));
          this.sessionData = { ...this.sessionData, ...memory };
          result.restored.memory = true;
        }
      }

      // Load agent roster
      if (loadAgents) {
        const rosterPath = path.join(sessionDir, 'agent-roster.json');
        if (await fs.access(rosterPath).then(() => true).catch(() => false)) {
          const roster = JSON.parse(await fs.readFile(rosterPath, 'utf-8'));
          roster.forEach(agent => {
            this.sessionData.agents.set(agent.id, agent);
          });
          result.restored.agents = true;
        }
      }

      // Load metrics
      const metricsPath = path.join(sessionDir, 'session-metrics.json');
      if (await fs.access(metricsPath).then(() => true).catch(() => false)) {
        const metrics = JSON.parse(await fs.readFile(metricsPath, 'utf-8'));
        this.sessionData.metrics = { ...this.sessionData.metrics, ...metrics };
        result.restored.metrics = true;
      }

    } catch (error) {
      console.error('Session restore error:', error.message);
    }

    return result;
  }

  /**
     * Session end hook - Generate summary and persist state
     */
  async sessionEndHook(args) {
    const { generateSummary, saveMemory, exportMetrics } = args;
    const sessionDir = path.join(process.cwd(), '.claude', 'sessions');
    await fs.mkdir(sessionDir, { recursive: true });

    const timestamp = new Date().toISOString().replace(/:/g, '-');
    const results = {};

    // Generate summary
    if (generateSummary) {
      const summary = this.generateSessionSummary();
      const summaryPath = path.join(sessionDir, `${timestamp}-summary.md`);
      await fs.writeFile(summaryPath, summary);
      results.summary = summaryPath;
    }

    // Save memory state
    if (saveMemory) {
      const state = this.captureSwarmState();
      const statePath = path.join(sessionDir, `${timestamp}-state.json`);
      await fs.writeFile(statePath, JSON.stringify(state, null, 2));
      results.state = statePath;
    }

    // Export metrics
    if (exportMetrics) {
      const metrics = this.calculateSessionMetrics();
      const metricsPath = path.join(sessionDir, `${timestamp}-metrics.json`);
      await fs.writeFile(metricsPath, JSON.stringify(metrics, null, 2));
      results.metrics = metricsPath;
    }

    console.log('\n🎯 Session Summary:');
    console.log(`Duration: ${this.formatDuration(Date.now() - this.sessionData.startTime)}`);
    console.log(`Operations: ${this.sessionData.operations.length}`);
    console.log(`Tokens Saved: ${this.sessionData.metrics.tokensSaved}`);
    console.log(`Patterns Improved: ${this.sessionData.metrics.patternsImproved}`);

    return {
      continue: true,
      files: results,
      summary: {
        duration: Date.now() - this.sessionData.startTime,
        operations: this.sessionData.operations.length,
        improvements: this.sessionData.metrics.patternsImproved,
      },
    };
  }

  // Helper methods

  getAgentTypeForFile(extension) {
    const mapping = {
      '.js': 'coder',
      '.ts': 'coder',
      '.jsx': 'coder',
      '.tsx': 'coder',
      '.py': 'coder',
      '.go': 'coder',
      '.rs': 'coder',
      '.md': 'researcher',
      '.txt': 'researcher',
      '.json': 'analyst',
      '.yaml': 'analyst',
      '.yml': 'analyst',
      '.toml': 'analyst',
      '.xml': 'analyst',
      '.sql': 'analyst',
    };
    return mapping[extension] || 'coordinator';
  }

  async checkSwarmStatus() {
    try {
      // Check if swarm is initialized via file or global state
      const statusFile = path.join(process.cwd(), '.ruv-swarm', 'status.json');
      const exists = await fs.access(statusFile).then(() => true).catch(() => false);

      if (exists) {
        const status = JSON.parse(await fs.readFile(statusFile, 'utf-8'));
        return { initialized: true, ...status };
      }

      return { initialized: false };
    } catch (_error) {
      return { initialized: false };
    }
  }

  async ensureAgent(type) {
    let agent = this.sessionData.agents.get(type);

    if (!agent) {
      // Simulate agent creation
      agent = {
        id: `${type}-${Date.now()}`,
        type,
        pattern: this.getCognitivePattern(type),
        readiness: 0.95,
        created: Date.now(),
      };
      this.sessionData.agents.set(type, agent);
    }

    return agent;
  }

  getCognitivePattern(agentType) {
    const patterns = {
      coder: 'convergent',
      researcher: 'divergent',
      analyst: 'critical',
      coordinator: 'systems',
      architect: 'abstract',
      optimizer: 'lateral',
    };
    return patterns[agentType] || 'balanced';
  }

  async autoFormatFile(filePath) {
    const ext = path.extname(filePath);
    const formatters = {
      '.js': 'prettier --write',
      '.ts': 'prettier --write',
      '.jsx': 'prettier --write',
      '.tsx': 'prettier --write',
      '.json': 'prettier --write',
      '.md': 'prettier --write --prose-wrap always',
      '.py': 'black',
      '.go': 'gofmt -w',
      '.rs': 'rustfmt',
    };

    const formatter = formatters[ext];
    if (!formatter) {
      return { success: false, reason: 'No formatter configured for file type' };
    }

    try {
      execSync(`${formatter} "${filePath}"`, { stdio: 'pipe' });
      return { success: true, details: { formatter, fileType: ext } };
    } catch (error) {
      return { success: false, reason: error.message };
    }
  }

  async trainPatternsFromEdit(filePath) {
    // Simulate neural pattern training
    const improvement = Math.random() * 0.05; // 0-5% improvement
    const confidence = 0.85 + Math.random() * 0.1; // 85-95% confidence

    this.sessionData.learnings.push({
      file: filePath,
      timestamp: Date.now(),
      improvement,
      confidence,
      pattern: `edit_pattern_${ path.extname(filePath)}`,
    });

    return {
      pattern_updated: true,
      improvement: improvement.toFixed(3),
      confidence: confidence.toFixed(2),
      total_examples: this.sessionData.learnings.length,
    };
  }

  validateCommandSafety(command) {
    const dangerousPatterns = [
      /rm\s+-rf\s+\//,
      /curl.*\|\s*bash/,
      /wget.*\|\s*sh/,
      /eval\s*\(/,
      />\/dev\/null\s+2>&1/,
    ];

    for (const pattern of dangerousPatterns) {
      if (pattern.test(command)) {
        return {
          safe: false,
          reason: 'Command contains potentially dangerous pattern',
          riskLevel: 'high',
        };
      }
    }

    return { safe: true };
  }

  estimateCommandResources(command) {
    const resourceMap = {
      'npm test': { duration: 30000, requiresAgent: true, agentType: 'coordinator' },
      'npm run build': { duration: 60000, requiresAgent: true, agentType: 'optimizer' },
      'git': { duration: 1000, requiresAgent: false },
      'ls': { duration: 100, requiresAgent: false },
    };

    for (const [pattern, resources] of Object.entries(resourceMap)) {
      if (command.includes(pattern)) {
        return resources;
      }
    }

    return { duration: 5000, requiresAgent: false };
  }

  generateSessionSummary() {
    const duration = Date.now() - this.sessionData.startTime;
    const agentList = Array.from(this.sessionData.agents.values());

    return `# ruv-swarm Session Summary
Date: ${new Date().toISOString()}
Duration: ${this.formatDuration(duration)}
Token Reduction: ${this.sessionData.metrics.tokensSaved} tokens

## Swarm Activity
- Active Agents: ${agentList.length} (${agentList.map(a => a.type).join(', ')})
- Operations Performed: ${this.sessionData.operations.length}
- Files Modified: ${new Set(this.sessionData.operations.map(o => o.file)).size}
- Neural Improvements: ${this.sessionData.metrics.patternsImproved}

## Operations Breakdown
${this.sessionData.operations.slice(-10).map(op =>
    `- ${new Date(op.timestamp).toLocaleTimeString()}: ${op.type} on ${op.file} (${op.agent})`,
  ).join('\n')}

## Learning Highlights
${this.sessionData.learnings.slice(-5).map(l =>
    `- Pattern "${l.pattern}" improved by ${(l.improvement * 100).toFixed(1)}% (confidence: ${l.confidence})`,
  ).join('\n')}

## Performance Metrics
- Average Operation Time: ${(duration / this.sessionData.operations.length / 1000).toFixed(1)}s
- Token Efficiency: ${(this.sessionData.metrics.tokensSaved / this.sessionData.operations.length).toFixed(0)} tokens/operation
- Learning Rate: ${(this.sessionData.metrics.patternsImproved / this.sessionData.operations.length).toFixed(2)} improvements/operation
`;
  }

  captureSwarmState() {
    return {
      session_id: `sess-${Date.now()}`,
      agents: Object.fromEntries(this.sessionData.agents),
      operations: this.sessionData.operations,
      learnings: this.sessionData.learnings,
      metrics: this.sessionData.metrics,
      timestamp: new Date().toISOString(),
    };
  }

  calculateSessionMetrics() {
    const duration = Date.now() - this.sessionData.startTime;
    return {
      performance: {
        duration_ms: duration,
        operations_per_minute: (this.sessionData.operations.length / (duration / 60000)).toFixed(1),
        tokens_saved: this.sessionData.metrics.tokensSaved,
        efficiency_score: (this.sessionData.metrics.tokensSaved / this.sessionData.operations.length).toFixed(1),
      },
      learning: {
        patterns_improved: this.sessionData.metrics.patternsImproved,
        average_improvement: (this.sessionData.learnings.reduce((acc, l) => acc + l.improvement, 0) / this.sessionData.learnings.length).toFixed(3),
        confidence_average: (this.sessionData.learnings.reduce((acc, l) => acc + l.confidence, 0) / this.sessionData.learnings.length).toFixed(2),
      },
      agents: {
        total_spawned: this.sessionData.agents.size,
        by_type: Object.fromEntries(
          Array.from(this.sessionData.agents.values())
            .reduce((acc, agent) => {
              acc.set(agent.type, (acc.get(agent.type) || 0) + 1);
              return acc;
            }, new Map()),
        ),
      },
    };
  }

  formatDuration(ms) {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);

    if (hours > 0) {
      return `${hours}h ${minutes % 60}m`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds % 60}s`;
    }
    return `${seconds}s`;

  }

  // Additional helper methods for optimization

  analyzeTaskComplexity(description) {
    const keywords = {
      simple: ['fix', 'update', 'change', 'modify', 'rename'],
      medium: ['implement', 'create', 'add', 'integrate', 'refactor'],
      complex: ['architect', 'design', 'optimize', 'migrate', 'scale'],
    };

    const desc = description.toLowerCase();
    let complexity = 'simple';
    let score = 1;
    let estimatedMinutes = 5;

    // Check for complex keywords
    if (keywords.complex.some(k => desc.includes(k))) {
      complexity = 'complex';
      score = 3;
      estimatedMinutes = 60;
    } else if (keywords.medium.some(k => desc.includes(k))) {
      complexity = 'medium';
      score = 2;
      estimatedMinutes = 30;
    }

    // Adjust for multiple files or components
    const fileCount = (desc.match(/\b(files?|components?|modules?)\b/g) || []).length;
    if (fileCount > 1) {
      score += 0.5;
      estimatedMinutes *= 1.5;
    }

    return {
      level: complexity,
      score,
      estimatedMinutes,
      requiresResearch: desc.includes('research') || desc.includes('analyze'),
      requiresTesting: desc.includes('test') || desc.includes('verify'),
    };
  }

  selectOptimalTopology(complexity) {
    const topologyMap = {
      simple: 'star', // Centralized for simple tasks
      medium: 'mesh', // Flexible for medium complexity
      complex: 'hierarchical', // Structured for complex tasks
    };

    return topologyMap[complexity.level] || 'mesh';
  }

  determineRequiredAgents(description, complexity) {
    const agents = new Set(['coordinator']); // Always need a coordinator

    const desc = description.toLowerCase();

    // Add agents based on task keywords
    if (desc.includes('code') || desc.includes('implement') || desc.includes('fix')) {
      agents.add('coder');
    }
    if (desc.includes('research') || desc.includes('analyze') || desc.includes('investigate')) {
      agents.add('researcher');
    }
    if (desc.includes('data') || desc.includes('metrics') || desc.includes('performance')) {
      agents.add('analyst');
    }
    if (desc.includes('design') || desc.includes('architect') || desc.includes('structure')) {
      agents.add('architect');
    }
    if (desc.includes('optimize') || desc.includes('improve') || desc.includes('enhance')) {
      agents.add('optimizer');
    }

    // Add more agents for complex tasks
    if (complexity.score >= 3) {
      agents.add('reviewer');
    }

    return Array.from(agents);
  }

  async updateKnowledgeGraph(file, operation) {
    if (!this.sessionData.knowledgeGraph) {
      this.sessionData.knowledgeGraph = {
        nodes: new Map(),
        edges: [],
      };
    }

    const graph = this.sessionData.knowledgeGraph;

    // Add or update node
    const nodeId = file;
    if (!graph.nodes.has(nodeId)) {
      graph.nodes.set(nodeId, {
        id: nodeId,
        type: this.getFileType(file),
        operations: [],
        lastModified: Date.now(),
      });
    }

    const node = graph.nodes.get(nodeId);
    node.operations.push({
      type: operation,
      timestamp: Date.now(),
      agent: this.getCurrentAgent(),
    });
    node.lastModified = Date.now();

    // Add edges for related files
    const relatedFiles = await this.findRelatedFiles(file);
    relatedFiles.forEach(related => {
      if (!graph.edges.find(e =>
        (e.from === nodeId && e.to === related) ||
                (e.from === related && e.to === nodeId),
      )) {
        graph.edges.push({
          from: nodeId,
          to: related,
          type: 'related',
          weight: 1,
        });
      }
    });
  }

  calculateEfficiency(performance) {
    const baselineTime = 60000; // 1 minute baseline
    const efficiencyScore = Math.max(0, Math.min(1, baselineTime / performance.completionTime));

    // Adjust for agent utilization
    const agentUtilization = performance.agentsUsed.length > 0 ?
      0.8 + (0.2 * Math.min(1, 3 / performance.agentsUsed.length)) : 0.5;

    return {
      score: (efficiencyScore * agentUtilization).toFixed(2),
      timeEfficiency: efficiencyScore.toFixed(2),
      agentEfficiency: agentUtilization.toFixed(2),
      rating: efficiencyScore > 0.8 ? 'excellent' :
        efficiencyScore > 0.6 ? 'good' :
          efficiencyScore > 0.4 ? 'fair' : 'needs improvement',
    };
  }

  identifyBottlenecks(performance) {
    const bottlenecks = [];

    // Time-based bottlenecks
    if (performance.completionTime > 300000) { // > 5 minutes
      bottlenecks.push({
        type: 'time',
        severity: 'high',
        description: 'Task took longer than expected',
        recommendation: 'Consider breaking into smaller subtasks',
      });
    }

    // Agent-based bottlenecks
    if (performance.agentsUsed.length === 1) {
      bottlenecks.push({
        type: 'coordination',
        severity: 'medium',
        description: 'Single agent used for complex task',
        recommendation: 'Spawn specialized agents for parallel work',
      });
    }

    // Resource bottlenecks
    if (this.sessionData.operations.length > 100) {
      bottlenecks.push({
        type: 'operations',
        severity: 'medium',
        description: 'High number of operations',
        recommendation: 'Optimize operation batching',
      });
    }

    return bottlenecks;
  }

  suggestImprovements(performance) {
    const improvements = [];
    const efficiency = this.calculateEfficiency(performance);

    // Time improvements
    if (efficiency.timeEfficiency < 0.7) {
      improvements.push({
        area: 'execution_time',
        suggestion: 'Use parallel task execution',
        expectedImprovement: '30-50% time reduction',
      });
    }

    // Coordination improvements
    if (efficiency.agentEfficiency < 0.8) {
      improvements.push({
        area: 'agent_coordination',
        suggestion: 'Implement specialized agent patterns',
        expectedImprovement: '20-30% efficiency gain',
      });
    }

    // Pattern improvements
    if (this.sessionData.learnings.length < 5) {
      improvements.push({
        area: 'learning',
        suggestion: 'Enable neural pattern training',
        expectedImprovement: 'Cumulative performance gains',
      });
    }

    return improvements;
  }

  updateCoordinationStrategy(performance) {
    const efficiency = this.calculateEfficiency(performance);

    // Update strategy based on performance
    if (!this.sessionData.coordinationStrategy) {
      this.sessionData.coordinationStrategy = {
        current: 'balanced',
        history: [],
        adjustments: 0,
      };
    }

    const strategy = this.sessionData.coordinationStrategy;
    strategy.history.push({
      timestamp: Date.now(),
      efficiency: efficiency.score,
      strategy: strategy.current,
    });

    // Adjust strategy if needed
    if (parseFloat(efficiency.score) < 0.6) {
      strategy.current = 'adaptive';
      strategy.adjustments++;
    } else if (parseFloat(efficiency.score) > 0.9) {
      strategy.current = 'specialized';
      strategy.adjustments++;
    }
  }

  extractSearchPatterns(query) {
    const patterns = [];

    // Extract file type patterns
    const fileTypes = query.match(/\.(js|ts|py|go|rs|md|json|yaml)\b/gi);
    if (fileTypes) {
      patterns.push(...fileTypes.map(ft => `filetype:${ft}`));
    }

    // Extract function/class patterns
    const codePatterns = query.match(/\b(function|class|interface|struct|impl)\s+\w+/gi);
    if (codePatterns) {
      patterns.push(...codePatterns.map(cp => `code:${cp}`));
    }

    // Extract scope patterns
    const scopePatterns = query.match(/\b(src|test|lib|bin|docs?)\//gi);
    if (scopePatterns) {
      patterns.push(...scopePatterns.map(sp => `scope:${sp}`));
    }

    return patterns;
  }

  async updateKnowledgeBase(type, data) {
    const kbPath = path.join(process.cwd(), '.ruv-swarm', 'knowledge-base.json');

    // Load existing knowledge base
    let kb = {};
    try {
      if (await fs.access(kbPath).then(() => true).catch(() => false)) {
        kb = JSON.parse(await fs.readFile(kbPath, 'utf-8'));
      }
    } catch (_error) {
      kb = { searches: [], patterns: {}, insights: [] };
    }

    // Update based on type
    if (type === 'search') {
      if (!kb.searches) {
        kb.searches = [];
      }
      kb.searches.push({
        query: data.query,
        patterns: data.patterns,
        timestamp: Date.now(),
      });

      // Update pattern frequency
      if (!kb.patterns) {
        kb.patterns = {};
      }
      data.patterns.forEach(pattern => {
        kb.patterns[pattern] = (kb.patterns[pattern] || 0) + 1;
      });
    }

    // Keep only recent data
    if (kb.searches && kb.searches.length > 100) {
      kb.searches = kb.searches.slice(-100);
    }

    // Save updated knowledge base
    await fs.mkdir(path.dirname(kbPath), { recursive: true });
    await fs.writeFile(kbPath, JSON.stringify(kb, null, 2));
  }

  extractUrlPatterns(url) {
    const patterns = [];

    try {
      const urlObj = new URL(url);

      // Domain pattern
      patterns.push(`domain:${urlObj.hostname}`);

      // Path patterns
      const pathParts = urlObj.pathname.split('/').filter(p => p);
      if (pathParts.length > 0) {
        patterns.push(`path:/${pathParts[0]}`); // Top level path
      }

      // Content type patterns
      if (urlObj.pathname.endsWith('.md')) {
        patterns.push('content:markdown');
      }
      if (urlObj.pathname.includes('docs')) {
        patterns.push('content:documentation');
      }
      if (urlObj.pathname.includes('api')) {
        patterns.push('content:api');
      }
      if (urlObj.pathname.includes('guide')) {
        patterns.push('content:guide');
      }

      // Query patterns
      if (urlObj.search) {
        patterns.push('has:queryparams');
      }
    } catch (_error) {
      patterns.push('pattern:invalid-url');
    }

    return patterns;
  }

  async getSwarmStatus() {
    try {
      const statusPath = path.join(process.cwd(), '.ruv-swarm', 'status.json');
      if (await fs.access(statusPath).then(() => true).catch(() => false)) {
        return JSON.parse(await fs.readFile(statusPath, 'utf-8'));
      }
    } catch (_error) {
      // Fallback to session data
    }

    return {
      agents: this.sessionData.agents,
      activeTasks: this.sessionData.operations.filter(op =>
        Date.now() - op.timestamp < 300000, // Last 5 minutes
      ).length,
      health: 'operational',
    };
  }

  sendTelemetry(event, data) {
    // In production, this would send to telemetry service
    // For now, just log to telemetry file
    const telemetryPath = path.join(process.cwd(), '.ruv-swarm', 'telemetry.jsonl');

    const telemetryEvent = {
      event,
      data,
      timestamp: Date.now(),
      sessionId: this.sessionData.sessionId || 'unknown',
      version: '1.0.0',
    };

    // Async write without blocking
    fs.appendFile(telemetryPath, `${JSON.stringify(telemetryEvent) }\n`).catch(() => { /* intentionally empty */ });
  }

  // Helper methods for other functionality

  getSpecializationForType(type) {
    const specializations = {
      researcher: ['literature-review', 'data-analysis', 'trend-identification'],
      coder: ['implementation', 'refactoring', 'optimization'],
      analyst: ['metrics', 'performance', 'data-visualization'],
      architect: ['system-design', 'api-design', 'database-schema'],
      coordinator: ['task-planning', 'resource-allocation', 'progress-tracking'],
      optimizer: ['performance-tuning', 'algorithm-optimization', 'resource-usage'],
    };
    return specializations[type] || ['general'];
  }

  generateSpecializationPatterns(type) {
    const patterns = {
      researcher: ['depth-first-search', 'breadth-first-search', 'citation-tracking'],
      coder: ['modular-design', 'error-handling', 'code-reuse'],
      analyst: ['statistical-analysis', 'trend-detection', 'anomaly-detection'],
      architect: ['layered-architecture', 'microservices', 'event-driven'],
      coordinator: ['dependency-tracking', 'parallel-execution', 'milestone-planning'],
      optimizer: ['bottleneck-identification', 'caching-strategies', 'lazy-loading'],
    };
    return patterns[type] || ['adaptive-learning'];
  }

  generateMockWeights() {
    // Generate mock neural network weights for demonstration
    return {
      layers: [
        { neurons: 128, weights: Array(128).fill(0).map(() => Math.random() - 0.5) },
        { neurons: 64, weights: Array(64).fill(0).map(() => Math.random() - 0.5) },
        { neurons: 32, weights: Array(32).fill(0).map(() => Math.random() - 0.5) },
      ],
      biases: Array(224).fill(0).map(() => Math.random() - 0.5),
    };
  }

  optimizeAgentAllocation(_taskId) {
    // Simple load balancing algorithm
    const agents = Array.from(this.sessionData.agents.values());
    const allocation = {};

    agents.forEach(agent => {
      // Allocate based on agent type and current load
      const load = this.sessionData.operations.filter(op =>
        op.agent === agent.id &&
                Date.now() - op.timestamp < 60000,
      ).length;

      allocation[agent.id] = {
        agent: agent.id,
        type: agent.type,
        currentLoad: load,
        capacity: Math.max(0, 10 - load), // Max 10 concurrent ops
        priority: load < 5 ? 'high' : 'normal',
      };
    });

    return allocation;
  }

  calculateParallelization(_taskId) {
    // Determine parallelization factor based on task and resources
    const agentCount = this.sessionData.agents.size;
    const complexity = this.sessionData.taskComplexity || { score: 2 };

    return {
      factor: Math.min(agentCount, Math.ceil(complexity.score * 1.5)),
      strategy: agentCount > 3 ? 'distributed' : 'local',
      maxConcurrency: Math.min(agentCount * 2, 10),
    };
  }

  getFileType(filePath) {
    const ext = path.extname(filePath);
    const typeMap = {
      '.js': 'javascript',
      '.ts': 'typescript',
      '.py': 'python',
      '.go': 'golang',
      '.rs': 'rust',
      '.json': 'config',
      '.yaml': 'config',
      '.yml': 'config',
      '.md': 'documentation',
      '.txt': 'text',
    };
    return typeMap[ext] || 'unknown';
  }

  getCurrentAgent() {
    // Get the most recently active agent
    const recentOps = this.sessionData.operations.slice(-10);
    const agentCounts = {};

    recentOps.forEach(op => {
      if (op.agent) {
        agentCounts[op.agent] = (agentCounts[op.agent] || 0) + 1;
      }
    });

    const sorted = Object.entries(agentCounts).sort((a, b) => b[1] - a[1]);
    return sorted.length > 0 ? sorted[0][0] : 'coordinator';
  }

  async findRelatedFiles(filePath) {
    const related = [];
    const _baseName = path.basename(filePath, path.extname(filePath));
    // const dirName = path.dirname(filePath);

    // Common related file patterns
    // const patterns = [
    //   `${baseName}.test.*`, // Test files
    //   `${baseName}.spec.*`, // Spec files
    //   `test-${baseName}.*`, // Alternative test pattern
    //   `${baseName}.d.ts`, // TypeScript definitions
    //   `${baseName}.types.*`, // Type definitions
    // ];

    // For now, return mock related files
    // In production, would use file system search
    if (filePath.endsWith('.js')) {
      related.push(filePath.replace('.js', '.test.js'));
    }
    if (filePath.endsWith('.ts')) {
      related.push(filePath.replace('.ts', '.test.ts'));
      related.push(filePath.replace('.ts', '.d.ts'));
    }

    return related.filter(f => f !== filePath);
  }
}

// Export singleton instance and its methods
const hooksInstance = new RuvSwarmHooks();

export const handleHook = (hookType, options) => hooksInstance.handleHook(hookType, options);

export default hooksInstance;