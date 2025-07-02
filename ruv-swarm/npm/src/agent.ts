/**
 * Agent implementation and wrappers
 */

import {
  Agent,
  AgentConfig,
  AgentState,
  AgentStatus,
  Task,
  Message,
  MessageType,
} from './types';
import { generateId, getDefaultCognitiveProfile } from './utils';

export class BaseAgent implements Agent {
  id: string;
  config: AgentConfig;
  state: AgentState;
  connections: string[] = [];
  
  private messageHandlers: Map<MessageType, (message: Message) => Promise<void>> = new Map();
  private wasmAgentId?: number;

  constructor(config: AgentConfig) {
    this.id = config.id || generateId('agent');
    this.config = {
      ...config,
      id: this.id,
      cognitiveProfile: config.cognitiveProfile || getDefaultCognitiveProfile(config.type),
    };
    
    this.state = {
      status: 'idle',
      load: 0,
      performance: {
        tasksCompleted: 0,
        tasksFailed: 0,
        averageExecutionTime: 0,
        successRate: 0,
      },
    };

    this.setupMessageHandlers();
  }

  private setupMessageHandlers(): void {
    this.messageHandlers.set('task_assignment', this.handleTaskAssignment.bind(this));
    this.messageHandlers.set('coordination', this.handleCoordination.bind(this));
    this.messageHandlers.set('knowledge_share', this.handleKnowledgeShare.bind(this));
    this.messageHandlers.set('status_update', this.handleStatusUpdate.bind(this));
  }

  async execute(task: Task): Promise<any> {
    const startTime = Date.now();
    
    try {
      this.update({ status: 'busy', currentTask: task.id });
      
      // Execute task based on agent type
      const result = await this.executeTaskByType(task);
      
      // Update performance metrics
      const executionTime = Date.now() - startTime;
      this.updatePerformanceMetrics(true, executionTime);
      
      this.update({ status: 'idle', currentTask: undefined });
      
      return result;
    } catch (error) {
      this.updatePerformanceMetrics(false, Date.now() - startTime);
      this.update({ status: 'error', currentTask: undefined });
      throw error;
    }
  }

  protected async executeTaskByType(task: Task): Promise<any> {
    // Base implementation - override in specialized agents
    console.log(`Agent ${this.id} executing task ${task.id}: ${task.description}`);
    
    // Simulate work
    await new Promise(resolve => setTimeout(resolve, 100 + Math.random() * 400));
    
    return {
      taskId: task.id,
      agentId: this.id,
      result: `Task completed by ${this.config.type} agent`,
      timestamp: Date.now(),
    };
  }

  async communicate(message: Message): Promise<void> {
    const handler = this.messageHandlers.get(message.type);
    if (handler) {
      await handler(message);
    } else {
      console.warn(`No handler for message type: ${message.type}`);
    }
  }

  update(state: Partial<AgentState>): void {
    this.state = { ...this.state, ...state };
  }

  private updatePerformanceMetrics(success: boolean, executionTime: number): void {
    const performance = this.state.performance;
    
    if (success) {
      performance.tasksCompleted++;
    } else {
      performance.tasksFailed++;
    }
    
    const totalTasks = performance.tasksCompleted + performance.tasksFailed;
    performance.successRate = totalTasks > 0 ? performance.tasksCompleted / totalTasks : 0;
    
    // Update average execution time
    const totalTime = performance.averageExecutionTime * (totalTasks - 1) + executionTime;
    performance.averageExecutionTime = totalTime / totalTasks;
  }

  private async handleTaskAssignment(message: Message): Promise<void> {
    const task = message.payload as Task;
    console.log(`Agent ${this.id} received task assignment: ${task.id}`);
    // Task execution is handled by the swarm coordinator
  }

  private async handleCoordination(message: Message): Promise<void> {
    console.log(`Agent ${this.id} received coordination message from ${message.from}`);
    // Handle coordination logic
  }

  private async handleKnowledgeShare(message: Message): Promise<void> {
    console.log(`Agent ${this.id} received knowledge share from ${message.from}`);
    // Store shared knowledge in memory
    if (this.config.memory) {
      this.config.memory.shortTerm.set(`knowledge_${message.id}`, message.payload);
    }
  }

  private async handleStatusUpdate(message: Message): Promise<void> {
    console.log(`Agent ${this.id} received status update from ${message.from}`);
    // Process status update
  }

  setWasmAgentId(id: number): void {
    this.wasmAgentId = id;
  }

  getWasmAgentId(): number | undefined {
    return this.wasmAgentId;
  }
}

/**
 * Specialized agent for research tasks
 */
export class ResearcherAgent extends BaseAgent {
  constructor(config: Omit<AgentConfig, 'type'>) {
    super({ ...config, type: 'researcher' });
  }

  protected async executeTaskByType(task: Task): Promise<any> {
    console.log(`Researcher ${this.id} analyzing: ${task.description}`);
    
    // Simulate research activities
    const phases = ['collecting_data', 'analyzing', 'synthesizing', 'reporting'];
    const results: any[] = [];
    
    for (const phase of phases) {
      await new Promise(resolve => setTimeout(resolve, 200));
      results.push({
        phase,
        timestamp: Date.now(),
        findings: `${phase} completed for ${task.description}`,
      });
    }
    
    return {
      taskId: task.id,
      agentId: this.id,
      type: 'research_report',
      phases: results,
      summary: `Research completed on: ${task.description}`,
      recommendations: ['Further investigation needed', 'Consider alternative approaches'],
    };
  }
}

/**
 * Specialized agent for coding tasks
 */
export class CoderAgent extends BaseAgent {
  constructor(config: Omit<AgentConfig, 'type'>) {
    super({ ...config, type: 'coder' });
  }

  protected async executeTaskByType(task: Task): Promise<any> {
    console.log(`Coder ${this.id} implementing: ${task.description}`);
    
    // Simulate coding activities
    const steps = ['design', 'implement', 'test', 'refactor'];
    const codeArtifacts: any[] = [];
    
    for (const step of steps) {
      await new Promise(resolve => setTimeout(resolve, 300));
      codeArtifacts.push({
        step,
        timestamp: Date.now(),
        artifact: `${step}_${task.id}.ts`,
      });
    }
    
    return {
      taskId: task.id,
      agentId: this.id,
      type: 'code_implementation',
      artifacts: codeArtifacts,
      summary: `Implementation completed for: ${task.description}`,
      metrics: {
        linesOfCode: Math.floor(Math.random() * 500) + 100,
        complexity: Math.floor(Math.random() * 10) + 1,
      },
    };
  }
}

/**
 * Specialized agent for analysis tasks
 */
export class AnalystAgent extends BaseAgent {
  constructor(config: Omit<AgentConfig, 'type'>) {
    super({ ...config, type: 'analyst' });
  }

  protected async executeTaskByType(task: Task): Promise<any> {
    console.log(`Analyst ${this.id} analyzing: ${task.description}`);
    
    // Simulate analysis activities
    await new Promise(resolve => setTimeout(resolve, 400));
    
    return {
      taskId: task.id,
      agentId: this.id,
      type: 'analysis_report',
      metrics: {
        dataPoints: Math.floor(Math.random() * 1000) + 100,
        confidence: Math.random() * 0.3 + 0.7,
      },
      insights: [
        'Pattern detected in data',
        'Anomaly found at timestamp X',
        'Recommendation for optimization',
      ],
      visualizations: ['chart_1.png', 'graph_2.svg'],
    };
  }
}

/**
 * Factory function to create specialized agents
 */
export function createAgent(config: AgentConfig): Agent {
  switch (config.type) {
  case 'researcher':
    return new ResearcherAgent(config);
  case 'coder':
    return new CoderAgent(config);
  case 'analyst':
    return new AnalystAgent(config);
  default:
    return new BaseAgent(config);
  }
}

/**
 * Agent pool for managing multiple agents
 */
export class AgentPool {
  private agents: Map<string, Agent> = new Map();
  private availableAgents: Set<string> = new Set();

  addAgent(agent: Agent): void {
    this.agents.set(agent.id, agent);
    if (agent.state.status === 'idle') {
      this.availableAgents.add(agent.id);
    }
  }

  removeAgent(agentId: string): void {
    this.agents.delete(agentId);
    this.availableAgents.delete(agentId);
  }

  getAgent(agentId: string): Agent | undefined {
    return this.agents.get(agentId);
  }

  getAvailableAgent(preferredType?: string): Agent | undefined {
    let selectedAgent: Agent | undefined;

    for (const agentId of this.availableAgents) {
      const agent = this.agents.get(agentId);
      if (!agent) continue;

      if (!preferredType || agent.config.type === preferredType) {
        selectedAgent = agent;
        break;
      }
    }

    if (!selectedAgent && this.availableAgents.size > 0) {
      const firstAvailable = Array.from(this.availableAgents)[0];
      selectedAgent = this.agents.get(firstAvailable);
    }

    if (selectedAgent) {
      this.availableAgents.delete(selectedAgent.id);
    }

    return selectedAgent;
  }

  releaseAgent(agentId: string): void {
    const agent = this.agents.get(agentId);
    if (agent && agent.state.status === 'idle') {
      this.availableAgents.add(agentId);
    }
  }

  getAllAgents(): Agent[] {
    return Array.from(this.agents.values());
  }

  getAgentsByType(type: string): Agent[] {
    return this.getAllAgents().filter(agent => agent.config.type === type);
  }

  getAgentsByStatus(status: AgentStatus): Agent[] {
    return this.getAllAgents().filter(agent => agent.state.status === status);
  }
}