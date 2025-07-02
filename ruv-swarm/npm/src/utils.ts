/**
 * Utility functions for RuvSwarm
 */

import { CognitiveProfile, SwarmTopology, AgentType, TaskPriority } from './types';

/**
 * Generate a unique ID for agents, tasks, and messages
 */
export function generateId(prefix: string = ''): string {
  const timestamp = Date.now().toString(36);
  const random = Math.random().toString(36).substring(2, 9);
  return prefix ? `${prefix}_${timestamp}_${random}` : `${timestamp}_${random}`;
}

/**
 * Create a default cognitive profile based on agent type
 */
export function getDefaultCognitiveProfile(type: AgentType): CognitiveProfile {
  const profiles: Record<AgentType, CognitiveProfile> = {
    researcher: {
      analytical: 0.9,
      creative: 0.6,
      systematic: 0.8,
      intuitive: 0.5,
      collaborative: 0.7,
      independent: 0.8,
    },
    coder: {
      analytical: 0.8,
      creative: 0.7,
      systematic: 0.9,
      intuitive: 0.4,
      collaborative: 0.6,
      independent: 0.7,
    },
    analyst: {
      analytical: 0.95,
      creative: 0.4,
      systematic: 0.9,
      intuitive: 0.3,
      collaborative: 0.6,
      independent: 0.8,
    },
    architect: {
      analytical: 0.8,
      creative: 0.8,
      systematic: 0.85,
      intuitive: 0.7,
      collaborative: 0.8,
      independent: 0.6,
    },
    reviewer: {
      analytical: 0.85,
      creative: 0.5,
      systematic: 0.9,
      intuitive: 0.4,
      collaborative: 0.7,
      independent: 0.7,
    },
    debugger: {
      analytical: 0.9,
      creative: 0.6,
      systematic: 0.85,
      intuitive: 0.6,
      collaborative: 0.5,
      independent: 0.8,
    },
    tester: {
      analytical: 0.8,
      creative: 0.6,
      systematic: 0.95,
      intuitive: 0.3,
      collaborative: 0.6,
      independent: 0.7,
    },
    documenter: {
      analytical: 0.7,
      creative: 0.7,
      systematic: 0.85,
      intuitive: 0.4,
      collaborative: 0.8,
      independent: 0.6,
    },
    optimizer: {
      analytical: 0.9,
      creative: 0.6,
      systematic: 0.8,
      intuitive: 0.5,
      collaborative: 0.5,
      independent: 0.8,
    },
    custom: {
      analytical: 0.5,
      creative: 0.5,
      systematic: 0.5,
      intuitive: 0.5,
      collaborative: 0.5,
      independent: 0.5,
    },
  };

  return profiles[type];
}

/**
 * Calculate cognitive diversity score between two profiles
 */
export function calculateCognitiveDiversity(
  profile1: CognitiveProfile,
  profile2: CognitiveProfile,
): number {
  const dimensions = Object.keys(profile1) as (keyof CognitiveProfile)[];
  let totalDifference = 0;

  for (const dimension of dimensions) {
    const diff = Math.abs(profile1[dimension] - profile2[dimension]);
    totalDifference += diff;
  }

  return totalDifference / dimensions.length;
}

/**
 * Determine optimal topology based on swarm characteristics
 */
export function recommendTopology(
  agentCount: number,
  taskComplexity: 'low' | 'medium' | 'high',
  coordinationNeeds: 'minimal' | 'moderate' | 'extensive',
): SwarmTopology {
  if (agentCount <= 5) {
    return 'mesh';
  }

  if (coordinationNeeds === 'extensive') {
    return 'hierarchical';
  }

  if (taskComplexity === 'high' && agentCount > 10) {
    return 'hybrid';
  }

  if (coordinationNeeds === 'minimal') {
    return 'distributed';
  }

  return 'centralized';
}

/**
 * Convert task priority to numeric value for sorting
 */
export function priorityToNumber(priority: TaskPriority): number {
  const priorityMap: Record<TaskPriority, number> = {
    low: 1,
    medium: 2,
    high: 3,
    critical: 4,
  };
  return priorityMap[priority];
}

/**
 * Format swarm metrics for display
 */
export function formatMetrics(metrics: {
  totalTasks: number;
  completedTasks: number;
  failedTasks: number;
  averageCompletionTime: number;
  throughput: number;
}): string {
  const successRate = metrics.totalTasks > 0 
    ? ((metrics.completedTasks / metrics.totalTasks) * 100).toFixed(1)
    : '0.0';

  return `
Swarm Metrics:
- Total Tasks: ${metrics.totalTasks}
- Completed: ${metrics.completedTasks}
- Failed: ${metrics.failedTasks}
- Success Rate: ${successRate}%
- Avg Completion Time: ${metrics.averageCompletionTime.toFixed(2)}ms
- Throughput: ${metrics.throughput.toFixed(2)} tasks/sec
  `.trim();
}

/**
 * Validate swarm options
 */
export function validateSwarmOptions(options: any): string[] {
  const errors: string[] = [];

  if (options.maxAgents !== undefined) {
    if (typeof options.maxAgents !== 'number' || options.maxAgents < 1) {
      errors.push('maxAgents must be a positive number');
    }
  }

  if (options.connectionDensity !== undefined) {
    if (
      typeof options.connectionDensity !== 'number' ||
      options.connectionDensity < 0 ||
      options.connectionDensity > 1
    ) {
      errors.push('connectionDensity must be a number between 0 and 1');
    }
  }

  if (options.topology !== undefined) {
    const validTopologies = ['mesh', 'hierarchical', 'distributed', 'centralized', 'hybrid'];
    if (!validTopologies.includes(options.topology)) {
      errors.push(`topology must be one of: ${validTopologies.join(', ')}`);
    }
  }

  return errors;
}

/**
 * Deep clone an object
 */
export function deepClone<T>(obj: T): T {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (obj instanceof Date) {
    return new Date(obj.getTime()) as any;
  }

  if (obj instanceof Array) {
    return obj.map(item => deepClone(item)) as any;
  }

  if (obj instanceof Map) {
    const cloned = new Map();
    obj.forEach((value, key) => {
      cloned.set(key, deepClone(value));
    });
    return cloned as any;
  }

  if (obj instanceof Set) {
    const cloned = new Set();
    obj.forEach(value => {
      cloned.add(deepClone(value));
    });
    return cloned as any;
  }

  const cloned = {} as T;
  for (const key in obj) {
    if (obj.hasOwnProperty(key)) {
      cloned[key] = deepClone(obj[key]);
    }
  }

  return cloned;
}

/**
 * Retry a function with exponential backoff
 */
export async function retryWithBackoff<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  initialDelay: number = 100,
): Promise<T> {
  let lastError: Error;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      lastError = error as Error;
      if (i < maxRetries - 1) {
        const delay = initialDelay * Math.pow(2, i);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
  }

  throw lastError!;
}