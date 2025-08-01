// Additional TypeScript utilities for ruv-swarm WASM

export interface SwarmConfig {
    maxAgents: number;
    topology: 'mesh' | 'star' | 'ring' | 'hierarchical';
    enableSIMD?: boolean;
    memoryBudgetMB?: number;
}

export interface AgentConfig {
    name: string;
    type: 'researcher' | 'coder' | 'analyst' | 'optimizer' | 'coordinator';
    cognitivePattern?: 'convergent' | 'divergent' | 'lateral' | 'systems' | 'critical' | 'abstract';
}

export interface TaskConfig {
    name: string;
    description: string;
    priority?: 'low' | 'medium' | 'high' | 'critical';
}

export interface PerformanceMetrics {
    agentCount: number;
    taskCount: number;
    memoryUsageMB: number;
    executionTimeMs: number;
    agentsPerMB: number;
}

// Helper functions
export function createOptimizedSwarm(config: SwarmConfig): Promise<any>;
export function benchmarkPerformance(testSize: number): Promise<any>;
