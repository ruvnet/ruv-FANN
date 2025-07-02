// Mock for WASM module imports in tests
module.exports = {
  init: jest.fn().mockResolvedValue(undefined),
  createSwarm: jest.fn().mockReturnValue(1),
  addAgent: jest.fn().mockReturnValue(1),
  assignTask: jest.fn(),
  getState: jest.fn().mockReturnValue({
    agents: new Map(),
    tasks: new Map(),
    topology: 'mesh',
    connections: [],
    metrics: {
      totalTasks: 0,
      completedTasks: 0,
      failedTasks: 0,
      averageCompletionTime: 0,
      agentUtilization: new Map(),
      throughput: 0,
    },
  }),
  destroy: jest.fn(),
};