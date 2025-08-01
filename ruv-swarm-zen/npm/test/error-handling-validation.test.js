/**
 * Comprehensive Error Handling and Validation Tests
 * Tests the new error handling system for all MCP tools
 */

const { jest } = require('@jest/globals');

// Mock dependencies
const mockRuvSwarm = {
  initialize: jest.fn(),
  createSwarm: jest.fn(),
  features: {
    neural_networks: true,
    forecasting: true,
    cognitive_diversity: true,
    simd_support: true,
  },
  wasmLoader: {
    getTotalMemoryUsage: jest.fn(() => 1024 * 1024),
    getModuleStatus: jest.fn(() => ({ core: { loaded: true } })),
  },
};

const mockPersistence = {
  createSwarm: jest.fn(),
  createAgent: jest.fn(),
  getActiveSwarms: jest.fn(() => []),
  getSwarmAgents: jest.fn(() => []),
};

// Import the modules with mocking
jest.unstable_mockModule('../src/index-enhanced.js', () => ({
  RuvSwarm: {
    initialize: jest.fn(() => mockRuvSwarm),
  },
}));

jest.unstable_mockModule('../src/persistence.js', () => ({
  SwarmPersistence: jest.fn(() => mockPersistence),
}));

// Now import the modules under test
const {
  ValidationError,
  SwarmError,
  AgentError,
  TaskError,
  NeuralError,
  WasmError,
  ErrorFactory,
  ErrorContext,
} = await import('../src/errors.js');

const { ValidationUtils } = await import('../src/schemas.js');
const { EnhancedMCPTools } = await import('../src/mcp-tools-enhanced.js');

describe('Error Handling System', () => {
  let mcpTools;

  beforeEach(() => {
    mcpTools = new EnhancedMCPTools();
    jest.clearAllMocks();
  });

  describe('Custom Error Classes', () => {
    test('ValidationError should include field and value information', () => {
      const error = new ValidationError('Invalid value', 'testField', 'badValue', 'string');

      expect(error.name).toBe('ValidationError');
      expect(error.code).toBe('VALIDATION_ERROR');
      expect(error.field).toBe('testField');
      expect(error.value).toBe('badValue');
      expect(error.expectedType).toBe('string');
      expect(error.getSuggestions()).toContain('Check the \'testField\' parameter');
    });

    test('SwarmError should include swarm context', () => {
      const error = new SwarmError('Swarm not found', 'test-swarm-id', 'initialization');

      expect(error.name).toBe('SwarmError');
      expect(error.code).toBe('SWARM_ERROR');
      expect(error.swarmId).toBe('test-swarm-id');
      expect(error.operation).toBe('initialization');
      expect(error.getSuggestions()).toContain('Verify the swarm ID is correct');
    });

    test('AgentError should include agent context', () => {
      const error = new AgentError('Agent not found', 'test-agent-id', 'researcher', 'spawn');

      expect(error.name).toBe('AgentError');
      expect(error.code).toBe('AGENT_ERROR');
      expect(error.agentId).toBe('test-agent-id');
      expect(error.agentType).toBe('researcher');
      expect(error.operation).toBe('spawn');
    });

    test('TaskError should include task context', () => {
      const error = new TaskError('Task timeout', 'test-task-id', 'analysis', 'execution');

      expect(error.name).toBe('TaskError');
      expect(error.code).toBe('TASK_ERROR');
      expect(error.taskId).toBe('test-task-id');
      expect(error.taskType).toBe('analysis');
      expect(error.operation).toBe('execution');
      expect(error.getSuggestions()).toContain('Increase task timeout duration');
    });

    test('NeuralError should include neural network context', () => {
      const error = new NeuralError('Training failed', 'test-nn-id', 'training', 'lstm');

      expect(error.name).toBe('NeuralError');
      expect(error.code).toBe('NEURAL_ERROR');
      expect(error.networkId).toBe('test-nn-id');
      expect(error.operation).toBe('training');
      expect(error.modelType).toBe('lstm');
    });

    test('WasmError should include module context', () => {
      const error = new WasmError('Module not loaded', 'core', 'initialization');

      expect(error.name).toBe('WasmError');
      expect(error.code).toBe('WASM_ERROR');
      expect(error.module).toBe('core');
      expect(error.operation).toBe('initialization');
      expect(error.getSuggestions()).toContain('Check WASM module availability');
    });
  });

  describe('ErrorFactory', () => {
    test('should create appropriate error types', () => {
      const validationError = ErrorFactory.createError('validation', 'Invalid input', {
        field: 'test',
        value: 'bad',
        expectedType: 'number',
      });
      expect(validationError).toBeInstanceOf(ValidationError);

      const swarmError = ErrorFactory.createError('swarm', 'Swarm failed', {
        swarmId: 'test-id',
        operation: 'init',
      });
      expect(swarmError).toBeInstanceOf(SwarmError);

      const agentError = ErrorFactory.createError('agent', 'Agent failed', {
        agentId: 'test-id',
        agentType: 'researcher',
      });
      expect(agentError).toBeInstanceOf(AgentError);
    });

    test('should wrap existing errors with context', () => {
      const originalError = new Error('Original error');
      const wrappedError = ErrorFactory.wrapError(originalError, 'wasm', {
        module: 'core',
        operation: 'load',
      });

      expect(wrappedError).toBeInstanceOf(WasmError);
      expect(wrappedError.message).toContain('WASM: Original error');
      expect(wrappedError.details.originalError.message).toBe('Original error');
    });
  });

  describe('ErrorContext', () => {
    test('should manage error context', () => {
      const context = new ErrorContext();
      context.set('operation', 'test');
      context.set('timestamp', '2023-01-01');

      expect(context.get('operation')).toBe('test');
      expect(context.toObject()).toEqual({
        operation: 'test',
        timestamp: '2023-01-01',
      });

      context.clear();
      expect(context.toObject()).toEqual({});
    });

    test('should enrich errors with context', () => {
      const context = new ErrorContext();
      context.set('tool', 'swarm_init');
      context.set('operation', 'test');

      const error = new ValidationError('Test error');
      const enrichedError = context.enrichError(error);

      expect(enrichedError.details.context).toEqual({
        tool: 'swarm_init',
        operation: 'test',
      });
    });
  });
});

describe('Validation System', () => {
  describe('ValidationUtils', () => {
    test('should validate swarm_init parameters correctly', () => {
      const validParams = {
        topology: 'mesh',
        maxAgents: 10,
        strategy: 'balanced',
      };

      const result = ValidationUtils.validateParams(validParams, 'swarm_init');
      expect(result.topology).toBe('mesh');
      expect(result.maxAgents).toBe(10);
      expect(result.strategy).toBe('balanced');
    });

    test('should apply default values for missing parameters', () => {
      const minimalParams = {};

      const result = ValidationUtils.validateParams(minimalParams, 'swarm_init');
      expect(result.topology).toBe('mesh');
      expect(result.maxAgents).toBe(5);
      expect(result.strategy).toBe('balanced');
    });

    test('should reject invalid enum values', () => {
      const invalidParams = {
        topology: 'invalid-topology',
      };

      expect(() => {
        ValidationUtils.validateParams(invalidParams, 'swarm_init');
      }).toThrow(ValidationError);
    });

    test('should reject out-of-range numbers', () => {
      const invalidParams = {
        maxAgents: 200, // Max is 100
      };

      expect(() => {
        ValidationUtils.validateParams(invalidParams, 'swarm_init');
      }).toThrow(ValidationError);
    });

    test('should validate agent_spawn parameters', () => {
      const validParams = {
        type: 'researcher',
        name: 'Test Agent',
        capabilities: ['analysis', 'research'],
      };

      const result = ValidationUtils.validateParams(validParams, 'agent_spawn');
      expect(result.type).toBe('researcher');
      expect(result.name).toBe('Test Agent');
      expect(result.capabilities).toEqual(['analysis', 'research']);
    });

    test('should validate neural_train parameters', () => {
      const validParams = {
        agentId: 'test-agent-123',
        iterations: 50,
        learningRate: 0.01,
        modelType: 'feedforward',
      };

      const result = ValidationUtils.validateParams(validParams, 'neural_train');
      expect(result.agentId).toBe('test-agent-123');
      expect(result.iterations).toBe(50);
      expect(result.learningRate).toBe(0.01);
      expect(result.modelType).toBe('feedforward');
    });

    test('should reject invalid learning rates', () => {
      const invalidParams = {
        agentId: 'test-agent',
        learningRate: 2.0, // Max is 1.0
      };

      expect(() => {
        ValidationUtils.validateParams(invalidParams, 'neural_train');
      }).toThrow(ValidationError);
    });

    test('should sanitize string inputs', () => {
      const maliciousInput = '<script>alert(\"xss\")</script>';
      const sanitized = ValidationUtils.sanitizeInput(maliciousInput);

      expect(sanitized).not.toContain('<script>');
      expect(sanitized).not.toContain('</script>');
    });
  });
});

describe('Enhanced MCP Tools Error Handling', () => {
  let mcpTools;

  beforeEach(() => {
    mcpTools = new EnhancedMCPTools();
    jest.clearAllMocks();
  });

  describe('Error Handler', () => {
    test('should handle and log errors properly', () => {
      const originalError = new ValidationError('Test validation error', 'testField');

      // Capture console output
      const consoleSpy = jest.spyOn(console, 'warn').mockImplementation();

      const handledError = mcpTools.handleError(originalError, 'swarm_init', 'test_operation', {});

      expect(handledError).toBeInstanceOf(ValidationError);
      expect(mcpTools.errorLog).toHaveLength(1);
      expect(mcpTools.errorLog[0].tool).toBe('swarm_init');
      expect(mcpTools.errorLog[0].operation).toBe('test_operation');
      expect(mcpTools.errorLog[0].severity).toBe('medium');

      consoleSpy.mockRestore();
    });

    test('should determine error severity correctly', () => {
      const validationError = new ValidationError('Validation failed');
      expect(mcpTools.determineSeverity(validationError)).toBe('medium');

      const wasmError = new WasmError('WASM module failed');
      expect(mcpTools.determineSeverity(wasmError)).toBe('high');

      const criticalError = new Error('Database corrupt');
      expect(mcpTools.determineSeverity(criticalError)).toBe('low');
    });

    test('should determine error recoverability', () => {
      const validationError = new ValidationError('Invalid input');
      expect(mcpTools.isRecoverable(validationError)).toBe(true);

      const timeoutError = new TaskError('Task timeout');
      expect(mcpTools.isRecoverable(timeoutError)).toBe(true);

      const genericError = new Error('Unknown error');
      expect(mcpTools.isRecoverable(genericError)).toBe(false);
    });
  });

  describe('Parameter Validation', () => {
    test('should validate tool parameters', () => {
      const params = {
        topology: 'mesh',
        maxAgents: 10,
      };

      const result = mcpTools.validateToolParams(params, 'swarm_init');
      expect(result.topology).toBe('mesh');
      expect(result.maxAgents).toBe(10);
    });

    test('should throw ValidationError for invalid parameters', () => {
      const params = {
        topology: 'invalid',
      };

      expect(() => {
        mcpTools.validateToolParams(params, 'swarm_init');
      }).toThrow(ValidationError);
    });
  });

  describe('Error Statistics', () => {
    test('should track error statistics', () => {
      // Simulate some errors
      mcpTools.handleError(new ValidationError('Error 1'), 'swarm_init', 'op1');
      mcpTools.handleError(new WasmError('Error 2'), 'agent_spawn', 'op2');
      mcpTools.handleError(new TaskError('Error 3'), 'task_orchestrate', 'op3');

      const stats = mcpTools.getErrorStats();

      expect(stats.total).toBe(3);
      expect(stats.bySeverity.medium).toBe(2); // ValidationError + TaskError
      expect(stats.bySeverity.high).toBe(1); // WasmError
      expect(stats.byTool.swarm_init).toBe(1);
      expect(stats.byTool.agent_spawn).toBe(1);
      expect(stats.byTool.task_orchestrate).toBe(1);
    });

    test('should return recent error logs', () => {
      // Add some errors
      for (let i = 0; i < 5; i++) {
        mcpTools.handleError(new Error(`Error ${i}`), 'test_tool', 'test_op');
      }

      const recentLogs = mcpTools.getErrorLogs(3);
      expect(recentLogs).toHaveLength(3);
      expect(recentLogs[2].error.message).toBe('Error 4'); // Most recent
    });
  });

  describe('Integration with MCP Tools', () => {
    test('swarm_init should use enhanced error handling', async() => {
      // Mock to throw an error
      mockRuvSwarm.createSwarm = jest.fn().mockRejectedValue(new Error('WASM module not loaded'));

      await expect(mcpTools.swarm_init({})).rejects.toThrow();

      // Check that error was logged
      expect(mcpTools.errorLog).toHaveLength(1);
      expect(mcpTools.errorLog[0].tool).toBe('swarm_init');
    });

    test('should handle validation errors in swarm_init', async() => {
      const invalidParams = {
        topology: 'invalid-topology',
        maxAgents: 'not-a-number',
      };

      await expect(mcpTools.swarm_init(invalidParams)).rejects.toThrow(ValidationError);
    });

    test('agent_spawn should use enhanced error handling', async() => {
      // First initialize a swarm
      mockRuvSwarm.createSwarm = jest.fn().mockResolvedValue({
        id: 'test-swarm',
        agents: new Map(),
        maxAgents: 5,
        spawn: jest.fn().mockRejectedValue(new Error('Neural network initialization failed')),
      });

      await mcpTools.swarm_init({});

      await expect(mcpTools.agent_spawn({ type: 'researcher' })).rejects.toThrow();

      // Check that error was logged with proper context
      const agentError = mcpTools.errorLog.find(log => log.tool === 'agent_spawn');
      expect(agentError).toBeDefined();
      expect(agentError.error.message).toContain('Neural network error');
    });
  });
});

describe('Schema Validation', () => {
  test('should provide schema documentation', () => {
    const doc = ValidationUtils.getSchemaDoc('swarm_init');

    expect(doc.tool).toBe('swarm_init');
    expect(doc.parameters.topology).toBeDefined();
    expect(doc.parameters.topology.type).toBe('string');
    expect(doc.parameters.topology.allowedValues).toContain('mesh');
    expect(doc.parameters.maxAgents.range.min).toBe(1);
    expect(doc.parameters.maxAgents.range.max).toBe(100);
  });

  test('should list all available schemas', () => {
    const schemas = ValidationUtils.getAllSchemas();

    expect(schemas).toContain('swarm_init');
    expect(schemas).toContain('agent_spawn');
    expect(schemas).toContain('task_orchestrate');
    expect(schemas).toContain('neural_train');
    expect(schemas.length).toBeGreaterThan(20); // Should have 25+ tools
  });

  test('should validate UUID format', () => {
    const validUUID = '123e4567-e89b-12d3-a456-426614174000';
    const invalidUUID = 'not-a-uuid';

    expect(ValidationUtils.isValidUUID(validUUID)).toBe(true);
    expect(ValidationUtils.isValidUUID(invalidUUID)).toBe(false);
  });
});
