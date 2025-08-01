/**
 * Custom Error Classes for RUV-Swarm MCP Tools
 * Provides detailed, actionable error messages with context
 */

/**
 * Base error class for all ruv-swarm MCP errors
 */
class RuvSwarmError extends Error {
  constructor(message, code = 'GENERAL_ERROR', details = {}) {
    super(message);
    this.name = this.constructor.name;
    this.code = code;
    this.details = details;
    this.timestamp = new Date().toISOString();
    this.stack = this.stack || (new Error()).stack;
  }

  toJSON() {
    return {
      name: this.name,
      message: this.message,
      code: this.code,
      details: this.details,
      timestamp: this.timestamp,
      stack: this.stack,
    };
  }

  /**
   * Get actionable suggestions for resolving this error
   */
  getSuggestions() {
    return [
      'Check the error details for specific information',
      'Verify your input parameters',
      'Consult the MCP tools documentation',
    ];
  }
}

/**
 * Validation errors for input parameters
 */
class ValidationError extends RuvSwarmError {
  constructor(message, field = null, value = null, expectedType = null) {
    const details = {
      field,
      value: typeof value === 'object' ? JSON.stringify(value) : value,
      expectedType,
      actualType: typeof value,
    };

    super(message, 'VALIDATION_ERROR', details);
    this.field = field;
    this.value = value;
    this.expectedType = expectedType;
  }

  getSuggestions() {
    const suggestions = [
      `Check the '${this.field}' parameter`,
      `Expected type: ${this.expectedType}, got: ${this.details.actualType}`,
    ];

    if (this.expectedType === 'number') {
      suggestions.push('Ensure the value is a valid number');
      suggestions.push('Check for NaN or Infinity values');
    } else if (this.expectedType === 'string') {
      suggestions.push('Ensure the value is a non-empty string');
      suggestions.push('Check for null or undefined values');
    } else if (this.expectedType === 'array') {
      suggestions.push('Ensure the value is a valid array');
      suggestions.push('Check array elements for correct types');
    } else if (this.expectedType === 'object') {
      suggestions.push('Ensure the value is a valid object');
      suggestions.push('Check for required object properties');
    }

    return suggestions;
  }
}

/**
 * Swarm-related errors
 */
class SwarmError extends RuvSwarmError {
  constructor(message, swarmId = null, operation = null) {
    const details = { swarmId, operation };
    super(message, 'SWARM_ERROR', details);
    this.swarmId = swarmId;
    this.operation = operation;
  }

  getSuggestions() {
    const suggestions = [];

    if (this.message.includes('not found')) {
      suggestions.push('Verify the swarm ID is correct');
      suggestions.push('Check if the swarm was properly initialized');
      suggestions.push('Use swarm_status to list available swarms');
    } else if (this.message.includes('capacity') || this.message.includes('full')) {
      suggestions.push('Increase the swarm maxAgents parameter');
      suggestions.push('Remove idle agents before adding new ones');
      suggestions.push('Consider using multiple swarms for load distribution');
    } else if (this.message.includes('initialization')) {
      suggestions.push('Call swarm_init before other swarm operations');
      suggestions.push('Check WASM module loading status');
      suggestions.push('Verify system resources are available');
    }

    suggestions.push('Check swarm logs for additional details');
    return suggestions;
  }
}

/**
 * Agent-related errors
 */
class AgentError extends RuvSwarmError {
  constructor(message, agentId = null, agentType = null, operation = null) {
    const details = { agentId, agentType, operation };
    super(message, 'AGENT_ERROR', details);
    this.agentId = agentId;
    this.agentType = agentType;
    this.operation = operation;
  }

  getSuggestions() {
    const suggestions = [];

    if (this.message.includes('not found')) {
      suggestions.push('Verify the agent ID is correct');
      suggestions.push('Check if the agent was properly spawned');
      suggestions.push('Use agent_list to see available agents');
    } else if (this.message.includes('busy') || this.message.includes('unavailable')) {
      suggestions.push('Wait for the agent to complete current tasks');
      suggestions.push('Spawn additional agents for parallel processing');
      suggestions.push('Check agent status before assignment');
    } else if (this.message.includes('capabilities')) {
      suggestions.push('Verify agent has required capabilities');
      suggestions.push('Spawn an agent with appropriate type');
      suggestions.push('Check capability matching logic');
    } else if (this.message.includes('neural')) {
      suggestions.push('Ensure neural networks are enabled');
      suggestions.push('Verify WASM neural module is loaded');
      suggestions.push('Check neural network configuration');
    }

    suggestions.push('Review agent configuration and requirements');
    return suggestions;
  }
}

/**
 * Task-related errors
 */
class TaskError extends RuvSwarmError {
  constructor(message, taskId = null, taskType = null, operation = null) {
    const details = { taskId, taskType, operation };
    super(message, 'TASK_ERROR', details);
    this.taskId = taskId;
    this.taskType = taskType;
    this.operation = operation;
  }

  getSuggestions() {
    const suggestions = [];

    if (this.message.includes('not found')) {
      suggestions.push('Verify the task ID is correct');
      suggestions.push('Check if the task was properly created');
      suggestions.push('Use task_status to list available tasks');
    } else if (this.message.includes('timeout')) {
      suggestions.push('Increase task timeout duration');
      suggestions.push('Break the task into smaller sub-tasks');
      suggestions.push('Optimize task execution logic');
    } else if (this.message.includes('dependency')) {
      suggestions.push('Check task dependency requirements');
      suggestions.push('Ensure prerequisite tasks are completed');
      suggestions.push('Review task execution order');
    } else if (this.message.includes('resources')) {
      suggestions.push('Check system resource availability');
      suggestions.push('Reduce task complexity or requirements');
      suggestions.push('Scale up available agents');
    }

    suggestions.push('Review task configuration and execution logs');
    return suggestions;
  }
}

/**
 * Neural network related errors
 */
class NeuralError extends RuvSwarmError {
  constructor(message, networkId = null, operation = null, modelType = null) {
    const details = { networkId, operation, modelType };
    super(message, 'NEURAL_ERROR', details);
    this.networkId = networkId;
    this.operation = operation;
    this.modelType = modelType;
  }

  getSuggestions() {
    const suggestions = [];

    if (this.message.includes('not available') || this.message.includes('not loaded')) {
      suggestions.push('Ensure neural network features are enabled');
      suggestions.push('Check WASM neural module loading');
      suggestions.push('Verify system supports neural operations');
    } else if (this.message.includes('training')) {
      suggestions.push('Check training data format and quality');
      suggestions.push('Adjust learning rate and iterations');
      suggestions.push('Verify neural network architecture');
    } else if (this.message.includes('memory')) {
      suggestions.push('Reduce neural network size or complexity');
      suggestions.push('Increase available system memory');
      suggestions.push('Use memory-efficient training algorithms');
    } else if (this.message.includes('convergence')) {
      suggestions.push('Increase training iterations');
      suggestions.push('Adjust learning rate');
      suggestions.push('Improve training data quality');
    }

    suggestions.push('Check neural network configuration and logs');
    return suggestions;
  }
}

/**
 * WASM-related errors
 */
class WasmError extends RuvSwarmError {
  constructor(message, module = null, operation = null) {
    const details = { module, operation };
    super(message, 'WASM_ERROR', details);
    this.module = module;
    this.operation = operation;
  }

  getSuggestions() {
    const suggestions = [];

    if (this.message.includes('not loaded') || this.message.includes('not found')) {
      suggestions.push('Check WASM module availability');
      suggestions.push('Verify module loading sequence');
      suggestions.push('Ensure WASM runtime is supported');
    } else if (this.message.includes('memory')) {
      suggestions.push('Increase WASM memory allocation');
      suggestions.push('Optimize memory usage in operations');
      suggestions.push('Check for memory leaks');
    } else if (this.message.includes('compilation')) {
      suggestions.push('Verify WASM module integrity');
      suggestions.push('Check browser/runtime WASM support');
      suggestions.push('Rebuild WASM modules if corrupted');
    } else if (this.message.includes('function')) {
      suggestions.push('Verify exported function names');
      suggestions.push('Check function parameter types');
      suggestions.push('Ensure WASM module is properly linked');
    }

    suggestions.push('Check WASM module logs and browser console');
    return suggestions;
  }
}

/**
 * Configuration errors
 */
class ConfigurationError extends RuvSwarmError {
  constructor(message, configKey = null, configValue = null) {
    const details = { configKey, configValue };
    super(message, 'CONFIGURATION_ERROR', details);
    this.configKey = configKey;
    this.configValue = configValue;
  }

  getSuggestions() {
    return [
      `Check the '${this.configKey}' configuration`,
      'Review configuration documentation',
      'Verify configuration file format',
      'Ensure all required configuration keys are present',
      'Check configuration value types and ranges',
    ];
  }
}

/**
 * Network/connectivity errors
 */
class NetworkError extends RuvSwarmError {
  constructor(message, endpoint = null, statusCode = null) {
    const details = { endpoint, statusCode };
    super(message, 'NETWORK_ERROR', details);
    this.endpoint = endpoint;
    this.statusCode = statusCode;
  }

  getSuggestions() {
    const suggestions = [];

    if (this.statusCode === 404) {
      suggestions.push('Verify the endpoint URL is correct');
      suggestions.push('Check if the service is running');
    } else if (this.statusCode === 401 || this.statusCode === 403) {
      suggestions.push('Check authentication credentials');
      suggestions.push('Verify API permissions');
    } else if (this.statusCode === 500) {
      suggestions.push('Check server logs for errors');
      suggestions.push('Retry the operation after a delay');
    } else if (this.statusCode === 408 || this.message.includes('timeout')) {
      suggestions.push('Increase request timeout');
      suggestions.push('Check network connectivity');
    }

    suggestions.push('Check network connectivity and firewall settings');
    suggestions.push('Verify service endpoint availability');
    return suggestions;
  }
}

/**
 * Database/persistence errors
 */
class PersistenceError extends RuvSwarmError {
  constructor(message, operation = null, table = null) {
    const details = { operation, table };
    super(message, 'PERSISTENCE_ERROR', details);
    this.operation = operation;
    this.table = table;
  }

  getSuggestions() {
    const suggestions = [];

    if (this.message.includes('constraint') || this.message.includes('unique')) {
      suggestions.push('Check for duplicate entries');
      suggestions.push('Verify unique key constraints');
      suggestions.push('Use update instead of insert for existing records');
    } else if (this.message.includes('not found') || this.message.includes('no such table')) {
      suggestions.push('Verify database schema is initialized');
      suggestions.push('Run database migrations');
      suggestions.push('Check table name spelling');
    } else if (this.message.includes('locked') || this.message.includes('busy')) {
      suggestions.push('Retry the operation after a delay');
      suggestions.push('Check for long-running transactions');
      suggestions.push('Optimize database queries');
    }

    suggestions.push('Check database connectivity and permissions');
    suggestions.push('Review database logs for additional details');
    return suggestions;
  }
}

/**
 * Resource/memory errors
 */
class ResourceError extends RuvSwarmError {
  constructor(message, resourceType = null, currentUsage = null, limit = null) {
    const details = { resourceType, currentUsage, limit };
    super(message, 'RESOURCE_ERROR', details);
    this.resourceType = resourceType;
    this.currentUsage = currentUsage;
    this.limit = limit;
  }

  getSuggestions() {
    const suggestions = [];

    if (this.resourceType === 'memory') {
      suggestions.push('Reduce memory usage in operations');
      suggestions.push('Implement memory cleanup procedures');
      suggestions.push('Use streaming for large data sets');
      suggestions.push('Optimize data structures');
    } else if (this.resourceType === 'cpu') {
      suggestions.push('Reduce computational complexity');
      suggestions.push('Use async operations to prevent blocking');
      suggestions.push('Implement caching for expensive operations');
    } else if (this.resourceType === 'storage') {
      suggestions.push('Clean up temporary files');
      suggestions.push('Implement data compression');
      suggestions.push('Archive old data');
    }

    suggestions.push('Monitor resource usage trends');
    suggestions.push('Consider scaling up available resources');
    return suggestions;
  }
}

/**
 * Concurrency/threading errors
 */
class ConcurrencyError extends RuvSwarmError {
  constructor(message, operation = null, conflictType = null) {
    const details = { operation, conflictType };
    super(message, 'CONCURRENCY_ERROR', details);
    this.operation = operation;
    this.conflictType = conflictType;
  }

  getSuggestions() {
    return [
      'Implement proper locking mechanisms',
      'Use atomic operations where possible',
      'Retry the operation with exponential backoff',
      'Check for race conditions in the code',
      'Consider using queues for serializing operations',
      'Review concurrent access patterns',
    ];
  }
}

/**
 * Error factory for creating appropriate error types
 */
class ErrorFactory {
  /**
   * Create an appropriate error based on the context
   */
  static createError(type, message, details = {}) {
    switch (type) {
      case 'validation':
        return new ValidationError(message, details.field, details.value, details.expectedType);
      case 'swarm':
        return new SwarmError(message, details.swarmId, details.operation);
      case 'agent':
        return new AgentError(message, details.agentId, details.agentType, details.operation);
      case 'task':
        return new TaskError(message, details.taskId, details.taskType, details.operation);
      case 'neural':
        return new NeuralError(message, details.networkId, details.operation, details.modelType);
      case 'wasm':
        return new WasmError(message, details.module, details.operation);
      case 'configuration':
        return new ConfigurationError(message, details.configKey, details.configValue);
      case 'network':
        return new NetworkError(message, details.endpoint, details.statusCode);
      case 'persistence':
        return new PersistenceError(message, details.operation, details.table);
      case 'resource':
        return new ResourceError(message, details.resourceType, details.currentUsage, details.limit);
      case 'concurrency':
        return new ConcurrencyError(message, details.operation, details.conflictType);
      default:
        return new RuvSwarmError(message, 'GENERAL_ERROR', details);
    }
  }

  /**
   * Wrap an existing error with additional context
   */
  static wrapError(originalError, type, additionalContext = {}) {
    const message = `${type.toUpperCase()}: ${originalError.message}`;
    const details = {
      ...additionalContext,
      originalError: {
        name: originalError.name,
        message: originalError.message,
        stack: originalError.stack,
      },
    };

    return this.createError(type, message, details);
  }
}

/**
 * Error context for logging and debugging
 */
class ErrorContext {
  constructor() {
    this.context = new Map();
  }

  set(key, value) {
    this.context.set(key, value);
  }

  get(key) {
    return this.context.get(key);
  }

  clear() {
    this.context.clear();
  }

  toObject() {
    return Object.fromEntries(this.context);
  }

  /**
   * Add context to an error
   */
  enrichError(error) {
    if (error instanceof RuvSwarmError) {
      error.details = {
        ...error.details,
        context: this.toObject(),
      };
    }
    return error;
  }
}

// Export all error classes and utilities
export {
  RuvSwarmError,
  ValidationError,
  SwarmError,
  AgentError,
  TaskError,
  NeuralError,
  WasmError,
  ConfigurationError,
  NetworkError,
  PersistenceError,
  ResourceError,
  ConcurrencyError,
  ErrorFactory,
  ErrorContext,
};

// Export default error factory
export default ErrorFactory;
