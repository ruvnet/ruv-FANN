/**
 * Input Validation Schemas for RUV-Swarm MCP Tools
 * Provides comprehensive validation for all 25+ MCP tools
 */

import { ValidationError } from './errors.js';

/**
 * Base validator class
 */
class BaseValidator {
  static validate(value, schema, fieldName = 'value') {
    try {
      return this.validateValue(value, schema, fieldName);
    } catch (error) {
      if (error instanceof ValidationError) {
        throw error;
      }
      throw new ValidationError(
        `Validation failed for ${fieldName}: ${error.message}`,
        fieldName,
        value,
      );
    }
  }

  static validateValue(value, schema, fieldName) {
    // Handle required fields
    if (schema.required && (value === undefined || value === null)) {
      throw new ValidationError(
        `${fieldName} is required`,
        fieldName,
        value,
        schema.type,
      );
    }

    // Handle optional fields
    if (!schema.required && (value === undefined || value === null)) {
      return schema.default;
    }

    // Type validation
    if (schema.type && !this.validateType(value, schema.type)) {
      throw new ValidationError(
        `${fieldName} must be of type ${schema.type}`,
        fieldName,
        value,
        schema.type,
      );
    }

    // Range validation for numbers
    if (schema.type === 'number') {
      if (schema.min !== undefined && value < schema.min) {
        throw new ValidationError(
          `${fieldName} must be at least ${schema.min}`,
          fieldName,
          value,
          schema.type,
        );
      }
      if (schema.max !== undefined && value > schema.max) {
        throw new ValidationError(
          `${fieldName} must be at most ${schema.max}`,
          fieldName,
          value,
          schema.type,
        );
      }
      if (schema.integer && !Number.isInteger(value)) {
        throw new ValidationError(
          `${fieldName} must be an integer`,
          fieldName,
          value,
          'integer',
        );
      }
    }

    // Length validation for strings and arrays
    if (schema.type === 'string' || schema.type === 'array') {
      const length = schema.type === 'string' ? value.length : value.length;
      if (schema.minLength !== undefined && length < schema.minLength) {
        throw new ValidationError(
          `${fieldName} must be at least ${schema.minLength} characters/items long`,
          fieldName,
          value,
          schema.type,
        );
      }
      if (schema.maxLength !== undefined && length > schema.maxLength) {
        throw new ValidationError(
          `${fieldName} must be at most ${schema.maxLength} characters/items long`,
          fieldName,
          value,
          schema.type,
        );
      }
    }

    // Enum validation
    if (schema.enum && !schema.enum.includes(value)) {
      throw new ValidationError(
        `${fieldName} must be one of: ${schema.enum.join(', ')}`,
        fieldName,
        value,
        `enum(${schema.enum.join('|')})`,
      );
    }

    // Pattern validation for strings
    if (schema.type === 'string' && schema.pattern) {
      const regex = new RegExp(schema.pattern);
      if (!regex.test(value)) {
        throw new ValidationError(
          `${fieldName} does not match the required pattern`,
          fieldName,
          value,
          'string(pattern)',
        );
      }
    }

    // Object property validation
    if (schema.type === 'object' && schema.properties) {
      for (const [propName, propSchema] of Object.entries(schema.properties)) {
        if (value[propName] !== undefined) {
          value[propName] = this.validateValue(value[propName], propSchema, `${fieldName}.${propName}`);
        } else if (propSchema.required) {
          throw new ValidationError(
            `${fieldName}.${propName} is required`,
            `${fieldName}.${propName}`,
            undefined,
            propSchema.type,
          );
        }
      }
    }

    // Array item validation
    if (schema.type === 'array' && schema.items) {
      for (let i = 0; i < value.length; i++) {
        value[i] = this.validateValue(value[i], schema.items, `${fieldName}[${i}]`);
      }
    }

    return value;
  }

  static validateType(value, expectedType) {
    switch (expectedType) {
      case 'string':
        return typeof value === 'string';
      case 'number':
        return typeof value === 'number' && !isNaN(value) && isFinite(value);
      case 'boolean':
        return typeof value === 'boolean';
      case 'array':
        return Array.isArray(value);
      case 'object':
        return typeof value === 'object' && value !== null && !Array.isArray(value);
      case 'function':
        return typeof value === 'function';
      default:
        return true;
    }
  }
}

/**
 * Schema definitions for all MCP tools
 */
const MCPSchemas = {
  // Core Swarm Management
  swarm_init: {
    topology: {
      type: 'string',
      enum: ['mesh', 'hierarchical', 'ring', 'star'],
      default: 'mesh',
    },
    maxAgents: {
      type: 'number',
      integer: true,
      min: 1,
      max: 100,
      default: 5,
    },
    strategy: {
      type: 'string',
      enum: ['balanced', 'specialized', 'adaptive'],
      default: 'balanced',
    },
    enableCognitiveDiversity: {
      type: 'boolean',
      default: true,
    },
    enableNeuralAgents: {
      type: 'boolean',
      default: true,
    },
    enableForecasting: {
      type: 'boolean',
      default: false,
    },
  },

  agent_spawn: {
    type: {
      type: 'string',
      enum: ['researcher', 'coder', 'analyst', 'optimizer', 'coordinator', 'tester', 'reviewer', 'documenter'],
      default: 'researcher',
    },
    name: {
      type: 'string',
      minLength: 1,
      maxLength: 100,
      required: false,
    },
    capabilities: {
      type: 'array',
      items: {
        type: 'string',
        minLength: 1,
      },
      required: false,
    },
    cognitivePattern: {
      type: 'string',
      enum: ['convergent', 'divergent', 'lateral', 'systems', 'critical', 'adaptive'],
      required: false,
    },
    swarmId: {
      type: 'string',
      pattern: '^[a-fA-F0-9-]+$',
      required: false,
    },
  },

  task_orchestrate: {
    task: {
      type: 'string',
      required: true,
      minLength: 1,
      maxLength: 1000,
    },
    priority: {
      type: 'string',
      enum: ['low', 'medium', 'high', 'critical'],
      default: 'medium',
    },
    strategy: {
      type: 'string',
      enum: ['parallel', 'sequential', 'adaptive'],
      default: 'adaptive',
    },
    maxAgents: {
      type: 'number',
      integer: true,
      min: 1,
      max: 50,
      required: false,
    },
    swarmId: {
      type: 'string',
      pattern: '^[a-fA-F0-9-]+$',
      required: false,
    },
    requiredCapabilities: {
      type: 'array',
      items: {
        type: 'string',
        minLength: 1,
      },
      required: false,
    },
    estimatedDuration: {
      type: 'number',
      min: 1000,
      max: 3600000, // 1 hour max
      required: false,
    },
  },

  swarm_status: {
    verbose: {
      type: 'boolean',
      default: false,
    },
    swarmId: {
      type: 'string',
      pattern: '^[a-fA-F0-9-]+$',
      required: false,
    },
  },

  task_status: {
    taskId: {
      type: 'string',
      pattern: '^[a-fA-F0-9-]+$',
      required: false,
    },
    detailed: {
      type: 'boolean',
      default: false,
    },
  },

  task_results: {
    taskId: {
      type: 'string',
      required: true,
      pattern: '^[a-fA-F0-9-]+$',
    },
    format: {
      type: 'string',
      enum: ['summary', 'detailed', 'raw', 'performance'],
      default: 'summary',
    },
    includeAgentResults: {
      type: 'boolean',
      default: true,
    },
  },

  agent_list: {
    filter: {
      type: 'string',
      enum: ['all', 'active', 'idle', 'busy'],
      default: 'all',
    },
    swarmId: {
      type: 'string',
      pattern: '^[a-fA-F0-9-]+$',
      required: false,
    },
  },

  agent_metrics: {
    agentId: {
      type: 'string',
      pattern: '^[a-fA-F0-9-]+$',
      required: false,
    },
    swarmId: {
      type: 'string',
      pattern: '^[a-fA-F0-9-]+$',
      required: false,
    },
    metric: {
      type: 'string',
      enum: ['all', 'cpu', 'memory', 'tasks', 'performance', 'neural'],
      default: 'all',
    },
  },

  benchmark_run: {
    type: {
      type: 'string',
      enum: ['all', 'wasm', 'swarm', 'agent', 'task', 'neural'],
      default: 'all',
    },
    iterations: {
      type: 'number',
      integer: true,
      min: 1,
      max: 100,
      default: 10,
    },
    includeNeuralBenchmarks: {
      type: 'boolean',
      default: true,
    },
    includeSwarmBenchmarks: {
      type: 'boolean',
      default: true,
    },
  },

  features_detect: {
    category: {
      type: 'string',
      enum: ['all', 'wasm', 'simd', 'memory', 'platform', 'neural', 'forecasting'],
      default: 'all',
    },
  },

  memory_usage: {
    detail: {
      type: 'string',
      enum: ['summary', 'detailed', 'by-agent'],
      default: 'summary',
    },
  },

  // Neural Network Tools
  neural_status: {
    agentId: {
      type: 'string',
      pattern: '^[a-fA-F0-9-]+$',
      required: false,
    },
  },

  neural_train: {
    agentId: {
      type: 'string',
      required: true,
      pattern: '^[a-fA-F0-9-]+$',
    },
    iterations: {
      type: 'number',
      integer: true,
      min: 1,
      max: 100,
      default: 10,
    },
    learningRate: {
      type: 'number',
      min: 0.00001,
      max: 1.0,
      default: 0.001,
    },
    modelType: {
      type: 'string',
      enum: ['feedforward', 'lstm', 'transformer', 'cnn', 'attention'],
      default: 'feedforward',
    },
    trainingData: {
      type: 'object',
      required: false,
    },
  },

  neural_patterns: {
    pattern: {
      type: 'string',
      enum: ['all', 'convergent', 'divergent', 'lateral', 'systems', 'critical', 'abstract', 'adaptive'],
      default: 'all',
    },
  },

  // DAA (Decentralized Autonomous Agents) Tools
  daa_init: {
    enableCoordination: {
      type: 'boolean',
      default: true,
    },
    enableLearning: {
      type: 'boolean',
      default: true,
    },
    persistenceMode: {
      type: 'string',
      enum: ['auto', 'memory', 'disk'],
      default: 'auto',
    },
  },

  daa_agent_create: {
    id: {
      type: 'string',
      required: true,
      minLength: 1,
      maxLength: 100,
    },
    capabilities: {
      type: 'array',
      items: {
        type: 'string',
        minLength: 1,
      },
      required: false,
    },
    cognitivePattern: {
      type: 'string',
      enum: ['convergent', 'divergent', 'lateral', 'systems', 'critical', 'adaptive'],
      required: false,
    },
    enableMemory: {
      type: 'boolean',
      default: true,
    },
    learningRate: {
      type: 'number',
      min: 0.001,
      max: 1.0,
      default: 0.1,
    },
  },

  daa_agent_adapt: {
    agent_id: {
      type: 'string',
      required: true,
      minLength: 1,
    },
    agentId: {
      type: 'string',
      required: false,
      minLength: 1,
    },
    feedback: {
      type: 'string',
      minLength: 1,
      maxLength: 1000,
      required: false,
    },
    performanceScore: {
      type: 'number',
      min: 0,
      max: 1,
      required: false,
    },
    suggestions: {
      type: 'array',
      items: {
        type: 'string',
        minLength: 1,
      },
      required: false,
    },
  },

  daa_workflow_create: {
    id: {
      type: 'string',
      required: true,
      minLength: 1,
      maxLength: 100,
    },
    name: {
      type: 'string',
      required: true,
      minLength: 1,
      maxLength: 200,
    },
    steps: {
      type: 'array',
      items: {
        type: 'object',
      },
      required: false,
    },
    dependencies: {
      type: 'object',
      required: false,
    },
    strategy: {
      type: 'string',
      enum: ['parallel', 'sequential', 'adaptive'],
      default: 'adaptive',
    },
  },

  daa_workflow_execute: {
    workflow_id: {
      type: 'string',
      required: true,
      minLength: 1,
    },
    workflowId: {
      type: 'string',
      required: false,
      minLength: 1,
    },
    agentIds: {
      type: 'array',
      items: {
        type: 'string',
        minLength: 1,
      },
      required: false,
    },
    parallelExecution: {
      type: 'boolean',
      default: true,
    },
  },

  daa_knowledge_share: {
    source_agent: {
      type: 'string',
      required: true,
      minLength: 1,
    },
    sourceAgentId: {
      type: 'string',
      required: false,
      minLength: 1,
    },
    target_agents: {
      type: 'array',
      items: {
        type: 'string',
        minLength: 1,
      },
      required: true,
      minItems: 1,
    },
    targetAgentIds: {
      type: 'array',
      items: {
        type: 'string',
        minLength: 1,
      },
      required: true,
      minLength: 1,
    },
    knowledgeDomain: {
      type: 'string',
      minLength: 1,
      required: false,
    },
    knowledgeContent: {
      type: 'object',
      required: false,
    },
  },

  daa_learning_status: {
    agentId: {
      type: 'string',
      required: false,
    },
    detailed: {
      type: 'boolean',
      default: false,
    },
  },

  daa_cognitive_pattern: {
    agentId: {
      type: 'string',
      required: false,
    },
    pattern: {
      type: 'string',
      enum: ['convergent', 'divergent', 'lateral', 'systems', 'critical', 'adaptive'],
      required: false,
    },
    analyze: {
      type: 'boolean',
      default: false,
    },
  },

  daa_meta_learning: {
    sourceDomain: {
      type: 'string',
      minLength: 1,
      required: false,
    },
    targetDomain: {
      type: 'string',
      minLength: 1,
      required: false,
    },
    transferMode: {
      type: 'string',
      enum: ['adaptive', 'direct', 'gradual'],
      default: 'adaptive',
    },
    agentIds: {
      type: 'array',
      items: {
        type: 'string',
        minLength: 1,
      },
      required: false,
    },
  },

  daa_performance_metrics: {
    category: {
      type: 'string',
      enum: ['all', 'system', 'performance', 'efficiency', 'neural'],
      default: 'all',
    },
    timeRange: {
      type: 'string',
      pattern: '^\\d+[hmd]$', // e.g., "1h", "24h", "7d"
      required: false,
    },
  },

  // Monitoring Tools
  swarm_monitor: {
    swarmId: {
      type: 'string',
      pattern: '^[a-fA-F0-9-]+$',
      required: false,
    },
    duration: {
      type: 'number',
      integer: true,
      min: 1,
      max: 3600, // 1 hour max
      default: 10,
    },
    interval: {
      type: 'number',
      integer: true,
      min: 1,
      max: 60,
      default: 1,
    },
    includeAgents: {
      type: 'boolean',
      default: true,
    },
    includeTasks: {
      type: 'boolean',
      default: true,
    },
    includeMetrics: {
      type: 'boolean',
      default: true,
    },
    realTime: {
      type: 'boolean',
      default: false,
    },
  },
};

/**
 * Validation utilities
 */
class ValidationUtils {
  /**
   * Validate parameters against a schema
   */
  static validateParams(params, toolName) {
    const schema = MCPSchemas[toolName];
    if (!schema) {
      throw new ValidationError(
        `No validation schema found for tool: ${toolName}`,
        'toolName',
        toolName,
        'string',
      );
    }

    // Handle empty or null params
    if (!params || typeof params !== 'object') {
      params = {};
    }

    const validatedParams = {};

    // Validate each parameter
    for (const [fieldName, fieldSchema] of Object.entries(schema)) {
      try {
        const value = params[fieldName];
        validatedParams[fieldName] = BaseValidator.validate(value, fieldSchema, fieldName);
      } catch (error) {
        // Add tool context to error
        if (error instanceof ValidationError) {
          error.details.tool = toolName;
          error.details.schema = fieldSchema;
        }
        throw error;
      }
    }

    // Check for unexpected parameters
    const allowedFields = Object.keys(schema);
    const providedFields = Object.keys(params);
    const unexpectedFields = providedFields.filter(field => !allowedFields.includes(field));

    if (unexpectedFields.length > 0) {
      console.warn(`Unexpected parameters for ${toolName}: ${unexpectedFields.join(', ')}`);
      // Note: We don't throw here to maintain backward compatibility
    }

    return validatedParams;
  }

  /**
   * Get schema documentation for a tool
   */
  static getSchemaDoc(toolName) {
    const schema = MCPSchemas[toolName];
    if (!schema) {
      return null;
    }

    const doc = {
      tool: toolName,
      parameters: {},
    };

    for (const [fieldName, fieldSchema] of Object.entries(schema)) {
      doc.parameters[fieldName] = {
        type: fieldSchema.type,
        required: fieldSchema.required || false,
        default: fieldSchema.default,
        description: this.generateFieldDescription(fieldName, fieldSchema),
      };

      if (fieldSchema.enum) {
        doc.parameters[fieldName].allowedValues = fieldSchema.enum;
      }
      if (fieldSchema.min !== undefined || fieldSchema.max !== undefined) {
        doc.parameters[fieldName].range = {
          min: fieldSchema.min,
          max: fieldSchema.max,
        };
      }
      if (fieldSchema.minLength !== undefined || fieldSchema.maxLength !== undefined) {
        doc.parameters[fieldName].length = {
          min: fieldSchema.minLength,
          max: fieldSchema.maxLength,
        };
      }
    }

    return doc;
  }

  /**
   * Generate human-readable description for a field
   */
  static generateFieldDescription(fieldName, schema) {
    let desc = `${fieldName} (${schema.type})`;

    if (schema.required) {
      desc += ' - Required';
    } else {
      desc += ' - Optional';
      if (schema.default !== undefined) {
        desc += `, default: ${schema.default}`;
      }
    }

    if (schema.enum) {
      desc += `. Allowed values: ${schema.enum.join(', ')}`;
    }

    if (schema.min !== undefined || schema.max !== undefined) {
      desc += `. Range: ${schema.min || 'any'} to ${schema.max || 'any'}`;
    }

    if (schema.minLength !== undefined || schema.maxLength !== undefined) {
      desc += `. Length: ${schema.minLength || 0} to ${schema.maxLength || 'unlimited'}`;
    }

    return desc;
  }

  /**
   * Get all available tool schemas
   */
  static getAllSchemas() {
    return Object.keys(MCPSchemas);
  }

  /**
   * Validate a UUID string
   */
  static isValidUUID(str) {
    const uuidRegex = /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
    return uuidRegex.test(str);
  }

  /**
   * Sanitize input to prevent injection attacks
   */
  static sanitizeInput(input) {
    if (typeof input === 'string') {
      // Remove potentially dangerous characters
      return input.replace(/[<>"'&\x00-\x1f\x7f-\x9f]/g, '');
    }
    return input;
  }
}

/**
 * Export validation schemas and utilities
 */
export {
  MCPSchemas,
  BaseValidator,
  ValidationUtils,
};

export default ValidationUtils;
