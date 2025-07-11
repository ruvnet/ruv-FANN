/**
 * Logger module for ruv-swarm with comprehensive logging capabilities
 */

import { randomUUID } from 'crypto';

export class Logger {
    constructor(options = {}) {
        this.name = options.name || 'ruv-swarm';
        this.level = options.level || 'INFO';
        this.enableStderr = options.enableStderr !== false;
        this.enableFile = options.enableFile || false;
        this.formatJson = options.formatJson || false;
        this.logDir = options.logDir || './logs';
        this.metadata = options.metadata || {};
        this.correlationId = null;
        this.operations = new Map();
    }

    setCorrelationId(id) {
        this.correlationId = id || randomUUID();
        return this.correlationId;
    }

    getCorrelationId() {
        return this.correlationId;
    }

    _log(level, message, data = {}) {
        const timestamp = new Date().toISOString();
        const prefix = this.correlationId ? `[${this.correlationId}] ` : '';
        
        const logEntry = {
            timestamp,
            level,
            name: this.name,
            message,
            correlationId: this.correlationId,
            ...this.metadata,
            ...data
        };

        if (this.formatJson) {
            const output = JSON.stringify(logEntry);
            if (this.enableStderr) {
                console.error(output);
            } else {
                console.log(output);
            }
        } else {
            const output = `${prefix}[${level}] ${message}`;
            if (this.enableStderr) {
                console.error(output, Object.keys(data).length > 0 ? data : '');
            } else {
                console.log(output, Object.keys(data).length > 0 ? data : '');
            }
        }
    }

    info(message, data = {}) {
        this._log('INFO', message, data);
    }

    warn(message, data = {}) {
        this._log('WARN', message, data);
    }

    error(message, data = {}) {
        this._log('ERROR', message, data);
    }

    debug(message, data = {}) {
        if (this.level === 'DEBUG' || process.env.DEBUG) {
            this._log('DEBUG', message, data);
        }
    }

    trace(message, data = {}) {
        if (this.level === 'TRACE' || process.env.DEBUG) {
            this._log('TRACE', message, data);
        }
    }

    success(message, data = {}) {
        this._log('SUCCESS', message, data);
    }

    fatal(message, data = {}) {
        this._log('FATAL', message, data);
    }

    startOperation(operationName) {
        const operationId = randomUUID();
        this.operations.set(operationId, {
            name: operationName,
            startTime: Date.now()
        });
        this.debug(`Starting operation: ${operationName}`, { operationId });
        return operationId;
    }

    endOperation(operationId, success = true, data = {}) {
        const operation = this.operations.get(operationId);
        if (operation) {
            const duration = Date.now() - operation.startTime;
            this.debug(`Operation ${success ? 'completed' : 'failed'}: ${operation.name}`, {
                operationId,
                duration,
                success,
                ...data
            });
            this.operations.delete(operationId);
        }
    }

    logConnection(event, sessionId, data = {}) {
        this.info(`Connection ${event}`, {
            sessionId,
            event,
            ...data
        });
    }

    logMcp(direction, method, data = {}) {
        this.debug(`MCP ${direction.toUpperCase()}: ${method}`, {
            direction,
            method,
            ...data
        });
    }

    logMemoryUsage(component) {
        const memUsage = process.memoryUsage();
        this.trace(`Memory usage for ${component}`, {
            component,
            rss: `${(memUsage.rss / 1024 / 1024).toFixed(2)}MB`,
            heapUsed: `${(memUsage.heapUsed / 1024 / 1024).toFixed(2)}MB`,
            heapTotal: `${(memUsage.heapTotal / 1024 / 1024).toFixed(2)}MB`,
            external: `${(memUsage.external / 1024 / 1024).toFixed(2)}MB`
        });
    }

    getConnectionMetrics() {
        return {
            activeOperations: this.operations.size,
            correlationId: this.correlationId,
            uptime: process.uptime()
        };
    }

    // Static methods for backward compatibility
    static info(message, ...args) {
        const logger = new Logger();
        logger.info(message, ...args);
    }

    static warn(message, ...args) {
        const logger = new Logger();
        logger.warn(message, ...args);
    }

    static error(message, ...args) {
        const logger = new Logger();
        logger.error(message, ...args);
    }

    static debug(message, ...args) {
        const logger = new Logger();
        logger.debug(message, ...args);
    }

    static success(message, ...args) {
        const logger = new Logger();
        logger.success(message, ...args);
    }

    static trace(message, ...args) {
        const logger = new Logger();
        logger.trace(message, ...args);
    }
}

export default Logger;