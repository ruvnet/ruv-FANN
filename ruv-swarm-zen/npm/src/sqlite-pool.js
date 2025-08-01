/**
 * High-Availability SQLite Connection Pool for ruv-swarm
 *
 * This implementation addresses the production readiness concerns:
 * - Connection exhaustion prevention
 * - Deadlock avoidance under load
 * - System availability during high concurrency
 * - Proper resource lifecycle management
 *
 * Design decisions:
 * - Single primary connection for writes (SQLite single-writer limitation)
 * - Multiple reader connections in WAL mode for concurrent reads
 * - Worker thread pool for CPU-intensive queries
 * - Connection health monitoring and auto-recovery
 * - Graceful degradation under pressure
 */

import Database from 'better-sqlite3';
import { Worker } from 'worker_threads';
import { EventEmitter } from 'events';
import path from 'path';
import fs from 'fs';
import os from 'os';

class SQLiteConnectionPool extends EventEmitter {
  constructor(dbPath, options = {}) {
    super();

    this.dbPath = dbPath;
    this.options = {
      // Pool configuration
      maxReaders: options.maxReaders || Math.max(4, os.cpus().length),
      maxWorkers: options.maxWorkers || Math.max(2, Math.floor(os.cpus().length / 2)),

      // Connection timeouts
      acquireTimeout: options.acquireTimeout || 30000, // 30 seconds
      idleTimeout: options.idleTimeout || 300000, // 5 minutes

      // Health monitoring
      healthCheckInterval: options.healthCheckInterval || 60000, // 1 minute
      maxRetries: options.maxRetries || 3,

      // Performance settings
      mmapSize: options.mmapSize || 268435456, // 256MB
      cacheSize: options.cacheSize || -64000, // 64MB

      // High availability
      enableBackup: options.enableBackup || false,
      backupInterval: options.backupInterval || 3600000, // 1 hour

      ...options,
    };

    // Connection pools
    this.writeConnection = null;
    this.readerConnections = [];
    this.availableReaders = [];
    this.busyReaders = new Set();

    // Worker thread pool
    this.workers = [];
    this.availableWorkers = [];
    this.busyWorkers = new Set();

    // Request queues
    this.readQueue = [];
    this.writeQueue = [];
    this.workerQueue = [];

    // Health monitoring
    this.isHealthy = true;
    this.healthCheckTimer = null;
    this.lastHealthCheck = Date.now();

    // Statistics
    this.stats = {
      totalReads: 0,
      totalWrites: 0,
      totalWorkerTasks: 0,
      failedConnections: 0,
      averageReadTime: 0,
      averageWriteTime: 0,
      activeConnections: 0,
    };

    // Prepared statements cache
    this.preparedStatements = new Map();

    this.initialize();
  }

  async initialize() {
    try {
      // Ensure database directory exists
      const dataDir = path.dirname(this.dbPath);
      if (!fs.existsSync(dataDir)) {
        fs.mkdirSync(dataDir, { recursive: true });
      }

      // Initialize write connection
      await this.initializeWriteConnection();

      // Initialize reader connections
      await this.initializeReaderConnections();

      // Initialize worker threads
      await this.initializeWorkerThreads();

      // Start health monitoring
      this.startHealthMonitoring();

      // Setup cleanup handlers
      this.setupCleanupHandlers();

      this.emit('ready');
    } catch (error) {
      this.emit('error', error);
      throw error;
    }
  }

  async initializeWriteConnection() {
    try {
      this.writeConnection = new Database(this.dbPath);
      this.configureConnection(this.writeConnection);

      // Test connection
      this.writeConnection.prepare('SELECT 1').get();

      this.emit('write-connection-ready');
    } catch (error) {
      this.stats.failedConnections++;
      throw new Error(`Failed to initialize write connection: ${error.message}`);
    }
  }

  async initializeReaderConnections() {
    for (let i = 0; i < this.options.maxReaders; i++) {
      try {
        const reader = new Database(this.dbPath, { readonly: true });
        this.configureReadOnlyConnection(reader);

        // Test connection
        reader.prepare('SELECT 1').get();

        this.readerConnections.push(reader);
        this.availableReaders.push(reader);

        this.emit('reader-connection-ready', i);
      } catch (error) {
        this.stats.failedConnections++;
        console.error(`Failed to initialize reader connection ${i}:`, error);
        // Continue with fewer readers rather than failing completely
      }
    }

    if (this.availableReaders.length === 0) {
      throw new Error('Failed to initialize any reader connections');
    }
  }

  async initializeWorkerThreads() {
    const workerScript = path.join(path.dirname(new URL(import.meta.url).pathname), 'sqlite-worker.js');

    for (let i = 0; i < this.options.maxWorkers; i++) {
      try {
        const worker = new Worker(workerScript, {
          workerData: {
            dbPath: this.dbPath,
            options: this.options,
          },
        });

        worker.on('error', (error) => {
          this.emit('worker-error', error);
          this.handleWorkerError(worker, error);
        });

        worker.on('exit', (code) => {
          if (code !== 0) {
            this.emit('worker-exit', code);
            this.handleWorkerExit(worker, code);
          }
        });

        this.workers.push(worker);
        this.availableWorkers.push(worker);

        this.emit('worker-ready', i);
      } catch (error) {
        console.error(`Failed to initialize worker ${i}:`, error);
        // Continue with fewer workers rather than failing completely
      }
    }
  }

  configureConnection(db) {
    // Essential SQLite optimizations for high availability
    db.pragma('journal_mode = WAL');      // Enable WAL mode for concurrent reads
    db.pragma('synchronous = NORMAL');    // Balance safety and performance
    db.pragma('temp_store = MEMORY');     // Use memory for temp tables
    db.pragma('mmap_size = ' + this.options.mmapSize);
    db.pragma('cache_size = ' + this.options.cacheSize);
    db.pragma('foreign_keys = ON');       // Enable foreign key constraints
    db.pragma('busy_timeout = 5000');     // 5 second timeout for busy database

    // Optimize query planner
    db.pragma('optimize');
  }

  configureReadOnlyConnection(db) {
    // Configuration for readonly connections - limited pragma statements
    try {
      // These are safe for readonly connections
      db.pragma('temp_store = MEMORY');     // Use memory for temp tables
      db.pragma('cache_size = ' + this.options.cacheSize);
      db.pragma('busy_timeout = 5000');     // 5 second timeout for busy database
    } catch (error) {
      // Some pragma statements might fail on readonly connections
      // This is expected behavior, continue without them
      console.debug('Some pragma statements skipped for readonly connection:', error.message);
    }
  }

  // Read operation with connection pooling
  async read(sql, params = []) {
    const startTime = Date.now();

    try {
      const connection = await this.acquireReaderConnection();
      const stmt = this.getPreparedStatement(connection, sql);
      const result = stmt.all(params);

      this.releaseReaderConnection(connection);

      // Update statistics
      this.stats.totalReads++;
      this.updateAverageTime('read', Date.now() - startTime);

      return result;
    } catch (error) {
      this.emit('read-error', error);
      throw error;
    }
  }

  // Write operation with queuing
  async write(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.writeQueue.push({
        sql,
        params,
        resolve,
        reject,
        timestamp: Date.now(),
      });

      this.processWriteQueue();
    });
  }

  // Transaction support
  async transaction(fn) {
    const startTime = Date.now();

    try {
      const result = this.writeConnection.transaction(fn)();

      this.stats.totalWrites++;
      this.updateAverageTime('write', Date.now() - startTime);

      return result;
    } catch (error) {
      this.emit('transaction-error', error);
      throw error;
    }
  }

  // CPU-intensive query execution in worker thread
  async executeInWorker(sql, params = []) {
    return new Promise((resolve, reject) => {
      this.workerQueue.push({
        sql,
        params,
        resolve,
        reject,
        timestamp: Date.now(),
      });

      this.processWorkerQueue();
    });
  }

  async acquireReaderConnection() {
    return new Promise((resolve, reject) => {
      if (this.availableReaders.length > 0) {
        const connection = this.availableReaders.pop();
        this.busyReaders.add(connection);
        resolve(connection);
      } else {
        // Queue the request
        this.readQueue.push({ resolve, reject, timestamp: Date.now() });

        // Timeout handling
        setTimeout(() => {
          const index = this.readQueue.findIndex(item => item.resolve === resolve);
          if (index !== -1) {
            this.readQueue.splice(index, 1);
            reject(new Error('Reader connection acquire timeout'));
          }
        }, this.options.acquireTimeout);
      }
    });
  }

  releaseReaderConnection(connection) {
    if (this.busyReaders.has(connection)) {
      this.busyReaders.delete(connection);

      // Process queued read requests
      if (this.readQueue.length > 0) {
        const { resolve } = this.readQueue.shift();
        this.busyReaders.add(connection);
        resolve(connection);
      } else {
        this.availableReaders.push(connection);
      }
    }
  }

  async processWriteQueue() {
    if (this.writeQueue.length === 0) return;

    const { sql, params, resolve, reject, timestamp } = this.writeQueue.shift();
    const startTime = Date.now();

    try {
      // Check for timeout
      if (Date.now() - timestamp > this.options.acquireTimeout) {
        reject(new Error('Write operation timeout'));
        return;
      }

      const stmt = this.getPreparedStatement(this.writeConnection, sql);
      const result = stmt.run(params);

      this.stats.totalWrites++;
      this.updateAverageTime('write', Date.now() - startTime);

      resolve(result);
    } catch (error) {
      this.emit('write-error', error);
      reject(error);
    }

    // Process next item in queue
    if (this.writeQueue.length > 0) {
      setImmediate(() => this.processWriteQueue());
    }
  }

  async processWorkerQueue() {
    if (this.workerQueue.length === 0 || this.availableWorkers.length === 0) return;

    const worker = this.availableWorkers.pop();
    const { sql, params, resolve, reject, timestamp } = this.workerQueue.shift();

    try {
      // Check for timeout
      if (Date.now() - timestamp > this.options.acquireTimeout) {
        reject(new Error('Worker task timeout'));
        this.availableWorkers.push(worker);
        return;
      }

      this.busyWorkers.add(worker);

      const messageHandler = (result) => {
        this.busyWorkers.delete(worker);
        this.availableWorkers.push(worker);
        this.stats.totalWorkerTasks++;

        if (result.error) {
          reject(new Error(result.error));
        } else {
          resolve(result.data);
        }

        // Process next item in queue
        if (this.workerQueue.length > 0) {
          setImmediate(() => this.processWorkerQueue());
        }
      };

      worker.once('message', messageHandler);
      worker.postMessage({ sql, params });

    } catch (error) {
      this.busyWorkers.delete(worker);
      this.availableWorkers.push(worker);
      reject(error);
    }
  }

  getPreparedStatement(connection, sql) {
    const key = `${connection._id || 'main'}_${sql}`;
    if (!this.preparedStatements.has(key)) {
      this.preparedStatements.set(key, connection.prepare(sql));
    }
    return this.preparedStatements.get(key);
  }

  startHealthMonitoring() {
    this.healthCheckTimer = setInterval(() => {
      this.performHealthCheck();
    }, this.options.healthCheckInterval);
  }

  async performHealthCheck() {
    const startTime = Date.now();

    try {
      // Check write connection
      this.writeConnection.prepare('SELECT 1').get();

      // Check reader connections
      for (const reader of this.readerConnections) {
        reader.prepare('SELECT 1').get();
      }

      // Check worker threads
      const workerHealthPromises = this.workers.map(worker => {
        return new Promise((resolve) => {
          const timeout = setTimeout(() => resolve(false), 1000);
          worker.once('message', (result) => {
            clearTimeout(timeout);
            resolve(result.health === 'ok');
          });
          worker.postMessage({ health: true });
        });
      });

      const workerHealthResults = await Promise.all(workerHealthPromises);
      const healthyWorkers = workerHealthResults.filter(Boolean).length;

      this.isHealthy = healthyWorkers >= Math.floor(this.options.maxWorkers / 2);
      this.lastHealthCheck = Date.now();

      this.emit('health-check', {
        healthy: this.isHealthy,
        duration: Date.now() - startTime,
        workers: healthyWorkers,
        readers: this.availableReaders.length,
        stats: this.getStats(),
      });

    } catch (error) {
      this.isHealthy = false;
      this.emit('health-check-error', error);
    }
  }

  handleWorkerError(worker, error) {
    // Remove failed worker from pools
    this.busyWorkers.delete(worker);
    const index = this.availableWorkers.indexOf(worker);
    if (index !== -1) {
      this.availableWorkers.splice(index, 1);
    }

    // Attempt to create replacement worker
    this.createReplacementWorker();
  }

  handleWorkerExit(worker, code) {
    // Remove exited worker from pools
    this.busyWorkers.delete(worker);
    const index = this.availableWorkers.indexOf(worker);
    if (index !== -1) {
      this.availableWorkers.splice(index, 1);
    }

    // Attempt to create replacement worker
    this.createReplacementWorker();
  }

  async createReplacementWorker() {
    try {
      const workerScript = path.join(path.dirname(new URL(import.meta.url).pathname), 'sqlite-worker.js');
      const worker = new Worker(workerScript, {
        workerData: {
          dbPath: this.dbPath,
          options: this.options,
        },
      });

      worker.on('error', (error) => {
        this.emit('worker-error', error);
        this.handleWorkerError(worker, error);
      });

      worker.on('exit', (code) => {
        if (code !== 0) {
          this.emit('worker-exit', code);
          this.handleWorkerExit(worker, code);
        }
      });

      this.workers.push(worker);
      this.availableWorkers.push(worker);

      this.emit('worker-replaced');
    } catch (error) {
      this.emit('worker-replacement-failed', error);
    }
  }

  updateAverageTime(type, time) {
    const statKey = `average${type.charAt(0).toUpperCase() + type.slice(1)}Time`;
    const countKey = `total${type.charAt(0).toUpperCase() + type.slice(1)}s`;

    if (this.stats[statKey] === 0) {
      this.stats[statKey] = time;
    } else {
      this.stats[statKey] = (this.stats[statKey] + time) / 2;
    }
  }

  getStats() {
    return {
      ...this.stats,
      activeConnections: this.busyReaders.size + (this.writeConnection ? 1 : 0),
      availableReaders: this.availableReaders.length,
      availableWorkers: this.availableWorkers.length,
      readQueueLength: this.readQueue.length,
      writeQueueLength: this.writeQueue.length,
      workerQueueLength: this.workerQueue.length,
      isHealthy: this.isHealthy,
      lastHealthCheck: this.lastHealthCheck,
    };
  }

  setupCleanupHandlers() {
    const cleanup = () => {
      this.close();
    };

    process.on('exit', cleanup);
    process.on('SIGINT', cleanup);
    process.on('SIGTERM', cleanup);
    process.on('uncaughtException', cleanup);
  }

  async close() {
    try {
      // Stop health monitoring
      if (this.healthCheckTimer) {
        clearInterval(this.healthCheckTimer);
      }

      // Close all connections
      if (this.writeConnection) {
        this.writeConnection.close();
      }

      for (const reader of this.readerConnections) {
        reader.close();
      }

      // Terminate worker threads
      for (const worker of this.workers) {
        await worker.terminate();
      }

      // Clear all queues
      this.readQueue.length = 0;
      this.writeQueue.length = 0;
      this.workerQueue.length = 0;

      this.emit('closed');
    } catch (error) {
      this.emit('close-error', error);
    }
  }
}

export { SQLiteConnectionPool };
