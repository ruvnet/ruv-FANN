/**
 * SQLite Worker Thread for CPU-intensive database operations
 *
 * This worker handles long-running queries and CPU-intensive operations
 * to prevent blocking the main thread during heavy database workloads.
 */

import { parentPort, workerData } from 'worker_threads';
import Database from 'better-sqlite3';

class SQLiteWorker {
  constructor(dbPath, options = {}) {
    this.dbPath = dbPath;
    this.options = options;
    this.db = null;
    this.preparedStatements = new Map();

    this.initialize();
  }

  initialize() {
    try {
      this.db = new Database(this.dbPath, { readonly: true });
      this.configureConnection();

      // Test connection
      this.db.prepare('SELECT 1').get();

      // Setup message handling
      if (parentPort) {
        parentPort.on('message', (message) => {
          this.handleMessage(message);
        });
      }

    } catch (error) {
      if (parentPort) {
        parentPort.postMessage({
          error: `Worker initialization failed: ${error.message}`,
        });
      }
    }
  }

  configureConnection() {
    // Configure for optimal read performance
    this.db.pragma('journal_mode = WAL');
    this.db.pragma('synchronous = NORMAL');
    this.db.pragma('temp_store = MEMORY');
    this.db.pragma('mmap_size = ' + (this.options.mmapSize || 268435456));
    this.db.pragma('cache_size = ' + (this.options.cacheSize || -64000));
    this.db.pragma('busy_timeout = 5000');

    // Optimize for queries
    this.db.pragma('optimize');
  }

  handleMessage(message) {
    try {
      if (message.health) {
        // Health check
        this.db.prepare('SELECT 1').get();
        parentPort.postMessage({ health: 'ok' });
        return;
      }

      if (message.sql && message.params !== undefined) {
        // Execute query
        const result = this.executeQuery(message.sql, message.params);
        parentPort.postMessage({
          data: result,
          error: null,
        });
      } else {
        parentPort.postMessage({
          error: 'Invalid message format',
        });
      }

    } catch (error) {
      parentPort.postMessage({
        error: error.message,
        data: null,
      });
    }
  }

  executeQuery(sql, params) {
    const stmt = this.getPreparedStatement(sql);

    // Determine operation type
    const operation = sql.trim().toUpperCase().split(' ')[0];

    switch (operation) {
      case 'SELECT':
        return stmt.all(params);
      case 'INSERT':
      case 'UPDATE':
      case 'DELETE':
        return stmt.run(params);
      default:
        // For other operations, try to execute and return result
        try {
          return stmt.all(params);
        } catch {
          return stmt.run(params);
        }
    }
  }

  getPreparedStatement(sql) {
    if (!this.preparedStatements.has(sql)) {
      this.preparedStatements.set(sql, this.db.prepare(sql));
    }
    return this.preparedStatements.get(sql);
  }

  cleanup() {
    if (this.db) {
      this.db.close();
    }
  }
}

// Initialize worker
const worker = new SQLiteWorker(workerData.dbPath, workerData.options);

// Cleanup on exit
process.on('exit', () => {
  worker.cleanup();
});

process.on('SIGINT', () => {
  worker.cleanup();
  process.exit(0);
});

process.on('SIGTERM', () => {
  worker.cleanup();
  process.exit(0);
});
