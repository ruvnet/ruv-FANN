/**
 * 🚀 UNIFIED LANCE-SWARM PERSISTENCE LAYER
 *
 * Revolutionary hybrid architecture combining:
 * - LanceDB vector storage for semantic similarity
 * - SQLite relational data for structured queries
 * - Graph relationships for hive-mind coordination
 * - Neural pattern storage for continuous learning
 *
 * This replaces both the Claude Zen memory plugin AND ruv-swarm persistence
 * with a single, ultra-high-performance unified backend.
 */

import { SwarmPersistencePooled } from './persistence-pooled.js';
import path from 'path';
import fs from 'fs/promises';

export class UnifiedLancePersistence {
  constructor(options = {}) {
    this.options = {
      // LanceDB configuration
      lanceDbPath: options.lanceDbPath || './.hive-mind/lance-db',
      collection: options.collection || 'unified_swarm_memory',

      // SQLite configuration (for structured data)
      sqliteDbPath: options.sqliteDbPath || './.hive-mind/swarm-relational.db',

      // Hybrid configuration
      enableVectorSearch: options.enableVectorSearch !== false,
      enableGraphTraversal: options.enableGraphTraversal !== false,
      enableNeuralPatterns: options.enableNeuralPatterns !== false,

      // Performance settings
      maxReaders: options.maxReaders || 8,
      maxWorkers: options.maxWorkers || 4,
      vectorDimensions: options.vectorDimensions || 384, // All-MiniLM embeddings

      ...options,
    };

    // Storage engines
    this.lanceDb = null;
    this.lanceTable = null;
    this.sqlitePersistence = null;

    // State management
    this.initialized = false;
    this.initializing = false;

    // Performance tracking
    this.stats = {
      vectorQueries: 0,
      relationQueries: 0,
      hybridQueries: 0,
      totalOperations: 0,
      averageResponseTime: 0,
    };
  }

  async initialize() {
    if (this.initialized) return;
    if (this.initializing) {
      // Wait for initialization to complete
      return new Promise((resolve, reject) => {
        const checkInitialized = () => {
          if (this.initialized) {
            resolve();
          } else if (!this.initializing) {
            reject(new Error('Initialization failed'));
          } else {
            setTimeout(checkInitialized, 100);
          }
        };
        checkInitialized();
      });
    }

    this.initializing = true;

    try {
      console.log('🚀 Initializing Unified Lance-Swarm Persistence Layer...');

      // Initialize LanceDB for vector storage
      if (this.options.enableVectorSearch) {
        await this.initializeLanceDB();
      }

      // Initialize SQLite for relational data
      await this.initializeSQLite();

      // Create unified schema
      await this.createUnifiedSchema();

      this.initialized = true;
      this.initializing = false;

      console.log('✅ Unified Lance-Swarm Persistence initialized successfully');
      console.log(`📊 Features: Vector=${this.options.enableVectorSearch}, Graph=${this.options.enableGraphTraversal}, Neural=${this.options.enableNeuralPatterns}`);

    } catch (error) {
      this.initializing = false;
      console.error('❌ Failed to initialize Unified Lance-Swarm Persistence:', error);
      throw error;
    }
  }

  async initializeLanceDB() {
    try {
      // Dynamic import for LanceDB
      const lancedb = await import('@lancedb/lancedb');

      // Ensure directory exists
      await fs.mkdir(this.options.lanceDbPath, { recursive: true });

      // Connect to LanceDB
      this.lanceDb = await lancedb.connect(this.options.lanceDbPath);

      // Create or open table
      try {
        this.lanceTable = await this.lanceDb.openTable(this.options.collection);
        console.log(`🧠 LanceDB table '${this.options.collection}' opened`);
      } catch (error) {
        // Table doesn't exist, create it with vector schema
        const sampleData = [{
          id: 'sample',
          content: 'Sample content for vector embedding',
          embedding: new Array(this.options.vectorDimensions).fill(0.0),
          entity_type: 'sample',
          entity_id: 'sample',
          namespace: 'sample',
          metadata: JSON.stringify({}),
          created_at: new Date().toISOString(),
        }];

        this.lanceTable = await this.lanceDb.createTable(this.options.collection, sampleData);
        console.log(`🧠 LanceDB table '${this.options.collection}' created with vector schema`);
      }

    } catch (error) {
      console.warn(`⚠️ LanceDB initialization failed: ${error.message}`);
      this.options.enableVectorSearch = false;
      this.lanceDb = null;
      this.lanceTable = null;
    }
  }

  async initializeSQLite() {
    // Use the existing pooled SQLite persistence
    this.sqlitePersistence = new SwarmPersistencePooled(this.options.sqliteDbPath, {
      maxReaders: this.options.maxReaders,
      maxWorkers: this.options.maxWorkers,
    });

    await this.sqlitePersistence.initialize();
    console.log('🗃️ SQLite relational layer initialized');
  }

  async createUnifiedSchema() {
    // The SQLite schema is already created by SwarmPersistencePooled
    // Add any additional tables for unified features

    if (this.options.enableGraphTraversal) {
      await this.sqlitePersistence.pool.write(`
        CREATE TABLE IF NOT EXISTS entity_relationships (
          id TEXT PRIMARY KEY,
          from_entity_type TEXT NOT NULL,
          from_entity_id TEXT NOT NULL,
          to_entity_type TEXT NOT NULL,
          to_entity_id TEXT NOT NULL,
          relationship_type TEXT NOT NULL,
          strength REAL DEFAULT 1.0,
          metadata TEXT,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          UNIQUE(from_entity_type, from_entity_id, to_entity_type, to_entity_id, relationship_type)
        )
      `);

      await this.sqlitePersistence.pool.write(`
        CREATE INDEX IF NOT EXISTS idx_relationships_from ON entity_relationships(from_entity_type, from_entity_id)
      `);

      await this.sqlitePersistence.pool.write(`
        CREATE INDEX IF NOT EXISTS idx_relationships_to ON entity_relationships(to_entity_type, to_entity_id)
      `);
    }

    if (this.options.enableNeuralPatterns) {
      await this.sqlitePersistence.pool.write(`
        CREATE TABLE IF NOT EXISTS neural_patterns (
          id TEXT PRIMARY KEY,
          pattern_type TEXT NOT NULL,
          pattern_name TEXT NOT NULL,
          pattern_data TEXT NOT NULL,
          success_rate REAL DEFAULT 0.0,
          usage_count INTEGER DEFAULT 0,
          last_used DATETIME DEFAULT CURRENT_TIMESTAMP,
          created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
          UNIQUE(pattern_type, pattern_name)
        )
      `);
    }

    console.log('📊 Unified schema created successfully');
  }

  // HYBRID STORAGE OPERATIONS

  /**
   * Store entity with both vector embedding and relational data
   */
  async storeEntity(entityType, entityId, data, options = {}) {
    await this.ensureInitialized();

    const startTime = Date.now();

    try {
      // Store in SQLite for relational queries
      await this.storeRelationalData(entityType, entityId, data, options);

      // Store in LanceDB for vector similarity (if enabled and has content)
      if (this.options.enableVectorSearch && (data.content || data.description)) {
        await this.storeVectorData(entityType, entityId, data, options);
      }

      // Create relationships (if specified)
      if (options.relationships && this.options.enableGraphTraversal) {
        await this.createRelationships(entityType, entityId, options.relationships);
      }

      this.updateStats('totalOperations', startTime);

      return {
        success: true,
        entityType,
        entityId,
        hasVector: this.options.enableVectorSearch && (data.content || data.description),
        hasRelations: options.relationships?.length || 0,
      };

    } catch (error) {
      console.error(`Failed to store entity ${entityType}:${entityId}:`, error);
      throw error;
    }
  }

  async storeRelationalData(entityType, entityId, data, options) {
    // Store based on entity type
    switch (entityType) {
      case 'swarm':
        return this.sqlitePersistence.createSwarm({
          id: entityId,
          ...data,
          metadata: options.metadata || {},
        });

      case 'agent':
        return this.sqlitePersistence.createAgent({
          id: entityId,
          ...data,
        });

      case 'task':
        return this.sqlitePersistence.createTask({
          id: entityId,
          ...data,
        });

      case 'memory':
        return this.sqlitePersistence.storeMemory(
          data.agentId,
          data.key,
          data.value,
          data.ttlSecs,
        );

      default:
        throw new Error(`Unknown entity type: ${entityType}`);
    }
  }

  async storeVectorData(entityType, entityId, data, options) {
    if (!this.lanceTable) return;

    const content = data.content || data.description || JSON.stringify(data);

    // Generate embedding (placeholder - in production would use actual embedding model)
    const embedding = await this.generateEmbedding(content);

    const vectorRecord = {
      id: `${entityType}:${entityId}`,
      content: content,
      embedding: embedding,
      entity_type: entityType,
      entity_id: entityId,
      namespace: options.namespace || 'default',
      metadata: JSON.stringify({
        ...data,
        ...(options.metadata || {}),
      }),
      created_at: new Date().toISOString(),
    };

    // Insert into LanceDB
    await this.lanceTable.add([vectorRecord]);
  }

  async generateEmbedding(text) {
    // Placeholder embedding generation
    // In production, use actual embedding model like sentence-transformers
    const embedding = new Array(this.options.vectorDimensions);

    // Simple hash-based embedding for demonstration
    let hash = 0;
    for (let i = 0; i < text.length; i++) {
      hash = ((hash << 5) - hash + text.charCodeAt(i)) & 0xffffffff;
    }

    for (let i = 0; i < this.options.vectorDimensions; i++) {
      embedding[i] = Math.sin(hash * (i + 1)) * 0.5;
    }

    return embedding;
  }

  /**
   * ULTRA HYBRID QUERY - Combines vector similarity + relational + graph
   */
  async hybridQuery(query, options = {}) {
    await this.ensureInitialized();

    const startTime = Date.now();
    const results = {
      vector_results: [],
      relational_results: [],
      graph_results: [],
      combined_score: [],
    };

    try {
      // 1. Vector similarity search (if enabled)
      if (this.options.enableVectorSearch && query.semantic) {
        results.vector_results = await this.vectorSimilaritySearch(
          query.semantic,
          options.vectorLimit || 10,
        );
      }

      // 2. Relational queries (always available)
      if (query.relational) {
        results.relational_results = await this.relationalSearch(
          query.relational,
          options.relationalLimit || 20,
        );
      }

      // 3. Graph traversal (if enabled)
      if (this.options.enableGraphTraversal && query.graph) {
        results.graph_results = await this.graphTraversal(
          query.graph.startEntity,
          query.graph.relationshipTypes,
          query.graph.maxDepth || 2,
        );
      }

      // 4. Combine and rank results
      results.combined_score = this.combineResults(results, options.rankingWeights);

      this.updateStats('hybridQueries', startTime);

      return results;

    } catch (error) {
      console.error('Hybrid query failed:', error);
      throw error;
    }
  }

  async vectorSimilaritySearch(queryText, limit = 10) {
    if (!this.lanceTable) return [];

    const queryEmbedding = await this.generateEmbedding(queryText);

    // Use LanceDB's vector search
    const results = await this.lanceTable
      .search(queryEmbedding)
      .limit(limit)
      .toArray();

    this.updateStats('vectorQueries');

    return results.map(result => ({
      id: result.id,
      entity_type: result.entity_type,
      entity_id: result.entity_id,
      content: result.content,
      similarity: result._distance ? (1 - result._distance) : 1.0,
      metadata: JSON.parse(result.metadata || '{}'),
    }));
  }

  async relationalSearch(query, limit = 20) {
    // Use SQLite for structured queries
    const { entityType, filters, orderBy } = query;

    let sql = '';
    const params = [];

    switch (entityType) {
      case 'swarms':
        sql = 'SELECT * FROM swarms WHERE 1=1';
        break;
      case 'agents':
        sql = 'SELECT * FROM agents WHERE 1=1';
        break;
      case 'tasks':
        sql = 'SELECT * FROM tasks WHERE 1=1';
        break;
      default:
        throw new Error(`Unknown entity type for relational search: ${entityType}`);
    }

    // Add filters
    if (filters) {
      Object.entries(filters).forEach(([key, value]) => {
        sql += ` AND ${key} = ?`;
        params.push(value);
      });
    }

    // Add ordering
    if (orderBy) {
      sql += ` ORDER BY ${orderBy}`;
    }

    sql += ` LIMIT ${limit}`;

    const results = await this.sqlitePersistence.pool.read(sql, params);

    this.updateStats('relationQueries');

    return results;
  }

  async graphTraversal(startEntity, relationshipTypes = [], maxDepth = 2) {
    if (!this.options.enableGraphTraversal) return [];

    const { entityType, entityId } = startEntity;
    const relatedEntities = [];

    let sql = `
      WITH RECURSIVE entity_graph(entity_type, entity_id, relationship_type, depth) AS (
        SELECT ?, ?, 'self', 0
        UNION ALL
        SELECT 
          r.to_entity_type,
          r.to_entity_id,
          r.relationship_type,
          eg.depth + 1
        FROM entity_relationships r
        JOIN entity_graph eg ON (
          r.from_entity_type = eg.entity_type AND 
          r.from_entity_id = eg.entity_id
        )
        WHERE eg.depth < ?
    `;

    const params = [entityType, entityId, maxDepth];

    if (relationshipTypes.length > 0) {
      sql += ` AND r.relationship_type IN (${relationshipTypes.map(() => '?').join(',')})`;
      params.push(...relationshipTypes);
    }

    sql += `
      )
      SELECT DISTINCT entity_type, entity_id, relationship_type, depth
      FROM entity_graph
      WHERE depth > 0
      ORDER BY depth, entity_type, entity_id
    `;

    const results = await this.sqlitePersistence.pool.read(sql, params);

    return results;
  }

  combineResults(results, weights = {}) {
    const defaultWeights = {
      vector: 0.4,
      relational: 0.4,
      graph: 0.2,
    };

    const finalWeights = { ...defaultWeights, ...weights };
    const combined = new Map();

    // Add vector results
    results.vector_results.forEach(item => {
      const key = `${item.entity_type}:${item.entity_id}`;
      combined.set(key, {
        ...item,
        combined_score: item.similarity * finalWeights.vector,
        sources: ['vector'],
      });
    });

    // Add relational results
    results.relational_results.forEach(item => {
      const key = `${item.entity_type || 'unknown'}:${item.id}`;
      const existing = combined.get(key);

      if (existing) {
        existing.combined_score += finalWeights.relational;
        existing.sources.push('relational');
      } else {
        combined.set(key, {
          entity_type: item.entity_type || 'unknown',
          entity_id: item.id,
          data: item,
          combined_score: finalWeights.relational,
          sources: ['relational'],
        });
      }
    });

    // Add graph results
    results.graph_results.forEach(item => {
      const key = `${item.entity_type}:${item.entity_id}`;
      const existing = combined.get(key);

      const graphScore = finalWeights.graph / (item.depth || 1); // Closer entities get higher scores

      if (existing) {
        existing.combined_score += graphScore;
        existing.sources.push('graph');
      } else {
        combined.set(key, {
          entity_type: item.entity_type,
          entity_id: item.entity_id,
          relationship_type: item.relationship_type,
          depth: item.depth,
          combined_score: graphScore,
          sources: ['graph'],
        });
      }
    });

    // Sort by combined score
    return Array.from(combined.values())
      .sort((a, b) => b.combined_score - a.combined_score);
  }

  // RELATIONSHIP MANAGEMENT

  async createRelationships(fromEntityType, fromEntityId, relationships) {
    if (!this.options.enableGraphTraversal) return;

    for (const rel of relationships) {
      const relationshipId = `rel_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      await this.sqlitePersistence.pool.write(`
        INSERT OR REPLACE INTO entity_relationships 
        (id, from_entity_type, from_entity_id, to_entity_type, to_entity_id, relationship_type, strength, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
      `, [
        relationshipId,
        fromEntityType,
        fromEntityId,
        rel.toEntityType,
        rel.toEntityId,
        rel.relationshipType,
        rel.strength || 1.0,
        JSON.stringify(rel.metadata || {}),
      ]);
    }
  }

  // NEURAL PATTERN MANAGEMENT

  async storeNeuralPattern(patternType, patternName, patternData, successRate = 0.0) {
    if (!this.options.enableNeuralPatterns) return;

    const patternId = `pattern_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    return this.sqlitePersistence.pool.write(`
      INSERT OR REPLACE INTO neural_patterns 
      (id, pattern_type, pattern_name, pattern_data, success_rate, usage_count)
      VALUES (?, ?, ?, ?, ?, 1)
    `, [
      patternId,
      patternType,
      patternName,
      JSON.stringify(patternData),
      successRate,
    ]);
  }

  async getNeuralPatterns(patternType, limit = 10) {
    if (!this.options.enableNeuralPatterns) return [];

    const patterns = await this.sqlitePersistence.pool.read(`
      SELECT * FROM neural_patterns 
      WHERE pattern_type = ? 
      ORDER BY success_rate DESC, usage_count DESC 
      LIMIT ?
    `, [patternType, limit]);

    return patterns.map(p => ({
      ...p,
      pattern_data: JSON.parse(p.pattern_data),
    }));
  }

  async updateNeuralPatternSuccess(patternType, patternName, wasSuccessful) {
    if (!this.options.enableNeuralPatterns) return;

    const pattern = await this.sqlitePersistence.pool.read(`
      SELECT success_rate, usage_count FROM neural_patterns 
      WHERE pattern_type = ? AND pattern_name = ?
    `, [patternType, patternName]);

    if (pattern.length === 0) return;

    const current = pattern[0];
    const newUsageCount = current.usage_count + 1;
    const newSuccessRate = wasSuccessful
      ? (current.success_rate * current.usage_count + 1) / newUsageCount
      : (current.success_rate * current.usage_count) / newUsageCount;

    await this.sqlitePersistence.pool.write(`
      UPDATE neural_patterns 
      SET success_rate = ?, usage_count = ?, last_used = CURRENT_TIMESTAMP
      WHERE pattern_type = ? AND pattern_name = ?
    `, [newSuccessRate, newUsageCount, patternType, patternName]);
  }

  // UTILITY METHODS

  async ensureInitialized() {
    if (!this.initialized) {
      await this.initialize();
    }
  }

  updateStats(operation, startTime = null) {
    this.stats[operation] = (this.stats[operation] || 0) + 1;
    this.stats.totalOperations++;

    if (startTime) {
      const duration = Date.now() - startTime;
      this.stats.averageResponseTime =
        (this.stats.averageResponseTime + duration) / 2;
    }
  }

  getStats() {
    return {
      ...this.stats,
      features: {
        vectorSearch: this.options.enableVectorSearch,
        graphTraversal: this.options.enableGraphTraversal,
        neuralPatterns: this.options.enableNeuralPatterns,
      },
      backends: {
        lancedb: !!this.lanceDb,
        sqlite: !!this.sqlitePersistence,
      },
    };
  }

  async cleanup() {
    if (this.sqlitePersistence) {
      await this.sqlitePersistence.cleanup();
    }

    if (this.lanceDb) {
      // LanceDB cleanup (if needed)
    }
  }

  async close() {
    if (this.sqlitePersistence) {
      await this.sqlitePersistence.close();
    }

    // Close LanceDB connections
    this.lanceDb = null;
    this.lanceTable = null;
  }
}

export default UnifiedLancePersistence;
