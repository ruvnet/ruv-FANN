# Schema Issues to Fix

## Current Problems with ruv-swarm-persistence SQLite Schema

### 1. Inconsistent Naming Conventions
- Some tables use `created_at`/`updated_at`, others use `timestamp`
- Should standardize on one approach

### 2. Redundant Data Storage
- Every table has a `data` column storing full JSON representation
- This duplicates all the other columns, wasting space
- Example: agents table stores agent info in columns AND in `data` JSON

### 3. Over-Engineering for Local Use Case
- `messages` table has read/unread status tracking
- `agent_groups` and `agent_group_members` tables for complex hierarchies
- These features are unnecessary for local swarm coordination

### 4. Missing Core Functionality
- No simple key-value storage for session memory (memory_store/memory_get)
- The `messages` table is for agent-to-agent communication, not session storage
- MCP tools expect different storage patterns

### 5. Type Mismatches
- Using INTEGER for timestamps instead of proper datetime
- JSON stored as TEXT without validation
- Priority stored as INTEGER in tasks but TEXT in messages

### 6. Poor Normalization
- Storing JSON arrays/objects in multiple columns
- Could be properly normalized into related tables
- Makes querying and indexing inefficient

### 7. Unnecessary Complexity
- Schema designed for distributed, persistent swarm
- But users run this locally for coordination
- Too many features for the actual use case

## Additional Issues Found

### 8. Multiple Database Files
- Found multiple .db files scattered across the project:
  - `/Users/lion/ruv-FANN/ruv-swarm-mcp.db` (MCP persistence)
  - `/Users/lion/ruv-FANN/.hive-mind/hive.db`
  - `/Users/lion/ruv-FANN/.hive-mind/memory.db`
  - `/Users/lion/ruv-FANN/data/ruv-swarm.db`
  - `/Users/lion/ruv-FANN/ruv-swarm/npm/data/ruv-swarm.db`
- No clear data management strategy
- Memory operations not connected to persistence layer

### 9. MCP Integration Broken
- `memory_store` operations don't persist to SQLite
- `memory_get` returns nothing even after storing data
- Session memory isn't actually using the persistence layer

## Recommended Simplifications

1. Add simple `session_memory` table for MCP memory operations:
   ```sql
   CREATE TABLE session_memory (
       key TEXT PRIMARY KEY,
       value TEXT NOT NULL, -- JSON
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

2. Remove redundant `data` columns
3. Standardize timestamp columns
4. Remove unnecessary features (groups, read status, etc.)
5. Focus on core swarm coordination needs