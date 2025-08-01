-- Database Cleanup Script for ruv-swarm.db
-- Purpose: Remove test swarms and optimize database performance

-- Begin transaction for safety
BEGIN TRANSACTION;

-- Step 1: Create temporary table for swarms to keep
CREATE TEMP TABLE swarms_to_keep AS
SELECT id FROM swarms 
WHERE (
    -- Keep recent non-test swarms (last 7 days)
    (created_at > datetime('now', '-7 days') 
     AND name NOT LIKE '%test%' 
     AND name NOT LIKE '%Test%' 
     AND name NOT LIKE 'Swarm_%')
    -- Keep any swarm with meaningful names from today
    OR (DATE(created_at) = DATE('now') 
        AND name NOT LIKE '%test%' 
        AND name NOT LIKE '%Test%' 
        AND name NOT LIKE 'Swarm_%'
        AND name NOT LIKE 'default-swarm')
    -- Keep the 10 most recent swarms regardless
    OR id IN (
        SELECT id FROM swarms 
        ORDER BY created_at DESC 
        LIMIT 10
    )
);

-- Step 2: Delete cascade - remove all related data for swarms not kept
DELETE FROM events WHERE swarm_id NOT IN (SELECT id FROM swarms_to_keep);
DELETE FROM metrics WHERE entity_id NOT IN (SELECT id FROM swarms_to_keep);
DELETE FROM task_results WHERE task_id IN (SELECT id FROM tasks WHERE swarm_id NOT IN (SELECT id FROM swarms_to_keep));
DELETE FROM tasks WHERE swarm_id NOT IN (SELECT id FROM swarms_to_keep);
DELETE FROM agent_memory WHERE agent_id IN (SELECT id FROM agents WHERE swarm_id NOT IN (SELECT id FROM swarms_to_keep));
DELETE FROM neural_networks WHERE id IN (SELECT neural_model_id FROM agents WHERE swarm_id NOT IN (SELECT id FROM swarms_to_keep));
DELETE FROM agents WHERE swarm_id NOT IN (SELECT id FROM swarms_to_keep);
DELETE FROM swarms WHERE id NOT IN (SELECT id FROM swarms_to_keep);

-- Step 3: Clean up orphaned records
DELETE FROM events WHERE swarm_id NOT IN (SELECT id FROM swarms);
DELETE FROM agents WHERE swarm_id NOT IN (SELECT id FROM swarms);
DELETE FROM tasks WHERE swarm_id NOT IN (SELECT id FROM swarms);
DELETE FROM neural_networks WHERE id NOT IN (SELECT neural_model_id FROM agents);

-- Commit the transaction
COMMIT;

-- Step 4: Optimize database
VACUUM;
ANALYZE;