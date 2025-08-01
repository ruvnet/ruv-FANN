-- Aggressive Database Cleanup Script
-- Remove all test swarms and keep only meaningful recent swarms

BEGIN TRANSACTION;

-- Create temp table with swarms to keep
CREATE TEMP TABLE keeper_swarms AS
SELECT id FROM swarms 
WHERE 
    -- Keep only non-test, non-default swarms from last 24 hours
    created_at > datetime('now', '-1 day')
    AND name NOT LIKE '%test%' 
    AND name NOT LIKE '%Test%' 
    AND name NOT LIKE 'Swarm_%'
    AND name NOT LIKE 'default-swarm'
    AND name != 'Test Swarm'
    
UNION

-- Keep the 5 most recent meaningful swarms
SELECT id FROM swarms 
WHERE name NOT LIKE '%test%' 
    AND name NOT LIKE '%Test%' 
    AND name NOT LIKE 'Swarm_%'
    AND name NOT LIKE 'default-swarm'
ORDER BY created_at DESC 
LIMIT 5;

-- Show what we're keeping
SELECT COUNT(*) as keeping_swarms FROM keeper_swarms;
SELECT id, name, created_at FROM swarms WHERE id IN (SELECT id FROM keeper_swarms);

-- Delete all related data for swarms we're not keeping
DELETE FROM events WHERE swarm_id NOT IN (SELECT id FROM keeper_swarms);
DELETE FROM metrics WHERE entity_id NOT IN (SELECT id FROM keeper_swarms) AND entity_type = 'swarm';
DELETE FROM task_results WHERE task_id IN (SELECT id FROM tasks WHERE swarm_id NOT IN (SELECT id FROM keeper_swarms));
DELETE FROM tasks WHERE swarm_id NOT IN (SELECT id FROM keeper_swarms);
DELETE FROM agent_memory WHERE agent_id IN (SELECT id FROM agents WHERE swarm_id NOT IN (SELECT id FROM keeper_swarms));
DELETE FROM neural_networks WHERE id IN (SELECT neural_model_id FROM agents WHERE swarm_id NOT IN (SELECT id FROM keeper_swarms));
DELETE FROM agents WHERE swarm_id NOT IN (SELECT id FROM keeper_swarms);
DELETE FROM swarms WHERE id NOT IN (SELECT id FROM keeper_swarms);

-- Show results
SELECT COUNT(*) as remaining_swarms FROM swarms;
SELECT COUNT(*) as remaining_agents FROM agents;
SELECT COUNT(*) as remaining_events FROM events;

COMMIT;

-- Optimize
VACUUM;
ANALYZE;