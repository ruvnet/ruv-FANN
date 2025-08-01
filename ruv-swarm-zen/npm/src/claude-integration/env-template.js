/**
 * Environment variable template for ruv-swarm projects
 */

const envTemplate = `# ruv-swarm Configuration
NODE_ENV=development

# Git Integration
RUV_SWARM_AUTO_COMMIT=true
RUV_SWARM_AUTO_PUSH=false
RUV_SWARM_COMMIT_PREFIX=feat
RUV_SWARM_GIT_AUTHOR=ruv-swarm

# Agent Reports
RUV_SWARM_GENERATE_REPORTS=true
RUV_SWARM_REPORT_DIR=.ruv-swarm/agent-reports

# Memory & Learning
RUV_SWARM_MEMORY_PERSIST=true
RUV_SWARM_NEURAL_LEARNING=true

# Performance Tracking
RUV_SWARM_PERFORMANCE_TRACKING=true
RUV_SWARM_TELEMETRY_ENABLED=true

# Hook Configuration
RUV_SWARM_HOOKS_ENABLED=true
RUV_SWARM_HOOK_DEBUG=false

# Coordination
RUV_SWARM_COORDINATION_MODE=adaptive
RUV_SWARM_AUTO_INIT=true

# Remote Execution
RUV_SWARM_REMOTE_EXECUTION=true
RUV_SWARM_REMOTE_READY=true
`;

module.exports = { envTemplate };
