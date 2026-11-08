/** SQL identifiers must come from the schema; placeholders protect only values. */
const taskColumns = Object.freeze({
  description: 'description', priority: 'priority', status: 'status',
  assignedAgents: 'assigned_agents', assigned_agents: 'assigned_agents',
  result: 'result', error: 'error', completedAt: 'completed_at', completed_at: 'completed_at',
  executionTimeMs: 'execution_time_ms', execution_time_ms: 'execution_time_ms',
});
const neuralColumns = Object.freeze({
  architecture: 'architecture', weights: 'weights',
  trainingData: 'training_data', training_data: 'training_data',
  performanceMetrics: 'performance_metrics', performance_metrics: 'performance_metrics',
});

export function buildUpdateFields(updates, kind) {
  if (!updates || typeof updates !== 'object' || Array.isArray(updates)) {
    throw new TypeError('Updates must be an object');
  }
  const columns = kind === 'task' ? taskColumns : neuralColumns;
  const fields = [];
  const values = [];
  const seen = new Set();
  for (const [key, value] of Object.entries(updates)) {
    if (!Object.hasOwn(columns, key)) {
      throw new TypeError(`Unknown ${kind} update field: ${key}`);
    }
    const column = columns[key];
    if (seen.has(column)) {
      throw new TypeError(`Duplicate update field: ${column}`);
    }
    seen.add(column);
    fields.push(`${column} = ?`);
    values.push(kind === 'neural' || column === 'assigned_agents' || column === 'result'
      ? JSON.stringify(value) : value);
  }
  if (fields.length === 0) {
    throw new TypeError('At least one update field is required');
  }
  return { fields, values };
}
