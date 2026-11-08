import { EventEmitter } from 'node:events';
export declare class DAAService extends EventEmitter {
  readonly initialized: false;
  getCapabilities(): { autonomousLearning: false; peerCoordination: false; neuralIntegration: false; cognitivePatterns: 0; supported: false };
  getStatus(): { initialized: false; supported: false };
  getPerformanceMetrics(): { available: false; reason: string };
  initialize(...args: unknown[]): Promise<never>;
  createAgent(...args: unknown[]): Promise<never>;
  createAgentInternal(...args: unknown[]): Promise<never>;
  adaptAgent(...args: unknown[]): Promise<never>;
  executeWorkflow(...args: unknown[]): Promise<never>;
  shareKnowledge(...args: unknown[]): Promise<never>;
  getAgentLearningStatus(...args: unknown[]): Promise<never>;
  getSystemLearningStatus(...args: unknown[]): Promise<never>;
  analyzeCognitivePatterns(...args: unknown[]): Promise<never>;
  setCognitivePattern(...args: unknown[]): Promise<never>;
  performMetaLearning(...args: unknown[]): Promise<never>;
  destroyAgent(...args: unknown[]): Promise<never>;
  makeDecision(...args: unknown[]): Promise<never>;
  createWorkflow(...args: unknown[]): Promise<never>;
  executeWorkflowStep(...args: unknown[]): Promise<never>;
  synchronizeStates(...args: unknown[]): Promise<never>;
  optimizeResources(...args: unknown[]): Promise<never>;
  batchCreateAgents(...args: unknown[]): Promise<never>;
  batchMakeDecisions(...args: unknown[]): Promise<never>;
  cleanup(...args: unknown[]): Promise<void>;
}
export declare const daaService: DAAService;
export default DAAService;
