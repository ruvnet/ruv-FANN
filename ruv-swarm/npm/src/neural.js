/** Real local WASM inference and statistical forecasting commands. */
import { infer, forecast, denseBenchmark } from './neural-runtime.js';
import { unsupported } from './unsupported.js';

const PATTERN_MEMORY_CONFIG = {
  convergent: { baseMemory: 260, poolSharing: 0.8, lazyLoading: true },
  divergent: { baseMemory: 275, poolSharing: 0.6, lazyLoading: true },
  lateral: { baseMemory: 270, poolSharing: 0.7, lazyLoading: true },
  systems: { baseMemory: 285, poolSharing: 0.5, lazyLoading: false },
  critical: { baseMemory: 265, poolSharing: 0.7, lazyLoading: true },
  abstract: { baseMemory: 280, poolSharing: 0.6, lazyLoading: false },
  attention: { baseMemory: 290, poolSharing: 0.4, lazyLoading: false },
  lstm: { baseMemory: 275, poolSharing: 0.5, lazyLoading: false },
  transformer: { baseMemory: 295, poolSharing: 0.3, lazyLoading: false },
  cnn: { baseMemory: 285, poolSharing: 0.5, lazyLoading: false },
  gru: { baseMemory: 270, poolSharing: 0.6, lazyLoading: true },
  autoencoder: { baseMemory: 265, poolSharing: 0.7, lazyLoading: true },
};

class NeuralCLI {
  constructor() {
    this.ruvSwarm = null;
  }
  async initialize() {
    if (!this.ruvSwarm) {
      const { RuvSwarm } = await import('./index.js');
      this.ruvSwarm = await RuvSwarm.initialize();
    }
    return this.ruvSwarm;
  }
  async status() {
    const runtime = await this.initialize();
    const status = { wasm_version: runtime._bindings.get_version(), dense_inference: true,
      training: false, statistical_forecasts: ['linear', 'mean', 'naive'],
      features: runtime.constructor.getRuntimeFeatures() };
    console.log(JSON.stringify(status, null, 2));
    return status;
  }
  async infer(args) {
    const runtime = await this.initialize();
    const result = infer(runtime, { layers: this.parseNumbers(args, '--layers'),
      input: this.parseNumbers(args, '--input'), weights: this.parseNumbers(args, '--weights'),
      activation: this.getArg(args, '--activation') || 'Linear' });
    console.log(JSON.stringify(result));
    return result;
  }
  async forecast(args) {
    const runtime = await this.initialize();
    const result = forecast(runtime, { input: this.parseNumbers(args, '--input'), model: this.getArg(args, '--model') || 'linear' });
    console.log(JSON.stringify(result));
    return result;
  }
  async benchmark(args) {
    const runtime = await this.initialize();
    const result = denseBenchmark(runtime, { iterations: Number(this.getArg(args, '--iterations') || 1000),
      type: this.getArg(args, '--test') || 'dense-inference' });
    console.log(JSON.stringify(result, null, 2));
    return result;
  }
  async train() {
    unsupported('Neural training');
  }
  async patterns() {
    unsupported('Learned cognitive-pattern analysis');
  }
  async export() {
    unsupported('Exporting a trained model from CLI state');
  }
  async getPatternMemoryUsage() {
    unsupported('Per-pattern memory measurement');
  }
  parseNumbers(args, flag) {
    const value = this.getArg(args, flag);
    if (!value || value.length > 65536 || value.split(',').some(item => item.trim() === '')) {
      throw new TypeError(`${flag} requires comma-separated finite numbers`);
    }
    const result = value.split(',').map(Number);
    if (!result.every(Number.isFinite)) {
      throw new TypeError(`${flag} requires finite numbers`);
    }
    return result;
  }
  getArg(args, flag) {
    const index = args.indexOf(flag);
    return index >= 0 ? args[index + 1] : null;
  }
  async cleanup() {
    this.ruvSwarm = null;
  }
}
const neuralCLI = new NeuralCLI();
export { neuralCLI, NeuralCLI, PATTERN_MEMORY_CONFIG };
