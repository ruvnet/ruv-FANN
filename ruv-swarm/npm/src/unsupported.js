export class UnsupportedFeatureError extends Error {
  constructor(feature) {
    super(`${feature} is not implemented by this distribution. No execution or measurement was performed.`);
    this.name = 'UnsupportedFeatureError';
    this.code = 'UNSUPPORTED_FEATURE';
  }
}

export function unsupported(feature) {
  throw new UnsupportedFeatureError(feature);
}
