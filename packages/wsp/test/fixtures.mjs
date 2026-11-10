export const xorModel = () => ({ schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 2, outputs: 2, weights: [1, 1, -1, -1], bias: [-0.5, 1.5], activation: 'Threshold', steepness: 1 },
  { inputs: 2, outputs: 1, weights: [0, 0], bias: [-1.5], activation: 'Threshold', steepness: 1 }
] });
export const xorData = () => ({ inputs: [0, 0, 0, 1, 1, 0, 1, 1], targets: [0, 1, 1, 0], rows: 4 });
export const linearModel = (weight = 0, bias) => ({ schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 1, outputs: 1, weights: [weight], ...(bias === undefined ? {} : { bias }), activation: 'Linear', steepness: 1 }
] });
export const edit = (value, layer = 0, parameter = 'weights', index = 0) => ({ layer, parameter, index, value });
export const patch = (baseHash, edits = [edit(2)]) => ({ schema: 'ruv-fann.weight-patch/1', baseHash, edits });
export const linearData = () => ({ inputs: [1, 2], targets: [2, 4], rows: 2 });
export const strictPolicy = () => ({ maxMSE: 0, maxRegressionMSE: 0 });
