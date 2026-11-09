export const dense = overrides => JSON.stringify({ schema: 'ruv-fann.dense-layers/1', layers: [
  { inputs: 2, outputs: 2, weights: [2, 1, 1, 3], bias: [0.5, -0.5], activation: 'Linear', steepness: 1, ...overrides }
] });

export function network({ sparse = false, activation = 'Linear' } = {}) {
  const neuron = (index, weights) => ({ sum: 0, value: 0, activation_steepness: 1,
    activation_function: activation, is_bias: false,
    connections: weights.map(([from_neuron, weight]) => ({ from_neuron, to_neuron: index, weight })) });
  const input = [neuron(0, []), neuron(1, []), { ...neuron(2, []), is_bias: true, value: 1, sum: 1 }];
  const first = sparse ? [[1, 1], [0, 2], [0, -0.5], [0, 0.5], [2, 0.5]] : [[0, 2], [1, 1], [2, 0.5]];
  return JSON.stringify({ layers: [{ neurons: input }, { neurons: [neuron(0, first), neuron(1, [[0, 1], [1, 3], [2, -0.5]])] }], connection_rate: 1 });
}
export const expected = input => [input[0] * 2 + input[1] + 0.5, input[0] + input[1] * 3 - 0.5];
