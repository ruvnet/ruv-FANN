// Deterministic proposal controls. This module never receives intact weights or test cases.
function medianMagnitude(values) {
  if (values.length === 0) return 0;
  const sorted = values.map(Math.abs).sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : sorted[middle - 1] / 2 + sorted[middle] / 2;
}

function parameterGroups(model) {
  return model.layers.flatMap((layer, index) => ['weights', 'bias'].map(parameter => ({
    layer: index,
    parameter,
    values: layer[parameter] ?? [],
  })).filter(group => group.values.length > 0));
}

export function medianClip(model, threshold = 15) {
  if (!Number.isFinite(threshold) || threshold <= 0) throw new RangeError('invalid clipping threshold');
  const edits = [];
  for (const { layer, parameter, values } of parameterGroups(model)) {
    const bound = threshold * medianMagnitude(values);
    if (bound === 0) continue;
    for (const [index, value] of values.entries()) {
      if (Math.abs(value) > bound) edits.push({ layer, parameter, index, value: Math.sign(value) * bound });
    }
  }
  return edits;
}

function assertSearchCap(maxEvaluations) {
  if (!Number.isInteger(maxEvaluations) || maxEvaluations < 0 || maxEvaluations > 20) {
    throw new RangeError('search allows between 0 and 20 evaluations');
  }
}

export async function coordinateSearch(model, evaluate, maxEvaluations = 20) {
  assertSearchCap(maxEvaluations);
  const coordinates = parameterGroups(model).flatMap(({ layer, parameter, values }) => {
    const scale = medianMagnitude(values.filter(value => value !== 0));
    if (!Number.isFinite(scale) || scale === 0) return [];
    return Array.from(values, (original, index) => ({
      layer, parameter, index, original, scale,
      priority: original === 0 ? 0 : Math.abs(original) / scale,
    }));
  }).sort((left, right) => right.priority - left.priority);

  // Interleave magnitude outliers with exact zeros. A contender is always a
  // complete patch against the damaged baseline, even after multiple steps.
  const nonzero = coordinates.filter(coordinate => coordinate.original !== 0);
  const zero = coordinates.filter(coordinate => coordinate.original === 0);
  let edits = [];
  let bestScore = -Infinity;
  let evaluations = 0;
  for (let ordinal = 0; ordinal < Math.max(nonzero.length, zero.length); ordinal++) {
    for (const coordinate of [nonzero[ordinal], zero[ordinal]]) {
      if (!coordinate) continue;
      const { layer, parameter, index, original, scale } = coordinate;
      const direction = Math.sign(original);
      const replacements = original === 0 ? [scale, -scale] : [0, -original, direction * scale, original / 2];
      let winner = edits;
      for (const value of new Set(replacements)) {
        if (value === original) continue;
        if (evaluations === maxEvaluations) return winner;
        const candidate = [...edits, { layer, parameter, index, value }];
        evaluations++;
        const score = await evaluate(candidate);
        if (Number.isFinite(score) && score > bestScore) {
          bestScore = score;
          winner = candidate;
        }
      }
      edits = winner;
    }
  }
  return edits;
}

function editRow(row, replacement) {
  const edits = [];
  for (const [offset, original] of row.values.entries()) {
    const value = replacement(original, offset);
    if (value !== original) edits.push({ layer: row.layer, parameter: 'weights', index: row.start + offset, value });
  }
  return edits;
}

export async function groupSearch(model, evaluate, maxEvaluations = 20) {
  assertSearchCap(maxEvaluations);
  const rows = model.layers.flatMap(({ weights, inputs, outputs }, layer) => {
    const typical = medianMagnitude(weights.filter(value => value !== 0));
    return Array.from({ length: outputs }, (_, output) => {
      const start = output * inputs;
      const values = weights.slice(start, start + inputs);
      const largest = values.reduce((maximum, value) => Math.max(maximum, Math.abs(value)), 0);
      return { layer, start, values, largest, ratio: typical === 0 ? 0 : largest / typical, typical };
    });
  }).sort((left, right) => right.ratio - left.ratio);

  const anomalous = rows.filter(row => row.ratio > 8);
  const empty = rows.filter(row => row.largest === 0);
  const proposals = [
    ...[15, 6, 3].map(threshold => medianClip(model, threshold)),
    ...anomalous.flatMap(row => [editRow(row, () => 0), editRow(row, value => value * (row.typical / row.largest))]),
    ...rows.map(row => editRow(row, value => -value)),
    ...empty.flatMap(row => {
      const donor = rows.findLast(other => other.layer === row.layer && other.largest > 0);
      return donor ? [editRow(row, (_, offset) => donor.values[offset]), editRow(row, (_, offset) => -donor.values[offset])] : [];
    }),
    ...rows.flatMap(row => [editRow(row, () => 0), editRow(row, value => 0.5 * value)]),
  ];

  // Search one group at a time. Small development splits can reward spurious
  // combinations, so the control does not compose the winning row edits.
  let bestScore = -Infinity;
  let bestEdits = [];
  const seen = new Set();
  for (const edits of proposals) {
    if (seen.size === maxEvaluations) break;
    if (edits.length === 0) continue;
    const key = JSON.stringify(edits);
    if (seen.has(key)) continue;
    seen.add(key);
    const score = await evaluate(edits);
    if (Number.isFinite(score) && score > bestScore) {
      bestScore = score;
      bestEdits = edits;
    }
  }
  return bestEdits;
}

export function refitReadout(model, activations, targets, ridge = 0.1) {
  const layer = model.layers.length - 1;
  const readout = model.layers[layer];
  if (readout.activation !== 'Linear' || readout.steepness !== 1) {
    throw new TypeError('readout refit requires a linear layer with unit steepness');
  }
  if (!Number.isFinite(ridge) || ridge <= 0) throw new RangeError('readout ridge must be positive');
  if (activations.length === 0 || activations.length !== targets.length) throw new RangeError('invalid refit rows');

  const { inputs, outputs } = readout;
  const width = inputs + 1;
  // Solve (XᵀX + λI)B = XᵀY. The last column of X is the
  // intercept and has no ridge penalty. The hidden layers stay frozen.
  const matrix = Array.from({ length: width }, () => new Float64Array(width + outputs));
  for (let row = 0; row < activations.length; row++) {
    const features = [...activations[row], 1];
    const labels = targets[row];
    if (features.length !== width || labels.length !== outputs ||
      !features.every(Number.isFinite) || !Array.from(labels).every(Number.isFinite)) {
      throw new RangeError('invalid refit feature or target shape');
    }
    for (let left = 0; left < width; left++) {
      for (let right = 0; right < width; right++) matrix[left][right] += features[left] * features[right];
      for (let output = 0; output < outputs; output++) matrix[left][width + output] += features[left] * labels[output];
    }
  }
  for (let index = 0; index < inputs; index++) matrix[index][index] += ridge;

  for (let column = 0; column < width; column++) {
    let pivot = column;
    for (let row = column + 1; row < width; row++) {
      if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivot][column])) pivot = row;
    }
    [matrix[column], matrix[pivot]] = [matrix[pivot], matrix[column]];
    const value = matrix[column][column];
    if (!Number.isFinite(value) || value === 0) throw new RangeError('readout refit is singular');
    for (let index = column; index < width + outputs; index++) matrix[column][index] /= value;
    for (let row = 0; row < width; row++) {
      if (row === column) continue;
      const factor = matrix[row][column];
      for (let index = column; index < width + outputs; index++) matrix[row][index] -= factor * matrix[column][index];
    }
  }

  const edits = [];
  for (let output = 0; output < outputs; output++) {
    for (let input = 0; input < inputs; input++) {
      const index = output * inputs + input;
      const value = matrix[input][width + output];
      if (value !== readout.weights[index]) edits.push({ layer, parameter: 'weights', index, value });
    }
    const value = matrix[inputs][width + output];
    if (value !== (readout.bias?.[output] ?? 0)) edits.push({ layer, parameter: 'bias', index: output, value });
  }
  if (!edits.every(edit => Number.isFinite(edit.value))) throw new RangeError('readout refit produced nonfinite weights');
  return edits;
}
