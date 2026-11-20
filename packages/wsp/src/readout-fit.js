// Solve a single readout correction on columns normalized to [-1, 1].
// Positive ridge keeps the Gram matrix positive definite for repeated neurons.
// The caller bounds the dimensions, work and allocations before inference.
export function fitReadout(features, rows, width, deltas) {
  const size = width + 1;
  const scales = new Float64Array(size);
  for (let i = 0; i < features.length; i++) {
    const column = i % width;
    scales[column] = Math.max(scales[column], Math.abs(features[i]));
  }
  // Inactive neurons stay at zero correction. The final column is the bias.
  for (let i = 0; i < size; i++) if (scales[i] === 0) scales[i] = 1;

  const gram = new Float64Array(size * size);
  const rhs = new Float64Array(size);
  const row = new Float64Array(size);
  row[width] = 1;
  for (let n = 0; n < rows; n++) {
    for (let i = 0; i < width; i++) row[i] = features[n * width + i] / scales[i];
    for (let i = 0; i < size; i++) {
      rhs[i] += row[i] * (deltas[n] / rows);
      for (let j = 0; j <= i; j++) gram[i * size + j] += row[i] * row[j];
    }
  }

  // Cholesky of X'X / rows + 1e-4 I, followed by two triangular solves.
  for (let i = 0; i < size; i++) {
    for (let j = 0; j <= i; j++) {
      let value = gram[i * size + j] / rows + (i === j ? 1e-4 : 0);
      for (let k = 0; k < j; k++) value -= gram[i * size + k] * gram[j * size + k];
      gram[i * size + j] = i === j ? Math.sqrt(value) : value / gram[j * size + j];
    }
    for (let j = 0; j < i; j++) rhs[i] -= gram[i * size + j] * rhs[j];
    rhs[i] /= gram[i * size + i];
  }
  for (let i = size - 1; i >= 0; i--) {
    for (let j = i + 1; j < size; j++) rhs[i] -= gram[j * size + i] * rhs[j];
    rhs[i] /= gram[i * size + i];
  }
  return rhs.map((coefficient, i) => coefficient / scales[i]);
}
