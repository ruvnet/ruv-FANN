/** Browser-safe adapter for caller-owned, actual WasmPreparedNetwork instances. */
export const ENCODER_LIMITS = Object.freeze({ maxDimension: 65536, maxRows: 4096,
  maxBatchRows: 128, maxBytes: 64 * 1024 * 1024 });
const typed = Object.getPrototypeOf(Float64Array.prototype);
const tag = Object.getOwnPropertyDescriptor(typed, Symbol.toStringTag).get;
const length = Object.getOwnPropertyDescriptor(typed, 'length').get;
const buffer = Object.getOwnPropertyDescriptor(typed, 'buffer').get;
const arrayBytes = Object.getOwnPropertyDescriptor(ArrayBuffer.prototype, 'byteLength').get;
const values = Float64Array.prototype.values;

export function checkedInteger(value, min, max, name) {
  if (!Number.isSafeInteger(value) || value < min || value > max)
    throw new RangeError(`${name} must be an integer from ${min} to ${max}`);
  return value;
}

/** Intrinsic brand/buffer checks reject proxies, tag spoofing, shared and detached views. */
export function checkedArray(value, name, type = 'Float64Array') {
  let size;
  try {
    if (tag.call(value) !== type) throw new TypeError();
    size = length.call(value);
    arrayBytes.call(buffer.call(value)); // SharedArrayBuffer is a different intrinsic brand.
    values.call(value).next(); // Detached zero-length views must not look like valid empty arrays.
  } catch { throw new TypeError(`${name} must be a nonshared, nondetached ${type}`); }
  return size;
}

export function preparedDimensions(prepared) {
  if (!prepared || typeof prepared.create_batch !== 'function' || typeof prepared.run !== 'function' ||
      (Object.hasOwn(prepared, '__wbg_ptr') && prepared.__wbg_ptr === 0))
    throw new TypeError('prepared must be a live WasmPreparedNetwork');
  const inputSize = checkedInteger(prepared.input_size, 1, ENCODER_LIMITS.maxDimension, 'input_size');
  const outputSize = checkedInteger(prepared.output_size, 1, ENCODER_LIMITS.maxDimension, 'output_size');
  if (prepared.estimated_bytes !== undefined)
    checkedInteger(prepared.estimated_bytes, 0, ENCODER_LIMITS.maxBytes, 'prepared estimated_bytes');
  return { inputSize, outputSize };
}

export function checkedInputs(inputs, inputSize, outputSize, maxRows) {
  checkedInteger(inputSize, 1, ENCODER_LIMITS.maxDimension, 'inputSize');
  checkedInteger(outputSize, 1, ENCODER_LIMITS.maxDimension, 'outputSize');
  checkedInteger(maxRows, 1, ENCODER_LIMITS.maxRows, 'maxRows');
  const size = checkedArray(inputs, 'inputs');
  if (size % inputSize !== 0) throw new RangeError('inputs must contain exact row-major input rows');
  const rows = size / inputSize;
  if (rows > maxRows || rows * (inputSize + outputSize) * 8 > ENCODER_LIMITS.maxBytes)
    throw new RangeError('input/output row or byte budget exceeded');
  for (let index = 0; index < size; index++)
    if (!Number.isFinite(inputs[index])) throw new RangeError('inputs must contain only finite numbers');
  return { values: inputs, rows };
}

/** Owns one reusable batch only; the supplied prepared snapshot always remains caller-owned. */
export function createEncoder(prepared, options = {}) {
  if (!options || typeof options !== 'object' || Array.isArray(options) ||
      Object.keys(options).some(key => !['batchRows', 'maxRows'].includes(key))) throw new TypeError('unknown encoder options');
  const batchRows = checkedInteger(options.batchRows ?? 128, 1, ENCODER_LIMITS.maxBatchRows, 'batchRows');
  const maxRows = checkedInteger(options.maxRows ?? 4096, 1, ENCODER_LIMITS.maxRows, 'maxRows');
  const { inputSize, outputSize } = preparedDimensions(prepared);
  if (batchRows * (inputSize + outputSize) * 8 > ENCODER_LIMITS.maxBytes)
    throw new RangeError('batch staging exceeds its byte budget');
  const batch = prepared.create_batch(batchRows);
  let input, staging, closed = false;
  try {
    if (!batch || typeof batch.run_into !== 'function' || typeof batch.free !== 'function' ||
        batch.input_size !== inputSize || batch.output_size !== outputSize || batch.max_rows !== batchRows)
      throw new TypeError('prepared returned an incompatible batch');
    if (batch.estimated_bytes !== undefined)
      checkedInteger(batch.estimated_bytes, 0, ENCODER_LIMITS.maxBytes, 'batch estimated_bytes');
    input = new Float64Array(batchRows * inputSize);
    staging = new Float64Array(batchRows * outputSize);
  } catch (error) { if (typeof batch?.free === 'function') batch.free(); throw error; }
  return Object.freeze({ inputSize, outputSize, batchRows,
    encode(inputs) {
      if (closed) throw new Error('encoder is closed');
      const { values, rows } = checkedInputs(inputs, inputSize, outputSize, maxRows);
      const output = new Float64Array(rows * outputSize);
      for (let first = 0; first < rows; first += batchRows) {
        const active = Math.min(batchRows, rows - first), count = active * inputSize;
        // Indexed copies avoid caller-overridden iterators, methods and species constructors.
        for (let index = 0; index < count; index++) input[index] = values[first * inputSize + index];
        const result = staging.subarray(0, active * outputSize);
        batch.run_into(input.subarray(0, count), result, active);
        for (let index = 0; index < result.length; index++)
          if (!Number.isFinite(result[index])) throw new RangeError('prepared inference produced a nonfinite output');
        output.set(result, first * outputSize);
      }
      return output;
    },
    close() { if (!closed) { closed = true; input = null; staging = null; batch.free(); } }
  });
}
