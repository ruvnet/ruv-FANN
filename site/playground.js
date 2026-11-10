// Packages are imported only after an explicit click. The site's import map pins
// their public npm releases to verified, self-hosted assets.
const mounts = new WeakMap();
const INPUTS = [0, 0, 0, 1, 1, 0, 1, 1];
const TARGETS = [0, 1, 1, 0];
const DATASET = { inputs: INPUTS, targets: TARGETS, rows: 4 };
const POLICY = { maxMSE: 1e-12, maxRegressionMSE: 0 };

function initialModel() {
  return { schema: 'ruv-fann.dense-layers/1', layers: [
    { inputs: 2, outputs: 2, weights: [1, -1, -1, 1], bias: [0, 0], activation: 'ReLU', steepness: 1 },
    { inputs: 2, outputs: 1, weights: [0, 1], bias: [0], activation: 'Linear', steepness: 1 }
  ] };
}

const number = value => Number(value.toFixed(6)).toString();

/** Mount the fixed four-row demonstration without loading or running WASM yet. */
export function mountPlayground(root = document) {
  const container = root.matches?.('[data-playground]') ? root : root.querySelector('[data-playground]');
  if (!container) return null;
  if (mounts.has(container)) return mounts.get(container);
  const find = id => {
    const element = container.querySelector(`#${id}`);
    if (!element) throw new Error(`Playground is missing #${id}`);
    return element;
  };
  const start = find('demo-start');
  const slider = find('weight-control');
  const evaluate = find('demo-evaluate');
  const promote = find('demo-promote');
  const rollback = find('demo-rollback');
  const reset = find('demo-reset');
  const status = find('demo-status');
  const outputs = Object.fromEntries(['weight', 'active-weight', 'mse', 'revision', 'backend', 'candidate-hash']
    .map(name => [name, find(`demo-${name}`)]));
  const rows = ['active', 'preview'].map(kind => Array.from({ length: 4 }, (_, index) => {
    const cell = container.querySelector(`[data-${kind}-output="${index}"]`);
    if (!cell) throw new Error(`Playground is missing ${kind} output ${index}`);
    return cell;
  }));
  let api, runtime, workspace, initialHash;
  let token = 0, ready = false, busy = false, previewReady = false, proposal = null;

  function message(text, state) {
    status.textContent = text;
    status.dataset.state = state;
    container.dataset.state = state;
  }

  function controls() {
    start.disabled = busy || ready;
    slider.disabled = !ready;
    evaluate.disabled = !ready || busy || !previewReady;
    promote.disabled = !ready || busy || !previewReady || !proposal?.evaluation.accepted;
    rollback.disabled = !ready || busy || workspace.snapshot().hash === initialHash;
    reset.disabled = !ready || busy;
  }

  function fail(error) {
    busy = false;
    previewReady = false;
    proposal = null;
    message(`The demonstration could not complete: ${error instanceof Error ? error.message : 'unknown error'}. ${ready ? 'Reset the example to try again.' : 'Use Start to retry.'}`, 'error');
    controls();
  }

  function predict(model) {
    const prepared = runtime.prepare(model, { maxRows: 4 });
    try {
      const values = Array.from(prepared.predict(new Float64Array(INPUTS), 4));
      if (values.length !== 4 || values.some(value => !Number.isFinite(value))) {
        throw new Error('WASM returned invalid predictions');
      }
      return values;
    } finally {
      prepared.dispose();
    }
  }

  function renderActive() {
    const active = workspace.snapshot();
    const values = predict(active.model);
    values.forEach((value, index) => { rows[0][index].textContent = number(value); });
    outputs['active-weight'].textContent = number(active.model.layers[1].weights[0]);
    outputs.revision.textContent = String(active.revision);
    outputs.backend.textContent = runtime.backend;
    return active;
  }

  function patchFor(active, value) {
    return { schema: 'ruv-fann.weight-patch/1', baseHash: active.hash,
      edits: [{ layer: 1, parameter: 'weights', index: 0, value }] };
  }

  function readWeight() {
    const value = Number(slider.value);
    if (!Number.isFinite(value) || value < 0 || value > 2) throw new Error('Choose a weight from 0 to 2');
    return value;
  }

  async function preview(text = 'Preview only. Evaluate these four examples before promoting a change.', state = 'preview') {
    const operation = ++token;
    proposal = null;
    previewReady = false;
    busy = true;
    controls();
    try {
      const value = readWeight();
      outputs.weight.textContent = number(value);
      outputs.mse.textContent = '…';
      outputs['candidate-hash'].textContent = 'Computing…';
      container.querySelectorAll('[data-demo-weight]').forEach(element => { element.textContent = number(value); });
      container.querySelectorAll('[data-demo-connection]').forEach(element => {
        element.style.setProperty('--demo-weight', String(value));
        element.style.opacity = String(0.25 + 0.3 * value);
      });
      const active = workspace.snapshot();
      const candidate = await api.applyWeightPatch(active.model, patchFor(active, value));
      const hash = await api.fingerprintModel(candidate);
      if (operation !== token) return;
      const predictions = predict(candidate);
      predictions.forEach((prediction, index) => { rows[1][index].textContent = number(prediction); });
      const mse = predictions.reduce((sum, prediction, index) => sum + (prediction - TARGETS[index]) ** 2, 0) / 4;
      outputs.mse.textContent = number(mse);
      outputs['candidate-hash'].textContent = hash;
      previewReady = true;
      busy = false;
      message(text, state);
      controls();
    } catch (error) {
      if (operation === token) fail(error);
    }
  }

  async function begin() {
    if (busy || ready) return;
    const operation = ++token;
    busy = true;
    controls();
    message('Loading the local WebAssembly runtime…', 'loading');
    try {
      const [wsp, core] = await Promise.all([import('ruv-fann-wsp'), import('ruv-fann')]);
      const loadedRuntime = await core.loadRuntime({ simd: 'auto' });
      const loadedWorkspace = await wsp.createWeightWorkspace(initialModel(), { simd: 'auto' });
      if (operation !== token) return;
      api = wsp;
      runtime = loadedRuntime;
      workspace = loadedWorkspace;
      initialHash = workspace.snapshot().hash;
      ready = true;
      slider.value = '0';
      renderActive();
      await preview('WASM is ready. The active model misses one case. Try weight 2, evaluate it, then try weight 1.', 'ready');
    } catch (error) {
      if (operation === token) fail(error);
    }
  }

  async function evaluatePreview() {
    if (evaluate.disabled) return;
    const operation = ++token;
    const active = workspace.snapshot();
    const value = readWeight();
    busy = true;
    proposal = null;
    controls();
    message('Evaluating the proposed weight against all four examples in WASM…', 'evaluating');
    try {
      const evaluated = await workspace.propose(patchFor(active, value), DATASET, POLICY);
      if (operation !== token) return;
      proposal = evaluated;
      busy = false;
      outputs.mse.textContent = number(evaluated.evaluation.candidate.mse);
      outputs['candidate-hash'].textContent = evaluated.evaluation.candidateHash;
      if (evaluated.evaluation.accepted) {
        message('Evaluation accepted: all four examples match. The active model is unchanged. Use Promote to activate this candidate.', 'accepted');
      } else {
        message(`Evaluation rejected: MSE ${number(evaluated.evaluation.candidate.mse)} exceeds the allowed error. The active model is unchanged. Try weight 1.`, 'rejected');
      }
      controls();
    } catch (error) {
      if (operation === token) fail(error);
    }
  }

  async function promotePreview() {
    if (promote.disabled || !proposal?.evaluation.accepted) return;
    try {
      // The workspace checks its own stored evidence, ID and revision. Displayed
      // metrics and the range input cannot serve as promotion authority.
      workspace.promote(proposal.id, proposal.revision);
      proposal = null;
      const active = renderActive();
      slider.value = String(active.model.layers[1].weights[0]);
      await preview(`Promoted to revision ${active.revision}. The active model now matches all four examples. Rollback restores the initial model as a new revision.`, 'promoted');
    } catch (error) { fail(error); }
  }

  async function restoreInitial() {
    if (rollback.disabled) return;
    try {
      const active = workspace.snapshot();
      const restored = workspace.rollback(1, active.revision);
      proposal = null;
      renderActive();
      slider.value = String(restored.model.layers[1].weights[0]);
      await preview(`Rolled back to the initial model as revision ${restored.revision}. The earlier revision history is preserved.`, 'rolled-back');
    } catch (error) { fail(error); }
  }

  async function resetExample() {
    if (reset.disabled) return;
    const operation = ++token;
    busy = true;
    proposal = null;
    previewReady = false;
    controls();
    message('Resetting this local example…', 'loading');
    try {
      const fresh = await api.createWeightWorkspace(initialModel(), { simd: 'auto' });
      if (operation !== token) return;
      workspace = fresh;
      initialHash = workspace.snapshot().hash;
      slider.value = '0';
      renderActive();
      await preview('Reset complete: a fresh workspace at revision 1. Try weight 2, evaluate it, then try weight 1.', 'ready');
    } catch (error) {
      if (operation === token) fail(error);
    }
  }

  const onInput = () => {
    if (!ready || slider.disabled) return;
    // Moving the range invalidates any pending/accepted proposal immediately.
    // A late async result can never re-enable promotion for an older value.
    void preview('Weight changed: earlier evaluation invalidated. This is a preview; evaluate again before promotion.');
  };
  start.addEventListener('click', begin);
  slider.addEventListener('input', onInput);
  evaluate.addEventListener('click', evaluatePreview);
  promote.addEventListener('click', promotePreview);
  rollback.addEventListener('click', restoreInitial);
  reset.addEventListener('click', resetExample);
  controls();
  const mounted = Object.freeze({
    destroy() {
      ++token;
      start.removeEventListener('click', begin);
      slider.removeEventListener('input', onInput);
      evaluate.removeEventListener('click', evaluatePreview);
      promote.removeEventListener('click', promotePreview);
      rollback.removeEventListener('click', restoreInitial);
      reset.removeEventListener('click', resetExample);
      ready = false;
      busy = false;
      proposal = null;
      controls();
      mounts.delete(container);
    }
  });
  mounts.set(container, mounted);
  return mounted;
}
