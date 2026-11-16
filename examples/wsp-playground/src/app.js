import { createLab, grid, benchmarkRuntime } from './engine.js';
import { drawScene, drawHeatmap } from './visuals.js';
import { walkthroughSteps, nextWalkthroughStep } from './walkthrough.js';

const $ = id => document.getElementById(id);
let current = 'robot', labs = {}, lab, candidate, evidence = null, busy = false, paused = matchMedia('(prefers-reduced-motion: reduce)').matches;
let scene = { name: current, inputs: [.85, .15], active: 0, candidate: 0 }, renderTicket = 0, traceTimer, toastTimer, clock = 0, lastTime;
let walkStep = 0, walkVisible = true;
const value = id => Number($(id).value);
const weights = () => [value('weight-a'), value('weight-b')];
const inputs = () => [value('sensor-a'), value('sensor-b')];
const format = (v, digits = 3) => Math.abs(v) < 1e-10 ? (0).toFixed(digits) : v.toFixed(digits);
function notify(message, error = false) {
  $('notice').textContent = message; $('notice').className = `visible${error ? ' error' : ''}`;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => { $('notice').className = ''; }, 6500);
}
function syncButtons() {
  for (const id of ['evaluate', 'repair', 'shortcut', 'search', 'weight-a', 'weight-b', 'export']) $(id).disabled = busy;
  document.querySelectorAll('.tab').forEach(b => { b.disabled = busy; });
  $('promote').disabled = busy || !evidence?.accepted;
  $('rollback').disabled = busy || lab?.snapshot().revision === 1;
  $('evaluate').firstChild.textContent = busy ? 'Running real WASM… ' : 'Evaluate candidate ';
  renderWalkthrough();
}
function renderWalkthrough() {
  $('walkthrough').dataset.step = String(walkStep);
  $('walkthrough').hidden = !walkVisible; $('walk-open').hidden = walkVisible;
  document.querySelectorAll('.walk-highlight').forEach(node => node.classList.remove('walk-highlight'));
  const steps = walkthroughSteps(current), step = steps[walkStep];
  $('walk-title').textContent = step.title; $('walk-copy').textContent = step.copy;
  $('walk-look').textContent = step.look;
  $('walk-step').textContent = walkStep ? `Step ${walkStep} of 7` : 'A guided first experiment';
  $('walk-action').textContent = busy ? 'Checking your experiment…' : step.action;
  $('walk-action').disabled = busy || !!(step.target && $(step.target).disabled);
  $('walk-restart').hidden = walkStep === 0; $('walk-restart').disabled = busy;
  $('walk-progress').setAttribute('aria-label', walkStep ? `Step ${walkStep} of 7` : 'Walkthrough not started');
  const bars = Array.from({ length: 7 }, (_, i) => { const bar = document.createElement('span'); bar.classList.toggle('complete', i < walkStep); return bar; });
  $('walk-progress').replaceChildren(...bars);
  if (walkVisible && walkStep && step.target) $(step.target).classList.add('walk-highlight');
}
function walkEvent(event) { walkStep = nextWalkthroughStep(walkStep, event); renderWalkthrough(); }
async function startWalkthrough() {
  const original = lab.scenario.initial, active = lab.snapshot().model.layers[1].weights;
  if (active.some((v, i) => v !== original[i])) lab.rollback();
  $('sensor-a').value = '0.85'; $('sensor-b').value = '0.15';
  await setWeights(original); walkStep = 1; walkVisible = true; renderWalkthrough();
}
$('walk-action').addEventListener('click', () => {
  if (busy) return;
  if (walkStep === 0) { operation(startWalkthrough); return; }
  if (walkStep === 7) { operation(() => selectTab(current === 'robot' ? 'room' : 'robot')); return; }
  const target = walkthroughSteps(current)[walkStep].target;
  if (target && !$(target).disabled) $(target).click();
});
$('walk-restart').addEventListener('click', () => operation(startWalkthrough));
$('walk-dismiss').addEventListener('click', () => { walkVisible = false; renderWalkthrough(); });
$('walk-open').addEventListener('click', () => { walkVisible = true; renderWalkthrough(); });
async function operation(fn) {
  if (busy) return;
  busy = true; syncButtons();
  try { await fn(); } catch (e) { notify(e.message, true); }
  finally { busy = false; syncButtons(); }
}
function resetGates() {
  evidence = null; lab.invalidate(); $('promote').disabled = true;
  $('gate-status').textContent = 'UNEVALUATED'; $('gate-status').className = 'status-tag';
  for (const row of $('gates').children) {
    row.querySelector('.gate-icon').textContent = '○'; row.querySelector('.gate-icon').className = 'gate-icon';
    row.querySelector('output').textContent = 'Waiting';
  }
  $('receipt-summary').textContent = 'Candidate changed. Evaluate it before promotion.';
}
function setWeights(w) {
  $('weight-a').value = w[0]; $('weight-b').value = w[1]; resetGates(); return render();
}
function decision(v) {
  return current === 'room' ? `${format(v, 2)} presence signal` : v > .05 ? 'Turn right' : v < -.05 ? 'Turn left' : 'Straight ahead';
}
async function render() {
  if (!lab || current === 'bench') return;
  const ticket = ++renderTicket, owner = lab, w = weights(), x = inputs();
  $('sensor-a-value').textContent = format(x[0], 2); $('sensor-b-value').textContent = format(x[1], 2);
  for (const [i, id] of ['weight-a', 'weight-b'].entries()) $(id + '-value').textContent = format(w[i], 2);
  const model = await owner.candidate(w);
  if (ticket !== renderTicket || lab !== owner) return;
  candidate = model;
  const snapshot = lab.snapshot();
  const active = lab.predict(snapshot.model, x)[0], proposed = lab.predict(model, x)[0], target = lab.scenario.expected(...x);
  scene = { name: current, inputs: x, active, candidate: proposed };
  $('active-output').textContent = format(active); $('candidate-output').textContent = format(proposed); $('target-output').textContent = format(target);
  $('active-action').textContent = decision(active); $('candidate-action').textContent = decision(proposed);
  $('network-wa').textContent = `w₁ ${format(w[0], 2)}`; $('network-wb').textContent = `w₂ ${format(w[1], 2)}`;
  $('neuron-out').textContent = format(proposed, 2);
  $('edge-a').setAttribute('stroke-width', 1 + Math.abs(w[0]) * 2); $('edge-b').setAttribute('stroke-width', 1 + Math.abs(w[1]) * 2);
  $('revision').textContent = `REV ${String(snapshot.revision).padStart(2, '0')}`;
  $('checkpoint').textContent = `Active SHA256 ${snapshot.hash}`;
  const domain = grid(lab.scenario), predictions = lab.predict(model, domain.inputs);
  const mse = predictions.reduce((sum, v, i) => sum + (v - domain.expected[i]) ** 2, 0) / predictions.length;
  $('grid-error').textContent = format(mse, 4); drawHeatmap($('heatmap'), predictions, domain.expected);
  drawScene($('scene'), scene, clock);
  clearTimeout(traceTimer); traceTimer = setTimeout(async () => {
    try {
      const trace = await owner.trace(model, x);
      if (ticket !== renderTicket || owner !== lab) return;
      $('neuron-a').textContent = format(trace.layers[0].activations[0], 2);
      $('neuron-b').textContent = format(trace.layers[0].activations[1], 2);
      $('trace').textContent = JSON.stringify(trace, null, 2);
    } catch (e) { notify(e.message, true); }
  }, 60);
}
async function selectTab(name) {
  ++renderTicket; current = name; walkStep = 0;
  document.querySelectorAll('.tab').forEach(button => {
    const selected = button.dataset.tab === name;
    button.classList.toggle('selected', selected); button.setAttribute('aria-selected', String(selected));
  });
  document.querySelector('.note-number').textContent = `${name === 'robot' ? '01' : name === 'room' ? '02' : '03'} / 03`;
  $('lab').hidden = name === 'bench'; $('bench').hidden = name !== 'bench';
  if (name === 'bench') return;
  lab = labs[name] ||= await createLab(name);
  $('scene-title').textContent = name === 'robot' ? 'Same world. Different decisions.' : 'A room that senses the difference.';
  $('scene-label').textContent = name === 'robot' ? 'ROBOT SENSOR FIELD' : 'RUVIEW · SIMULATED AMBIENT ROOM';
  $('sensor-a-name').textContent = lab.scenario.inputs[0]; $('sensor-b-name').textContent = lab.scenario.inputs[1];
  $('scenario-instruction').textContent = lab.scenario.instruction;
  $('search-note').textContent = 'Search sees one development case. Separate cases can reject its shortcut.';
  await setWeights(lab.snapshot().model.layers[1].weights); syncButtons();
}
for (const id of ['sensor-a', 'sensor-b']) $(id).addEventListener('input', () => { render().catch(e => notify(e.message, true)); });
for (const id of ['weight-a', 'weight-b']) $(id).addEventListener('input', () => { resetGates(); walkEvent('edit'); render().catch(e => notify(e.message, true)); });
document.querySelectorAll('.tab').forEach(b => b.addEventListener('click', () => operation(() => selectTab(b.dataset.tab))));
$('shortcut').addEventListener('click', () => operation(async () => { await setWeights(lab.scenario.shortcut); walkEvent('shortcut'); notify('This fits the development case. Will the other cases agree?'); }));
$('repair').addEventListener('click', () => operation(async () => { await setWeights(lab.scenario.good); walkEvent('repair'); notify('Repair loaded as a candidate. Evaluate it before promotion.'); }));
$('search').addEventListener('click', () => operation(async () => {
  const found = await lab.search(); await setWeights(found.weights);
  walkEvent('edit');
  $('search-note').textContent = `${found.result.evaluations} candidate screens · development error ${format(found.result.bestScore, 4)}. Separate tests still required.`;
  notify('Bounded search finished. A development win can still fail the gates.');
}));
$('evaluate').addEventListener('click', () => operation(async () => {
  evidence = null;
  const result = await lab.evaluate(weights());
  const response = await fetch('/api/evaluate', { method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ scenario: current, weights: weights(), baseWeights: lab.snapshot().model.layers[1].weights }) });
  if (!response.ok) throw new Error('Edge evaluation unavailable. Active model unchanged; retry evaluation.');
  const edge = await response.json();
  if (edge.candidateHash !== result.candidateHash || edge.baseHash !== result.baseHash || edge.accepted !== result.accepted ||
      edge.invariant.suiteHash !== result.invariant.suiteHash || edge.regression.suites.some((s, i) =>
        s.evaluation.datasetHash !== result.regression.suites[i].evaluation.datasetHash ||
        s.evaluation.policyHash !== result.regression.suites[i].evaluation.policyHash)) throw new Error('Browser and edge evidence disagree. Promotion blocked.');
  result.edge = edge; evidence = result;
  lab.log.push({ action: 'edge-evaluate', at: new Date().toISOString(), ...edge });
  const results = [result.regression.suites[0].evaluation.accepted, result.regression.suites[1].evaluation.accepted, result.invariant.accepted];
  [...$('gates').children].forEach((row, i) => {
    row.querySelector('.gate-icon').textContent = results[i] ? '✓' : '×';
    row.querySelector('.gate-icon').className = `gate-icon ${results[i] ? 'pass' : 'fail'}`;
    row.querySelector('output').textContent = i < 2 ? `${results[i] ? 'PASS' : 'FAIL'} · MSE ${format(result.regression.suites[i].evaluation.candidate.mse, 4)}` : `${results[i] ? 'PASS' : 'FAIL'} · ${result.invariant.violationCount} violations`;
  });
  $('gate-status').textContent = result.accepted ? 'READY TO PROMOTE' : 'REJECTED';
  $('gate-status').className = `status-tag ${result.accepted ? 'pass' : 'fail'}`;
  $('receipt-summary').textContent = result.accepted ? 'Browser and edge gates agree. Active weights are unchanged until you choose Promote.' : 'Browser and edge rejected this candidate. Active weights remain unchanged.';
  walkEvent(result.accepted ? 'accepted' : 'rejected');
  notify(result.accepted ? 'Candidate passed all gates. You can now promote it.' : 'Shortcut rejected. The active model did not change.', !result.accepted);
}));
$('promote').addEventListener('click', () => operation(async () => {
  const next = lab.promote(); evidence = null;
  $('receipt-summary').textContent = `Promoted revision ${next.revision}. The active model now uses the evaluated weights.`;
  $('gate-status').textContent = 'PROMOTED'; await render(); notify(`Revision ${next.revision} is active in your local session.`);
  walkEvent('promote');
}));
$('rollback').addEventListener('click', () => operation(async () => {
  const next = lab.rollback(); await setWeights(next.model.layers[1].weights);
  $('receipt-summary').textContent = `Revision ${next.revision} restores the exact original checkpoint. History is retained.`;
  $('gate-status').textContent = 'ROLLED BACK'; notify('Original weights restored. The revision number moves forward.');
  walkEvent('rollback');
}));
$('export').addEventListener('click', () => {
  const exported = { schema: 'ruv-fann.playground-receipt/1', scenario: current, generatedAt: new Date().toISOString(),
    active: lab.snapshot(), candidate, weights: weights(), events: lab.log, history: lab.workspace.history(),
    scope: 'Local session; public synthetic finite-domain cases; no production authorization or generalization claim.' };
  const url = URL.createObjectURL(new Blob([JSON.stringify(exported, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a'); a.href = url; a.download = `wsp-${current}-experiment.json`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
});
$('pause').addEventListener('click', () => {
  paused = !paused; $('pause').textContent = paused ? '▷' : 'Ⅱ'; $('pause').setAttribute('aria-label', paused ? 'Resume animation' : 'Pause animation');
});
$('run-benchmark').addEventListener('click', async () => {
  const button = $('run-benchmark'); button.disabled = true; button.textContent = 'Measuring real WASM…';
  try {
    await new Promise(resolve => setTimeout(resolve, 40));
    const results = [];
    for (const simd of [false, true]) {
      await new Promise(resolve => setTimeout(resolve, 40));
      results.push(await benchmarkRuntime(simd));
    }
    [...$('benchmark-results').children].forEach((card, i) => {
      const r = results[i]; card.replaceChildren();
      const title = document.createElement('span'); title.textContent = r.backend.toUpperCase(); card.append(title);
      const heading = document.createElement('h3'); heading.textContent = `${Math.round(r.predictionsPerSecond).toLocaleString()} predictions / sec`; card.append(heading);
      const info = document.createElement('p'); info.textContent = 'Observed on this device. 256 rows per batch.'; card.append(info);
      const dl = document.createElement('dl');
      for (const [k, v] of [['Median batch', `${r.medianBatchMs.toFixed(4)} ms`], ['p95 batch', `${r.p95BatchMs.toFixed(4)} ms`], ['Reference max error', r.maxError.toExponential(2)], ['Measured batches', '3,000']]) {
        const dt = document.createElement('dt'), dd = document.createElement('dd'); dt.textContent = k; dd.textContent = v; dl.append(dt, dd);
      }
      card.append(dl);
    });
    notify('Both WASM backends measured against the same arithmetic reference.');
  } catch (e) { notify(e.message, true); }
  finally { button.disabled = false; button.textContent = 'Run benchmark ↗'; }
});
function animate(time) {
  if (lastTime === undefined) lastTime = time;
  if (!paused && !document.hidden) clock += Math.min(time - lastTime, 100) / 1000;
  lastTime = time;
  if (lab && current !== 'bench' && !document.hidden) drawScene($('scene'), scene, clock);
  requestAnimationFrame(animate);
}
window.addEventListener('resize', () => { render().catch(e => notify(e.message, true)); });
try {
  await selectTab('robot'); $('loading').hidden = true;
  $('runtime').replaceChildren(); const dot = document.createElement('i'); $('runtime').append(dot, `${lab.runtime.backend.toUpperCase()} ONLINE`);
  $('pause').textContent = paused ? '▷' : 'Ⅱ'; $('pause').setAttribute('aria-label', paused ? 'Resume animation' : 'Pause animation');
  requestAnimationFrame(animate);
  document.documentElement.dataset.ready = 'true';
} catch (e) {
  $('loading').hidden = false; $('loading').textContent = `WASM initialization failed: ${e.message}. Please reload to retry.`;
  $('runtime').textContent = 'WASM unavailable'; notify(e.message, true);
}
