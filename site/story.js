import { mountPlayground } from './playground.js';

// Native scrolling remains in charge. The nearest chapter updates one shared
// illustration; every explanation and result is also readable without scripts.
const scenes = [
  ['input', 'Two inputs enter. The connections decide how much each one matters.', '0', 'w = 0'],
  ['training', 'Training adjusts weights from examples. The network keeps the same shape.', '0', 'w = 0'],
  ['edit', 'A direct edit changes one connection. It is still only a candidate.', '1', 'w = 1'],
  ['review', 'Evaluate first. Promote explicitly. Keep the previous model available.', '1', 'w = 1']
];
const chapters = scenes.map(([id]) => document.querySelector(`[data-scene="${id}"]`));
const visual = document.querySelector('#story-visual');
const progress = document.querySelector('#reading-progress');
let scheduled = false;
function updateScroll() {
  scheduled = false;
  const range = document.documentElement.scrollHeight - innerHeight;
  if (progress) progress.style.transform = `scaleX(${range > 0 ? Math.min(1, Math.max(0, scrollY / range)) : 0})`;
  const anchor = innerHeight * (innerWidth <= 600 ? .72 : .52);
  let closest = 0, distance = Infinity;
  chapters.forEach((chapter, index) => {
    if (!chapter) return;
    const rect = chapter.getBoundingClientRect();
    const next = Math.abs((rect.top + rect.bottom) / 2 - anchor);
    if (next < distance) { closest = index; distance = next; }
  });
  const [id, caption, output, weight] = scenes[closest];
  if (visual && visual.dataset.activeScene !== id) {
    visual.dataset.activeScene = id;
    document.querySelector('#scene-number').textContent = `0${closest + 1} / 04`;
    document.querySelector('#scene-caption').textContent = caption;
    document.querySelector('#story-output').textContent = output;
    document.querySelector('#story-weight').textContent = weight;
    document.querySelectorAll('.scene-pagination a').forEach((link, index) => {
      if (index === closest) link.setAttribute('aria-current', 'step');
      else link.removeAttribute('aria-current');
    });
  }
}
function scheduleScroll() {
  if (!scheduled) { scheduled = true; requestAnimationFrame(updateScroll); }
}
addEventListener('scroll', scheduleScroll, { passive: true });
addEventListener('resize', scheduleScroll, { passive: true });
document.fonts?.ready.then(scheduleScroll);
updateScroll();

const motion = document.querySelector('#motion-toggle');
motion.addEventListener('click', () => {
  const paused = motion.getAttribute('aria-pressed') !== 'true';
  document.documentElement.dataset.motion = paused ? 'paused' : 'running';
  motion.setAttribute('aria-pressed', String(paused));
  motion.querySelector('[data-motion-label]').textContent = paused ? 'Resume motion' : 'Pause motion';
  motion.querySelector('.pause-icon').textContent = paused ? '▶' : 'Ⅱ';
});

const models = ['majority', 'centroid', 'linear', 'mlp', 'trees'];
const benchmarks = {
  iris: { name: 'Iris', values: [33.33, 84.67, 80.67, 96, 94.67], takeaway: 'On Iris, the small MLP averages 96.00%. That is a result on these splits, not a universal ranking.' },
  wine: { name: 'Wine', values: [38.46, 97.95, 97.44, 95.90, 99.49], takeaway: 'On Wine, ExtraTrees averages 99.49%; the small MLP reaches 95.90%. The comparison includes where the network loses.' }
};
document.querySelectorAll('[data-benchmark]').forEach(button => button.addEventListener('click', () => {
  const data = benchmarks[button.dataset.benchmark];
  if (!data) return;
  document.querySelectorAll('[data-benchmark]').forEach(other => other.setAttribute('aria-pressed', String(other === button)));
  models.forEach((model, index) => {
    const width = data.values[index] * 4.32;
    document.querySelector(`[data-bar="${model}"]`).setAttribute('width', String(width));
    const label = document.querySelector(`[data-bar-value="${model}"]`);
    label.setAttribute('x', String(160 + width + 10));
    label.textContent = `${data.values[index].toFixed(2)}%`;
  });
  document.querySelector('#benchmark-svg-title').textContent = `${data.name} mean held-out accuracy`;
  document.querySelector('#benchmark-svg-desc').textContent = `Across five fixed splits: ${models.map((model, index) => `${model} ${data.values[index].toFixed(2)}%`).join(', ')}. The complete results for both datasets follow in a table.`;
  document.querySelector('#chart-takeaway').textContent = data.takeaway;
}));

document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
  const target = document.getElementById(button.dataset.copy);
  if (!target) return;
  try {
    await navigator.clipboard.writeText(target.textContent.trim());
    document.querySelector('#copy-status').textContent = `Copied ${button.dataset.copy === 'quickstart-code' ? 'JavaScript example' : 'command'} to clipboard.`;
    button.textContent = 'Copied';
    setTimeout(() => { button.textContent = button.dataset.copy === 'quickstart-code' ? 'Copy code' : 'Copy'; }, 1800);
  } catch {
    document.querySelector('#copy-status').textContent = 'Copy is unavailable in this browser. Select the command or example to copy it.';
  }
}));

try { mountPlayground(document); }
catch (error) {
  document.querySelector('#demo-status').textContent = `The demonstration could not initialize: ${error.message}. The rest of this guide remains available.`;
}
