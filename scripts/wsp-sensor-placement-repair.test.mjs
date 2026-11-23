import assert from 'node:assert/strict';
import { test } from 'node:test';
import { evaluateScores, passesGate, passesTrigger, subjectRoles } from './wsp-sensor-placement-repair.mjs';

const subjects = Array.from({ length: 10 }, (_, index) => index + 1);
const activities = Array.from({ length: 12 }, (_, index) => index + 1);

test('rotation allocates every subject once per fold without cross-role leakage', () => {
  const seen = new Map(subjects.map(subject => [subject, { test: 0, trigger: 0, fit: 0, gate: 0, source: 0 }]));
  for (const subject of subjects) {
    const roles = subjectRoles(subject);
    assert.equal(roles.test, subject);
    assert.equal(roles.fit.length, 2);
    assert.equal(roles.gate.length, 2);
    assert.equal(roles.source.length, 4);
    const assigned = Object.values(roles).flat();
    assert.equal(assigned.length, subjects.length);
    assert.deepEqual([...new Set(assigned)].sort((a, b) => a - b), subjects);
    for (const [role, participants] of Object.entries(roles))
      for (const participant of [participants].flat()) seen.get(participant)[role]++;
  }
  for (const frequency of seen.values())
    assert.deepEqual(frequency, { test: 1, trigger: 1, fit: 2, gate: 2, source: 4 });
  assert.throws(() => subjectRoles(0), /unknown test subject/);
});

test('balanced accuracy averages per-class recall and argmax ties choose the first class', () => {
  const labels = [...Array(11).fill(1), ...activities];
  const predictions = labels.flatMap(label => activities.map(activity =>
    label === 2 ? 0 : Number(label === activity)));
  const result = evaluateScores(labels, predictions);
  assert.equal(result.balancedAccuracy, 11 / 12);
  assert.equal(result.accuracy, 22 / 23);
  assert.equal(result.mse, 1 / (23 * 12));
  assert.deepEqual(result.classCounts, [12, ...Array(11).fill(1)]);
  assert.deepEqual(result.classCorrect, [12, 0, ...Array(10).fill(1)]);
});

test('metric rejects missing class coverage, wrong score size and nonfinite values', () => {
  assert.throws(() => evaluateScores([1], Array(12).fill(0)), /requires every class/);
  assert.throws(() => evaluateScores(activities, Array(143).fill(0)), /strictly equal/);
  const predictions = Array(144).fill(0);
  predictions[13] = NaN;
  assert.throws(() => evaluateScores(activities, predictions), /nonfinite score/);
});

test('trigger requires at least ten points of balanced accuracy loss', () => {
  const reference = { balancedAccuracy: 0.8 };
  assert.equal(passesTrigger(reference, { balancedAccuracy: 0.7 }), true);
  assert.equal(passesTrigger(reference, { balancedAccuracy: 0.7001 }), false);
  assert.equal(passesTrigger(reference, { balancedAccuracy: 0.8 }), false);
  assert.equal(passesTrigger(reference, { balancedAccuracy: 0.9 }), false);
});

test('acceptance gate requires three points of gain and no MSE regression', () => {
  const previous = { balancedAccuracy: 0.70, mse: 0.2 };
  assert.equal(passesGate(previous, { balancedAccuracy: 0.73, mse: 0.2 }), true);
  assert.equal(passesGate(previous, { balancedAccuracy: 0.75, mse: 0.19 }), true);
  assert.equal(passesGate(previous, { balancedAccuracy: 0.7299, mse: 0.1 }), false);
  assert.equal(passesGate(previous, { balancedAccuracy: 0.75, mse: 0.200001 }), false);
});
