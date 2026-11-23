import assert from 'node:assert/strict';
import { test } from 'node:test';
import { coordinateSearch, groupSearch, medianClip, refitReadout } from './wsp-blind-repair-controls.mjs';

const model = {
  layers: [
    { inputs: 3, outputs: 3, activation: 'Tanh', steepness: 1, weights: [1, 2, 90, -1, -2, -3, 0, 0, 0], bias: [0, 0, 0] },
    { inputs: 3, outputs: 2, activation: 'Linear', steepness: 1, weights: [1, 2, 3, -1, -2, -3], bias: [0, 0] }
  ]
};

test('median clipping respects the per-array median and leaves the source unchanged', () => {
  const copy = structuredClone(model);
  assert.deepEqual(medianClip(model), [{ layer: 0, parameter: 'weights', index: 2, value: 15 }]);
  assert.deepEqual(model, copy);
  assert.throws(() => medianClip(model, Infinity), RangeError);
});

for (const [name, search] of [['coordinate', coordinateSearch], ['group', groupSearch]]) {
  test(name + ' search obeys evaluation limits even when every candidate fails', async () => {
    const copy = structuredClone(model);
    let attempts = 0;
    const reject = async edits => {
      assert.ok(edits.length);
      const locations = edits.map(edit => [edit.layer, edit.parameter, edit.index].join(' '));
      assert.equal(new Set(locations).size, locations.length);
      attempts++;
      return null;
    };
    assert.deepEqual(await search(model, reject, 0), []);
    assert.equal(attempts, 0);
    assert.deepEqual(await search(model, reject, 7), []);
    assert.equal(attempts, 7);
    assert.deepEqual(model, copy);
    await assert.rejects(search(model, reject, 21), RangeError);
  });
}

test('group search can undo a whole-row sign flip in one proposal', async () => {
  const expected = [
    { layer: 0, parameter: 'weights', index: 3, value: 1 },
    { layer: 0, parameter: 'weights', index: 4, value: 2 },
    { layer: 0, parameter: 'weights', index: 5, value: 3 }
  ];
  let attempts = 0;
  const patch = await groupSearch(model, edits => {
    attempts++;
    return Number(JSON.stringify(edits) === JSON.stringify(expected));
  });
  assert.deepEqual(patch, expected);
  assert.ok(attempts > 0 && attempts <= 20);
});

test('readout ridge refit matches the closed form for two centered points', () => {
  const linear = {
    layers: [{ inputs: 1, outputs: 2, activation: 'Linear', steepness: 1, weights: [0, 0], bias: [0, 0] }]
  };
  assert.deepEqual(refitReadout(linear, [[-1], [1]], [[3, -2], [7, 4]], 2), [
    { layer: 0, parameter: 'weights', index: 0, value: 1 },
    { layer: 0, parameter: 'bias', index: 0, value: 5 },
    { layer: 0, parameter: 'weights', index: 1, value: 1.5 },
    { layer: 0, parameter: 'bias', index: 1, value: 1 }
  ]);
  assert.throws(() => refitReadout(linear, [[1, 2]], [[3, 4]]), RangeError);
});
