# Transfer a correction into one receiver readout

The development checkout exports `proposeReadoutDelta`. Give it a receiver, an
ordered batch of shared probes, and the desired change in one output per probe.
Use the receiver's `baseHash` and `inputHash` from `traceWeightModel`. The
function returns an ordinary weight patch or `null` for a no-op.

```js
const { baseHash, inputHash } = await traceWeightModel(receiver, probes);
const patch = await proposeReadoutDelta(receiver, probes, {
  baseHash, inputHash, output: 2, deltas: senderOutputChanges
});
if (patch) {
  const evaluation = await evaluateWeightPatch(receiver, patch, holdout, policy);
  console.log(evaluation.accepted, evaluation.candidate);
}
```

The receiver runs its frozen hidden network in WASM, scales each feature by
its maximum absolute probe activation, adds a bias column, and solves mean
squared residual error with an L2 penalty of 0.0001 on all scaled coefficients.
The fit uses Cholesky factorization. Only the selected row of the final unit
steepness Linear layer and its bias can change. The row including bias must fit
the edit budget (128 by default) and have at least that many probes. Zero
support in a feature produces zero correction there. Calibration fit alone
cannot establish transfer on fresh inputs. The caller supplies holdout and
retention cases and controls promotion.

[LatentMesh experiment and measured controls](https://github.com/ruvnet/LatentMesh/pull/34)
