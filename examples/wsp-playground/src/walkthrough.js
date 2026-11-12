// Tutorial text stays independent of model authority. Only completed real actions
// advance it; a passing message cannot enable promotion.
export function walkthroughSteps(name) {
  const room = name === 'room';
  return [
    { title: 'Change a decision. See why it changed.', copy: 'Weights are numbers that shape a model’s decisions. This tour lets you try a change, check whether it works, and undo it. No coding or AI background needed.', look: 'Starts from the original weights for this lab. Your other lab keeps its own experiment.', action: 'Start walkthrough →' },
    { title: 'Meet the two versions.', copy: room ? 'The gray model turns motion and background noise into a presence signal. The green model is your candidate. Changing its weights changes the strength of that signal.' : 'The gray robot is the active model. The green robot is your candidate. Both see the same obstacles. Changing two weights changes how the green robot steers.', look: 'Look at the active and candidate outputs below. A target is the answer we want for these sensor readings.', action: 'Load a shortcut →', target: 'shortcut' },
    { title: 'It looks right here. Will it work elsewhere?', copy: room ? 'This shortcut gets the current motion example right, but it also responds to noise when it should not. We need more than one example to spot that mistake.' : 'This shortcut steers away from a left obstacle. But it ignores an obstacle on the other side. One good example can hide a bad decision.', look: 'The active model has not changed. Now test the candidate on other sensor combinations.', action: 'Test the shortcut →', target: 'evaluate' },
    { title: 'The checks caught the mistake.', copy: 'The shortcut passed its practice example but failed other cases. Rejected means the candidate stays separate. The active version keeps its original weights.', look: 'The test results below show what failed. Coral squares mark input combinations where the candidate gives the wrong answer.', action: 'Load a better repair →', target: 'repair' },
    { title: 'Two numbers can change the behavior.', copy: room ? 'This repair keeps motion that rises above noise and removes the noise response. It changes the output connections directly. The candidate is still a proposal.' : 'This repair connects both sides of the steering decision. The robot can turn away from an obstacle on either side. The candidate is still a proposal.', look: 'The green output should now match the target. Matching this one view is not enough: run all the checks.', action: 'Test the repair →', target: 'evaluate' },
    { title: 'The repair passed. You still choose.', copy: 'The browser and Cloudflare checked the same candidate. It passed the practice case, five separate cases and 121 sensor combinations. Passing tests does not switch the active model.', look: 'Promote means use this tested version in your local session. These checks cover the examples shown, not every possible real world situation.', action: 'Make the repair active →', target: 'promote' },
    { title: 'The tested version is now active.', copy: 'You explicitly promoted the repair. The gray and green versions now use the same weights, so their outputs agree. The revision number records that a change happened.', look: 'Now undo it. Rollback brings back the exact original weights and keeps the history of your experiment.', action: 'Undo the change →', target: 'rollback' },
    { title: 'You changed a model and brought it back.', copy: 'You tried a shortcut, saw it rejected, tested a repair, chose to use it, and restored the original. That is the basic weight space programming workflow.', look: 'Keep exploring: move the sensors, adjust the weights yourself, or export a receipt with the model and its test results.', action: room ? 'Explore the robot lab →' : 'Explore the ambient room →' },
  ];
}

export function nextWalkthroughStep(step, event) {
  if (step === 0) return 0;
  if (event === 'edit') return 1;
  if (event === 'shortcut') return 2;
  if (event === 'repair') return 4;
  if (event === 'rejected') return 3;
  if (event === 'accepted') return 5;
  if (event === 'promote') return 6;
  if (event === 'rollback') return 7;
  return step;
}
