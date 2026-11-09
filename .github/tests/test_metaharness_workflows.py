"""MetaHarness workflow structure and executable evidence-gate regressions.

Synthetic receipts test rejection behavior only. The workflow separately runs
actual Darwin evolveNumeric against both committed production WASM variants.
"""
import copy
import json
import os
import pathlib
import re
import subprocess
import tempfile
import unittest

from test_rune_workflows import ROOT, load


def validate(workflow, ci):
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)

    require('workflow_call' in workflow.get('on', {}), 'reusable workflow missing')
    for event in ('push', 'pull_request'):
        paths = workflow.get('on', {}).get(event, {}).get('paths', [])
        require(all(value in paths for value in ('tools/metaharness/**', 'ruv-swarm/npm/wasm/**',
            'src/**', '.github/workflows/metaharness.yml')), 'standalone source/tool coverage missing')
    require(workflow.get('permissions') == {'contents': 'read'}, 'least permissions missing')
    caller = ci.get('jobs', {}).get('metaharness', {})
    require(caller.get('uses') == './.github/workflows/metaharness.yml', 'CI does not require MetaHarness')
    require(not caller.get('if') and not caller.get('continue-on-error'), 'CI can skip or ignore MetaHarness')
    require(not caller.get('secrets'), 'caller passes provider secrets')
    require(ci.get('jobs', {}).get('rune', {}).get('uses') == './.github/workflows/rune.yml', 'Rune gate was removed')
    job = workflow.get('jobs', {}).get('numeric', {})
    require(1 <= job.get('timeout-minutes', 0) <= 15, 'job deadline missing')
    require(not job.get('if') and not job.get('continue-on-error'), 'numeric job is optional')
    require(job.get('strategy', {}).get('matrix', {}).get('node') == ['22.13.0', '24'], 'required Node matrix missing')
    require(not job.get('permissions') or job['permissions'] == {'contents': 'read'}, 'job permission escalation')
    steps = job.get('steps', [])
    by_id = {value['id']: value for value in steps if 'id' in value}
    required = ('install', 'audit', 'boundaries', 'core', 'simd', 'evidence', 'unchanged')
    require(all(name in by_id for name in required), 'required bounded stage missing')
    if all(name in by_id for name in required):
        ids = [value.get('id') for value in steps]
        require([ids.index(name) for name in required] == sorted(ids.index(name) for name in required),
            'install/audit/test/actual execution ordering changed')
        require(by_id['install'].get('run') == 'npm ci --ignore-scripts --omit=optional --prefix tools/metaharness',
            'installation escapes the private locked tool or enables scripts')
        audit = by_id['audit'].get('run', '')
        require('npm audit' in audit and '--prefix tools/metaharness' in audit and '--audit-level=low' in audit
            and '--omit=dev' not in audit and '--production' not in audit, 'installed development graph is not audited')
        require(by_id['boundaries'].get('run') == 'node --test tools/metaharness/test/*.test.mjs',
            'adapter boundary tests missing')
        for variant in ('core', 'simd'):
            command = by_id[variant].get('run', '')
            require(all(value in command for value in ('node tools/metaharness/src/cli.mjs optimize',
                '--repo .', f'--variant {variant}', '--model tiny', '--generations 1', '--children 1',
                '--samples 5', '--total-rows 128', '--timeout-ms 60000', f'"$RUNNER_TEMP/metaharness-{variant}"')),
                'real bounded core/SIMD experiment missing')
        evidence = by_id['evidence'].get('run', '')
        require(all(value in evidence for value in ('modelCalls', 'sourceApplied', 'trialBudget', 'trials',
            'inputsAfter', 'inputsBefore', 'baselineSamplesNs', 'candidateSamplesNs', 'not_improved', 'workloadSha256')),
            'correctness/bounds/confirmation evidence missing')
        require('git status --porcelain --untracked-files=all' in by_id['unchanged'].get('run', ''),
            'source immutability guard missing')
    uploads = []
    for step in steps:
        require(not step.get('continue-on-error'), 'step failure can be ignored')
        require(not step.get('if') or step['if'] == 'always()', 'stage can silently skip')
        require('${{ secrets.' not in str(step), 'provider or publication secret is passed')
        action = step.get('uses', '')
        if action:
            require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[a-f0-9]{40}', action) is not None,
                'action commit pin missing')
            require('cache' not in action.lower(), 'dependency cache is trusted')
            if action.startswith('actions/checkout@'):
                require(step.get('with', {}).get('persist-credentials') is False, 'checkout credentials persisted')
            if action.startswith('actions/setup-node@'):
                require(step.get('with', {}).get('package-manager-cache') is False, 'automatic package cache enabled')
            if action.startswith('actions/upload-artifact@'):
                uploads.append(step)
        require(not re.search(r'\bnpm\s+(?:publish|audit\s+fix)\b|\bgit\s+push\b', step.get('run', '')),
            'source/publication mutation enabled')
    require(len(uploads) == 1, 'machine-readable artifact upload missing')
    if len(uploads) == 1:
        require(uploads[0].get('if') == 'always()', 'failure evidence discarded')
        options = uploads[0].get('with', {})
        require(options.get('if-no-files-found') == 'error', 'missing evidence is accepted')
        require(all(f'metaharness-{name}/*.json' in options.get('path', '') for name in ('core', 'simd')),
            'both experiment receipts are not preserved')
    return errors


def receipt(variant, faster=False, baseline_rows=1):
    context = dict(variant=variant, model='tiny', totalRows=128, layers=[5, 7, 3], activation='Linear',
        precision='f64', **{key: 'a' * 64 for key in ('inputSha256', 'weightsSha256', 'outputSha256',
            'sourceSha256', 'binarySha256', 'glueSha256', 'receiptSha256', 'workloadSha256')})
    inputs = {f'repo/{path}': 'a' * 64 for path in ('Cargo.lock', 'ruv-swarm/Cargo.lock',
        'neuro-divergent/Cargo.lock', 'cuda-wasm/Cargo.lock', 'opencv-rust/Cargo.lock',
        'config/rune-probe/Cargo.lock', 'package-lock.json', 'ruv-swarm/npm/package-lock.json',
        'cuda-wasm/package-lock.json', 'ruv-swarm/npm/test-npm-install-v106/package-lock.json',
        'ruv-swarm/npm/wasm/ruv_swarm_wasm_bg.wasm', 'ruv-swarm/npm/wasm/ruv_swarm_simd.wasm',
        'ruv-swarm/npm/wasm/ruv_swarm_wasm.js', 'ruv-swarm/npm/wasm/build-info.json')}
    inputs.update({'tool/package.json': 'a' * 64, 'tool/package-lock.json': 'a' * 64})
    baseline_samples = [120, 121, 119, 120, 122] if faster else [100, 101, 99, 100, 102]
    candidate_samples = [100, 101, 99, 100, 102] if faster else [103, 104, 102, 103, 105]
    trials = []
    for name, rows in [('baseline', baseline_rows), ('candidate', 3)]:
        score = dict(variantId=name, primary=1000, regressed=False, noopRate=0, costPerWin=0,
            raw=dict(context, maximumAbsoluteError=0, batchRows=rows, cacheHit=False,
                observationId=name, samplesNs=[100, 101, 99, 100, 102]))
        trials.append(dict(variantId=name, genome=dict(batchRows=rows), score=score))
    return dict(schema='ruv-fann.metaharness/1', status='passed', modelCalls=0, sourceApplied=False,
        options=dict(baselineBatchRows=baseline_rows),
        pins=dict(referenceCommit='b27878cf39d3ada6b2c4d1bcc3e9258a70c6cfd6',
            metaharness=dict(version='0.4.21', npmIntegrity='sha512-AAAA'),
            darwin=dict(version='0.13.0', npmIntegrity='sha512-AAAA')),
        inputsBefore=inputs, inputsAfter=copy.deepcopy(inputs), evaluatorContext=context,
        trialBudget=dict(absoluteLimit=12, worstCaseDispatches=2, dispatchCount=2,
            actualMeasurementCount=2, cacheHits=0, concurrency=1), trials=trials,
        baseline=copy.deepcopy(trials[0]), winner=copy.deepcopy(trials[1]),
        confirmation=dict(verified=True, faster=faster, status='passed' if faster else 'not_improved',
            meanSpeedRatio=sum(baseline_samples) / sum(candidate_samples), ci95=[1.1, 1.3] if faster else [0.9, 1.05],
            maximumAbsoluteError=0, workloadSha256=context['workloadSha256'], minimumImprovement=0.05,
            baselineBatchRows=baseline_rows, candidateBatchRows=3, baselineSamplesNs=baseline_samples,
            candidateSamplesNs=candidate_samples), recommendation=dict(batchRows=3 if faster else baseline_rows))


class MetaHarnessWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = load(ROOT / '.github/workflows/metaharness.yml')
        self.ci = load(ROOT / '.github/workflows/ci.yml')

    def stage(self, name):
        return next(step for step in self.workflow['jobs']['numeric']['steps'] if step.get('id') == name)

    def invalid(self, message):
        self.assertIn(message, validate(self.workflow, self.ci))

    def test_committed_workflow_contract(self):
        self.assertEqual(validate(self.workflow, self.ci), [])

    def test_all_actual_shell_steps_parse(self):
        for step in self.workflow['jobs']['numeric']['steps']:
            if 'run' in step:
                result = subprocess.run(['bash', '-n'], input=step['run'], text=True,
                    capture_output=True, timeout=10, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_reusable_gate_fails(self):
        self.ci['jobs'].pop('metaharness')
        self.invalid('CI does not require MetaHarness')

    def test_rune_gate_is_preserved(self):
        self.ci['jobs'].pop('rune')
        self.invalid('Rune gate was removed')

    def test_incomplete_node_matrix_fails(self):
        self.workflow['jobs']['numeric']['strategy']['matrix']['node'] = ['24']
        self.invalid('required Node matrix missing')

    def test_install_scripts_or_root_graph_changes_fail(self):
        self.stage('install')['run'] = 'npm ci --prefix .'
        self.invalid('installation escapes the private locked tool or enables scripts')

    def test_omitting_development_dependencies_from_audit_fails(self):
        self.stage('audit')['run'] += ' --omit=dev'
        self.invalid('installed development graph is not audited')

    def test_mutable_action_reference_fails(self):
        self.workflow['jobs']['numeric']['steps'][0]['uses'] = 'actions/checkout@main'
        self.invalid('action commit pin missing')

    def test_ignored_optimizer_failure_fails(self):
        self.stage('core')['continue-on-error'] = True
        self.invalid('step failure can be ignored')

    def test_unbounded_or_synthetic_only_smoke_fails(self):
        self.stage('simd')['run'] = 'node --test tools/metaharness/test/*.test.mjs'
        self.invalid('real bounded core/SIMD experiment missing')

    def test_provider_secret_fails(self):
        self.stage('core')['env'] = dict(OPENAI_API_KEY='${{ secrets.OPENAI_API_KEY }}')
        self.invalid('provider or publication secret is passed')

    def run_gate(self, mutate=None, faster=False, baseline_rows=1):
        with tempfile.TemporaryDirectory(prefix='metaharness-ci-test-') as directory:
            for variant in ('core', 'simd'):
                value = receipt(variant, faster, baseline_rows)
                if mutate and variant == 'core':
                    mutate(value)
                output = pathlib.Path(directory) / f'metaharness-{variant}'
                output.mkdir()
                (output / 'receipt.json').write_text(json.dumps(value))
            return subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c',
                self.stage('evidence')['run']], cwd=ROOT, env={**os.environ, 'RUNNER_TEMP': directory},
                text=True, capture_output=True, timeout=10, check=False)

    def test_real_gate_accepts_no_improvement_and_keeps_baseline(self):
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_gate_accepts_independently_confirmed_improvement(self):
        result = self.run_gate(faster=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_gate_accepts_configured_baseline128_retention(self):
        result = self.run_gate(baseline_rows=128)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_gate_rejects_invalid_configured_baseline(self):
        for value in (0, 129, 1.5, '128', None):
            with self.subTest(baseline=value):
                result = self.run_gate(lambda r: r['options'].update(baselineBatchRows=value))
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('invalid configured baseline', result.stderr)

    def test_real_gate_rejects_fabricated_measured_baseline(self):
        result = self.run_gate(lambda r: r['options'].update(baselineBatchRows=128))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('measured baseline differs', result.stderr)

    def test_real_gate_rejects_confirmation_from_another_baseline(self):
        result = self.run_gate(lambda r: r['confirmation'].update(baselineBatchRows=1), baseline_rows=128)
        self.assertNotEqual(result.returncode, 0)

    def test_real_gate_rejects_source_or_supplier_changes(self):
        result = self.run_gate(lambda r: r['inputsAfter'].update({'repo/Cargo.lock': 'b' * 64}))
        self.assertNotEqual(result.returncode, 0)

    def test_real_gate_rejects_missing_legacy_lock_even_with_tool_lock_present(self):
        def remove(value):
            for name in ('inputsBefore', 'inputsAfter'):
                value[name].pop('repo/config/rune-probe/Cargo.lock')
        result = self.run_gate(remove)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('legacy lock not captured', result.stderr)

    def test_real_gate_rejects_trial_overrun(self):
        result = self.run_gate(lambda r: r['trialBudget'].update(worstCaseDispatches=13))
        self.assertNotEqual(result.returncode, 0)

    def test_real_gate_rejects_wrong_wasm_output(self):
        result = self.run_gate(lambda r: r['trials'][1]['score']['raw'].update(maximumAbsoluteError=1))
        self.assertNotEqual(result.returncode, 0)

    def test_real_gate_rejects_missing_fresh_confirmation_samples(self):
        result = self.run_gate(lambda r: r['confirmation'].pop('candidateSamplesNs'))
        self.assertNotEqual(result.returncode, 0)

    def test_real_gate_rejects_changed_trial_workload(self):
        result = self.run_gate(lambda r: r['trials'][1]['score']['raw'].update(totalRows=256))
        self.assertNotEqual(result.returncode, 0)

    def test_real_gate_rejects_different_confirmation_workload(self):
        result = self.run_gate(lambda r: r['confirmation'].update(workloadSha256='b' * 64))
        self.assertNotEqual(result.returncode, 0)

    def test_real_gate_rejects_fabricated_confirmation_ratio(self):
        result = self.run_gate(lambda r: r['confirmation'].update(meanSpeedRatio=10))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('disagrees with samples', result.stderr)

    def test_real_gate_rejects_regressed_winner(self):
        result = self.run_gate(lambda r: r['winner']['score'].update(regressed=True))
        self.assertNotEqual(result.returncode, 0)

    def test_real_gate_rejects_unconfirmed_candidate_recommendation(self):
        result = self.run_gate(lambda r: r.update(recommendation={'batchRows': 3}))
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
