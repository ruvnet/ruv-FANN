"""Reject CI bypasses and malformed integration evidence using actual shell gates.

Synthetic receipts prove rejection behavior only. The workflow separately tests
the real production FANN binaries, RuVector backends and Darwin numeric engine.
"""
import copy
import hashlib
import json
import os
import pathlib
import re
import subprocess
import tempfile
import unittest
from functools import lru_cache

from test_rune_workflows import ROOT, load


LOCKS = ('Cargo.lock', 'ruv-swarm/Cargo.lock', 'neuro-divergent/Cargo.lock',
    'cuda-wasm/Cargo.lock', 'opencv-rust/Cargo.lock', 'config/rune-probe/Cargo.lock',
    'package-lock.json', 'ruv-swarm/npm/package-lock.json', 'cuda-wasm/package-lock.json',
    'ruv-swarm/npm/test-npm-install-v106/package-lock.json', 'tools/metaharness/package-lock.json')
ARTIFACTS = tuple('ruv-swarm/npm/wasm/' + name for name in
    ('ruv_swarm_wasm_bg.wasm', 'ruv_swarm_simd.wasm', 'ruv_swarm_wasm.js', 'build-info.json'))


@lru_cache(maxsize=1)
def input_hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in LOCKS + ARTIFACTS}


def validate(workflow, ci):
    errors = []

    def require(condition, message):
        if not condition:
            errors.append(message)

    events = workflow.get('on', {})
    require(all(name in events for name in ('workflow_call', 'workflow_dispatch')), 'reusable/manual entry missing')
    for name in ('push', 'pull_request'):
        event = events.get(name, {})
        require(event.get('branches') == ['main', 'develop'], 'standalone branch coverage missing')
        require(all(path in event.get('paths', []) for path in ('integrations/ruv-fann/**',
            'ruv-swarm/npm/wasm/**', 'src/**', '.github/workflows/ruv-fann-integrations.yml')),
            'standalone integration/source coverage missing')
    require(workflow.get('permissions') == {'contents': 'read'}, 'least permissions missing')
    jobs = ci.get('jobs', {})
    caller = jobs.get('ruv-fann-integrations', {})
    require(caller.get('uses') == './.github/workflows/ruv-fann-integrations.yml', 'CI integration gate missing')
    require(not caller.get('if') and not caller.get('continue-on-error') and not caller.get('secrets'),
        'CI integration gate can skip, ignore failure or receive secrets')
    for name in ('rune', 'metaharness'):
        require(jobs.get(name, {}).get('uses') == f'./.github/workflows/{name}.yml', f'prior {name} gate removed')
    job = workflow.get('jobs', {}).get('integrations', {})
    require(1 <= job.get('timeout-minutes', 0) <= 15, 'bounded job deadline missing')
    require(not job.get('if') and not job.get('continue-on-error'), 'integration job is optional')
    require(job.get('strategy', {}).get('matrix', {}).get('node') == ['22.13.0', '24'], 'Node compatibility matrix missing')
    require(job.get('defaults', {}).get('run', {}).get('shell') == 'bash', 'pipefail shell missing')
    require(not job.get('permissions') or job['permissions'] == {'contents': 'read'}, 'job permissions escalate')
    steps = job.get('steps', [])
    stages = {step['id']: step for step in steps if 'id' in step}
    required = ('snapshot', 'install', 'audit', 'types', 'unit', 'e2e', 'package', 'evidence', 'unchanged')
    require(all(name in stages for name in required), 'required execution/evidence stage missing')
    if all(name in stages for name in required):
        order = [step.get('id') for step in steps]
        require([order.index(name) for name in required] == sorted(order.index(name) for name in required),
            'snapshot/install/types/tests/package/evidence order changed')
        commands = {
            'install': 'npm ci --ignore-scripts --omit=optional --prefix integrations/ruv-fann',
            'audit': 'npm audit --audit-level=low --omit=optional --prefix integrations/ruv-fann',
            'types': 'npm run build --prefix integrations/ruv-fann',
            'unit': 'npm test --prefix integrations/ruv-fann',
            'e2e': 'npm run test:e2e --prefix integrations/ruv-fann',
            'package': 'node integrations/ruv-fann/scripts/check-package.mjs --output "$RUNNER_TEMP/ruv-fann-integration-package"',
        }
        for name, command in commands.items():
            run = stages[name].get('run', '')
            require(command in run and '| tee "$RUNNER_TEMP/ruv-fann-integrations/' in run,
                f'{name} command or external failure log missing')
            require(not re.search(r'\|\||;|set\s+\+e|\bexit\s+0\b', run), 'required command failure can be masked')
        require('--omit=dev' not in stages['audit']['run'] and '--production' not in stages['audit']['run'],
            'development dependency audit omitted')
        require(stages['e2e'].get('env', {}).get('RUV_FANN_INTEGRATION_EVIDENCE') ==
            '${{ runner.temp }}/ruv-fann-source-evidence.json', 'external source evidence missing')
        require(all(name in stages['snapshot'].get('run', '') for name in LOCKS), 'existing lock snapshot incomplete')
        evidence = stages['evidence'].get('run', '')
        require(all(name in evidence for name in ('integration-e2e/1', 'integration-package/1', 'sourceValidation',
            'actualDarwin', 'actualRuVector', 'actualNeuralInference', 'source(packaged.sourceEvidence)',
            'outputMaximumAbsoluteError', 'retrievalActualId', 'queryDistance', 'darwin', 'inputsAfter',
            'batch-tuning/1', 'worstCaseDispatches', 'baselineSamplesNs', 'candidateSamplesNs', 'not_improved', 'temporaryArchiveRemoved')),
            'actual source/package evidence gate missing')
        require('git diff --exit-code' in stages['unchanged'].get('run', '') and
            'git diff --cached --exit-code' in stages['unchanged'].get('run', '') and
            'git status --porcelain --untracked-files=all' in stages['unchanged'].get('run', ''),
            'source immutability guard missing')
    uploads = []
    for step in steps:
        require(not step.get('continue-on-error'), 'step failure ignored')
        require(not step.get('if') or step['if'] == 'always()', 'required step can silently skip')
        require('${{ secrets.' not in str(step), 'provider/publication secret supplied')
        action = step.get('uses', '')
        if action:
            require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[a-f0-9]{40}', action) is not None,
                'mutable action reference')
            require('cache' not in action.lower(), 'dependency cache trusted')
            if action.startswith('actions/checkout@'):
                require(step.get('with', {}).get('persist-credentials') is False, 'checkout credentials retained')
            if action.startswith('actions/setup-node@'):
                require(step.get('with', {}).get('package-manager-cache') is False, 'automatic package cache enabled')
            if action.startswith('actions/upload-artifact@'):
                uploads.append(step)
        require(not re.search(r'\bnpm\s+(?:publish|audit\s+fix)\b|\bgit\s+push\b', step.get('run', '')),
            'source/publication mutation enabled')
    require(len(uploads) == 1, 'evidence upload missing')
    if len(uploads) == 1:
        upload = uploads[0]
        require(upload.get('if') == 'always()', 'failure artifacts discarded')
        options = upload.get('with', {})
        require(options.get('if-no-files-found') == 'error', 'missing artifacts accepted')
        require(all(path in options.get('path', '') for path in ('ruv-fann-integrations/',
            'ruv-fann-source-evidence.json', 'ruv-fann-integration-package/*.json')), 'source/package logs or receipts omitted')
    return errors


def tuning_receipt(faster=False, baseline=128):
    context = dict(inputSize=3, outputSize=2, totalRows=131, precision='f64', modelId='fixture-label',
        **{name: 'a' * 64 for name in ('inputSha256', 'outputSha256', 'workloadSha256')})
    trials = []
    for name, rows in [('baseline', baseline), ('candidate', 17)]:
        trials.append(dict(variantId=name, genome=dict(batchRows=rows), score=dict(variantId=name,
            primary=1000, regressed=False, noopRate=0, costPerWin=0, raw=dict(
                **{key: context[key] for key in ('inputSha256', 'outputSha256', 'workloadSha256')},
                samplesNs=[10] * 5, maximumAbsoluteError=0, batchRows=rows, cacheHit=False, observationId=name))))
    return dict(schema='ruv-fann.batch-tuning/1', status='passed', modelId='fixture-label', modelIdScope='label-only',
        modelCalls=0, sourceApplied=False, temporaryArchiveRemoved=True,
        engine=dict(package='@metaharness/darwin', version='0.13.0', algorithm='evolveNumeric'),
        limits=dict(maxNumericBufferBytes=64 * 1024 * 1024, numericBufferBytes=14568,
            encoderStagingBytes=5120, cooperativeDeadline=True),
        options=dict(baselineBatchRows=baseline, generations=1, children=1, samples=5, seed=7, timeoutMs=30000),
        context=context, trialBudget=dict(absoluteLimit=12, worstCaseDispatches=2, dispatchCount=2,
            actualMeasurementCount=2, cacheHits=0, concurrency=1), trials=trials,
        baseline=copy.deepcopy(trials[0]), winner=copy.deepcopy(trials[1]),
        confirmation=dict(verified=True, status='passed' if faster else 'not_improved', faster=faster,
            meanSpeedRatio=2 if faster else 1, ci95=[1.8, 2.2] if faster else [.9, 1.1],
            baselineBatchRows=baseline, candidateBatchRows=17, minimumImprovement=.05,
            baselineSamplesNs=[20] * 5, candidateSamplesNs=[10 if faster else 20] * 5,
            maximumAbsoluteError=0, workloadSha256=context['workloadSha256']),
        recommendation=dict(batchRows=17 if faster else baseline))


def source_receipt(faster=False):
    artifacts = {name: digest for name, digest in input_hashes().items() if name in ARTIFACTS}
    cases = [dict(variant=variant, fixture=fixture, indexType=index, records=33,
        outputMaximumAbsoluteError=0, retrievalExpectedId='row-7', retrievalActualId='row-7',
        queryDistance=0, dimensions=dict(input=3, output=2), modelJsonSha256=('a' if fixture == 'dense' else 'b') * 64,
        binarySha256=artifacts['ruv-swarm/npm/wasm/' + ('ruv_swarm_wasm_bg.wasm' if variant == 'core' else 'ruv_swarm_simd.wasm')])
        for variant in ('core', 'simd') for fixture in ('dense', 'sparse') for index in ('flat', 'hnsw')]
    return dict(schema='ruv-fann.integration-e2e/1', status='passed', sourceValidation='passed', modelCalls=0,
        actualNeuralInference=True, actualRuVector=True, actualDarwin=True, coreAndSimd=True, unchanged=True,
        inputsBefore=copy.deepcopy(artifacts), inputsAfter=copy.deepcopy(artifacts),
        cases=cases, darwin=[dict(variant=variant, receipt=tuning_receipt(faster)) for variant in ('core', 'simd')])


class IntegrationsWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = load(ROOT / '.github/workflows/ruv-fann-integrations.yml')
        self.ci = load(ROOT / '.github/workflows/ci.yml')

    def stage(self, name):
        return next(step for step in self.workflow['jobs']['integrations']['steps'] if step.get('id') == name)

    def invalid(self, message):
        self.assertIn(message, validate(self.workflow, self.ci))

    def test_actual_workflow_contract(self):
        self.assertEqual(validate(self.workflow, self.ci), [])

    def test_all_actual_shell_programs_parse(self):
        for step in self.workflow['jobs']['integrations']['steps']:
            if 'run' in step:
                result = subprocess.run(['bash', '-n'], input=step['run'], text=True,
                    capture_output=True, timeout=10, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_required_pipelines_propagate_a_failed_real_command(self):
        with tempfile.TemporaryDirectory(prefix='integrations-ci-fail-') as directory:
            temporary = pathlib.Path(directory)
            (temporary / 'ruv-fann-integrations').mkdir()
            for command in ('npm', 'node'):
                stub = temporary / command
                stub.write_text('#!/bin/sh\nexit 17\n')
                stub.chmod(0o700)
            for name in ('install', 'audit', 'types', 'unit', 'e2e', 'package'):
                result = subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', self.stage(name)['run']],
                    env={**os.environ, 'RUNNER_TEMP': directory, 'PATH': directory + os.pathsep + os.environ['PATH']},
                    text=True, capture_output=True, timeout=10, check=False)
                self.assertEqual(result.returncode, 17, (name, result.stderr))

    def test_skippable_or_ignored_stages_fail(self):
        self.stage('e2e')['continue-on-error'] = True
        self.invalid('step failure ignored')

    def test_missing_package_consumer_stage_fails(self):
        self.workflow['jobs']['integrations']['steps'].remove(self.stage('package'))
        self.invalid('required execution/evidence stage missing')

    def test_masked_pipeline_failure_fails(self):
        self.stage('package')['run'] += ' || true'
        self.invalid('required command failure can be masked')

    def test_unsafe_install_or_missing_development_audit_fails(self):
        self.stage('install')['run'] = 'npm ci --prefix .'
        self.invalid('install command or external failure log missing')
        self.stage('audit')['run'] += ' --omit=dev'
        self.invalid('development dependency audit omitted')

    def test_prior_gates_must_remain(self):
        for name in ('rune', 'metaharness'):
            self.ci['jobs'].pop(name)
            self.invalid(f'prior {name} gate removed')

    def test_required_node_versions_and_action_pins(self):
        self.workflow['jobs']['integrations']['strategy']['matrix']['node'] = ['24']
        self.invalid('Node compatibility matrix missing')
        self.workflow['jobs']['integrations']['steps'][0]['uses'] = 'actions/checkout@main'
        self.invalid('mutable action reference')

    def test_provider_secrets_and_missing_uploads_fail(self):
        self.stage('e2e')['env']['OPENAI_API_KEY'] = '${{ secrets.OPENAI_API_KEY }}'
        self.invalid('provider/publication secret supplied')
        self.workflow['jobs']['integrations']['steps'].pop()
        self.invalid('evidence upload missing')

    def run_gate(self, mutate=None, package_mutate=None, snapshot_mutate=None):
        with tempfile.TemporaryDirectory(prefix='integrations-ci-evidence-') as directory:
            temporary = pathlib.Path(directory)
            logs = temporary / 'ruv-fann-integrations'
            logs.mkdir()
            before = copy.deepcopy(input_hashes())
            if snapshot_mutate:
                snapshot_mutate(before)
            (logs / 'inputs-before.json').write_text(json.dumps(before))
            source = source_receipt()
            if mutate:
                mutate(source)
            (temporary / 'ruv-fann-source-evidence.json').write_text(json.dumps(source))
            packaged = dict(schema='ruv-fann.integration-package/1', status='passed', actualConsumers=1,
                actualNeuralInference=True, actualRuVector=True, actualDarwin=True, coreAndSimd=True,
                sourceEvidence=source_receipt())
            if package_mutate:
                package_mutate(packaged)
            package = temporary / 'ruv-fann-integration-package'
            package.mkdir()
            (package / 'receipt.json').write_text(json.dumps(packaged))
            return subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', self.stage('evidence')['run']],
                cwd=ROOT, env={**os.environ, 'RUNNER_TEMP': directory}, text=True,
                capture_output=True, timeout=10, check=False)

    def test_actual_gate_accepts_complete_source_and_consumer_evidence(self):
        result = self.run_gate()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_gate_rejects_missing_mode_or_wrong_nearest_record(self):
        for change in (lambda r: r['cases'].pop(), lambda r: r['cases'][0].update(retrievalActualId='wrong')):
            result = self.run_gate(change)
            self.assertNotEqual(result.returncode, 0)

    def test_actual_gate_rejects_duplicate_mode_or_inference_error(self):
        for change in (lambda r: r['cases'].__setitem__(0, copy.deepcopy(r['cases'][1])),
            lambda r: r['cases'][0].update(outputMaximumAbsoluteError=1)):
            result = self.run_gate(change)
            self.assertNotEqual(result.returncode, 0)

    def test_actual_gate_rejects_changed_or_stale_source(self):
        for change in (lambda r: r.update(sourceValidation='failed'),
            lambda r: r['inputsAfter'].update({'0': 'b' * 64})):
            result = self.run_gate(change)
            self.assertNotEqual(result.returncode, 0)

    def test_actual_gate_rejects_missing_darwin_or_model_calls(self):
        for change in (lambda r: r['darwin'].pop(), lambda r: r['darwin'][0]['receipt'].update(modelCalls=1)):
            result = self.run_gate(change)
            self.assertNotEqual(result.returncode, 0)

    def test_actual_gate_accepts_confirmed_gain_without_requiring_one(self):
        result = self.run_gate(lambda r: r.update(darwin=source_receipt(True)['darwin']))
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_gate_rejects_tuner_budget_overrun_or_missing_confirmation(self):
        for change in (lambda r: r['darwin'][0]['receipt']['trialBudget'].update(dispatchCount=13),
            lambda r: r['darwin'][0]['receipt']['confirmation'].pop('candidateSamplesNs')):
            result = self.run_gate(change)
            self.assertNotEqual(result.returncode, 0)

    def test_actual_gate_rejects_wrong_baseline_or_unconfirmed_recommendation(self):
        for change in (lambda r: r['darwin'][0]['receipt']['options'].update(baselineBatchRows=1),
            lambda r: r['darwin'][0]['receipt'].update(recommendation=dict(batchRows=17))):
            result = self.run_gate(change)
            self.assertNotEqual(result.returncode, 0)

    def test_actual_gate_rejects_unremoved_archive_or_changed_tuner_workload(self):
        for change in (lambda r: r['darwin'][0]['receipt'].update(temporaryArchiveRemoved=False),
            lambda r: r['darwin'][0]['receipt']['trials'][0]['score']['raw'].update(outputSha256='b' * 64)):
            result = self.run_gate(change)
            self.assertNotEqual(result.returncode, 0)

    def test_actual_gate_rejects_wrong_engine_or_numeric_buffer_overrun(self):
        for change in (lambda r: r['darwin'][0]['receipt']['engine'].update(version='0.12.0'),
            lambda r: r['darwin'][0]['receipt']['limits'].update(numericBufferBytes=64 * 1024 * 1024 + 1)):
            result = self.run_gate(change)
            self.assertNotEqual(result.returncode, 0)

    def test_actual_gate_requires_every_previous_lock_including_private_tool(self):
        result = self.run_gate(snapshot_mutate=lambda before: before.pop('tools/metaharness/package-lock.json'))
        self.assertNotEqual(result.returncode, 0)

    def test_actual_gate_rejects_no_real_extracted_consumer(self):
        result = self.run_gate(package_mutate=lambda r: r.update(actualConsumers=0))
        self.assertNotEqual(result.returncode, 0)

    def test_actual_gate_rejects_wrong_extracted_consumer_results(self):
        result = self.run_gate(package_mutate=lambda r: r['sourceEvidence']['cases'][0].update(retrievalActualId='wrong'))
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
