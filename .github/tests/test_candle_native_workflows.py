"""Failure-propagation and genuine-evidence contracts for Candle/native CI.

Synthetic receipts below only exercise the real workflow's rejection paths;
the workflow separately builds and executes actual Rust, Candle, WASM and Node.
"""
import copy
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest

from test_rune_workflows import ROOT, load

DOWNLOAD = 'actions/download-artifact@37930b1c2abaa49bbe596cd826c3c89aef350131'
CORE_TESTS = ('prepared_dense', 'prepared_dense_kernel', 'prepared_batch_equivalence',
    'prepared_batch_work', 'prepared_allocations', 'autograd', 'autograd_allocations')
SOURCE_NAMES = ('Cargo.lock', 'bindings/napi/Cargo.lock', 'src/prepared_dense.rs',
    'src/autograd/pass.rs', 'src/autograd/optim.rs', 'bindings/napi/src/gateway.rs', 'bindings/napi/src/training.rs')
JOB_ENVIRONMENTS = {
    'core': {'CARGO_TARGET_DIR': 'candle-native-core-target'},
    'candle': {'CARGO_TARGET_DIR': 'candle-native-oracle-target'},
    'native': {'RUV_FANN_CANDLE_FIXTURE': 'candle-native-oracle/candle-f64-fixture.json'},
    'browser': {'NODE_PATH': 'candle-native-browser-tools/node_modules',
        'PLAYWRIGHT_BROWSERS_PATH': 'candle-native-browser-binaries',
        'RUV_FANN_CANDLE_FIXTURE': 'candle-native-oracle/candle-f64-fixture.json'},
}
BROWSER_VERSION = '151.0.7922.34'
BROWSER_SOURCE = 'playwright-core/browsers.json'


def validate(workflow, ci):
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)

    events = workflow.get('on', {})
    require(all(key in events for key in ('workflow_call', 'workflow_dispatch')), 'reusable/manual entry missing')
    for event in ('push', 'pull_request'):
        value = events.get(event, {})
        require(value.get('branches') == ['main', 'develop'], 'standalone branches missing')
        require(all(name in value.get('paths', []) for name in ('bindings/napi/**', 'integrations/candle/**',
            'src/**', 'tests/autograd.rs', 'tests/autograd_allocations.rs', 'tests/autograd/**',
            '.github/workflows/candle-native.yml')), 'standalone inference/training paths missing')
    require(workflow.get('permissions') == {'contents': 'read'}, 'least permissions missing')
    jobs = ci.get('jobs', {})
    gate = jobs.get('candle-native', {})
    require(gate.get('uses') == './.github/workflows/candle-native.yml', 'main CI native acceptance missing')
    require(not gate.get('if') and not gate.get('continue-on-error') and not gate.get('secrets'), 'main CI native acceptance can bypass or receive secrets')
    for name, source in (('wasm-targets', 'wasm-build'), ('rune', 'rune'), ('metaharness', 'metaharness'),
        ('ruv-fann-integrations', 'ruv-fann-integrations')):
        require(jobs.get(name, {}).get('uses') == f'./.github/workflows/{source}.yml', f'prior {name} gate removed')
    tasks = workflow.get('jobs', {})
    require(all(name in tasks for name in ('core', 'candle', 'native', 'browser')), 'required numerical/runtime job missing')
    for name, job in tasks.items():
        require(not re.search(r'\b(?:runner|env|steps|job)\s*(?:\.|\[)', str(job.get('env', {}))),
            'job environment uses unavailable context')
        require(1 <= job.get('timeout-minutes', 0) <= 30, 'job deadline missing')
        require(not job.get('if') and not job.get('continue-on-error'), 'required job can skip or ignore failure')
        require(job.get('defaults', {}).get('run', {}).get('shell') == 'bash', 'pipefail shell missing')
        require(not job.get('permissions') or job['permissions'] == {'contents': 'read'}, 'job permissions escalate')
        steps = job.get('steps', [])
        environment = next((step.get('run', '') for step in steps if step.get('id') == 'environment'), '')
        require('"$GITHUB_ENV"' in environment and all(key + '=' in environment and suffix in environment
            for key, suffix in JOB_ENVIRONMENTS.get(name, {}).items()), 'runner environment initialization missing')
        uploads = []
        for step in steps:
            require(not step.get('continue-on-error'), 'required step failure ignored')
            require(not step.get('if') or step['if'] == 'always()', 'required step can silently skip')
            require('${{ secrets.' not in str(step), 'provider/publication secret passed')
            action = step.get('uses', '')
            if action:
                require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[a-f0-9]{40}', action) is not None, 'mutable action reference')
                require('cache' not in action.lower(), 'unverified dependency cache trusted')
                if action.startswith('actions/checkout@'):
                    require(step.get('with', {}).get('persist-credentials') is False, 'checkout credentials retained')
                if action.startswith('actions/setup-node@'):
                    require(step.get('with', {}).get('package-manager-cache') is False, 'automatic npm cache enabled')
                if action.startswith('actions/upload-artifact@'):
                    uploads.append(step)
            program = step.get('run', '')
            require(not re.search(r'\|\|\s*(?:true|echo)|set\s+\+e|\bexit\s+0\b', program), 'command failure masked')
            require(not re.search(r'\bnpm\s+(?:publish|audit\s+fix)\b|\bgit\s+(?:push|merge)\b', program), 'source/publication mutation enabled')
        require(len(uploads) == 1 and uploads[0].get('if') == 'always()' and
            uploads[0].get('with', {}).get('if-no-files-found') == 'error', 'failure evidence upload missing')
        unchanged = next((step for step in steps if step.get('id') == 'unchanged'), {})
        require(all(command in unchanged.get('run', '') for command in ('git diff --exit-code',
            'git diff --cached --exit-code', 'git status --porcelain --untracked-files=all')), 'source immutability guard missing')

    def stages(name):
        return {step['id']: step for step in tasks.get(name, {}).get('steps', []) if 'id' in step}
    core = stages('core')
    require(all(f'--test {name}' in core.get('dense', {}).get('run', '') for name in CORE_TESTS) and
        'cargo +1.88.0 test --locked --lib' in core.get('dense', {}).get('run', ''), 'canonical core/autograd acceptance incomplete')
    require('cargo +1.81.0 check --locked --lib --no-default-features' in core.get('msrv-minimal', {}).get('run', '') and
        '--features serde' in core.get('msrv-serde', {}).get('run', ''), 'core MSRV feature checks missing')
    candle = stages('candle')
    require('cargo +1.99.1 test --locked --manifest-path integrations/candle/Cargo.toml' in candle.get('oracle', {}).get('run', ''), 'actual pinned Candle oracle missing')
    require('cargo +1.99.1 run --quiet --locked' in candle.get('fixture', {}).get('run', '') and
        '--example export_fixture' in candle.get('fixture', {}).get('run', ''), 'actual Candle fixture export missing')
    native = stages('native')
    needed = ('install', 'sdk-install', 'audit', 'fetch', 'build', 'native-tests', 'wasm-tests', 'package', 'wasm-package', 'tuner', 'provenance', 'evidence', 'unchanged')
    require(all(name in native for name in needed), 'native build/source/package/evidence stage missing')
    if all(name in native for name in needed):
        order = [step.get('id') for step in tasks['native']['steps']]
        require([order.index(name) for name in needed] == sorted(order.index(name) for name in needed), 'native correctness and quiet tuning order changed')
        commands = {'install': 'npm ci --ignore-scripts --omit=optional --prefix bindings/napi',
            'sdk-install': 'npm ci --ignore-scripts --omit=optional --prefix integrations/ruv-fann',
            'fetch': 'cargo +1.88.0 fetch --locked --manifest-path bindings/napi/Cargo.toml',
            'build': 'node bindings/napi/scripts/build.mjs --target-dir "$RUNNER_TEMP/candle-native-napi-target"',
            'native-tests': 'node --test bindings/napi/test/*.test.mjs',
            'wasm-tests': 'node --test ruv-swarm/npm/test/wasm-dense-import.test.js ruv-swarm/npm/test/wasm-autograd.test.js',
            'package': 'node bindings/napi/test/packed-consumer.mjs --output "$RUNNER_TEMP/candle-native-package"',
            'wasm-package': 'node ruv-swarm/npm/test/check-training-package.mjs --output "$RUNNER_TEMP/candle-native-napi/wasm-package"',
            'tuner': 'node bindings/napi/test/tuner-compat.mjs --output "$RUNNER_TEMP/candle-native-napi/tuner.json"'}
        for name, command in commands.items():
            require(command in native[name].get('run', '') and '| tee ' in native[name].get('run', ''), f'{name} actual bounded command/log missing')
        audit = native['audit'].get('run', '')
        require(all(f'npm audit --audit-level=low --omit=optional --prefix {prefix}' in audit for prefix in ('bindings/napi', 'integrations/ruv-fann')) and
            '--omit=dev' not in audit and '--production' not in audit, 'installed development dependencies not audited')
        require(native['build'].get('env', {}).get('CARGO_NET_OFFLINE') == 'true', 'native build is not offline after explicit fetch')
    require(tasks.get('native', {}).get('strategy', {}).get('matrix', {}).get('node') == ['22.13.0', '24'], 'Node compatibility matrix missing')
    require(tasks.get('native', {}).get('needs') == 'candle', 'native does not depend on actual Candle fixture')
    require('RUV_FANN_CANDLE_FIXTURE=' in native.get('environment', {}).get('run', ''), 'actual Candle fixture handoff missing')
    require(any(step.get('uses') == DOWNLOAD and step.get('with', {}).get('name') == 'candle-native-oracle'
        for step in tasks.get('native', {}).get('steps', [])), 'pinned fixture download missing')
    browser = stages('browser')
    require(tasks.get('browser', {}).get('needs') == 'candle', 'browser does not require actual Candle fixture')
    require(all(name in browser for name in ('sdk-install', 'audit', 'setup', 'training', 'evidence', 'unchanged')), 'mandatory browser acceptance incomplete')
    require('playwright@1.62.1' in browser.get('setup', {}).get('run', '') and
        'install --with-deps chromium' in browser.get('setup', {}).get('run', '') and
        'CHROMIUM_PATH=' in browser.get('setup', {}).get('run', ''), 'pinned actual browser setup missing')
    require('node ruv-swarm/npm/test/browser-training.cjs --output "$RUNNER_TEMP/candle-native-browser/browser.json"' in browser.get('training', {}).get('run', ''), 'actual browser training not executed')
    require('npm ci --ignore-scripts --omit=optional --prefix integrations/ruv-fann' in browser.get('sdk-install', {}).get('run', ''), 'browser alters a root dependency graph')
    require('npm audit --audit-level=low' in browser.get('audit', {}).get('run', '') and
        '--omit=dev' not in browser.get('audit', {}).get('run', ''), 'browser development dependencies not audited')
    return errors


def fixture():
    return dict(oracle='candle-core/candle-nn 0.11.1 CPU f64', parameter_gradients=[.1, .2],
        input_gradients=[.3], sgd_expected_parameters=[.4, .5], adamw_expected_parameters=[.6, .7],
        sgd_steps=[{}] * 3, adamw_steps=[{}] * 3)


def tuning(faster=False):
    trials = []
    for name, rows in (('baseline', 128), ('candidate', 17)):
        trials.append(dict(variantId=name, genome=dict(batchRows=rows), score=dict(primary=1000,
            regressed=False, raw=dict(maximumAbsoluteError=0, workloadSha256='a' * 64, samplesNs=[10] * 5))))
    return dict(schema='ruv-fann.batch-tuning/1', status='passed', modelCalls=0, sourceApplied=False,
        temporaryArchiveRemoved=True, engine=dict(package='@metaharness/darwin', version='0.13.0', algorithm='evolveNumeric'),
        options=dict(baselineBatchRows=128, samples=5, generations=1, children=1), context=dict(workloadSha256='a' * 64),
        trialBudget=dict(absoluteLimit=12, concurrency=1, worstCaseDispatches=2, dispatchCount=2,
            actualMeasurementCount=2, cacheHits=0), trials=trials, baseline=copy.deepcopy(trials[0]), winner=copy.deepcopy(trials[1]),
        confirmation=dict(verified=True, maximumAbsoluteError=0, baselineBatchRows=128, candidateBatchRows=17,
            workloadSha256='a' * 64, minimumImprovement=.05, baselineSamplesNs=[20] * 5,
            candidateSamplesNs=[10 if faster else 20] * 5, meanSpeedRatio=2 if faster else 1,
            ci95=[1.8, 2.2] if faster else [.9, 1.1], faster=faster, status='passed' if faster else 'not_improved'),
        recommendation=dict(batchRows=17 if faster else 128))


class CandleNativeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = load(ROOT / '.github/workflows/candle-native.yml')
        self.ci = load(ROOT / '.github/workflows/ci.yml')

    def stage(self, job, name):
        return next(step for step in self.workflow['jobs'][job]['steps'] if step.get('id') == name)

    def invalid(self, message):
        self.assertIn(message, validate(self.workflow, self.ci))

    def test_actual_workflow_contract(self):
        self.assertEqual(validate(self.workflow, self.ci), [])

    def test_actual_pinned_dependency_manifests(self):
        candle = tomllib.loads((ROOT / 'integrations/candle/Cargo.toml').read_text())
        for name in ('candle-core', 'candle-nn'):
            self.assertEqual(candle['dependencies'][name]['version'], '=0.11.1')
            self.assertFalse(candle['dependencies'][name]['default-features'])
        native = tomllib.loads((ROOT / 'bindings/napi/Cargo.toml').read_text())
        version = lambda dependency: dependency.get('version') if isinstance(dependency, dict) else dependency
        self.assertEqual(native['package']['rust-version'], '1.88')
        self.assertEqual(native['dependencies']['napi']['version'], '=3.16.1')
        self.assertEqual(version(native['dependencies']['napi-derive']), '=3.7.0')
        self.assertEqual(version(native['build-dependencies']['napi-build']), '=2.6.0')
        self.assertEqual(native['dependencies']['napi']['features'], ['napi8'])

    def test_all_actual_shell_programs_parse(self):
        for job in self.workflow['jobs'].values():
            for step in job['steps']:
                if 'run' in step:
                    result = subprocess.run(['bash', '-n'], input=step['run'], text=True, capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_required_pipeline_failures_propagate(self):
        with tempfile.TemporaryDirectory(prefix='candle-native-ci-fail-') as directory:
            temporary = pathlib.Path(directory)
            for name in ('candle-native-core', 'candle-native-oracle', 'candle-native-napi', 'candle-native-browser'):
                (temporary / name).mkdir()
            for name in ('cargo', 'npm', 'node'):
                program = temporary / name
                program.write_text('#!/bin/sh\nexit 19\n'); program.chmod(0o700)
            for job, names in (('core', ('dense', 'msrv-minimal', 'msrv-serde')), ('candle', ('oracle', 'fixture')),
                ('native', ('install', 'sdk-install', 'audit', 'fetch', 'build', 'native-tests', 'wasm-tests', 'package', 'wasm-package', 'tuner')),
                ('browser', ('sdk-install', 'audit', 'setup', 'training'))):
                for name in names:
                    result = subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', self.stage(job, name)['run']],
                        env={**os.environ, 'RUNNER_TEMP': directory, 'PATH': directory + os.pathsep + os.environ['PATH']},
                        text=True, capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 19, (job, name, result.stderr))

    def test_training_and_old_acceptance_gates_cannot_disappear(self):
        self.stage('core', 'dense')['run'] = self.stage('core', 'dense')['run'].replace('--test autograd_allocations', '')
        self.invalid('canonical core/autograd acceptance incomplete')
        for name in ('wasm-targets', 'rune'):
            self.ci['jobs'].pop(name)
            self.invalid(f'prior {name} gate removed')

    def test_mutable_or_skippable_or_networked_build_fails(self):
        self.stage('native', 'build')['env']['CARGO_NET_OFFLINE'] = 'false'
        self.invalid('native build is not offline after explicit fetch')
        self.stage('native', 'package')['continue-on-error'] = True
        self.invalid('required step failure ignored')
        self.workflow['jobs']['native']['steps'][0]['uses'] = 'actions/checkout@main'
        self.invalid('mutable action reference')

    def test_missing_matrix_fixture_or_development_audit_fails(self):
        self.workflow['jobs']['native']['strategy']['matrix']['node'] = ['24']
        self.invalid('Node compatibility matrix missing')
        self.stage('native', 'environment')['run'] = self.stage('native', 'environment')['run'].replace('RUV_FANN_CANDLE_FIXTURE=', 'MISSING_FIXTURE=')
        self.invalid('actual Candle fixture handoff missing')
        self.stage('native', 'audit')['run'] += ' --omit=dev'
        self.invalid('installed development dependencies not audited')

    def test_runner_and_step_contexts_are_rejected_at_job_environment_scope(self):
        for context in ('runner', 'env', 'steps', 'job'):
            self.workflow['jobs']['native']['env'] = {'BROKEN': '${{ ' + context + '.temp }}'}
            self.invalid('job environment uses unavailable context')

    def test_environment_initializers_materialize_external_paths(self):
        with tempfile.TemporaryDirectory(prefix='candle native environment ') as directory:
            root = pathlib.Path(directory)
            for job, expected in JOB_ENVIRONMENTS.items():
                destination = root / (job + '.env')
                result = subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', self.stage(job, 'environment')['run']],
                    env={**os.environ, 'RUNNER_TEMP': directory, 'GITHUB_ENV': str(destination)},
                    text=True, capture_output=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                values = dict(line.split('=', 1) for line in destination.read_text().splitlines())
                self.assertEqual(values, {key: str(root / suffix) for key, suffix in expected.items()})

    def discovery(self, metadata=None, package_version='1.62.1', core_version='1.62.1'):
        program = re.search(r"node <<'JS'\n(.*?)\nJS", self.stage('browser', 'setup')['run'], re.S).group(1)
        with tempfile.TemporaryDirectory(prefix='chromium environment ') as directory:
            root = pathlib.Path(directory)
            module = root / 'node_modules/playwright'
            module.mkdir(parents=True)
            (module / 'package.json').write_text(json.dumps(dict(version=package_version)))
            (module / 'index.js').write_text('module.exports={chromium:{executablePath:()=>process.env.RUV_FANN_TEST_CHROMIUM}};\n')
            core = root / 'node_modules/playwright-core'
            core.mkdir()
            (core / 'package.json').write_text(json.dumps(dict(version=core_version)))
            if metadata is None:
                metadata = dict(browsers=[dict(name='chromium', browserVersion=BROWSER_VERSION),
                    dict(name='chromium-headless-shell', browserVersion=BROWSER_VERSION)])
            (core / 'browsers.json').write_text(json.dumps(metadata))
            executable = root / 'browser bin/chromium'
            executable.parent.mkdir()
            executable.write_text('#!/bin/sh\nexit 0\n')
            executable.chmod(0o700)
            destination = root / 'github.env'
            result = subprocess.run(['node'], input=program, cwd=root, text=True, capture_output=True, timeout=10,
                env={**os.environ, 'NODE_PATH': str(root / 'node_modules'), 'GITHUB_ENV': str(destination),
                    'RUV_FANN_TEST_CHROMIUM': str(executable)})
            data = destination.read_bytes() if destination.exists() else b''
            if result.returncode == 0:
                values = dict(line.split('=', 1) for line in data.decode().splitlines())
                self.assertTrue(pathlib.Path(values['CHROMIUM_PATH']).is_file())
            return result, data, {'CHROMIUM_PATH': str(executable), 'EXPECTED_CHROMIUM_VERSION': BROWSER_VERSION,
                'CHROMIUM_VERSION_SOURCE': BROWSER_SOURCE}

    def test_browser_discovery_exports_metadata_and_real_newlines_with_space_safe_path(self):
        result, data, expected = self.discovery()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(data.endswith(bytes([10])))
        self.assertEqual(dict(line.split('=', 1) for line in data.decode().splitlines()), expected)

    def test_browser_discovery_rejects_missing_duplicate_malformed_metadata_and_wrong_package(self):
        for entries in ([], [dict(name='chromium')] * 2, [dict(name='chromium')],
            [dict(name='chromium', browserVersion='151.0')]):
            result, data, _ = self.discovery(dict(browsers=entries))
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(data, b'')
        for kwargs in ({'package_version': '1.62.0'}, {'core_version': '1.62.0'}):
            result, data, _ = self.discovery(**kwargs)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(data, b'')

    def test_browser_proof_cannot_be_removed_or_replaced_with_synthetic_test(self):
        self.stage('browser', 'training')['run'] = 'echo browser passed'
        self.invalid('actual browser training not executed')
        self.workflow['jobs'].pop('browser')
        self.invalid('required numerical/runtime job missing')

    def run_gate(self, mutate=None, faster=False):
        with tempfile.TemporaryDirectory(prefix='candle-native-ci-evidence-') as directory:
            temporary = pathlib.Path(directory)
            for name in ('candle-native-napi', 'candle-native-package'):
                (temporary / name).mkdir()
            sha = lambda data: hashlib.sha256(data).hexdigest()
            project = temporary / 'synthetic-checkout'
            for name in SOURCE_NAMES + ('bindings/napi/ruv_fann.node',):
                target = project / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(('Synthetic validation-only bytes: ' + name).encode())
            binary = sha((project / 'bindings/napi/ruv_fann.node').read_bytes())
            sources = {name: sha((project / name).read_bytes()) for name in SOURCE_NAMES}
            build = dict(schema='ruv-fann.napi-build/1', nodeApi=8, platform='linux', arch='x64',
                rustc='rustc 1.88.0 (synthetic structural fixture)', rustcPath=sys.executable,
                rustcSha256=sha(pathlib.Path(sys.executable).read_bytes()),
                pins=dict(napi='3.16.1', derive='3.7.0', build='2.6.0'), inputsBefore=sources, inputsAfter=copy.deepcopy(sources),
                coreLockSha256=sources['Cargo.lock'], napiLockSha256=sources['bindings/napi/Cargo.lock'], binarySha256=binary,
                commands=[dict(program='cargo', args=['--release', '--locked', '--offline', '--message-format=json-render-diagnostics'], exitCode=0)] * 4,
                flags=dict(optLevel=3, debugAssertions=False, lto='thin', codegenUnits=1, RUSTFLAGS='', CARGO_ENCODED_RUSTFLAGS=''))
            oracle_bytes = json.dumps(fixture()).encode()
            oracle_path = temporary / 'fixture.json'; oracle_path.write_bytes(oracle_bytes)
            consumer = dict(schema='ruv-fann.napi-consumer/1', status='passed', nativeLoaded=True, unchangedBinary=True,
                binarySha256=binary, buildSourceHashes=copy.deepcopy(sources), artifact=dict(sha256='a' * 64, size=1024),
                commands=[dict(label=name, exitCode=0) for name in ('pack', 'install-local', 'consume')],
                checks={key: True for key in ('esm', 'cjs', 'directBinding', 'alias', 'atomic', 'lifecycle')},
                training=dict(**{key: True for key in ('actualAutograd', 'finiteDifferences', 'sgdLearning', 'adamwLearning',
                    'atomicUpdates', 'immutableSnapshot', 'closedLifecycle')}, firstLoss=.085, lastLoss=.00001, adamFirst=.085, adamLast=.0001,
                    candle=dict(verified=True, oracle=fixture()['oracle'], fixtureSha256=sha(oracle_bytes), tolerance=1e-12,
                        sgdStepCount=3, adamwStepCount=3, parameterGradientError=0, inputGradientError=0, sgdError=0, adamwError=0)))
            tuner = dict(schema='ruv-fann.napi-tuner/1', status='passed', nativeLoaded=True, actualDarwin=True,
                unchangedBinary=True, binarySha256=binary, receipt=tuning(faster))
            if mutate:
                mutate(build, consumer, tuner)
            for path, value in (('candle-native-napi/build-info.json', build), ('candle-native-package/receipt.json', consumer),
                ('candle-native-napi/tuner.json', tuner)):
                (temporary / path).write_text(json.dumps(value))
            return subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', self.stage('native', 'evidence')['run']],
                cwd=project, env={**os.environ, 'RUNNER_TEMP': directory, 'RUV_FANN_CANDLE_FIXTURE': str(oracle_path)},
                text=True, capture_output=True, timeout=15)

    def test_real_gate_accepts_baseline_retention_and_confirmed_gain(self):
        for faster in (False, True):
            result = self.run_gate(faster=faster)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_gate_rejects_binary_compiler_and_source_mismatches(self):
        for change in (lambda b, c, t: c.update(binarySha256='b' * 64),
            lambda b, c, t: b.update(rustcSha256='b' * 64),
            lambda b, c, t: b['inputsAfter'].update({'src/autograd/pass.rs': 'b' * 64})):
            result = self.run_gate(change); self.assertNotEqual(result.returncode, 0)

    def test_real_gate_rejects_absent_learning_or_candle_proof(self):
        for change in (lambda b, c, t: c['training'].update(lastLoss=.085),
            lambda b, c, t: c['training']['candle'].update(verified=False),
            lambda b, c, t: c['training']['candle'].update(adamwStepCount=1),
            lambda b, c, t: c['training'].update(atomicUpdates=False)):
            result = self.run_gate(change); self.assertNotEqual(result.returncode, 0)

    def test_real_gate_rejects_nonfinite_gradient_error_and_tolerance(self):
        for key in ('parameterGradientError', 'tolerance'):
            result = self.run_gate(lambda b, c, t: c['training']['candle'].update({key: None}))
            self.assertNotEqual(result.returncode, 0)

    def test_real_gate_rejects_tuner_overrun_or_unconfirmed_application(self):
        for change in (lambda b, c, t: t['receipt']['trialBudget'].update(dispatchCount=13),
            lambda b, c, t: t['receipt'].update(recommendation=dict(batchRows=17)),
            lambda b, c, t: t['receipt']['confirmation'].pop('candidateSamplesNs')):
            result = self.run_gate(change); self.assertNotEqual(result.returncode, 0)

    def browser_gate(self, mutate=None):
        with tempfile.TemporaryDirectory(prefix='candle-native-browser-evidence-') as directory:
            temporary = pathlib.Path(directory); (temporary / 'candle-native-browser').mkdir()
            oracle_bytes = json.dumps(fixture()).encode()
            oracle = temporary / 'fixture.json'; oracle.write_bytes(oracle_bytes)
            sha = lambda data: hashlib.sha256(data).hexdigest()
            before = {name: sha((ROOT / name).read_bytes()) for name in (
                'ruv-swarm/npm/wasm/ruv_swarm_wasm_bg.wasm', 'ruv-swarm/npm/wasm/ruv_swarm_simd.wasm',
                'ruv-swarm/npm/wasm/ruv_swarm_wasm.js')}
            before.update(browserDriver=sha((ROOT / 'ruv-swarm/npm/test/browser-training.cjs').read_bytes()), candleFixture=sha(oracle_bytes))
            cases = [dict(variant=variant, simdSupport=variant == 'simd',
                candle=dict(oracle=fixture()['oracle'], optimizers=[dict(optimizer=name, steps=3,
                    backwardPasses=3, parameterUpdates=3, maximumAbsoluteError=0, freezeAfterTrainerFree=True,
                    hnsw=dict(indexType='hnsw', records=17, actualId='sample-7', distance=0, deleted=True))
                    for name in ('sgd', 'adamw')]), learning=dict(optimizer='sgd', steps=200, initialLoss=4, finalLoss=1e-14),
                boundaries=dict(arrayRejections=23, scalarRejections=25, invalidJsonRejections=8,
                    metadataGetterReads=0, sharedArrayBufferTested=True, detachedTested=True, wasmMemoryTested=True,
                    atomicParametersPreserved=True, poisonedMetadataIgnored=True)) for variant in ('core', 'simd')]
            receipt = dict(schema='ruv-fann.browser-training/1', status='passed', actualWasmTraining=True,
                actualCandle=True, actualRuVectorHnsw=True, coreAndSimd=True, crossOriginIsolated=True,
                unchanged=True, playwright='1.62.1', chromium=BROWSER_VERSION, expectedChromiumVersion=BROWSER_VERSION,
                chromiumVersionSource=BROWSER_SOURCE, externalRequests=0, fixtureSha256=sha(oracle_bytes),
                inputsBefore=before, inputsAfter=copy.deepcopy(before), cases=cases)
            if mutate:
                mutate(receipt)
            (temporary / 'candle-native-browser/browser.json').write_text(json.dumps(receipt))
            return subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', self.stage('browser', 'evidence')['run']],
                cwd=ROOT, env={**os.environ, 'RUNNER_TEMP': directory, 'RUV_FANN_CANDLE_FIXTURE': str(oracle),
                    'EXPECTED_CHROMIUM_VERSION': BROWSER_VERSION, 'CHROMIUM_VERSION_SOURCE': BROWSER_SOURCE},
                text=True, capture_output=True, timeout=15)

    def test_real_browser_gate_accepts_complete_training_and_retrieval(self):
        result = self.browser_gate(); self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_browser_gate_rejects_version_or_source_mismatch(self):
        for mutate in (lambda r: r.update(chromium='156.0.8078.79'),
            lambda r: r.update(expectedChromiumVersion='156.0.8078.79'),
            lambda r: r.update(chromiumVersionSource='actual-browser')):
            result = self.browser_gate(mutate); self.assertNotEqual(result.returncode, 0)

    def test_real_browser_gate_rejects_missing_training_boundary_or_candle_step(self):
        for mutate in (lambda r: r['cases'].pop(), lambda r: r['cases'][0]['candle']['optimizers'][0].update(parameterUpdates=1),
            lambda r: r['cases'][0]['boundaries'].update(atomicParametersPreserved=False),
            lambda r: r['cases'][0]['candle']['optimizers'][0]['hnsw'].update(actualId='wrong')):
            result = self.browser_gate(mutate); self.assertNotEqual(result.returncode, 0)

    def test_real_browser_gate_rejects_network_callbacks_and_stale_source(self):
        for mutate in (lambda r: r.update(externalRequests=1),
            lambda r: r['cases'][0]['boundaries'].update(metadataGetterReads=1),
            lambda r: r['inputsAfter'].update(browserDriver='a' * 64)):
            result = self.browser_gate(mutate); self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
