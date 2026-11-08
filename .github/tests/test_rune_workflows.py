"""Parse actual Actions YAML and reject mutations that bypass the Rune gate.

These structural checks supplement the real pinned Rune/probe execution in the
workflow; they do not replace build, reproduction or runtime correctness proof.
"""
import copy
import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

import yaml


class ActionsLoader(yaml.SafeLoader):
    pass


# Actions uses YAML 1.2 booleans: the event key `on` is not boolean True.
ActionsLoader.yaml_implicit_resolvers = copy.deepcopy(yaml.SafeLoader.yaml_implicit_resolvers)
for key, resolvers in ActionsLoader.yaml_implicit_resolvers.items():
    ActionsLoader.yaml_implicit_resolvers[key] = [
        item for item in resolvers if item[0] != 'tag:yaml.org,2002:bool'
    ]
ActionsLoader.add_implicit_resolver('tag:yaml.org,2002:bool',
    re.compile(r'^(?:true|True|TRUE|false|False|FALSE)$'), list('tTfF'))

ROOT = pathlib.Path(__file__).resolve().parents[2]
REVISION = '92425512f0f742e21b747f58f7555104446fc9de'
LOCK_SHA256 = '6509be23921039376d94f688822cd624167f42a5886b2f067c89ad97a6ecde28'


def load(path):
    return yaml.load(path.read_text(), Loader=ActionsLoader)


def validate(rune, ci):
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)

    events = rune.get('on', {})
    require('workflow_call' in events, 'reusable workflow_call missing')
    for event in ('push', 'pull_request'):
        paths = events.get(event, {}).get('paths', [])
        require(all(path in paths for path in ('**/*.rs', 'config/rune.json',
            'config/rune-probe/**', 'scripts/rune*.py', '.github/workflows/rune.yml')),
            'standalone Rust/Rune path coverage missing')
    require(rune.get('permissions') == {'contents': 'read'}, 'least permissions missing')
    env = rune.get('env', {})
    require(env.get('RUNE_REVISION') == REVISION, 'Rune source revision is not pinned')
    require(env.get('RUNE_LOCK_SHA256') == LOCK_SHA256, 'Rune source lock is not pinned')
    require(str(env.get('RUSTUP_TOOLCHAIN')) == '1.88.0', 'Rust compiler is not pinned')
    acceptance = ci.get('jobs', {}).get('rune', {})
    require(acceptance.get('uses') == './.github/workflows/rune.yml', 'CI acceptance does not call Rune')
    require(not acceptance.get('if') and not acceptance.get('continue-on-error'), 'CI acceptance can skip or ignore Rune')
    require(not acceptance.get('secrets'), 'CI passes secrets to Rune')

    job = rune.get('jobs', {}).get('verify', {})
    require(1 <= job.get('timeout-minutes', 0) <= 45, 'verification lacks bounded timeout')
    require(not job.get('continue-on-error') and not job.get('if'), 'verification job is optional')
    require(not job.get('permissions') or job['permissions'] == {'contents': 'read'}, 'job permissions escalate')
    steps = job.get('steps', [])
    ids = {step.get('id'): step for step in steps if step.get('id')}
    order = [step.get('id') for step in steps]
    needed = ['plan', 'source', 'prepare', 'prefetch', 'setup', 'verify', 'evidence', 'unchanged']
    require(all(key in ids for key in needed), 'required evidence stage missing')
    if all(key in ids for key in needed):
        require([order.index(key) for key in needed] == sorted(order.index(key) for key in needed),
            'prefetch/offline/evidence order is unsafe')
        for key in ('setup', 'verify'):
            require(str(ids[key].get('env', {}).get('CARGO_NET_OFFLINE')).lower() == 'true', 'Cargo offline enforcement missing')
            require(ids[key].get('env', {}).get('CARGO_HOME') == '${{ runner.temp }}/rune-dependencies', 'isolated dependency cache missing')
        require(ids['prefetch'].get('run', '').count('cargo fetch --locked --manifest-path') == 3,
            'explicit three-graph prefetch missing')
        source = ids['source'].get('run', '')
        require('rev-parse HEAD' in source and 'sha256sum --check' in source and '"$RUNE_REVISION"' in source,
            'source identity check missing')
        require('scripts/rune.py setup --source' in ids['setup'].get('run', '') and '--timeout 1200' in ids['setup'].get('run', ''),
            'actual bounded source setup missing')
        verify = ids['verify'].get('run', '')
        require(all(value in verify for value in ('scripts/rune.py verify --repo .', '--tool-receipt',
            '"$RUNNER_TEMP/rune-evidence"', '--timeout 1200')), 'actual bounded offline verification missing')
        evidence = ids['evidence'].get('run', '')
        require(all(name in evidence for name in ('status', 'passed', 'stale_source', 'plan_tamper',
            'object_tamper', 'correctness', 'runtime_probe')), 'acceptance receipt checks incomplete')
        require('git diff --exit-code' in ids['unchanged'].get('run', '') and
            'git status --porcelain --untracked-files=all' in ids['unchanged'].get('run', ''),
            'source immutability check missing')

    uploads = []
    for step in steps:
        require(not step.get('continue-on-error'), 'step failure may be ignored')
        require(not step.get('if') or step['if'] == 'always()', 'stage may silently skip')
        action = step.get('uses', '')
        if action:
            require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[a-f0-9]{40}', action) is not None,
                'action is not pinned by commit')
            require('cache' not in action.lower(), 'cached tool/target trust is forbidden')
            if action.startswith('actions/checkout@'):
                require(step.get('with', {}).get('persist-credentials') is False, 'checkout persists credentials')
            if action.startswith('actions/upload-artifact@'):
                uploads.append(step)
        command = step.get('run', '')
        require(re.search(r'\brune(?:\.py)?\s+(?:improve|edit|eval|promote|publish|rollback)\b', command) is None,
            'paid/mutating/promotion operation is enabled')
        require('${{ secrets.' not in str(step), 'provider/publication secret passed')
    require(len(uploads) == 1, 'evidence upload missing or ambiguous')
    if len(uploads) == 1:
        require(uploads[0].get('if') == 'always()', 'failed-gate evidence is discarded')
        options = uploads[0].get('with', {})
        require(options.get('if-no-files-found') == 'error', 'missing artifact evidence is accepted')
        require('rune-evidence/receipt.json' in options.get('path', '') and 'rune-evidence/logs/**' in options.get('path', ''),
            'actual evidence files are not uploaded')
        require('workspace/**' not in options.get('path', ''), 'source snapshot is uploaded')
    return errors


class RuneWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.rune = load(ROOT / '.github/workflows/rune.yml')
        self.ci = load(ROOT / '.github/workflows/ci.yml')

    def invalid(self, message):
        self.assertIn(message, validate(self.rune, self.ci))

    def stage(self, name):
        return next(step for step in self.rune['jobs']['verify']['steps'] if step.get('id') == name)

    def test_committed_workflows_enforce_the_contract(self):
        self.assertEqual(validate(self.rune, self.ci), [])

    def test_actual_step_shell_programs_parse(self):
        for step in self.rune['jobs']['verify']['steps']:
            if 'run' in step:
                with self.subTest(stage=step.get('id', step.get('name'))):
                    result = subprocess.run(['bash', '-n'], input=step['run'], text=True,
                        capture_output=True, timeout=10, check=False)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_reusable_acceptance_gate_fails(self):
        self.ci['jobs'].pop('rune')
        self.invalid('CI acceptance does not call Rune')

    def test_acceptance_cannot_ignore_rejections(self):
        self.ci['jobs']['rune']['continue-on-error'] = True
        self.invalid('CI acceptance can skip or ignore Rune')

    def test_verify_failure_cannot_be_suppressed(self):
        self.stage('verify')['continue-on-error'] = True
        self.invalid('step failure may be ignored')

    def test_mutable_action_ref_fails(self):
        self.rune['jobs']['verify']['steps'][0]['uses'] = 'actions/checkout@main'
        self.invalid('action is not pinned by commit')

    def test_mutable_tool_identity_fails(self):
        self.rune['env']['RUNE_REVISION'] = 'main'
        self.invalid('Rune source revision is not pinned')

    def test_dependency_fetch_after_offline_verify_fails(self):
        steps = self.rune['jobs']['verify']['steps']
        prefetch = self.stage('prefetch')
        steps.remove(prefetch)
        steps.append(prefetch)
        self.invalid('prefetch/offline/evidence order is unsafe')

    def test_network_enabled_verification_fails(self):
        self.stage('verify')['env'].pop('CARGO_NET_OFFLINE')
        self.invalid('Cargo offline enforcement missing')

    def test_no_runtime_or_negative_evidence_fails(self):
        self.stage('evidence')['run'] = "print('passed')"
        self.invalid('acceptance receipt checks incomplete')

    def test_rust_only_change_still_requires_standalone_gate(self):
        self.rune['on']['push']['paths'].remove('**/*.rs')
        self.invalid('standalone Rust/Rune path coverage missing')

    def test_provider_secret_is_rejected(self):
        self.stage('verify')['env']['OPENROUTER_API_KEY'] = '${{ secrets.OPENROUTER_API_KEY }}'
        self.invalid('provider/publication secret passed')

    def test_promotion_command_is_rejected(self):
        self.stage('verify')['run'] += '\nrune promote'
        self.invalid('paid/mutating/promotion operation is enabled')

    def test_failure_evidence_must_be_uploaded(self):
        upload = self.rune['jobs']['verify']['steps'][-1]
        upload.pop('if')
        self.invalid('failed-gate evidence is discarded')

    def run_evidence_gate(self, mutate=None):
        receipt = {
            'status': 'passed', 'upstream': {'commit': REVISION},
            'fresh_verify': True, 'unchanged_core_lock': True, 'offline': True, 'model_calls': 0,
            'product': {'name': 'ruv-fann-rune-probe', 'stored_object_verified': True, 'sha256': 'a' * 64},
            'rune_receipt': {'artifact_digest': 'a' * 64, 'offline': True, 'model_calls': 0},
            'negative_checks': {name: {'status': 'passed'} for name in ('stale_source', 'plan_tamper', 'object_tamper')},
            'correctness': {'status': 'passed'}, 'runtime_probe': {'status': 'passed'},
            'commands': [{'label': name, 'status': 'passed'} for name in ('freeze', 'replay', 'fresh-verify',
                'explain', 'runtime-probe', 'release-source-correctness', 'reject-stale-source', 'reject-plan-tamper')],
            'source': {'files': [{'path': 'core/src/lib.rs'}, {'path': 'src/main.rs'}]},
        }
        if mutate:
            mutate(receipt)
        command = self.stage('evidence')['run']
        program = command.split("python3 - <<'PY'\n", 1)[1].rsplit('\nPY', 1)[0]
        with tempfile.TemporaryDirectory(prefix='rune-workflow-test-') as directory:
            output = pathlib.Path(directory) / 'rune-evidence'
            output.mkdir()
            (output / 'receipt.json').write_text(json.dumps(receipt))
            return subprocess.run([sys.executable, '-c', program], check=False, capture_output=True,
                text=True, timeout=10, env={**os.environ, 'RUNNER_TEMP': directory,
                    'RUNE_REVISION': REVISION, 'PYTHONOPTIMIZE': '1'})

    def test_actual_evidence_program_accepts_complete_receipt(self):
        result = self.run_evidence_gate()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_actual_evidence_program_rejects_missing_runtime_even_with_python_optimization(self):
        result = self.run_evidence_gate(lambda r: r.pop('runtime_probe'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('runtime_probe', result.stderr)

    def test_actual_evidence_program_rejects_failed_tamper_probe(self):
        result = self.run_evidence_gate(lambda r: r['negative_checks']['plan_tamper'].update(status='failed'))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('plan_tamper', result.stderr)

    def test_actual_evidence_program_rejects_unverified_artifact(self):
        result = self.run_evidence_gate(lambda r: r['product'].update(sha256='b' * 64))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Rune artifact receipt missing', result.stderr)

    def test_actual_evidence_program_rejects_nonoffline_or_nonreproducing_run(self):
        for field in ('offline', 'fresh_verify'):
            with self.subTest(field=field):
                result = self.run_evidence_gate(lambda r: r.update({field: False}))
                self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
