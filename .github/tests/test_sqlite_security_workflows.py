"""Required real SQLite/MCP stages and fail-closed evidence acceptance.

Synthetic files exercise the exact workflow evidence programs without loading a
native addon or replacing the workflow's mandatory actual build/runtime stages.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import tomllib
import unittest

from test_rune_workflows import ROOT, load

RUST_COMMANDS = {
    'pool': '-p ruv-swarm-persistence --lib sqlite_pool::tests',
    'persistence': '-p ruv-swarm-persistence --test sqlite_release_gate',
    'benchmark': '-p ruv-swarm-benchmarking --lib storage::tests',
    'codec': '-p ruv-swarm-mcp --lib tests::stdio_transport',
    'build': '-p ruv-swarm-mcp --bin ruv-swarm-mcp-stdio',
}
SOURCE_ID = '2026-10-16 14:45:20 ' + 'a' * 64
PROTOCOL_CHECKS = ['initialize_and_legacy_negotiation', 'real_tool_catalogue_and_schemas',
    'real_memory_round_trip', 'real_agent_spawn_and_persistence', 'invalid_arguments_return_tool_error',
    'oversized_line_not_dispatched_and_stream_recovers', 'graceful_eof_shutdown']
BUILDERS = ['sqlite-build.mjs', 'sqlite-build-source.mjs', 'sqlite-build-contract.mjs', 'sqlite-build-process.mjs', 'sqlite-source.json']


def validate(workflow, ci):
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(message)
    events = workflow.get('on', {})
    require(all(name in events for name in ('workflow_call', 'workflow_dispatch')), 'reusable/manual entry missing')
    for name in ('push', 'pull_request'):
        event = events.get(name, {})
        require(event.get('branches') == ['main', 'develop'] and
            all(path in event.get('paths', []) for path in ('ruv-swarm/**', '**/Cargo.lock', '.github/tests/**')),
            'standalone security trigger missing')
    require(workflow.get('permissions') == {'contents': 'read'}, 'least permissions missing')
    gate = ci.get('jobs', {}).get('sqlite-security', {})
    require(gate.get('uses') == './.github/workflows/sqlite-security.yml' and not gate.get('if') and
        not gate.get('continue-on-error') and not gate.get('secrets'), 'mandatory main CI security gate missing')
    for name, source in [('candle-native', 'candle-native'), ('rune', 'rune'), ('wasm-targets', 'wasm-build')]:
        require(ci.get('jobs', {}).get(name, {}).get('uses') == './.github/workflows/' + source + '.yml', 'prior acceptance gate removed')
    matrix = ci.get('jobs', {}).get('workspaces', {}).get('strategy', {}).get('matrix', {}).get('include', [])
    require(any(row.get('workspace') == 'ruv-swarm' and row.get('toolchain') == '1.89.0' for row in matrix), 'native workspace MSRV not updated')
    jobs = workflow.get('jobs', {})
    require(set(jobs) == {'rust', 'advisories', 'node'}, 'required security job missing')
    for name, job in jobs.items():
        require(1 <= job.get('timeout-minutes', 0) <= 45, 'job deadline missing')
        require(not job.get('if') and not job.get('continue-on-error'), 'required job can skip or fail open')
        require(job.get('defaults', {}).get('run', {}).get('shell') == 'bash', 'pipefail shell missing')
        require(not re.search(r'\b(?:runner|env|steps|job)\s*(?:\.|\[)', str(job.get('env', {}))), 'invalid job environment context')
        uploads = []
        for step in job.get('steps', []):
            require(not step.get('continue-on-error') and (not step.get('if') or step['if'] == 'always()'), 'required step can skip or fail open')
            require('${{ secrets.' not in str(step), 'secret passed to validation')
            if action := step.get('uses'):
                require(re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[a-f0-9]{40}', action) is not None, 'mutable action reference')
                if action.startswith('actions/checkout@'):
                    require(step.get('with', {}).get('persist-credentials') is False, 'checkout credentials retained')
                if action.startswith('actions/setup-node@'):
                    require(step.get('with', {}).get('package-manager-cache') is False, 'automatic dependency cache trusted')
                if action.startswith('actions/upload-artifact@'):
                    uploads.append(step)
            program = step.get('run', '')
            require(not re.search(r'\|\|\s*(?:true|echo)|set\s+\+e|\bexit\s+0\b', program), 'command failure masked')
            require(not re.search(r'\bnpm\s+(?:publish|audit\s+fix)\b|\bgit\s+(?:push|merge)\b', program), 'publication/source mutation enabled')
        require(len(uploads) == 1 and uploads[0].get('if') == 'always()' and
            uploads[0].get('with', {}).get('if-no-files-found') == 'error', 'failure evidence upload missing')
        stages = {step['id']: step for step in job.get('steps', []) if 'id' in step}
        unchanged = stages.get('unchanged', {}).get('run', '')
        require(all(command in unchanged for command in ('git diff --exit-code', 'git diff --cached --exit-code',
            'git status --porcelain --untracked-files=all')), 'source immutability missing')
    def stages(name):
        return {step['id']: step for step in jobs.get(name, {}).get('steps', []) if 'id' in step}
    rust = stages('rust')
    require('cargo +1.89.0 fetch --locked' in rust.get('fetch', {}).get('run', ''), 'explicit locked native fetch missing')
    for name, command in RUST_COMMANDS.items():
        program = rust.get(name, {}).get('run', '')
        require('cargo +1.89.0 ' in program and '--locked --offline' in program and command in program and '| tee ' in program,
            'actual locked/offline Rust acceptance missing: ' + name)
    require('test_stdio.py --binary "$CARGO_TARGET_DIR/debug/ruv-swarm-mcp-stdio"' in rust.get('protocol', {}).get('run', ''), 'actual compiled MCP protocol proof missing')
    audit = stages('advisories')
    require('cargo-audit --version 0.22.2 --locked' in audit.get('setup', {}).get('run', ''), 'pinned audit tool missing')
    audit_program = audit.get('audit', {}).get('run', '')
    require('python3 scripts/verify-packages.py --audit --report ' in audit_program and
        '--workspace' not in audit_program and '--ignore' not in audit_program, 'all five locks without exceptions not audited')
    extra = audit.get('extra-audits', {}).get('run', '')
    require(all(f'cargo audit --file {name}/Cargo.lock --deny unsound --json' in extra
        for name in ['integrations/candle', 'bindings/napi']) and '--ignore' not in extra,
        'independent Candle/Node addon locks not audited')
    node = stages('node')
    require(jobs.get('node', {}).get('strategy', {}).get('matrix', {}).get('node') == ['22.13.0', '24'], 'Node compatibility matrix missing')
    commands = {'install': 'npm ci --ignore-scripts --prefix ruv-swarm/npm',
        'build': 'node ruv-swarm/npm/scripts/sqlite-build.mjs',
        'verify': 'node ruv-swarm/npm/scripts/sqlite-build.mjs --verify',
        'focused': 'node --test ruv-swarm/npm/test/sqlite-engine.test.js',
        'security': 'npm run test:security --prefix ruv-swarm/npm',
        'package': 'node ruv-swarm/npm/test/check-sqlite-package.mjs --output "$RUNNER_TEMP/sqlite-security-node/package"',
        'audit': 'npm audit --audit-level=low --prefix ruv-swarm/npm'}
    for name, command in commands.items():
        require(command in node.get(name, {}).get('run', ''), 'actual Node security command missing: ' + name)
    audit_program = node.get('audit', {}).get('run', '')
    require('--omit=dev' not in audit_program and '--production' not in audit_program, 'development dependency audit omitted')
    order = [step.get('id') for step in jobs.get('node', {}).get('steps', [])]
    require(all(name in order for name in commands) and
        [order.index(name) for name in commands if name in order] == sorted(order.index(name) for name in commands if name in order), 'Node install/build/runtime/security order changed')
    return errors


class SqliteSecurityWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = json.loads(subprocess.check_output(['node', '-p', 'JSON.stringify({platform:process.platform,arch:process.arch,nodeAbi:process.versions.modules,nodeVersion:process.versions.node})'], text=True))

    def setUp(self):
        self.workflow = load(ROOT / '.github/workflows/sqlite-security.yml')
        self.ci = load(ROOT / '.github/workflows/ci.yml')
        self.package_mutation = None

    def stage(self, job, name):
        return next(step for step in self.workflow['jobs'][job]['steps'] if step.get('id') == name)

    def test_actual_workflow_contract(self):
        self.assertEqual(validate(self.workflow, self.ci), [])

    def test_native_msrv_keeps_minimal_wasm_at_188(self):
        native = tomllib.loads((ROOT / 'ruv-swarm/Cargo.toml').read_text())
        wasm = tomllib.loads((ROOT / 'ruv-swarm/crates/ruv-swarm-wasm/Cargo.toml').read_text())
        self.assertEqual(native['workspace']['package']['rust-version'], '1.89')
        self.assertEqual(wasm['package']['rust-version'], '1.88')

    def test_actual_shell_programs_parse(self):
        for job in self.workflow['jobs'].values():
            for step in job['steps']:
                if 'run' in step:
                    result = subprocess.run(['bash', '-n'], input=step['run'], text=True, capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_runtime_offline_or_all_lock_gate_rejected(self):
        self.stage('rust', 'persistence')['run'] = self.stage('rust', 'persistence')['run'].replace('--offline', '')
        self.assertIn('actual locked/offline Rust acceptance missing: persistence', validate(self.workflow, self.ci))
        self.stage('advisories', 'audit')['run'] += ' --workspace ruv-swarm'
        self.assertIn('all five locks without exceptions not audited', validate(self.workflow, self.ci))
        self.stage('node', 'verify')['run'] = 'echo passed'
        self.assertIn('actual Node security command missing: verify', validate(self.workflow, self.ci))

    def test_skippable_mutable_or_removed_prior_gate_rejected(self):
        self.stage('node', 'build')['continue-on-error'] = True
        self.assertIn('required step can skip or fail open', validate(self.workflow, self.ci))
        self.workflow['jobs']['node']['steps'][0]['uses'] = 'actions/checkout@main'
        self.assertIn('mutable action reference', validate(self.workflow, self.ci))
        self.ci['jobs'].pop('candle-native')
        self.assertIn('prior acceptance gate removed', validate(self.workflow, self.ci))

    def test_pipeline_failures_propagate(self):
        with tempfile.TemporaryDirectory(prefix='sqlite-security-failure-') as directory:
            root = Path(directory)
            for name in ['rust', 'audit', 'node']:
                (root / ('sqlite-security-' + name)).mkdir()
            for name in ['cargo', 'node', 'npm', 'python3']:
                file = root / name
                file.write_text('#!/bin/sh\nexit 19\n'); file.chmod(0o700)
            for job, stages in [('rust', ['fetch', *RUST_COMMANDS, 'protocol']),
                ('advisories', ['setup', 'audit', 'extra-audits']), ('node', ['install', 'build', 'verify', 'focused', 'security', 'package', 'audit'])]:
                for name in stages:
                    result = subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', self.stage(job, name)['run']],
                        env={**os.environ, 'PATH': directory + os.pathsep + os.environ['PATH'], 'RUNNER_TEMP': directory, 'CARGO_TARGET_DIR': directory},
                        text=True, capture_output=True, timeout=10)
                    self.assertEqual(result.returncode, 19, (job, name, result.stderr))

    def node_gate(self, mutate=None):
        with tempfile.TemporaryDirectory(prefix='sqlite-security-synthetic-') as directory:
            root = Path(directory); package = root / 'ruv-swarm/npm'; scripts = package / 'scripts'; scripts.mkdir(parents=True)
            evidence = root / 'sqlite-security-node'; evidence.mkdir()
            sha = lambda data: hashlib.sha256(data).hexdigest()
            source = dict(version='3.54.1', sourceId=SOURCE_ID, archive=dict(sha256='a' * 64),
                files={name: dict(sha256='b' * 64) for name in ['sqlite3.c', 'sqlite3.h', 'sqlite3ext.h']})
            for name in BUILDERS:
                (scripts / name).write_text(json.dumps(source) if name == 'sqlite-source.json' else 'Synthetic gate fixture; never executed.\n')
            binary = package / 'native/sqlite' / (self.runtime['platform'] + '-' + self.runtime['arch'] + '-abi' + self.runtime['nodeAbi']) / 'better_sqlite3.node'
            binary.parent.mkdir(parents=True); binary.write_bytes(b'Synthetic validation-only bytes, never loaded.')
            receipt = dict(schema='ruv-swarm.sqlite-build/1', betterSqlite3Version='12.11.1', nodeGypVersion='12.4.0',
                **self.runtime, sqliteVersion='3.54.1', sqliteSourceId=SOURCE_ID, binaryBytes=binary.stat().st_size,
                binarySha256=sha(binary.read_bytes()), archiveSha256=source['archive']['sha256'],
                sourceFiles={name: entry['sha256'] for name, entry in source['files'].items()},
                builderFiles={name: sha((scripts / name).read_bytes()) for name in BUILDERS})
            proof = dict(schema='ruv-swarm.sqlite-verification/1', status='passed', sqliteVersion='3.54.1',
                sqliteSourceId=SOURCE_ID, nativeBinding=os.path.relpath(binary, package), receipt=receipt)
            packed = dict(schema='ruv-swarm.sqlite-package/1', status='passed', actualLifecycleInstall=True,
                actualCliPersistence=True, unchanged=True, sqliteVersion='3.54.1', sourceId=SOURCE_ID,
                nodeAbi=self.runtime['nodeAbi'], binarySha256='a' * 64, tarballSha256='b' * 64)
            packed['engine'] = copy.deepcopy(proof)
            packed['engine']['receipt']['binarySha256'] = packed['binarySha256']
            logs = {name: 'TAP version 13\n# tests 17\n# pass 17\n# fail 0\n# skipped 0\n' for name in ['focused', 'security']}
            if mutate:
                mutate(proof, logs)
            if self.package_mutation:
                self.package_mutation(packed)
            (evidence / 'runtime.json').write_text(json.dumps(proof))
            (evidence / 'package').mkdir()
            (evidence / 'package/receipt.json').write_text(json.dumps(packed))
            for name, text in logs.items():
                (evidence / (name + '.log')).write_text(text)
            return subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', self.stage('node', 'evidence')['run']],
                cwd=root, env={**os.environ, 'RUNNER_TEMP': directory}, text=True, capture_output=True, timeout=15)

    def test_real_node_gate_accepts_matching_runtime_and_dynamic_counts(self):
        result = self.node_gate(); self.assertEqual(result.returncode, 0, result.stderr)

    def test_real_node_gate_rejects_old_supplier_source_abi_and_binary(self):
        for mutate in [lambda p, l: p.update(sqliteVersion='3.53.4'),
            lambda p, l: p.update(sqliteSourceId='wrong'), lambda p, l: p['receipt'].update(nodeAbi='1'),
            lambda p, l: p['receipt'].update(binarySha256='c' * 64),
            lambda p, l: p['receipt']['builderFiles'].update({'sqlite-build.mjs': 'c' * 64})]:
            result = self.node_gate(mutate); self.assertNotEqual(result.returncode, 0)

    def test_real_node_gate_rejects_zero_failed_or_skipped_tests(self):
        for text in ['# tests 0\n# pass 0\n# fail 0\n# skipped 0\n',
            '# tests 17\n# pass 16\n# fail 1\n# skipped 0\n', '# tests 17\n# pass 16\n# fail 0\n# skipped 1\n']:
            result = self.node_gate(lambda p, logs: logs.update(security=text)); self.assertNotEqual(result.returncode, 0)

    def test_real_node_gate_rejects_missing_package_lifecycle_persistence_source_or_abi(self):
        for mutate in [lambda p: p.update(actualLifecycleInstall=False), lambda p: p.update(actualCliPersistence=False),
            lambda p: p.update(sourceId='wrong'), lambda p: p.update(nodeAbi='1'),
            lambda p: p['engine']['receipt']['builderFiles'].update({'sqlite-build.mjs': 'c' * 64}),
            lambda p: p['engine']['receipt'].update(binarySha256='c' * 64),
            lambda p: p['engine']['receipt'].update(betterSqlite3Version='11.10.0')]:
            self.package_mutation = mutate
            result = self.node_gate(); self.assertNotEqual(result.returncode, 0)

    def rust_gate(self, mutate=None):
        with tempfile.TemporaryDirectory(prefix='sqlite-rust-synthetic-') as directory:
            root = Path(directory); evidence = root / 'sqlite-security-rust'; evidence.mkdir()
            scripts = root / 'ruv-swarm/npm/scripts'; scripts.mkdir(parents=True)
            (scripts / 'sqlite-source.json').write_text(json.dumps(dict(version='3.54.1', sourceId=SOURCE_ID)))
            binary = root / 'debug/ruv-swarm-mcp-stdio'; binary.parent.mkdir(); binary.write_bytes(b'Synthetic gate fixture, never executed.')
            protocol = dict(status='passed', sdk='rmcp 3.9.0', stdioLineLimit=1048576, checks=PROTOCOL_CHECKS.copy(),
                binarySha256=hashlib.sha256(binary.read_bytes()).hexdigest())
            logs = {name: 'test result: ok. 3 passed; 0 failed; 0 ignored;\n' for name in RUST_COMMANDS if name != 'build'}
            logs['persistence'] += 'SQLite runtime: version=3.54.1, version_number=3054001, source_id=' + SOURCE_ID + '\n'
            if mutate:
                mutate(protocol, logs)
            (evidence / 'stdio.json').write_text(json.dumps(protocol))
            for name, text in logs.items():
                (evidence / (name + '.log')).write_text(text)
            return subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', self.stage('rust', 'evidence')['run']],
                cwd=root, env={**os.environ, 'RUNNER_TEMP': directory, 'CARGO_TARGET_DIR': directory},
                text=True, capture_output=True, timeout=10)

    def test_real_rust_gate_requires_actual_supplier_nonzero_tests_and_full_protocol(self):
        result = self.rust_gate(); self.assertEqual(result.returncode, 0, result.stderr)
        for mutate in [lambda p, l: l.update(pool='test result: ok. 0 passed; 0 failed;'),
            lambda p, l: l.update(persistence=l['persistence'].replace('version_number=3054001', 'version_number=3053004')),
            lambda p, l: p['checks'].pop(), lambda p, l: p.update(binarySha256='b' * 64)]:
            result = self.rust_gate(mutate); self.assertNotEqual(result.returncode, 0)

    def audit_gate(self, mutate=None, extra_mutation=None):
        with tempfile.TemporaryDirectory(prefix='sqlite-audit-synthetic-') as directory:
            root = Path(directory) / 'sqlite-security-audit'; root.mkdir()
            records = [dict(workspace=name, exit_code=0, command=['cargo', 'audit', '--file', 'Cargo.lock', '--deny', 'unsound'])
                for name in ['.', 'ruv-swarm', 'neuro-divergent', 'cuda-wasm', 'opencv-rust']]
            if mutate:
                mutate(records)
            (root / 'audit.json').write_text(json.dumps(records))
            extras = {name: dict(vulnerabilities=dict(found=False, count=0, list=[]), settings=dict(ignore=[], severity=None, target_arch=[], target_os=[]))
                for name in ['candle', 'napi']}
            if extra_mutation:
                extra_mutation(extras)
            for name, report in extras.items():
                (root / (name + '.json')).write_text(json.dumps(report))
            return subprocess.run(['bash', '--noprofile', '--norc', '-eo', 'pipefail', '-c', self.stage('advisories', 'evidence')['run']],
                env={**os.environ, 'RUNNER_TEMP': directory}, text=True, capture_output=True, timeout=10)

    def test_real_audit_gate_accepts_all_five_and_rejects_omission_failure_or_exception(self):
        result = self.audit_gate(); self.assertEqual(result.returncode, 0, result.stderr)
        for mutate in [lambda r: r.pop(), lambda r: r[0].update(exit_code=1),
            lambda r: r[0]['command'].extend(['--ignore', 'RUSTSEC-2026-0459'])]:
            result = self.audit_gate(mutate); self.assertNotEqual(result.returncode, 0)

    def test_real_audit_gate_requires_both_independent_graphs_and_no_exceptions(self):
        for mutate in [lambda r: r.pop('napi'), lambda r: r['candle']['vulnerabilities'].update(found=True, count=1, list=[{}]),
            lambda r: r['napi']['settings'].update(ignore=['RUSTSEC-2026-0459']),
            lambda r: r['candle']['settings'].update(severity='high'), lambda r: r['napi']['settings'].update(target_os=['windows'])]:
            result = self.audit_gate(extra_mutation=mutate); self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
