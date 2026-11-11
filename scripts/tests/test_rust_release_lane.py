import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


selector = load('rust_candidate_selector', 'select-rust-candidate.py')
public = load('rust_public_consumer', 'verify-public-rust.py')
release = load('rust_github_release', 'create-rust-github-release.py')
SOURCE = 'a' * 40


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        self.run = {'head_sha': SOURCE, 'event': 'push', 'head_branch': 'release/rust',
                    'path': '.github/workflows/ci.yml', 'status': 'completed', 'conclusion': 'success',
                    'id': 1, 'html_url': 'https://github.com/ruvnet/ruv-FANN/actions/runs/1'}
        self.jobs = [{'name': 'actual archive verification', 'status': 'completed', 'conclusion': 'success'}]
        self.artifacts = [{'name': name, 'expired': False, 'digest': 'sha256:' + 'b' * 64} for name in selector.ARTIFACTS]

    def check(self):
        return selector.validate_run(self.run, self.jobs, self.artifacts, SOURCE)

    def test_accepts_source_bound_success(self):
        self.assertEqual(self.check()['sourceSha'], SOURCE)

    def test_rejects_wrong_source_branch_event_or_workflow(self):
        for key, value in [('head_sha', 'b' * 40), ('head_branch', 'main'), ('event', 'pull_request'), ('path', 'other.yml')]:
            with self.subTest(key=key), patch.dict(self.run, {key: value}):
                with self.assertRaises(ValueError):
                    self.check()

    def test_rejects_skipped_failed_pending_and_empty_jobs(self):
        for conclusion in ('skipped', 'failure', None):
            with self.subTest(conclusion=conclusion), patch.dict(self.jobs[0], {'conclusion': conclusion}):
                with self.assertRaises(ValueError):
                    self.check()
        self.jobs.clear()
        with self.assertRaises(ValueError):
            self.check()

    def test_rejects_expired_duplicate_missing_and_unsealed_artifacts(self):
        for mutate in (lambda rows: rows.pop(), lambda rows: rows.append(rows[0]),
                       lambda rows: rows[0].update(expired=True), lambda rows: rows[0].update(digest='')):
            artifacts = copy.deepcopy(self.artifacts)
            mutate(artifacts)
            with self.assertRaises(ValueError):
                selector.validate_run(self.run, self.jobs, artifacts, SOURCE)


def receipts(base):
    (base / 'cargo-public-archives').mkdir()
    (base / 'cargo-candidates').mkdir()
    rows = []
    for index in range(27):
        name, version = ('ruv-fann', '0.2.3') if index == 0 else (('ruv-fann-candle', '0.1.1') if index == 1 else (f'crate-{index}', '1.0.0'))
        archive = base / 'cargo-public-archives' / f'{name}-{version}.crate'
        archive.write_bytes(f'public bytes {index}'.encode())
        rows.append({'name': name, 'version': version, 'sha256': release.digest(archive), 'status': 'published-and-byte-verified'})
    publication = {'sourceSha': SOURCE, 'crates': rows}
    commands = []
    argv = [['cargo', 'generate-lockfile'], ['cargo', 'metadata', '--locked', '--format-version=1'],
            ['cargo', 'run', '--locked', '--release', '--quiet', '--', '/fixture.json']]
    for index in range(3):
        log = base / f'public-rust-{index}.log'
        log.write_text('actual command output fixture')
        commands.append({'command': argv[index], 'exit_code': 0, 'logSha256': release.digest(log)})
    fixture = base / 'candle-f64-fixture.json'
    fixture.write_text('{"oracle":"unit test fixture"}')
    consumer = {'schema': 'ruv-fann.public-rust-consumer/1', 'sourceSha': SOURCE,
                'versions': {'ruv-fann': '0.2.3', 'ruv-fann-candle': '0.1.1'},
                'fixtureSha256': release.digest(fixture), 'commands': commands,
                'consumerSourceSha256': release.digest(ROOT / 'scripts/fixtures/public-rust-consumer.rs'),
                'result': {'cases': public.REQUIRED_CASES, 'tests': 3, 'checks': 300, 'failures': 0, 'skipped': 0},
                'crates': [{key: row[key] for key in ('name', 'version', 'sha256')} for row in rows]}
    ci = {'schema': 'ruv-fann.rust-candidate-ci/1', 'sourceSha': SOURCE, 'jobs': [{'conclusion': 'success'}]}
    for name, data in [('cargo-publication', publication), ('cargo-candidate-check', publication), ('public-rust-consumers', consumer), ('validated-ci', ci)]:
        (base / f'{name}.json').write_text(json.dumps(data))
    for name in ('receipt.json', 'receipt.sha256', 'publication-order.json', 'snapshot-files.json'):
        (base / 'cargo-candidates' / name).write_text('{}')
    return consumer, publication, fixture


class PublicRustReceiptTests(unittest.TestCase):
    def test_rejects_source_version_empty_skipped_failed_and_digest_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            consumer, publication, _ = receipts(Path(directory))
            public.validate(consumer, publication, SOURCE)
            mutations = [lambda r: r.update(sourceSha='b' * 40), lambda r: r['versions'].update({'ruv-fann': '0.2.2'}),
                         lambda r: r['result'].update(checks=0), lambda r: r['result'].update(skipped=1),
                         lambda r: r['result'].update(failures=1), lambda r: r['commands'][0].update(exit_code=1),
                         lambda r: r['commands'][1].update(command=['true']), lambda r: r.update(consumerSourceSha256='0' * 64),
                         lambda r: r['crates'].pop(), lambda r: r['crates'][0].update(sha256='0' * 64)]
            for mutate in mutations:
                changed = copy.deepcopy(consumer)
                mutate(changed)
                with self.assertRaises(ValueError):
                    public.validate(changed, publication, SOURCE)


class RustGitHubReleaseTests(unittest.TestCase):
    def exercise(self, corruption=None, wrong_tag=False, existing=False):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            _, _, fixture = receipts(base)
            calls = []
            def gh(*args):
                calls.append(args)
                if args[0] == 'api':
                    if 'issues?' in args[1]:
                        return '[]'
                    sha = 'b' * 40 if wrong_tag and '/tags/' in args[1] else SOURCE
                    return json.dumps({'object': {'type': 'commit', 'sha': sha}})
                if args[:2] == ('release', 'download'):
                    folder = Path(args[args.index('--dir') + 1])
                    shutil.copytree(base / 'github-release-assets', folder)
                    if corruption:
                        (folder / corruption).write_text('corrupt')
                return ''
            def lookup(*args, **kwargs):
                return SimpleNamespace(returncode=0 if existing else 1, stdout='{"draft":false}', stderr='HTTP 404: Not Found')
            env = {'GITHUB_ACTIONS': 'true', 'GITHUB_WORKFLOW': 'protected-release', 'GITHUB_REF': 'refs/heads/main',
                   'GITHUB_SHA': SOURCE, 'RUV_FANN_CANDLE_FIXTURE': str(fixture)}
            with patch.dict(os.environ, env), patch.object(release.sys, 'argv', ['script', str(base)]), patch.object(release, 'run', side_effect=gh), patch.object(release.subprocess, 'run', side_effect=lookup):
                if corruption or wrong_tag:
                    with self.assertRaises(ValueError):
                        release.main()
                    self.assertFalse(any(row[:2] == ('release', 'edit') for row in calls))
                else:
                    release.main()
                    result = json.loads((base / 'github-release.json').read_text())
                    self.assertEqual(result['tagSourceSha'], SOURCE)
                    self.assertEqual(len(result['assets']), 40)
                    self.assertEqual(result['status'], 'already-released' if existing else 'published-and-byte-verified')
    def test_every_public_asset_and_source_tag_verified(self):
        self.exercise()
    def test_corrupt_archive_cannot_publish(self):
        self.exercise(corruption='ruv-fann-0.2.3.crate')
    def test_corrupt_receipt_cannot_publish(self):
        self.exercise(corruption='public-rust-consumers.json')
    def test_wrong_tag_cannot_publish(self):
        self.exercise(wrong_tag=True)
    def test_existing_release_requires_every_receipt_and_asset_identical(self):
        self.exercise(existing=True)
        self.exercise(existing=True, corruption='cargo-publication.json')


class RustWorkflowTests(unittest.TestCase):
    def test_rust_default_has_no_unguarded_npm_publication(self):
        workflow = (ROOT / '.github/workflows/protected-release.yml').read_text()
        self.assertIn('default: false\n        type: boolean', workflow)
        for step in workflow.split('      - name: ')[1:]:
            if any(command in step for command in ('--prepare --directory', '--publish --directory', '--verify-public --directory')):
                self.assertIn('if: inputs.publish_npm', step)
        self.assertNotIn('uses: ./.github/workflows/ci.yml', workflow)
        self.assertIn('PUBLISH_SPACING_SECONDS: \'360\'', workflow)
        self.assertIn('timeout-minutes: 180', workflow)
