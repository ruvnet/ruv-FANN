"""Reject altered source semantics, untrusted CI and rewritten candidate evidence."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('tensor_release_tested', ROOT / 'scripts/tensor-release.py')
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)
SHA = 'a' * 40


def normalized():
    data = tomllib.loads((release.PACKAGE / 'Cargo.toml').read_text())
    data.pop('workspace')
    data['package'].update(build=False, autolib=False, autobins=False, autoexamples=False, autotests=False, autobenches=False)
    data['lib'] = {'name': release.NAME.replace('-', '_'), 'path': 'src/lib.rs'}
    data['bin'] = [{'name': release.NAME, 'path': 'src/main.rs'}]
    data['test'] = [{'name': p.stem, 'path': 'tests/' + p.name} for p in sorted((release.PACKAGE / 'tests').glob('*.rs'))]
    for key in ('dependencies', 'dev-dependencies'):
        data[key] = {name: {'version': v} if isinstance(v, str) else v for name, v in data[key].items()}
    return data


class TensorReleaseTests(unittest.TestCase):
    def test_normalized_dependency_target_build_feature_and_extra_fields_reject(self):
        release.normalized_manifest(normalized())
        mutations = [
            lambda m: m['dependencies']['half'].update(version='999.0.0'),
            lambda m: m['package'].update(build='scripts/exfiltrate.rs'),
            lambda m: m['lib'].update(path='src/main.rs'),
            lambda m: m.update(features={'default': ['half/serde']}),
            lambda m: m['bin'][0].update(path='../outside.rs'),
        ]
        for mutate in mutations:
            data = normalized()
            mutate(data)
            with self.assertRaisesRegex(ValueError, 'Normalized'):
                release.normalized_manifest(data)

    def test_real_archive_source_and_normalized_manifest_binding(self):
        # Derive the normalized Cargo fixture with Cargo's existing TOML emitter,
        # independently mutate only that file, and preserve original/vcs payloads.
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / release.ARCHIVE
            files = {}
            for path in release.PACKAGE.rglob('*'):
                relative = path.relative_to(release.PACKAGE)
                if not path.is_file() or any(p in ('target', '__pycache__', 'validation') for p in relative.parts):
                    continue
                if path.suffix == '.log' or path.name == '.gitignore':
                    continue
                files[str(relative)] = path.read_bytes()
            files['Cargo.toml.orig'] = files.pop('Cargo.toml')
            files['.cargo_vcs_info.json'] = json.dumps({'git': {'sha1': SHA}, 'path_in_vcs': 'crates/wsp-tensor'}).encode()
            def write(data, duplicate=False):
                files['Cargo.toml'] = release.archive_tools.dump_toml(data).encode()
                with tarfile.open(archive, 'w:gz') as stream:
                    for name, content in files.items():
                        info = tarfile.TarInfo(f'{release.NAME}-{release.VERSION}/' + name)
                        info.size = len(content)
                        stream.addfile(info, io.BytesIO(content))
                    if duplicate:
                        stream.addfile(info, io.BytesIO(content))
            write(normalized())
            release.archive_binding(archive, SHA)
            data = normalized(); data['dependencies']['half']['version'] = '999.0.0'
            write(data)
            with self.assertRaisesRegex(ValueError, 'Normalized'):
                release.archive_binding(archive, SHA)
            write(normalized(), duplicate=True)
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                release.archive_binding(archive, SHA)
            write(normalized())
            with self.assertRaisesRegex(ValueError, 'VCS'):
                release.archive_binding(archive, 'b' * 40)

    def test_complete_nonzero_rust_and_python_results_required(self):
        good = 'test result: ok. 10 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.0s\n'
        self.assertEqual(release.evidence(good, 'rust'), {'testsPassed': 10})
        for output in (good.replace('10 passed', '0 passed'), good.replace('0 ignored', '1 ignored'),
                       good.replace('0 filtered', '5 filtered'), good + good, ''):
            with self.assertRaises(ValueError): release.evidence(output, 'rust')
        release.evidence('Ran 3 tests in 0.01s\n\nOK\n', 'python')
        for output in ('Ran 0 tests in 0s\nOK\n', 'Ran 1 test in 0s\nOK (skipped=1)\n', 'Ran 3 tests in 0s\nFAILED\n'):
            with self.assertRaises(ValueError): release.evidence(output, 'python')

    def ci_fixture(self):
        run = dict(id=12, head_sha=SHA, event='push', head_branch='release/tensor', path=release.WORKFLOW, status='completed', conclusion='success')
        jobs = [dict(name=n, status='completed', conclusion='success') for n in
                ('qualify', 'portability (windows-latest)', 'portability (macos-latest)')]
        artifact = dict(id=42, name='tensor-candidate-' + SHA, expired=False, digest='sha256:' + 'd' * 64)
        return run, jobs, [artifact]

    def test_missing_pending_skipped_stale_and_duplicate_ci_reject(self):
        run, jobs, artifacts = self.ci_fixture()
        release.validate_ci(run, jobs, artifacts, SHA)
        for key, value in [('head_sha', 'b' * 40), ('event', 'pull_request'), ('head_branch', 'main'),
                           ('path', '.github/workflows/ci.yml'), ('status', 'queued'), ('conclusion', 'failure')]:
            changed = dict(run); changed[key] = value
            with self.assertRaises(ValueError): release.validate_ci(changed, jobs, artifacts, SHA)
        for changed in (jobs[:-1], [dict(jobs[0], status='queued')] + jobs[1:], [dict(jobs[0], conclusion='skipped')] + jobs[1:]):
            with self.assertRaises(ValueError): release.validate_ci(run, changed, artifacts, SHA)
        for changed in ([], artifacts * 2, [dict(artifacts[0], expired=True)], [dict(artifacts[0], digest='none')]):
            with self.assertRaises(ValueError): release.validate_ci(run, jobs, changed, SHA)

    def test_protected_identity_environment_and_main_lease(self):
        environment = dict(GITHUB_ACTIONS='true', GITHUB_WORKFLOW='protected-tensor-release', GITHUB_REF='refs/heads/main', GITHUB_SHA=SHA)
        def api(endpoint):
            return {'object': {'sha': SHA}} if '/git/ref/' in endpoint else []
        with patch.object(release, 'identity'), patch.object(release, 'gh', side_effect=api), patch.dict(os.environ, environment, clear=True):
            release.protected(SHA)
            for key in environment:
                with patch.dict(os.environ, {key: 'invalid'}):
                    with self.assertRaises(ValueError): release.protected(SHA)
        with patch.object(release, 'identity'), patch.dict(os.environ, environment, clear=True), patch.object(release, 'gh', return_value={'object': {'sha': 'b'*40}}):
            with self.assertRaisesRegex(ValueError, 'Main moved'): release.protected(SHA)

    def test_rewritten_downloaded_candidate_rejects_even_with_rewritten_local_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); candidate = base / 'candidate'; candidate.mkdir()
            names = {release.ARCHIVE, 'candidate.json', *[role + '.log' for role, *_ in release.command_plan(candidate)]}
            for name in names: (candidate / name).write_bytes(b'original')
            with zipfile.ZipFile(base / 'candidate.zip', 'w') as stream:
                for name in names: stream.write(candidate / name, name)
            run, jobs, artifacts = self.ci_fixture()
            artifacts[0]['digest'] = 'sha256:' + release.sha(base / 'candidate.zip')
            (base / 'ci.json').write_text(json.dumps({'sourceSha': SHA, 'run': run, 'artifact': artifacts[0], 'repositoryCi': {'runId': 99}}))
            def pages(endpoint, field): return jobs if field == 'jobs' else artifacts
            with patch.object(release, 'protected'), patch.object(release, 'gh', return_value=run), patch.object(release.ci_tools, 'pages', side_effect=pages), patch.object(release.ci_tools, 'validate_run'), patch.object(release, 'validate', return_value={'valid': True}):
                self.assertEqual(release.validated_inputs(base, SHA), {'valid': True})
                (candidate / 'candidate.json').write_text('{"archiveSha256":"forged with new log hashes"}')
                with self.assertRaisesRegex(ValueError, 'bytes changed'):
                    release.validated_inputs(base, SHA)
                (base / 'candidate.zip').write_bytes(b'forged zip')
                with self.assertRaisesRegex(ValueError, 'ZIP seal changed'):
                    release.validated_inputs(base, SHA)

    def test_cli_requires_actual_roundtrip_and_no_inference_claim(self):
        data = {'schema': 'ruv-fann.tensor-e2e/1', 'roundtrip': 'passed', 'overwrite_rejected': True, 'pan2_inference': 'not_run'}
        release.evidence(json.dumps(data), 'cli')
        for key, value in [('roundtrip', 'skipped'), ('overwrite_rejected', False), ('pan2_inference', 'passed')]:
            changed = dict(data); changed[key] = value
            with self.assertRaises(ValueError): release.evidence(json.dumps(changed), 'cli')


if __name__ == '__main__':
    unittest.main()
