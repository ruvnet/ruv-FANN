"""Reject altered source semantics, untrusted CI and rewritten candidate evidence."""
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import unittest
from unittest.mock import Mock, patch
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
    def test_binary_asset_read_preserves_ansi_without_terminal_rendering(self):
        data = b'\x1b[32mcompiler output\x1b[0m\n\x00'
        with patch.object(release.subprocess, 'check_output', return_value=data) as read:
            self.assertEqual(release.github_asset_bytes(42), data)
            self.assertIn('--allow-escape-sequences', read.call_args.args[0])
            self.assertNotIn('text', read.call_args.kwargs)

    def release_fixture(self, base, empty=True):
        (base / 'candidate').mkdir(parents=True)
        (base / 'fresh').mkdir()
        (base / 'public.crate').write_bytes(b'canonical archive fixture')
        for name in ('ci.json', 'publication.json', 'public-consumers.json'):
            (base / name).write_text(json.dumps({'fixture': name}))
        (base / 'candidate/candidate.json').write_text('{"fixture":"unchanged candidate receipt"}')
        (base / 'candidate/fmt.log').write_bytes(b'' if empty else b'formatter output')
        (base / 'candidate/tests.log').write_bytes(b'original source test output')
        (base / 'fresh/build.log').write_bytes(b'' if empty else b'build output')
        (base / 'fresh/tests.log').write_bytes(b'original public test output')
        originals = {'candidate-' + p.name: (p, 'candidate/' + p.name) for p in (base / 'candidate').glob('*.log')}
        originals.update({p.name: (p, 'fresh/' + p.name) for p in (base / 'fresh').glob('*.log')})
        return {'archiveSha256': release.sha(base / 'public.crate')}, originals

    def test_empty_logs_and_receipts_preserved_in_deterministic_nonempty_container(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); receipt, originals = self.release_fixture(base)
            raw = {str(p.relative_to(base)): p.read_bytes() for p in base.rglob('*') if p.is_file()}
            assets, manifest = release.github_assets(base, SHA, receipt)
            release.validate_empty_log_assets(assets, originals, SHA, receipt['archiveSha256'])
            self.assertTrue(all(p.stat().st_size > 0 for p in assets.iterdir()))
            self.assertNotIn('candidate-fmt.log', manifest); self.assertNotIn('build.log', manifest)
            for name, data in raw.items(): self.assertEqual((base / name).read_bytes(), data)
            mapping = json.loads((assets / 'empty-log-encoding.json').read_text())
            self.assertEqual(mapping['sourceSha'], SHA)
            with zipfile.ZipFile(assets / 'empty-logs.zip') as stream:
                for row in mapping['entries']:
                    data = stream.read(row['member'])
                    self.assertEqual(data, (base / row['origin']).read_bytes())
                    self.assertEqual(len(data), row['size'])
                    self.assertEqual(__import__('hashlib').sha256(data).hexdigest(), row['sha256'])
            first = {p.name: p.read_bytes() for p in assets.iterdir()}
            shutil.rmtree(assets)
            again, manifest_again = release.github_assets(base, SHA, receipt)
            self.assertEqual(manifest, manifest_again)
            self.assertEqual(first, {p.name: p.read_bytes() for p in again.iterdir()})

    def test_empty_log_map_source_checksum_and_container_corruption_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); receipt, originals = self.release_fixture(base)
            assets, _ = release.github_assets(base, SHA, receipt)
            path = assets / 'empty-log-encoding.json'; mapping = json.loads(path.read_text())
            for key, value in [('sourceSha', 'b' * 40), ('archiveSha256', '0' * 64), ('containerSha256', '0' * 64), ('entries', [])]:
                path.write_text(json.dumps(dict(mapping, **{key: value})))
                with self.assertRaises(ValueError): release.validate_empty_log_assets(assets, originals, SHA, receipt['archiveSha256'])
            for content in (b'fabricated formatter success', b''):
                with zipfile.ZipFile(assets / 'empty-logs.zip', 'w') as stream:
                    stream.writestr('candidate-fmt.log', content)
                    stream.writestr('unexpected.log', b'')
                # Even rewriting the local map hash cannot excuse altered member bytes/inventory.
                mapping['containerSha256'] = release.sha(assets / 'empty-logs.zip')
                path.write_text(json.dumps(mapping))
                with self.assertRaises(ValueError): release.validate_empty_log_assets(assets, originals, SHA, receipt['archiveSha256'])
            # A hash-rewritten ZIP with matching names but symlink entries is also unsafe.
            with zipfile.ZipFile(assets / 'empty-logs.zip', 'w') as stream:
                for row in mapping['entries']:
                    info = zipfile.ZipInfo(row['member']); info.create_system = 3
                    info.external_attr = 0o120777 << 16
                    stream.writestr(info, b'')
            mapping['containerSha256'] = release.sha(assets / 'empty-logs.zip')
            path.write_text(json.dumps(mapping))
            with self.assertRaises(ValueError): release.validate_empty_log_assets(assets, originals, SHA, receipt['archiveSha256'])

    def test_no_empty_logs_keeps_raw_assets_and_rejects_unnecessary_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); receipt, originals = self.release_fixture(base, empty=False)
            assets, manifest = release.github_assets(base, SHA, receipt)
            self.assertIn('candidate-fmt.log', manifest); self.assertNotIn('empty-logs.zip', manifest)
            (assets / 'empty-log-encoding.json').write_text('{}')
            with self.assertRaises(ValueError): release.validate_empty_log_assets(assets, originals, SHA, receipt['archiveSha256'])

    def test_partial_draft_uploads_only_missing_exact_assets_and_resume_is_noop(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); receipt, _ = self.release_fixture(base)
            assets, manifest = release.github_assets(base, SHA, receipt)
            metadata = {'id': 42, 'tag_name': release.TAG, 'target_commitish': SHA, 'draft': True}
            rows, payloads = [], {}
            def upload(identifier, path):
                self.assertEqual(identifier, 42)
                data = path.read_bytes(); i = len(rows) + 1
                rows.append({'id': i, 'name': path.name, 'state': 'uploaded', 'size': len(data)})
                payloads[i] = data
            initial = next(iter(manifest)); upload(42, assets / initial)
            def api(endpoint): return rows[:] if '/assets?' in endpoint else metadata.copy()
            with patch.object(release, 'gh', side_effect=api), patch.object(release, 'github_asset_bytes', side_effect=payloads.get) as fetch, \
                 patch.object(release, 'github_upload', side_effect=upload) as mutation:
                release.github_resume(42, SHA, assets, manifest, Mock())
                self.assertEqual(mutation.call_count, len(manifest) - 1)
                self.assertNotIn(initial, [call.args[1].name for call in mutation.call_args_list])
                self.assertEqual(fetch.call_count, len(manifest))
                mutation.reset_mock()
                release.github_resume(42, SHA, assets, manifest, Mock())
                mutation.assert_not_called()

    def test_draft_wrong_source_unknown_conflict_or_publication_stops_all_uploads(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); receipt, _ = self.release_fixture(base)
            assets, manifest = release.github_assets(base, SHA, receipt)
            name = next(iter(manifest)); expected = (assets / name).read_bytes()
            row = {'id': 1, 'name': name, 'size': len(expected), 'state': 'uploaded'}
            metadata = {'id': 42, 'tag_name': release.TAG, 'target_commitish': SHA, 'draft': True}
            for changed, rows, data in [
                (dict(metadata, target_commitish='b' * 40), [], expected),
                (dict(metadata, tag_name='other'), [], expected),
                (dict(metadata, draft=False), [], expected),
                (metadata, [dict(row, name='unknown.log')], expected),
                (metadata, [row], b'x' * len(expected)),
                (metadata, [dict(row, state='starter')], expected),
            ]:
                def api(endpoint): return rows if '/assets?' in endpoint else changed
                with patch.object(release, 'gh', side_effect=api), patch.object(release, 'github_asset_bytes', return_value=data), \
                     patch.object(release, 'github_upload') as mutation:
                    with self.assertRaises(ValueError): release.github_resume(42, SHA, assets, manifest, Mock())
                    mutation.assert_not_called()

    def test_existing_public_release_only_downloads_exact_assets_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); receipt, _ = self.release_fixture(base)
            assets, manifest = release.github_assets(base, SHA, receipt)
            metadata = {'id': 42, 'tag_name': release.TAG, 'target_commitish': SHA, 'draft': False, 'html_url': 'https://example.test/release'}
            rows = [{'id': i, 'name': name, 'state': 'uploaded', 'size': (assets / name).stat().st_size} for i, name in enumerate(manifest)]
            payloads = {row['id']: (assets / row['name']).read_bytes() for row in rows}
            tools = Mock(); tools.existing_release.return_value = metadata; tools.tag_source.return_value = SHA
            def api(endpoint): return rows if '/assets?' in endpoint else metadata
            with patch.object(release, 'validate_public', return_value=receipt), patch.object(release.registry, 'downloaded', return_value=(base / 'public.crate').read_bytes()), \
                 patch.object(release, 'github_assets', return_value=(assets, manifest)), patch.object(release, 'load', return_value=tools), \
                 patch.object(release, 'protected'), patch.object(release, 'gh', side_effect=api), \
                 patch.object(release, 'github_asset_bytes', side_effect=payloads.get), patch.object(release, 'github_upload') as upload, \
                 patch.object(release.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0)) as command:
                release.github_release(base, SHA)
                upload.assert_not_called()
                # The sole subprocess call is the read-only tag lookup, never create/PATCH/DELETE.
                self.assertEqual(command.call_count, 1)
                self.assertEqual(command.call_args.args[0][:2], ['gh', 'api'])
                self.assertNotIn('--method', command.call_args.args[0])
            self.assertEqual({p.name: release.sha(p) for p in (base / 'github-before').iterdir()}, manifest)
            self.assertEqual({p.name: release.sha(p) for p in (base / 'github-after').iterdir()}, manifest)

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
