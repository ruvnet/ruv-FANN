"""Guard actual controller admission, immutable evidence and partial-draft recovery."""
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings
import zipfile

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('tensor_recovery', ROOT / 'scripts/tensor-gh-recovery.py')
r = importlib.util.module_from_spec(spec); spec.loader.exec_module(r)


class RecoveryTests(unittest.TestCase):
    def test_artifact_source_run_and_checksum_pins(self):
        item = {'id': r.PUBLICATION_ID, 'digest': 'sha256:' + r.PUBLICATION_SHA, 'expired': False,
                'name': 'tensor-publication-' + r.SOURCE, 'workflow_run': {'id': r.FAILED_RUN, 'head_sha': r.SOURCE}}
        check = lambda row: r.artifact_identity(row, r.PUBLICATION_ID, r.PUBLICATION_SHA, r.FAILED_RUN, r.SOURCE, item['name'])
        check(item)
        for key, value in [('id', 1), ('digest', 'sha256:' + '0'*64), ('expired', True), ('name', 'unrelated'),
                           ('workflow_run', {'id': 1, 'head_sha': r.SOURCE}), ('workflow_run', {'id': r.FAILED_RUN, 'head_sha': 'a'*40})]:
            with self.assertRaises(ValueError): check(dict(item, **{key: value}))

    def failed_fixture(self):
        row = {'id': r.FAILED_RUN, 'head_sha': r.SOURCE, 'head_branch': 'main', 'event': 'workflow_dispatch',
               'path': r.WORKFLOW, 'status': 'completed', 'conclusion': 'failure', 'run_attempt': 1}
        steps = [{'name': n, 'conclusion': 'success'} for n in
                 ['Revalidate exact source and download the sealed pre-main candidate',
                  'Publish only the qualified immutable Cargo archive',
                  'Verify canonical bytes and fresh public Cargo and CLI consumers without registry credentials']]
        steps.append({'name': 'Create exact-source release and verify every downloaded asset', 'conclusion': 'failure'})
        return row, [{'name': 'publish', 'status': 'completed', 'conclusion': 'failure', 'steps': steps}]

    def test_original_failure_is_retained_and_all_earlier_gates_required(self):
        row, jobs = self.failed_fixture(); r.validate_failed_run(row, jobs)
        for key, value in [('head_sha', 'a'*40), ('conclusion', 'success'), ('event', 'push'), ('run_attempt', 2)]:
            with self.assertRaises(ValueError): r.validate_failed_run(dict(row, **{key: value}), jobs)
        for status in ['failure', 'skipped', None]:
            changed = copy.deepcopy(jobs); changed[0]['steps'][2]['conclusion'] = status
            with self.assertRaises(ValueError): r.validate_failed_run(row, changed)
        changed = copy.deepcopy(jobs); changed[0]['steps'].append({'name': 'other', 'conclusion': 'failure'})
        with self.assertRaises(ValueError): r.validate_failed_run(row, changed)

    def test_real_environment_controller_identity_no_source_sha_spoof(self):
        controller = 'b'*40
        environment = {'GITHUB_ACTIONS': 'true', 'GITHUB_EVENT_NAME': 'workflow_dispatch',
                       'GITHUB_WORKFLOW': 'protected-tensor-release', 'GITHUB_REF': r.REF,
                       'GITHUB_SHA': controller, 'GITHUB_REPOSITORY': r.REPO,
                       'GITHUB_WORKFLOW_REF': f'{r.REPO}/{r.WORKFLOW}@{r.REF}'}
        with patch.object(r, 'checkout_identity'), patch.object(r, 'gh', return_value={'object': {'sha': controller}}), patch.dict(os.environ, environment, clear=True):
            r.controller_guard(controller)
            for name in environment:
                with patch.dict(os.environ, {name: 'wrong'}):
                    with self.assertRaises(ValueError): r.controller_guard(controller)
            with patch.dict(os.environ, {'GITHUB_SHA': r.SOURCE}):
                with self.assertRaises(ValueError): r.controller_guard(controller)

    def test_source_main_tag_and_release_blocker_guards(self):
        def api(endpoint):
            if '/heads/main' in endpoint: return {'object': {'sha': r.SOURCE}}
            if '/tags/' in endpoint: return {'object': {'type': 'commit', 'sha': r.SOURCE}}
            return []
        with patch.object(r, 'gh', side_effect=api): r.live_source()
        for selected in ['main', 'tag', 'issues']:
            def mutant(endpoint):
                row = api(endpoint)
                if selected == 'main' and '/heads/main' in endpoint: row['object']['sha'] = 'a'*40
                if selected == 'tag' and '/tags/' in endpoint: row['object']['sha'] = 'a'*40
                if selected == 'issues' and '/issues?' in endpoint: row = [{'number': 1}]
                return row
            with patch.object(r, 'gh', side_effect=mutant), self.assertRaises(ValueError): r.live_source()

    def test_zip_duplicate_traversal_symlink_and_size_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for index, names in enumerate([['../outside'], ['/outside'], ['a/./b'], ['a\\b'], ['C:/outside'], ['same', 'same']]):
                archive = base / f'{index}.zip'
                with warnings.catch_warnings(), zipfile.ZipFile(archive, 'w') as stream:
                    warnings.simplefilter('ignore', UserWarning)
                    for name in names: stream.writestr(name, b'x')
                with self.assertRaises(ValueError): r.safe_extract(archive, base / f'out{index}')
            archive = base / 'link.zip'
            with zipfile.ZipFile(archive, 'w') as stream:
                info = zipfile.ZipInfo('link'); info.external_attr = (stat.S_IFLNK | 0o777) << 16
                stream.writestr(info, 'outside')
            with self.assertRaises(ValueError): r.safe_extract(archive, base / 'outlink')
            archive = base / 'oversized.zip'
            with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as stream:
                stream.writestr('oversized', b'x' * 10_000_001)
            with self.assertRaises(ValueError): r.safe_extract(archive, base / 'outsize')

    def test_original_download_bytes_must_match_pinned_seal(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'archive.zip'
            def transfer(*args, **kwargs): kwargs['stdout'].write(b'corrupt original seal')
            with patch.object(r.subprocess, 'run', side_effect=transfer), self.assertRaisesRegex(ValueError, 'checksum'):
                r.download_artifact(r.PUBLICATION_ID, r.PUBLICATION_SHA, output)

    def test_actual_unittest_nonzero_unskipped_log_required(self):
        code = 'import unittest\nT=type("T",(unittest.TestCase,),{f"test_{i}":lambda self:None for i in range(8)})\nunittest.main()'
        process = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(r.test_evidence(process.stdout + process.stderr), 8)
        code = code.replace('lambda self:None', 'unittest.skip("fixture skip")(lambda self:None)')
        process = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        with self.assertRaises(ValueError): r.test_evidence(process.stdout + process.stderr)
        for text in ['', 'Ran 0 tests in 0s\nOK\n', 'Ran 10 tests in 0s\nFAILED\n']:
            with self.assertRaises(ValueError): r.test_evidence(text)

    def asset_fixture(self, base):
        folder = base / 'assets'; folder.mkdir()
        (folder / 'a.log').write_bytes(b'original log')
        (folder / 'SHA256SUMS').write_bytes(b'new sums')
        return {p.name: p for p in folder.iterdir()}

    def test_known_partial_draft_only_original_sums_may_be_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = self.asset_fixture(Path(directory)); payloads = {1: b'original log', 2: b'old sums'}
            rows = [{'id': 1, 'name': 'a.log', 'state': 'uploaded', 'size': len(payloads[1])},
                    {'id': 2, 'name': 'SHA256SUMS', 'state': 'uploaded', 'size': len(payloads[2])}]
            self.assertEqual(r.classify_assets(rows, assets, b'old sums', True, payloads.get), 2)
            payloads[1] = b'altered log'; rows[0]['size'] = len(payloads[1])
            with self.assertRaises(ValueError): r.classify_assets(rows, assets, b'old sums', True, payloads.get)
            with self.assertRaises(ValueError): r.classify_assets(rows[1:], assets, b'old sums', False, payloads.get)
            with self.assertRaises(ValueError): r.classify_assets([dict(rows[0], name='unknown')], assets, b'old sums', True, payloads.get)
            with self.assertRaises(ValueError): r.classify_assets([dict(rows[0], state='starter')], assets, b'old sums', True, payloads.get)

    def test_resume_exact_final_assets_and_published_release_are_immutable(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = self.asset_fixture(Path(directory))
            rows = [{'id': i, 'name': n, 'state': 'uploaded', 'size': p.stat().st_size} for i, (n, p) in enumerate(assets.items())]
            payloads = {row['id']: assets[row['name']].read_bytes() for row in rows}
            self.assertIsNone(r.classify_assets(rows[:1], assets, b'old sums', True, payloads.get))
            self.assertIsNone(r.classify_assets(rows, assets, b'old sums', True, payloads.get))
            self.assertIsNone(r.classify_assets(rows, assets, b'old sums', False, payloads.get))
            with self.assertRaises(ValueError): r.classify_assets(rows[:1], assets, b'old sums', False, payloads.get)

    def test_observable_draft_or_asset_drift_stops_before_next_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = self.asset_fixture(Path(directory))
            original = {'id': 1, 'name': 'a.log', 'state': 'uploaded', 'size': len(b'original log')}
            for change in ('published', 'extra', 'conflict'):
                rows = [original]
                release = {'draft': change != 'published'}
                if change == 'extra': rows = rows + [dict(original, id=3, name='unexpected')]
                if change == 'conflict': rows = [dict(original, id=2)]
                payloads = {1: b'original log', 2: b'changed log!'}
                with patch.object(r, 'release_inventory', side_effect=[({'draft': True}, []), (release, rows)]), \
                     patch.object(r, 'asset_bytes', side_effect=payloads.get), patch.object(r, 'upload_asset') as upload, \
                     patch.object(r.subprocess, 'run') as mutation:
                    with self.assertRaises(ValueError): r.repair_draft(assets, b'old sums', Mock())
                    self.assertEqual(upload.call_count, 1)
                    mutation.assert_not_called()
            # Detect publication before the first write as well.
            with patch.object(r, 'release_inventory', return_value=({'draft': False}, [])), \
                 patch.object(r, 'upload_asset') as upload, patch.object(r.subprocess, 'run') as mutation:
                with self.assertRaises(ValueError): r.repair_draft(assets, b'old sums', Mock())
                upload.assert_not_called(); mutation.assert_not_called()

    def test_old_manifest_delete_then_upload_is_resumable_and_cached_ids_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = self.asset_fixture(Path(directory)); rows = []
            payloads = {1: b'original log', 2: b'old sums', 3: b'new sums'}
            row = lambda i, n: {'id': i, 'name': n, 'state': 'uploaded', 'size': len(payloads[i])}
            a, old, new = row(1, 'a.log'), row(2, 'SHA256SUMS'), row(3, 'SHA256SUMS')
            with patch.object(r, 'release_inventory', side_effect=[({'draft': True}, [a, old]),
                         ({'draft': True}, [a]), ({'draft': True}, [a, new])]), \
                 patch.object(r, 'asset_bytes', side_effect=payloads.get) as fetch, \
                 patch.object(r.subprocess, 'run') as delete, patch.object(r, 'upload_asset') as upload:
                r.repair_draft(assets, b'old sums', Mock())
                self.assertEqual(delete.call_count, 1)
                self.assertIn('/releases/assets/2', delete.call_args.args[0][-1])
                upload.assert_called_once_with(assets['SHA256SUMS'])
                self.assertEqual(fetch.call_count, 3)  # unchanged asset 1 is read only once

    def test_publication_rechecks_full_manifest_and_draft_after_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            assets = self.asset_fixture(Path(directory))
            rows = [{'id': i, 'name': n, 'state': 'uploaded', 'size': p.stat().st_size} for i, (n, p) in enumerate(assets.items())]
            payloads = {row['id']: assets[row['name']].read_bytes() for row in rows}
            for release, changed in [({'draft': False}, rows), ({'draft': True}, rows[:1]),
                    ({'draft': True}, rows + [dict(rows[0], id=50, name='extra')])]:
                with patch.object(r, 'release_inventory', return_value=(release, changed)), \
                     patch.object(r, 'asset_bytes', side_effect=payloads.get), patch.object(r.subprocess, 'run') as mutation:
                    with self.assertRaises(ValueError): r.publish_draft(assets, b'old sums', Mock())
                    mutation.assert_not_called()
            with patch.object(r, 'release_inventory', return_value=({'draft': True}, rows)), \
                 patch.object(r, 'asset_bytes', side_effect=payloads.get), patch.object(r.subprocess, 'run') as mutation:
                r.publish_draft(assets, b'old sums', Mock())
                self.assertEqual(mutation.call_count, 1)
                self.assertIn('PATCH', mutation.call_args.args[0])

    def test_upload_uses_known_release_id_and_exact_nonempty_raw_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'safe.log'; path.write_bytes(b'exact original bytes')
            response = Mock(status=201)
            response.read.return_value = json.dumps({'name': path.name, 'state': 'uploaded', 'size': path.stat().st_size}).encode()
            response.__enter__ = Mock(return_value=response); response.__exit__ = Mock(return_value=False)
            opener = Mock(); opener.open.return_value = response
            with patch.object(r.urllib.request, 'build_opener', return_value=opener), patch.dict(os.environ, {'GH_TOKEN': 'unit-test-only'}):
                r.upload_asset(path)
                request = opener.open.call_args.args[0]
                self.assertEqual(request.full_url, f'https://uploads.github.com/repos/{r.REPO}/releases/{r.DRAFT}/assets?name=safe.log')
                self.assertEqual(request.method, 'POST'); self.assertEqual(request.data, path.read_bytes())
                path.write_bytes(b'')
                with self.assertRaises(ValueError): r.upload_asset(path)
                self.assertEqual(opener.open.call_count, 1)

    def test_empty_log_is_carried_unchanged_in_original_candidate_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); original = base / 'evidence/publication/release-assets'; original.mkdir(parents=True)
            (original / 'SHA256SUMS').write_bytes(b'original empty-file hash manifest')
            (original / 'candidate-fmt.log').write_bytes(b'')
            (original / 'other.log').write_bytes(b'unchanged')
            with zipfile.ZipFile(base / 'evidence/candidate.zip', 'w') as stream: stream.writestr('fmt.log', b'')
            (base / 'evidence/publication.zip').write_bytes(b'publication sealed fixture')
            (base / 'qualification.zip').write_bytes(b'qualification sealed fixture')
            with patch.object(r, 'controller_files', return_value={'source': 'hash'}):
                manifest = r.make_assets(base, 'b'*40, {'runId': 1})
            self.assertNotIn('candidate-fmt.log', manifest)
            self.assertTrue(all(p.stat().st_size for p in (base / 'assets').iterdir()))
            mapping = json.loads((base / 'assets/empty-log-encoding.json').read_text())['entries'][0]
            with zipfile.ZipFile(base / 'assets' / mapping['container']) as stream:
                data = stream.read(mapping['member'])
                self.assertEqual(data, b''); self.assertEqual(hashlib.sha256(data).hexdigest(), mapping['sha256'])
            self.assertEqual((base / 'assets/original-SHA256SUMS').read_bytes(), (original / 'SHA256SUMS').read_bytes())
            self.assertEqual((base / 'assets/other.log').read_bytes(), b'unchanged')
            shutil = __import__('shutil'); shutil.rmtree(base / 'assets')
            (original / 'candidate-fmt.log').write_bytes(b'fake passed text')
            with patch.object(r, 'controller_files', return_value={}), self.assertRaises(ValueError): r.make_assets(base, 'b'*40, {'runId': 1})

    def test_recovery_workflow_separates_source_and_controller_without_cargo_credentials(self):
        text = (ROOT / r.WORKFLOW).read_text()
        self.assertIn('options: [qualify, recover]', text)
        self.assertIn('ref: ${{ github.sha }}', text)
        self.assertIn('ref: ' + r.SOURCE, text)
        self.assertIn('group: protected-registry-release', text)
        self.assertIn('path: controller', text); self.assertIn('path: qualified-source', text)
        self.assertIn('tensor-recovery-qualification-${{ github.sha }}', text)
        self.assertNotIn('CARGO_REGISTRY_TOKEN', text); self.assertNotIn('cargo publish', text)
        self.assertNotIn('GITHUB_SHA:', text)


if __name__ == '__main__':
    unittest.main()
