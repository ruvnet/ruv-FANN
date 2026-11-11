import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('publisher', ROOT / 'scripts/publish-candidates.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublicationTests(unittest.TestCase):
    def test_publish_main_preflights_entire_inventory_before_any_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            argv = ['publisher', '--publish', '--artifacts', directory, '--expected-sha', 'reviewed',
                    '--output', str(Path(directory) / 'publication.json')]
            environment = {'GITHUB_ACTIONS': 'true', 'GITHUB_WORKFLOW': 'protected-release',
                           'GITHUB_REF': 'refs/heads/main', 'GITHUB_SHA': 'reviewed',
                           'CARGO_REGISTRY_TOKEN': 'unit-test-only'}
            with patch.dict(os.environ, environment), patch.object(publisher.subprocess, 'check_output', side_effect=['reviewed', b'']), \
                 patch.object(publisher, 'candidates', return_value=['first', 'last']) as inventory, \
                 patch.object(publisher, 'preflight', side_effect=ValueError('late inventory drift')) as preflight, \
                 patch.object(publisher, 'publish') as upload, patch('sys.argv', argv):
                with self.assertRaisesRegex(ValueError, 'late inventory drift'):
                    publisher.main()
                preflight.assert_called_once_with(['first', 'last'])
                upload.assert_not_called()
                self.assertFalse((Path(directory) / 'publication.json').exists())

    def archive(self, directory, manifest):
        archive = Path(directory) / 'fixture-1.0.0.crate'
        with tarfile.open(archive, 'w:gz') as tar:
            for name, text in [('Cargo.toml', manifest), ('README.md', 'Fixture readme')]:
                data = text.encode()
                info = tarfile.TarInfo('fixture-1.0.0/' + name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return archive

    def test_normalized_metadata_preserves_renamed_target_and_dev_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.archive(directory, '''
[package]
name="fixture"
version="1.0.0"
readme="README.md"
rust-version="1.81"
license="MIT"
[dependencies.alias]
package="real"
version="=2.0.0"
default-features=false
optional=true
features=["std"]
[target.'cfg(unix)'.build-dependencies]
build="1"
[dev-dependencies]
test="3"
[features]
optional=["dep:alias"]
''')
            result = publisher.metadata(archive)
            self.assertEqual(result['readme'], 'Fixture readme')
            self.assertEqual(result['rust_version'], '1.81')
            deps = {d['name']: d for d in result['deps']}
            self.assertEqual(deps['real']['explicit_name_in_toml'], 'alias')
            self.assertEqual(deps['real']['version_req'], '=2.0.0')
            self.assertFalse(deps['real']['default_features'])
            self.assertTrue(deps['real']['optional'])
            self.assertEqual(deps['build']['target'], 'cfg(unix)')
            self.assertEqual(deps['build']['kind'], 'build')
            self.assertEqual(deps['test']['kind'], 'dev')

    def test_path_dependencies_cannot_be_uploaded(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.archive(directory, '[package]\nname="fixture"\nversion="1.0.0"\n[dependencies.local]\nversion="1"\npath="../local"')
            with self.assertRaisesRegex(ValueError, 'Unpublishable'):
                publisher.metadata(archive)

    def test_existing_version_must_match_exact_bytes_without_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.archive(directory, '[package]\nname="fixture"\nversion="1.0.0"')
            data = publisher.metadata(archive)
            with patch.object(publisher, 'downloaded', return_value=archive.read_bytes()), patch.object(publisher.urllib.request, 'urlopen') as upload:
                self.assertEqual(publisher.publish(archive, data, publisher.sha(archive.read_bytes()), 'test-only'), 'already-verified')
                upload.assert_not_called()
            with patch.object(publisher, 'downloaded', return_value=b'other'), patch.object(publisher.urllib.request, 'urlopen') as upload:
                with self.assertRaisesRegex(ValueError, 'different bytes'):
                    publisher.publish(archive, data, publisher.sha(archive.read_bytes()), 'test-only')
                upload.assert_not_called()

    def test_changed_archive_is_rejected_before_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.archive(directory, '[package]\nname="fixture"\nversion="1.0.0"')
            with patch.object(publisher, 'downloaded', return_value=None), patch.object(publisher.urllib.request, 'urlopen') as upload:
                with self.assertRaisesRegex(ValueError, 'changed before upload'):
                    publisher.publish(archive, publisher.metadata(archive), '0'*64, 'test-only')
                upload.assert_not_called()


    def test_public_download_waits_past_two_minutes_with_backoff(self):
        clock = Clock()
        calls = []
        def fake(name, version):
            calls.append((name, version))
            return b'crate' if clock.now >= 300 else None
        with patch.object(publisher, 'downloaded', side_effect=fake):
            self.assertEqual(publisher.wait_public('fixture', '1.0.0', sleep=clock.sleep, clock=clock.time), b'crate')
        self.assertEqual(clock.sleeps[:5], [5, 10, 20, 30, 60])
        self.assertTrue(all(delay <= 60 for delay in clock.sleeps))
        self.assertEqual(calls[0], ('fixture', '1.0.0'))

    def test_public_download_gives_up_at_the_deadline(self):
        clock = Clock()
        with patch.object(publisher, 'downloaded', return_value=None):
            self.assertIsNone(publisher.wait_public('fixture', '1.0.0', sleep=clock.sleep, clock=clock.time))
        self.assertEqual(clock.now, publisher.VISIBILITY_TIMEOUT)

    def test_upload_timeout_reports_visibility_not_byte_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.archive(directory, '[package]\nname="fixture"\nversion="1.0.0"')
            response = io.BytesIO(b'{}')
            response.__enter__ = lambda *a: response
            response.__exit__ = lambda *a: False
            with patch.object(publisher, 'downloaded', return_value=None), \
                 patch.object(publisher.urllib.request, 'urlopen', return_value=response), \
                 patch.object(publisher, 'wait_public', return_value=None):
                with self.assertRaisesRegex(RuntimeError, 'not publicly downloadable'):
                    publisher.publish(archive, publisher.metadata(archive), publisher.sha(archive.read_bytes()), 'test-only')


    def test_preflight_rejects_every_drifted_published_version_before_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = self.archive(directory, '[package]\nname="fixture"\nversion="1.0.0"')
            data = publisher.metadata(archive)
            digest = publisher.sha(archive.read_bytes())
            other = dict(data, name='other')
            public = {'fixture': archive.read_bytes(), 'other': b'old bytes'}
            with patch.object(publisher, 'downloaded', side_effect=lambda name, version: public.get(name)), \
                 patch.object(publisher.urllib.request, 'urlopen') as upload:
                self.assertEqual(publisher.preflight([(archive, data, digest)]), ['already-verified'])
                with self.assertRaisesRegex(ValueError, 'nothing was uploaded.*other 1.0.0'):
                    publisher.preflight([(archive, data, digest), (archive, other, digest)])
                upload.assert_not_called()
            with patch.object(publisher, 'downloaded', return_value=None):
                self.assertEqual(publisher.preflight([(archive, data, digest)]), ['verified-candidate'])


    def test_pace_spaces_only_consecutive_uploads(self):
        clock = Clock()
        publisher._last_upload[0] = None
        env = {publisher.UPLOAD_SPACING_ENV: '420'}
        publisher.pace(sleep=clock.sleep, clock=clock.time, environ=env)
        self.assertEqual(clock.sleeps, [])
        clock.now += 100
        publisher.pace(sleep=clock.sleep, clock=clock.time, environ=env)
        self.assertEqual(clock.sleeps, [320])
        publisher._last_upload[0] = None
        publisher.pace(sleep=clock.sleep, clock=clock.time, environ={})
        publisher.pace(sleep=clock.sleep, clock=clock.time, environ={})
        self.assertEqual(clock.sleeps, [320])
        with self.assertRaisesRegex(ValueError, 'between 0 and 3600'):
            publisher.pace(sleep=clock.sleep, clock=clock.time, environ={publisher.UPLOAD_SPACING_ENV: '-1'})
        publisher._last_upload[0] = None


    def crate_bytes(self, files, name='lib', version='1.0.0'):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
            for path, text in files.items():
                data = text.encode()
                info = tarfile.TarInfo(f'{name}-{version}/{path}')
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
        return buffer.getvalue()

    LIB = {'Cargo.toml': '[package]\nname="lib"\nversion="1.0.0"\n', 'src/lib.rs': 'pub fn f() {}', 'Cargo.lock': 'rustix 1.0.7'}

    def test_library_with_only_lockfile_drift_is_skipped_not_failed(self):
        ours = self.crate_bytes({**self.LIB, 'Cargo.lock': 'rustix 1.1.6'})
        theirs = self.crate_bytes(self.LIB)
        self.assertTrue(publisher.lockfile_only_library_drift(ours, theirs))
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / 'lib-1.0.0.crate'
            archive.write_bytes(ours)
            data = {'name': 'lib', 'vers': '1.0.0'}
            with patch.object(publisher, 'downloaded', return_value=theirs), \
                 patch.object(publisher.urllib.request, 'urlopen') as upload:
                self.assertEqual(publisher.preflight([(archive, data, publisher.sha(ours))]), [publisher.LOCKFILE_ONLY_DRIFT])
                self.assertEqual(publisher.publish(archive, data, publisher.sha(ours), 'test-only'), publisher.LOCKFILE_ONLY_DRIFT)
                upload.assert_not_called()
            kept = publisher.keep_published(Path(directory), data, theirs)
            self.assertEqual(kept, publisher.sha(theirs))
            self.assertEqual((Path(directory) / 'published-lockfile-drift/lib-1.0.0.crate').read_bytes(), theirs)

    def test_binaries_must_still_be_bumped_for_lockfile_drift(self):
        for extra in ({'src/main.rs': 'fn main() {}'}, {'src/bin/tool.rs': 'fn main() {}'},
                      {'Cargo.toml': self.LIB['Cargo.toml'] + '[[bin]]\nname="tool"\npath="tools/tool.rs"\n', 'tools/tool.rs': 'fn main() {}'}):
            base = {**self.LIB, **extra}
            ours = self.crate_bytes({**base, 'Cargo.lock': 'rustix 1.1.6'})
            self.assertFalse(publisher.lockfile_only_library_drift(ours, self.crate_bytes(base)), extra)
            with tempfile.TemporaryDirectory() as directory:
                archive = Path(directory) / 'lib-1.0.0.crate'
                archive.write_bytes(ours)
                with patch.object(publisher, 'downloaded', return_value=self.crate_bytes(base)):
                    with self.assertRaisesRegex(ValueError, 'different bytes'):
                        publisher.preflight([(archive, {'name': 'lib', 'vers': '1.0.0'}, publisher.sha(ours))])

    def test_autobins_false_keeps_a_main_rs_library(self):
        base = {**self.LIB, 'Cargo.toml': self.LIB['Cargo.toml'] + 'autobins=false\n', 'src/main.rs': 'fn main() {}'}
        base['Cargo.toml'] = '[package]\nname="lib"\nversion="1.0.0"\nautobins=false\n'
        self.assertTrue(publisher.lockfile_only_library_drift(self.crate_bytes({**base, 'Cargo.lock': 'x'}), self.crate_bytes(base)))

    def test_any_other_difference_is_still_drift(self):
        theirs = self.crate_bytes(self.LIB)
        for ours in (self.crate_bytes({**self.LIB, 'Cargo.lock': 'new', 'src/lib.rs': 'pub fn g() {}'}),
                     self.crate_bytes({**self.LIB, 'Cargo.lock': 'new', 'README.md': 'added'}),
                     self.crate_bytes({k: v for k, v in self.LIB.items() if k != 'Cargo.lock'})):
            self.assertFalse(publisher.lockfile_only_library_drift(ours, theirs))


class Clock:
    def __init__(self):
        self.now = 0
        self.sleeps = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


if __name__ == '__main__': unittest.main()
