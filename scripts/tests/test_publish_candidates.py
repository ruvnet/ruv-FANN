import importlib.util
import io
import json
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


if __name__ == '__main__': unittest.main()
