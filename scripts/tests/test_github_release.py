import importlib.util
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('github_release', Path(__file__).resolve().parents[1] / 'create-github-release.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class GitHubReleaseTests(unittest.TestCase):
    def exercise(self, corrupt=False, consumer_passed=True):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / 'cargo-candidates').mkdir()
            crates, packages = [], []
            for index in range(29):
                archive = base / ('cargo-candidates' if index < 27 else '.') / f'package-{index}.{"crate" if index < 27 else "tgz"}'
                archive.write_bytes(f'exact published bytes {index}'.encode())
                row = {'name': f'package-{index}', 'sha256': module.digest(archive), 'archive': str(archive), 'status': 'published-and-byte-verified'}
                (crates if index < 27 else packages).append(row)
            for name, field, rows in [('cargo-publication', 'crates', crates), ('npm-publication', 'packages', packages)]:
                (base / (name + '.json')).write_text(json.dumps({'sourceSha': 'reviewed', field: rows}))
            (base / 'public-consumers.json').write_text(json.dumps({'sourceSha': 'reviewed', 'status': 'passed' if consumer_passed else 'failed', **dict.fromkeys(('npmNormalInstall','coreAndSimdTraining','persistentCli','cudaPublicCompiler','cargoPublicDenseTraining'), True)}))
            calls = []
            def gh(*args):
                calls.append(args)
                if args[:2] == ('release', 'download'):
                    folder = Path(args[args.index('--dir') + 1])
                    shutil.copytree(base / 'github-release-assets', folder)
                    if corrupt:
                        (folder / 'package-0.crate').write_bytes(b'wrong bytes')
                return ''
            with patch.dict(os.environ, {'GITHUB_ACTIONS': 'true', 'GITHUB_WORKFLOW': 'protected-release', 'GITHUB_REF': 'refs/heads/main', 'GITHUB_SHA': 'reviewed'}), patch.object(module.sys, 'argv', ['script', str(base)]), patch.object(module, 'run', side_effect=gh), patch.object(module.subprocess, 'run', return_value=SimpleNamespace(returncode=1)):
                if corrupt or not consumer_passed:
                    with self.assertRaises(ValueError):
                        module.main()
                    self.assertFalse(any(c[:2] == ('release', 'edit') for c in calls))
                else:
                    module.main()
                    receipt = json.loads((base / 'github-release.json').read_text())
                    self.assertEqual(receipt['status'], 'published-and-byte-verified')
                    self.assertEqual(len(receipt['assets']), 33)
                    self.assertEqual(sum(c[:2] == ('release', 'download') for c in calls), 2)
    def test_exact_assets_are_verified_before_and_after_publication(self):
        self.exercise()
    def test_corrupt_draft_cannot_be_published(self):
        self.exercise(corrupt=True)
    def test_failed_consumers_cannot_create_release(self):
        self.exercise(consumer_passed=False)
