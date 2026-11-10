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
    def exercise(self, corrupt=False, consumer_passed=True, existing=None, swarm_version='2.0.2'):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            (base / 'cargo-candidates').mkdir()
            crates, packages = [], []
            for index in range(29):
                archive = base / ('cargo-candidates' if index < 27 else '.') / f'package-{index}.{"crate" if index < 27 else "tgz"}'
                archive.write_bytes(f'exact published bytes {index}'.encode())
                name = f'package-{index}' if index < 27 else ('ruv-swarm', 'cuda-wasm')[index - 27]
                version = '1.0.0' if index < 27 else (swarm_version, '2.0.1')[index - 27]
                row = {'name': name, 'version': version, 'sha256': module.digest(archive), 'archive': str(archive), 'status': 'published-and-byte-verified'}
                (crates if index < 27 else packages).append(row)
            for name, field, rows in [('cargo-publication', 'crates', crates), ('npm-publication', 'packages', packages)]:
                (base / (name + '.json')).write_text(json.dumps({'sourceSha': 'reviewed', field: rows}))
            (base / 'public-consumers.json').write_text(json.dumps({'sourceSha': 'reviewed', 'status': 'passed' if consumer_passed else 'failed', **dict.fromkeys(('npmNormalInstall','coreAndSimdTraining','persistentCli','cudaPublicCompiler','cargoPublicDenseTraining'), True)}))
            calls = []
            def gh_lookup(argv):
                if existing is None:
                    return SimpleNamespace(returncode=1, stdout='')
                return SimpleNamespace(returncode=0, stdout=json.dumps({'isDraft': existing == 'draft'}))
            def gh(*args):
                calls.append(args)
                if args[:2] == ('release', 'download'):
                    folder = Path(args[args.index('--dir') + 1])
                    if existing is not None and folder.name == 'github-existing-download':
                        shutil.copytree(base / 'github-release-assets', folder)
                        (folder / 'cargo-publication.json').write_text('{"sourceSha": "older commit"}')
                        if existing == 'different':
                            (folder / 'package-3.crate').write_bytes(b'other release bytes')
                        return ''
                    shutil.copytree(base / 'github-release-assets', folder)
                    if corrupt:
                        (folder / 'package-0.crate').write_bytes(b'wrong bytes')
                return ''
            with patch.dict(os.environ, {'GITHUB_ACTIONS': 'true', 'GITHUB_WORKFLOW': 'protected-release', 'GITHUB_REF': 'refs/heads/main', 'GITHUB_SHA': 'reviewed'}), patch.object(module.sys, 'argv', ['script', str(base)]), patch.object(module, 'run', side_effect=gh), patch.object(module.subprocess, 'run', side_effect=lambda argv, **kw: gh_lookup(argv)):
                if existing == 'same':
                    module.main()
                    receipt = json.loads((base / 'github-release.json').read_text())
                    self.assertEqual(receipt['status'], 'already-released')
                    self.assertEqual(receipt['tag'], f'ruv-fann-registry-v{swarm_version}')
                    self.assertFalse(any(c[:2] in (('release', 'create'), ('release', 'edit')) for c in calls))
                elif corrupt or not consumer_passed or existing in ('different', 'draft') or swarm_version is None:
                    with self.assertRaises(ValueError):
                        module.main()
                    self.assertFalse(any(c[:2] == ('release', 'edit') for c in calls))
                    if not corrupt:
                        self.assertFalse(any(c[:2] == ('release', 'create') for c in calls))
                else:
                    module.main()
                    receipt = json.loads((base / 'github-release.json').read_text())
                    self.assertEqual(receipt['status'], 'published-and-byte-verified')
                    self.assertEqual(len(receipt['assets']), 33)
                    self.assertEqual(sum(c[:2] == ('release', 'download') for c in calls), 2)
                    create = next(c for c in calls if c[:2] == ('release', 'create'))
                    self.assertEqual(create[2], f'ruv-fann-registry-v{swarm_version}')
    def test_exact_assets_are_verified_before_and_after_publication(self):
        self.exercise()
    def test_corrupt_draft_cannot_be_published(self):
        self.exercise(corrupt=True)
    def test_failed_consumers_cannot_create_release(self):
        self.exercise(consumer_passed=False)
    def test_rerun_with_identical_archives_is_already_released(self):
        self.exercise(existing='same')
    def test_existing_release_with_different_archives_fails(self):
        self.exercise(existing='different')
    def test_existing_draft_release_fails(self):
        self.exercise(existing='draft')
    def test_missing_ruv_swarm_version_fails(self):
        self.exercise(swarm_version=None)
