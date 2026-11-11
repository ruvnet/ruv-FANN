"""Release-oracle regressions; fake CLI tests are explicitly not native proof."""
import contextlib
import importlib
import io
import os
import pathlib
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock

import benchmark
import e2e


class CliScripts(unittest.TestCase):
    def test_explicit_binary_override_selects_exact_file(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / 'installed-cli'
            executable.write_bytes(b'fixture binary path only')
            with mock.patch.dict(os.environ, {'WSP_TENSOR_BIN': str(executable)}):
                self.assertEqual(e2e.binary(), executable.resolve())

    def test_target_directory_override_resolves_standalone_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / 'release/ruv-fann-wsp-tensor'
            executable.parent.mkdir()
            executable.write_bytes(b'fixture binary path only')
            with mock.patch.dict(os.environ, {'CARGO_TARGET_DIR': directory}):
                os.environ.pop('WSP_TENSOR_BIN', None)
                self.assertEqual(e2e.binary(), executable)

    def test_windows_target_directory_resolves_exe_without_changing_explicit_path(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = pathlib.Path(directory) / 'release/ruv-fann-wsp-tensor.exe'
            executable.parent.mkdir()
            executable.write_bytes(b'Windows path selection fixture only')
            explicit = pathlib.Path(directory) / 'explicit-installed-name'
            explicit.write_bytes(b'explicit selection fixture only')
            windows = types.SimpleNamespace(name='nt', environ={'CARGO_TARGET_DIR': directory})
            with mock.patch.object(e2e, 'os', windows):
                self.assertEqual(e2e.binary(), executable)
                windows.environ['WSP_TENSOR_BIN'] = str(explicit)
                self.assertEqual(e2e.binary(), explicit)

    def test_optimized_python_rejects_wrong_native_candidate_bytes(self):
        # Fake external CLI reports correct hashes of a deliberately unchanged
        # candidate. The independent BF16 oracle must still fail under -O.
        with tempfile.TemporaryDirectory() as directory:
            fake = pathlib.Path(directory) / 'fake-cli'
            fake.write_text('''#!/usr/bin/env python3
import hashlib,json,pathlib,sys
command,source=sys.argv[1:3];data=pathlib.Path(source).read_bytes();sha=hashlib.sha256(data).hexdigest()
if command=='inspect': print(json.dumps({'sha256':sha}))
elif command=='apply':
 pathlib.Path(sys.argv[3]).write_bytes(data)
 print(json.dumps({'schema':'ruv-fann.tensor-receipt/1','base_sha256':sha,'candidate_sha256':sha,'behavioral_validation':'not_run'}))
else: raise RuntimeError('oracle should have rejected before rollback')
''')
            fake.chmod(0o700)
            env = dict(os.environ, WSP_TENSOR_BIN=str(fake))
            result = subprocess.run([sys.executable, '-O', str(pathlib.Path(e2e.__file__))],
                                    env=env, capture_output=True, text=True, timeout=20)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('candidate differs outside expected BF16 edit', result.stderr)

    def test_cli_process_has_timeout(self):
        with tempfile.TemporaryDirectory() as directory:
            fake = pathlib.Path(directory) / 'fixture'
            fake.write_bytes(b'path fixture')
            with mock.patch.dict(os.environ, {'WSP_TENSOR_BIN': str(fake)}), \
                 mock.patch.object(e2e.subprocess, 'run', side_effect=subprocess.TimeoutExpired('fixture', 60)) as run:
                with self.assertRaises(subprocess.TimeoutExpired):
                    e2e.run('inspect', '/fixture')
                self.assertEqual(run.call_args.kwargs['timeout'], 60)

    def test_import_benchmark_does_not_start_benchmark(self):
        with mock.patch.object(benchmark.subprocess, 'run', side_effect=RuntimeError('import spawned CLI')):
            importlib.reload(benchmark)

    def test_benchmark_rejects_unbounded_work_before_loading_cli(self):
        with mock.patch.object(benchmark, 'binary', side_effect=RuntimeError('must validate first')), \
             contextlib.redirect_stderr(io.StringIO()):
            for argv in (['--repeats', '0'], ['--repeats', '101'], ['--sizes-mib', '0'], ['--timeout', '0']):
                with self.assertRaises(SystemExit):
                    benchmark.main(argv)


if __name__ == '__main__':
    unittest.main()
