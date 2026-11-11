"""CPU-only harness regressions. The mock launcher does NOT run PAN2/CUDA."""
import contextlib
import hashlib
import io
import json
import os
import pathlib
import signal
import struct
import subprocess
import tempfile
import time
import unittest
from unittest import mock

import pan2_compare as pan


class PairedHarness(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        self.repo = self.root / 'repo'
        self.repo.mkdir()
        self.base = self.root / 'baseline/transformer'
        self.candidate = self.root / 'candidate/transformer'
        self.cache = self.root / 'hf-home'
        for path in (self.base, self.candidate, self.cache):
            path.mkdir(parents=True)
        header = json.dumps({'weight': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]}}).encode()
        header += b' ' * (-len(header) % 8)
        data = struct.pack('<Q', len(header)) + header + bytes.fromhex('803e003f')
        edited = data[:-4] + bytes.fromhex('403f003f')
        (self.base / 'model.safetensors').write_bytes(data)
        (self.candidate / 'model.safetensors').write_bytes(edited)
        for path in (self.base, self.candidate):
            (path / 'config.json').write_text('{"identical":true}')
        self.receipt = self.root / 'receipt.json'
        self.receipt.write_text(json.dumps({'schema': 'ruv-fann.tensor-receipt/1',
            'base_sha256': hashlib.sha256(data).hexdigest(), 'candidate_sha256': hashlib.sha256(edited).hexdigest(),
            'edits': [{'offset': len(data)-4, 'before': [128, 62], 'after': [64, 63]}],
            'behavioral_validation': 'not_run'}))
        self.prompts = self.root / 'prompts.json'
        self.prompts.write_text('[{"type":"t2v","prompt":"CPU mock only"}]')
        (self.cache / 'hub').mkdir()
        (self.cache / 'hub/model').write_bytes(b'offline mock model asset')
        (self.repo / 'inference').mkdir()
        # Real subprocess, fake model: outputs argument sidecars, never claims CUDA.
        (self.repo / 'inference/inference.sh').write_text('''#!/bin/bash
set -eu
python3 - "$@" <<'MOCK'
import json, pathlib, sys
checkpoint, output, prompts = sys.argv[1:]
out=pathlib.Path(output);out.mkdir()
(out/'clip.mp4').write_bytes(b'CPU MOCK, NOT VIDEO')
args={'checkpoint_path':checkpoint,'prompt_json':prompts,'seed':123,'num_inference_steps':50,'flow_shift':9,'guidance_scale':2.5,'video_height':704,'video_width':1248,'fps':24,'dtype':'bf16'}
(out/'clip_config.json').write_text(json.dumps({'arguments':args}))
MOCK
''')
        self.git('init', '-q')
        self.git('add', '.')
        self.git('-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')
        self.pin = self.git('rev-parse', 'HEAD').strip()
        self.args = ['--repo', str(self.repo), '--baseline', str(self.base), '--candidate', str(self.candidate),
                     '--prompts', str(self.prompts), '--receipt', str(self.receipt), '--hf-home', str(self.cache),
                     '--output', str(self.root / 'result')]

    def git(self, *args):
        return subprocess.check_output(['git', '-C', str(self.repo), *args], text=True)

    def invoke(self, after_run=None):
        real = pan.subprocess.check_output
        def checked(command, **kwargs):
            if command == ['nvidia-smi']:
                return 'CPU-ONLY MOCK GPU INVENTORY'
            return real(command, **kwargs)
        launch = pan.run_process
        def run(*a, **kw):
            launch(*a, **kw)
            if after_run:
                after_run(kw)
        with mock.patch.object(pan, 'PIN', self.pin), mock.patch.object(pan.shutil, 'which', return_value='/mock/nvidia-smi'), \
             mock.patch.object(pan.subprocess, 'check_output', side_effect=checked), \
             mock.patch.object(pan, 'run_process', side_effect=run), contextlib.redirect_stdout(io.StringIO()):
            return pan.main(self.args)

    def test_mock_pair_records_bound_inputs_without_gpu_claim(self):
        result = self.invoke()
        self.assertEqual(set(result['runs']), {'baseline', 'candidate'})
        self.assertIn('requires independent scoring', result['behavioral_validation'])
        self.assertEqual(result['gpu_inventory'], 'CPU-ONLY MOCK GPU INVENTORY')
        self.assertTrue((self.root / 'result/receipt.json').is_file())

    def test_git_assume_unchanged_cannot_hide_modified_launcher(self):
        self.git('update-index', '--assume-unchanged', 'inference/inference.sh')
        (self.repo / 'inference/inference.sh').write_text('malicious launcher')
        self.assertEqual(self.git('status', '--porcelain').strip(), '')
        with self.assertRaisesRegex(RuntimeError, 'must not hide files'):
            self.invoke()
        self.assertFalse((self.root / 'result').exists())

    def test_different_nonshard_config_rejected_before_launch(self):
        (self.candidate / 'config.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(RuntimeError, 'exactly one edited shard'):
            self.invoke()
        self.assertFalse((self.root / 'result').exists())

    def test_nested_or_wrong_loader_layout_rejected(self):
        (self.candidate / 'nested').mkdir()
        (self.candidate / 'model.safetensors').rename(self.candidate / 'nested/model.safetensors')
        with self.assertRaisesRegex(RuntimeError, 'top-level'):
            self.invoke()
        with self.assertRaisesRegex(RuntimeError, 'named transformer'):
            pan.checkpoint(self.root / 'baseline')

    def test_loader_fallback_file_rejected(self):
        (self.base.parent / 'model_state.pt').write_bytes(b'alternate weights')
        with self.assertRaisesRegex(RuntimeError, 'bypass'):
            self.invoke()

    def test_symlink_and_fifo_inputs_rejected_without_blocking(self):
        (self.base / 'alias').symlink_to(self.base / 'model.safetensors')
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            pan.checkpoint(self.base)
        (self.base / 'alias').unlink()
        os.mkfifo(self.base / 'pipe')
        with self.assertRaisesRegex(RuntimeError, 'special file'):
            pan.checkpoint(self.base)

    def test_hf_snapshot_internal_link_allowed_external_link_rejected(self):
        (self.cache / 'snapshot').symlink_to(self.cache / 'hub/model')
        self.assertIn('link', pan.inventory(self.cache, cache=True)['snapshot'])
        (self.cache / 'escape').symlink_to(self.receipt)
        with self.assertRaisesRegex(RuntimeError, 'unsafe symlink'):
            pan.inventory(self.cache, cache=True)

    def test_hf_credentials_rejected_without_opening_them(self):
        token = self.cache / 'token'
        token.write_text('fixture credential - must not be read')
        real = pan.digest
        def checked(path):
            if path == token:
                self.fail('credential was opened for hashing')
            return real(path)
        with mock.patch.object(pan, 'digest', side_effect=checked), self.assertRaisesRegex(RuntimeError, 'no credentials'):
            self.invoke()

    def test_receipt_hashes_alone_cannot_lie_about_undo_bytes(self):
        receipt = json.loads(self.receipt.read_text())
        receipt['edits'][0]['before'] = [0, 0]
        self.receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(RuntimeError, 'undo does not restore'):
            self.invoke()

    def test_receipt_change_after_baseline_rejects_before_candidate(self):
        calls = []
        def tamper(kw):
            calls.append(kw)
            self.receipt.write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'inputs changed'):
            self.invoke(tamper)
        self.assertEqual(len(calls), 1)
        self.assertFalse((self.root / 'result/comparison.json').exists())

    def test_candidate_change_after_baseline_rejects_before_candidate(self):
        calls = []
        def tamper(kw):
            calls.append(kw)
            (self.candidate / 'model.safetensors').write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError, 'inputs changed'):
            self.invoke(tamper)
        self.assertEqual(len(calls), 1)

    def test_prompt_reference_image_change_is_detected(self):
        image = self.root / 'image.png'
        image.write_bytes(b'image fixture')
        self.prompts.write_text(json.dumps([{'type': 'i2v', 'prompt': 'CPU mock', 'image': str(image)}]))
        with self.assertRaisesRegex(RuntimeError, 'inputs changed'):
            self.invoke(lambda kw: image.write_bytes(b'tampered'))

    def test_cache_change_after_baseline_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'inputs changed'):
            self.invoke(lambda kw: (self.cache / 'hub/model').write_bytes(b'tampered'))

    def test_checkout_change_after_baseline_rejected(self):
        with self.assertRaisesRegex(RuntimeError, 'inputs changed'):
            self.invoke(lambda kw: (self.repo / 'inference/inference.sh').write_text('tampered'))

    def test_overlapping_output_rejected(self):
        self.args[-1] = str(self.base / 'results')
        with self.assertRaisesRegex(RuntimeError, 'disjoint'):
            self.invoke()

    def test_environment_removes_shell_injection_and_separates_kernel_cache(self):
        namespace = type('Args', (), dict(repo=self.repo, output=self.root / 'out', hf_home=self.cache, seed=123))
        with mock.patch.dict(os.environ, {'BASH_ENV': '/evil', 'PYTHONPATH': '/evil', 'CONDA_ENV': 'wrong', 'EXTRA_ARGS': '--wrong', 'TRITON_CACHE_DIR': '/shared'}):
            baseline = pan.environment(namespace, 'baseline')
            candidate = pan.environment(namespace, 'candidate')
        self.assertNotIn('BASH_ENV', baseline)
        self.assertNotIn('PYTHONPATH', baseline)
        self.assertEqual(baseline['CONDA_ENV'], 'pan')
        self.assertEqual(baseline['EXTRA_ARGS'], '')
        self.assertNotEqual(baseline['TRITON_CACHE_DIR'], candidate['TRITON_CACHE_DIR'])
        self.assertEqual(baseline['HF_HUB_OFFLINE'], '1')

    def test_matching_but_incomplete_output_sets_cannot_qualify(self):
        self.prompts.write_text('[{"prompt":"first"},{"prompt":"second"}]')
        with self.assertRaisesRegex(RuntimeError, 'one video per prompt'):
            self.invoke()
        self.assertFalse((self.root / 'result/candidate').exists())

    def test_wrong_output_checkpoint_settings_rejected(self):
        def tamper(kw):
            out = self.root / 'result/baseline/clip_config.json'
            out.write_text('{"arguments":{"checkpoint_path":"wrong"}}')
        with self.assertRaisesRegex(RuntimeError, 'settings'):
            self.invoke(tamper)

    def test_successful_launcher_also_kills_leftover_workers(self):
        marker = self.root / 'leftover-pid'
        with (self.root / 'success.log').open('w') as log:
            pan.run_process(['bash', '-c', 'sleep 60 & echo $! > "$1"', 'fixture', str(marker)],
                            cwd=self.root, env=os.environ.copy(), log=log, timeout=5)
        pid = int(marker.read_text())
        for _ in range(100):
            proc = pathlib.Path(f'/proc/{pid}/stat')
            try:
                state = proc.read_text().split()[2]
            except (FileNotFoundError, ProcessLookupError):
                break
            if state == 'Z':
                break
            time.sleep(0.01)
        else:
            self.fail('worker survived successful launcher cleanup')

    def test_symlink_video_cannot_hash_an_external_file(self):
        def tamper(kw):
            video = self.root / 'result/baseline/clip.mp4'
            video.unlink()
            video.symlink_to(self.receipt)
        with self.assertRaisesRegex(RuntimeError, 'symlink'):
            self.invoke(tamper)

    def test_prompt_file_swap_after_parsing_rejected_before_launch(self):
        snapshot = pan.json_snapshot
        def swap(path):
            result = snapshot(path)
            if path == self.prompts:
                self.prompts.write_text('[{"type":"t2v","prompt":"swapped"}]')
            return result
        with mock.patch.object(pan, 'json_snapshot', side_effect=swap), \
             self.assertRaisesRegex(RuntimeError, 'inputs changed'):
            self.invoke()
        self.assertFalse((self.root / 'result/baseline').exists())

    def test_timeout_kills_worker_process_group(self):
        marker = self.root / 'worker-pid'
        script = self.root / 'timeout.sh'
        script.write_text('sleep 60 &\necho $! > "$1"\nwait\n')
        with (self.root / 'timeout.log').open('w') as log:
            with self.assertRaises(subprocess.TimeoutExpired):
                pan.run_process(['bash', str(script), str(marker)], cwd=self.root,
                                env=os.environ.copy(), log=log, timeout=0.2)
        pid = int(marker.read_text())
        for _ in range(100):
            proc = pathlib.Path(f'/proc/{pid}/stat')
            try:
                state = proc.read_text().split()[2]
            except (FileNotFoundError, ProcessLookupError):
                break
            if state == 'Z':
                break
            time.sleep(0.01)
        else:
            self.fail('worker survived timeout cleanup')


if __name__ == '__main__':
    unittest.main()
