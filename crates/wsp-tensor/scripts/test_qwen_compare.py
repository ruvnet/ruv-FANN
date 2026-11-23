"""CPU-only tests of the Qwen smoke runner's input and report checks."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

import qwen_compare as qwen


class QwenComparison(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.base, self.candidate = root / "baseline", root / "candidate"
        for model in (self.base, self.candidate):
            model.mkdir()
            (model / "config.json").write_text(json.dumps({
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
            }))
            (model / "tokenizer.json").write_text("{}")
        self.shard = "model.safetensors"
        header = b'{"weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}'
        header += b" " * (-len(header) % 8)
        prefix = struct.pack("<Q", len(header)) + header
        base = prefix + bytes.fromhex("803e003f")
        edited = prefix + bytes.fromhex("403f003f")
        (self.base / self.shard).write_bytes(base)
        (self.candidate / self.shard).write_bytes(edited)
        self.receipt = root / "receipt.json"
        self.receipt_data = {
            "schema": "ruv-fann.tensor-receipt/1",
            "base_sha256": hashlib.sha256(base).hexdigest(),
            "candidate_sha256": hashlib.sha256(edited).hexdigest(),
            "edits": [{"offset": len(prefix), "before": [128, 62], "after": [64, 63]}],
            "behavioral_validation": "not_run",
        }
        self.receipt.write_text(json.dumps(self.receipt_data))
        self.output = root / "report.json"

    def run_harness(self, *, selected_shard=None, verify_only=True):
        args = ["--baseline", str(self.base), "--candidate", str(self.candidate),
                "--receipt", str(self.receipt), "--output", str(self.output),
                "--shard", selected_shard or self.shard]
        return qwen.main(args + (["--verify-only"] if verify_only else []))

    def test_cpu_verification_records_exact_receipt_without_loading_cuda(self):
        with mock.patch.object(qwen, "infer", side_effect=AssertionError("inference invoked")):
            report = self.run_harness()
        self.assertEqual(report["status"], "verified_only")
        self.assertEqual(report["promotion"], "not_performed")
        self.assertEqual(report["edit_receipt_sha256"], qwen.digest(self.receipt))
        self.assertTrue(report["rollback_bytes_verified"])
        self.assertEqual(report, json.loads(self.output.read_text()))
        with self.assertRaisesRegex(RuntimeError, "output path must be new"):
            self.run_harness()

    def test_selects_exact_shard_and_rejects_changed_config(self):
        with self.assertRaisesRegex(RuntimeError, "selected shard must be the only changed"):
            self.run_harness(selected_shard="other.safetensors")
        with (self.candidate / "config.json").open("a") as config:
            config.write("\n")
        with self.assertRaisesRegex(RuntimeError, "selected shard must be the only changed"):
            self.run_harness()
        self.assertFalse(self.output.exists())

    def test_refuses_forged_undo_bytes_and_invalid_receipt_marker(self):
        self.receipt_data["edits"][0]["before"] = [0, 0]
        self.receipt.write_text(json.dumps(self.receipt_data))
        with self.assertRaisesRegex(RuntimeError, "undo does not restore baseline"):
            self.run_harness()
        self.receipt_data["behavioral_validation"] = "validated"
        self.receipt.write_text(json.dumps(self.receipt_data))
        with self.assertRaisesRegex(RuntimeError, "unmodified ruv-FANN tensor edit receipt"):
            self.run_harness()

    def test_refuses_symlink_in_checkpoint(self):
        (self.candidate / "link").symlink_to(self.base / "config.json")
        with self.assertRaisesRegex(RuntimeError, "no pickle or symlinks"):
            self.run_harness()

    def test_refuses_alternate_weight_format_even_if_both_trees_contain_it(self):
        for model in (self.base, self.candidate):
            (model / "pytorch_model.bin").write_bytes(b"not Safetensors")
        with self.assertRaisesRegex(RuntimeError, "no pickle or symlinks"):
            self.run_harness()

    def test_sharded_index_matches_all_and_only_local_weights(self):
        self.shard = "model-00001-of-00002.safetensors"
        other_shard = "model-00002-of-00002.safetensors"
        for model in (self.base, self.candidate):
            (model / "model.safetensors").rename(model / self.shard)
            (model / other_shard).write_bytes(b"identical fixture byte stream")
            (model / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"weight": self.shard, "other_weight": other_shard},
            }))
        checkpoint = qwen.checkpoint(self.base)
        self.assertIn(other_shard, checkpoint)
        (self.candidate / "model.safetensors.index.json").write_text(json.dumps({
            "weight_map": {"weight": self.shard, "other_weight": "../unbound.safetensors"},
        }))
        with self.assertRaisesRegex(RuntimeError, "does not match the weight index"):
            self.run_harness()

    def test_mocked_inference_mutation_cannot_write_a_result(self):
        def mutate(*_):
            (self.candidate / "tokenizer.json").write_text("mutated")
            return {}, {}

        with mock.patch.object(qwen, "infer", side_effect=mutate):
            with self.assertRaisesRegex(RuntimeError, "inputs changed during evaluation"):
                self.run_harness(verify_only=False)
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
