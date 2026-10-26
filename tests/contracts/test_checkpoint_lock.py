"""Validate the pinned-checkpoint JSON Schema independently of Rust parsing."""

import json
import unittest
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "contracts" / "checkpoint-lock.schema.json"
EXAMPLE = {
    "version": 1,
    "repo": "example/checkpoint",
    "revision": "0123456789abcdef0123456789abcdef01234567",
    "sha256": {
        "config.json": "a" * 64,
        "encoder_config/config.json": "b" * 64,
        "tokenizer.json": "c" * 64,
        "model.safetensors": "d" * 64,
    },
}


class CheckpointLockContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_browser_manifest_is_complete_and_uses_exact_repository_contract(self):
        import re

        root = SCHEMA_PATH.parents[1]
        locks = json.loads((root / "website/src/lib/checkpoint-locks.json").read_text(encoding="utf-8"))
        declared = set(re.findall(
            r'hfRepo:\s*"([^"]+)"',
            (root / "website/src/lib/models.ts").read_text(encoding="utf-8"),
        ))
        repos = [lock["repo"] for lock in locks]
        self.assertEqual(len(repos), len(set(repos)), "duplicate checkpoint locks")
        self.assertEqual(set(repos), declared, "every browser model must be pinned")
        for lock in locks:
            with self.subTest(repo=lock["repo"]):
                self.validator.validate(lock)

    def test_complete_fixture_is_structurally_valid(self):
        self.validator.validate(EXAMPLE)

    def test_rejects_unpinned_revision(self):
        data = deepcopy(EXAMPLE)
        data["revision"] = "main"
        self.assertFalse(self.validator.is_valid(data))

    def test_rejects_unexpected_and_missing_hashes(self):
        data = deepcopy(EXAMPLE)
        del data["sha256"]["model.safetensors"]
        self.assertFalse(self.validator.is_valid(data))
        data["sha256"]["model.safetensors"] = "d" * 64
        data["sha256"]["unexpected.json"] = "a" * 64
        self.assertFalse(self.validator.is_valid(data))

    def test_rejects_malformed_digests(self):
        data = deepcopy(EXAMPLE)
        data["sha256"]["config.json"] = "not-a-hash"
        self.assertFalse(self.validator.is_valid(data))


if __name__ == "__main__":
    unittest.main()
