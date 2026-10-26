"""Structural contract tests for the public adaptive batch route."""
import json
import unittest
from pathlib import Path
from jsonschema import Draft202012Validator

SCHEMA = json.loads((Path(__file__).resolve().parents[2] / "contracts/http-extract-batch.schema.json").read_text())

class HttpBatchContractTest(unittest.TestCase):
    def test_schema_itself_is_valid(self):
        Draft202012Validator.check_schema(SCHEMA)

    def test_public_request_and_response_shapes(self):
        request = {"texts": ["Alice works for Acme.", "电机运行在480伏。"], "entities": ["person", "company"],
                   "threshold": 0.5, "spans": True, "confidence": True}
        response = {"model": "small", "results": [{"entities": {}}, {"entities": {}}]}
        Draft202012Validator(SCHEMA).validate(request)
        Draft202012Validator({**SCHEMA, "$ref": "#/$defs/response"}).validate(response)

    def test_reject_missing_texts_and_invalid_schema_shape(self):
        validator = Draft202012Validator(SCHEMA)
        for request in (
            {"texts": [], "entities": ["person"]},
            {"texts": ["text"], "entities": []},
            {"texts": ["text"], "entities": ["person"], "text": "not allowed"},
            {"texts": ["text"], "entities": ["person"], "threshold": 2},
            {"texts": ["text"] * 65, "entities": ["person"]},
            {"texts": ["text"], "relations": ["works_for"], "bogus": True},
        ):
            self.assertFalse(validator.is_valid(request), request)

if __name__ == "__main__":
    unittest.main()
