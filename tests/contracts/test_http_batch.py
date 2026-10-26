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

    def test_odcs_companion_matches_json_schema(self):
        odcs = json.loads((Path(__file__).resolve().parents[2] / "contracts/http-extract-batch.odcs.json").read_text())
        self.assertEqual(odcs["kind"], "DataContract")
        self.assertEqual(odcs["apiVersion"], "v3.1.0")
        request, response = odcs["schema"]
        self.assertEqual([x["name"] for x in odcs["schema"]], ["BatchExtractRequest", "BatchExtractResponse"])
        for obj, ref in ((request, SCHEMA["$defs"]["request"]), (response, SCHEMA["$defs"]["response"])):
            properties = {x["name"]: x for x in obj["properties"]}
            self.assertEqual(set(properties), set(ref["properties"]))
            self.assertEqual({k for k, x in properties.items() if x["required"]}, set(ref.get("required", [])))
            for name, odcs_field in properties.items():
                json_type = ref["properties"][name]["type"]
                if isinstance(json_type, list):
                    json_type = next(x for x in json_type if x != "null")
                self.assertEqual(odcs_field["logicalType"], json_type, name)

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
