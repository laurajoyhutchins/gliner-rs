"""Docling cross-repository contract, page identity, and table-parity tests."""
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from run_docling import extract_page, parse_pages
from prepare_corpus import make_chunks

CONTRACTS = Path(__file__).resolve().parent / "contracts"


class DoclingBridgeTests(unittest.TestCase):
    def setUp(self):
        self.document = {
            "schema_name": "DoclingDocument",
            "version": "1.10.0",
            "pages": {"40": {"page_no": 40, "size": {"width": 612, "height": 792}}},
            "texts": [{"text": "A test equipment label", "prov": [{"page_no": 40}]}],
            "tables": [{
                "prov": [{"page_no": 40}],
                "data": {"table_cells": [
                    {"text": "Voltage", "start_row_offset_idx": 0, "start_col_offset_idx": 0},
                    {"text": "Current", "start_row_offset_idx": 0, "start_col_offset_idx": 1},
                    {"text": "480 V", "start_row_offset_idx": 1, "start_col_offset_idx": 0},
                    {"text": "20 A", "start_row_offset_idx": 1, "start_col_offset_idx": 1}
                ]}
            }]
        }

    def test_original_page_and_structured_table_rows_preserved(self):
        blocks, tables = extract_page(self.document, 40)
        self.assertEqual(tables, 1)
        self.assertEqual(blocks, ["A test equipment label", "Voltage | Current", "480 V | 20 A"])
        chunks = make_chunks({40: blocks}, {40}, 120)
        self.assertTrue(chunks)
        self.assertTrue(all(chunk["page"] == 40 for chunk in chunks))

    def test_invalid_provenance_is_not_silently_relabelled(self):
        with self.assertRaises(ValueError):
            extract_page(self.document, 41)
        self.document["texts"][0]["prov"] = [{"page_no": 2}]
        with self.assertRaises(ValueError):
            extract_page(self.document, 40)

    def test_table_missing_structural_grid_is_rejected(self):
        self.document["tables"][0]["data"] = {}
        with self.assertRaises(ValueError):
            extract_page(self.document, 40)

    def test_specified_page_ranges_are_validated(self):
        self.assertEqual(parse_pages("40,80-82,40"), [40, 80, 81, 82])
        for invalid in ["0", "10-5", "a", "1, ", "999-2001"]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_pages(invalid)

    def test_docling_document_json_schema(self):
        validator = Draft202012Validator(json.loads(
            (CONTRACTS / "docling-export.schema.json").read_text()
        ))
        validator.validate(self.document)
        self.document["texts"][0]["prov"] = []
        with self.assertRaises(Exception):
            validator.validate(self.document)

    def test_docling_corpus_json_schema(self):
        h0 = "a" * 64
        h1 = "b" * 64
        corpus = {
            "format_version": 1, "source_sha256": h0,
            "extraction": "docling_document_json", "pages": [40],
            "chunks": [{"id": "p0040-c0000", "page": 40, "text": "Test",
                        "sha256": h1}],
            "docling": {"source_revision": "abcdef",
                        "binary_sha256": h1, "mode": "standard", "tables": 1,
                        "page_receipts": [{"page": 40, "docling_json_sha256": h0,
                                           "text_elements": 1, "tables": 1,
                                           "docling_schema_version": "1.10.0"}]}
        }
        validator = Draft202012Validator(json.loads(
            (CONTRACTS / "docling-corpus.schema.json").read_text()))
        validator.validate(corpus)
        corpus["source_sha256"] = "mismatched"
        with self.assertRaises(Exception):
            validator.validate(corpus)


if __name__ == "__main__":
    unittest.main()
