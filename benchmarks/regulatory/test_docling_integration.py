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
            "body": {"children": [{"$ref": "#/texts/0"}, {"$ref": "#/tables/0"}]},
            "groups": [],
            "texts": [{"self_ref": "#/texts/0", "text": "A test equipment label", "prov": [{"page_no": 40}]}],
            "tables": [{
                "self_ref": "#/tables/0",
                "prov": [{"page_no": 40}],
                "data": {"num_rows": 2, "num_cols": 2, "table_cells": [
                    {"text": "Voltage", "start_row_offset_idx": 0, "start_col_offset_idx": 0},
                    {"text": "Current", "start_row_offset_idx": 0, "start_col_offset_idx": 1},
                    {"text": "480 V", "start_row_offset_idx": 1, "start_col_offset_idx": 0},
                    {"text": "20 A", "start_row_offset_idx": 1, "start_col_offset_idx": 1}
                ]}
            }]
        }

    def test_original_page_and_structured_table_rows_preserved(self):
        structured = []
        blocks, tables = extract_page(self.document, 40, tables_out=structured)
        self.assertEqual(tables, 1)
        self.assertEqual(blocks, [
            "A test equipment label",
            "Table #/tables/0 row 0: column 0: Voltage | column 1: Current",
            "Table #/tables/0 row 1: column 0: 480 V | column 1: 20 A"
        ])
        self.assertEqual(structured[0]["num_rows"], 2)
        self.assertEqual(structured[0]["num_cols"], 2)
        self.assertEqual(structured[0]["cells"][0]["row_span"], 1)
        self.assertEqual(len(structured[0]["cells"]), 4)
        chunks = make_chunks({40: blocks}, {40}, 120)
        self.assertTrue(chunks)
        self.assertTrue(all(chunk["page"] == 40 for chunk in chunks))

    def test_nested_groups_preserve_reading_order(self):
        self.document["groups"] = [{"self_ref": "#/groups/0", "children": [
            {"$ref": "#/tables/0"}, {"$ref": "#/texts/0"}]}]
        self.document["body"]["children"] = [{"$ref": "#/groups/0"}]
        blocks, count = extract_page(self.document, 40)
        self.assertEqual(count, 1)
        self.assertEqual(blocks, [
            "Table #/tables/0 row 0: column 0: Voltage | column 1: Current",
            "Table #/tables/0 row 1: column 0: 480 V | column 1: 20 A",
            "A test equipment label"
        ])

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

    def test_rejects_misaligned_cell_spans(self):
        self.document["tables"][0]["data"]["table_cells"][0]["row_span"] = 10
        with self.assertRaises(ValueError):
            extract_page(self.document, 40)

    def test_skips_furniture_without_losing_page_provenance(self):
        self.document["texts"].append({
            "self_ref": "#/texts/1", "text": "Page footer",
            "content_layer": "furniture", "prov": [{"page_no": 40}]
        })
        self.document["body"]["children"].append({"$ref": "#/texts/1"})
        blocks, _ = extract_page(self.document, 40)
        self.assertNotIn("Page footer", blocks)

    def test_docling_table_caption_precedes_rows(self):
        self.document["texts"].append({
            "self_ref": "#/texts/1", "text": "Table 1: device ratings",
            "content_layer": "body", "prov": [{"page_no": 40}]
        })
        self.document["tables"][0]["captions"] = [{"$ref": "#/texts/1"}]
        self.document["tables"][0]["children"] = [{"$ref": "#/texts/1"}]
        blocks, _ = extract_page(self.document, 40)
        self.assertEqual(blocks[1], "Table 1: device ratings")
        self.assertTrue(blocks[2].startswith("Table #/tables/0 row 0:"))

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
                        "table_cells": 4, "tables_sha256": h0,
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
