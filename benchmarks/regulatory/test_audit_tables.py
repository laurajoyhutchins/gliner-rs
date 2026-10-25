"""Synthetic PDF tests of the independent cell-text cross-check. No NEC text."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import fitz
from audit_tables import audit_tables, normalize


class TableTextAuditTests(unittest.TestCase):
    def sample(self, extracted_text="480 V"):
        temp = tempfile.TemporaryDirectory()
        root = Path(temp.name)
        pdf_path = root / "source.pdf"
        doc = fitz.open()
        page = doc.new_page(width=250, height=120)
        page.insert_text((20, 40), "480 V", fontsize=12)
        doc.save(pdf_path)
        doc.close()
        with fitz.open(pdf_path) as pdf:
            words = pdf[0].get_text("words")
        l, t, _, b = words[0][:4]
        right = words[-1][2]
        table = {"format_version": 1,
                 "source_sha256": hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
                 "tables": [{"page": 1, "self_ref": "#/tables/0", "num_rows": 1,
                             "num_cols": 1, "bbox": None, "cells": [{
                                 "row": 0, "column": 0, "row_span": 1, "column_span": 1,
                                 "text": extracted_text, "row_header": False,
                                 "column_header": False,
                                 "bbox": {"l": l - 0.5, "t": t - 0.5,
                                          "r": right + 0.5, "b": b + 0.5,
                                          "coord_origin": "TOPLEFT"}}]}]}
        table_path = root / "tables.json"
        table_path.write_text(json.dumps(table))
        return temp, pdf_path, table_path

    def test_normalization_preserves_electrical_numbers(self):
        self.assertEqual(normalize("3 VA/m²"), normalize("3 VA / m2"))
        self.assertEqual(normalize("201–325"), normalize("201-325"))
        self.assertNotEqual(normalize("480.0 V"), normalize("4800 V"))
        self.assertNotEqual(normalize("120/240 V"), normalize("120-240 V"))

    def test_exact_match_keeps_raw_text_private(self):
        scope, pdf, tables = self.sample()
        with scope:
            safe, detailed = audit_tables(pdf, tables)
            self.assertEqual(safe["cells_total"], 1)
            self.assertEqual(safe["normalized_exact"], 1)
            self.assertTrue(safe["audit_passed_without_disagreements"])
            self.assertNotIn("480 V", json.dumps(safe))
            self.assertIn("480 V", json.dumps(detailed))

    def test_incorrect_rating_is_flagged_not_counted_as_correct(self):
        scope, pdf, tables = self.sample("460 V")
        with scope:
            safe, _ = audit_tables(pdf, tables)
            self.assertEqual(safe["needs_review"], 1)
            self.assertFalse(safe["audit_passed_without_disagreements"])
            self.assertFalse(safe["independent_human_review_complete"])

    def test_source_revision_hash_mismatch_rejected(self):
        scope, pdf, tables = self.sample()
        with scope:
            data = json.loads(tables.read_text())
            data["source_sha256"] = "0" * 64
            tables.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "PDF source hash"):
                audit_tables(pdf, tables)

    def test_cell_without_geometry_needs_review(self):
        scope, pdf, tables = self.sample()
        with scope:
            data = json.loads(tables.read_text())
            data["tables"][0]["cells"][0]["bbox"] = None
            tables.write_text(json.dumps(data))
            safe, _ = audit_tables(pdf, tables)
            self.assertEqual(safe["counts_by_status"]["missing_or_unsupported_bbox"], 1)


if __name__ == "__main__":
    unittest.main()
