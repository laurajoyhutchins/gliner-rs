"""Offline corpus/provenance and reviewed-accuracy tests, no licensed code text."""
import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from prepare_corpus import docling_pages, make_chunks
from benchmark_regulatory import numerically_equal, score_gold, select_chunks


class RegulatoryBenchmarkTests(unittest.TestCase):
    def test_docling_page_provenance(self):
        document = {"texts": [
            {"text": "Example one", "prov": [{"page_no": 9}]},
            {"text": "Example two", "prov": [{"page_no": 10}]},
            {"text": "Duplicate annotation", "prov": [{"page_no": 9}, {"page_no": 9}]},
            {"text": "Unpositioned", "prov": []}
        ]}
        pages = docling_pages(document)
        self.assertEqual(pages[9], ["Example one", "Duplicate annotation"])
        self.assertEqual(pages[10], ["Example two"])

    def test_chunks_have_bounded_size_and_provenance(self):
        chunks = make_chunks({3: ["hello world " * 110], 7: ["secondary block"]}, {3, 7}, 160)
        self.assertTrue(all(0 < len(x["text"]) <= 160 for x in chunks))
        self.assertEqual({3, 7}, {x["page"] for x in chunks})
        self.assertTrue(all(x["sha256"] == hashlib.sha256(x["text"].encode()).hexdigest() for x in chunks))

    def test_sampling_is_balanced_across_pages(self):
        chunks = [{"page": p, "id": f"{p}-{j}"} for p in (3, 4, 5, 6) for j in range(5)]
        selected = select_chunks(chunks, 8)
        self.assertEqual(len(selected), 8)
        self.assertEqual({3, 4, 5, 6}, {x["page"] for x in selected})

    def test_confidence_tolerance_and_semantic_differences(self):
        self.assertTrue(numerically_equal({"confidence": 0.500001}, {"confidence": 0.500002}))
        self.assertFalse(numerically_equal({"a": [1]}, {"a": [1, 2]}))
        self.assertFalse(numerically_equal({"a": 1}, {"b": 1}))

    def test_gold_scoring_fails_closed_on_wrong_revision(self):
        cases = [{"id": "example", "sha256": "doc1"}]
        predictions = {"example": {"prediction": {"entities": {
            "equipment": [{"start": 1, "end": 4}]
        }}}}
        gold = {"cases": [{"case_id": "example", "chunk_sha256": "doc1",
                           "complete": True, "labels": [{"label": "equipment", "start": 1, "end": 4}]}]}
        self.assertEqual(score_gold(cases, predictions, gold)["f1"], 1)
        gold["cases"][0]["chunk_sha256"] = "doc2"
        with self.assertRaises(ValueError):
            score_gold(cases, predictions, gold)

    def test_no_reviewed_cases_cannot_claim_accuracy(self):
        self.assertIsNone(score_gold([{"id": "a", "sha256": "x"}], {}, {"cases": []})["f1"])


if __name__ == "__main__":
    unittest.main()
