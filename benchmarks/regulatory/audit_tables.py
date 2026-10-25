#!/usr/bin/env python3
"""Cross-check Docling table cells with independently extracted PDF text.

This tests embedded-PDF transcription agreement, NOT human-reviewed correctness.
Private reports include text; source-safe receipts never include cell content.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize(text: str) -> str:
    """Remove only typography differences, preserving numeric punctuation."""
    value = unicodedata.normalize("NFKC", text).casefold()
    for dash in ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"):
        value = value.replace(dash, "-")
    return re.sub(r"\s+", "", value)


def audit_tables(pdf_path: Path, tables_path: Path) -> tuple[dict, dict]:
    import fitz

    record = json.loads(tables_path.read_text(encoding="utf-8"))
    source_hash = file_sha256(pdf_path)
    if source_hash != record["source_sha256"]:
        raise ValueError("PDF source hash differs from Docling table evidence")
    cases = []
    with fitz.open(pdf_path) as pdf:
        by_page = {}
        for table in record["tables"]:
            page = table["page"]
            if not 1 <= page <= len(pdf):
                raise ValueError("table refers to missing PDF page")
            if page not in by_page:
                opened = pdf[page - 1]
                by_page[page] = (opened.rect, opened.get_text("words"))
            page_rect, words = by_page[page]
            for cell in table["cells"]:
                position = {"page": page, "table_ref": table["self_ref"],
                            "row": cell["row"], "column": cell["column"]}
                bbox = cell.get("bbox")
                if not isinstance(bbox, dict) or bbox.get("coord_origin") != "TOPLEFT":
                    cases.append({**position, "status": "missing_or_unsupported_bbox",
                                  "docling_text": cell["text"], "pdf_text": None})
                    continue
                left, top, right, bottom = (bbox[key] for key in ("l", "t", "r", "b"))
                if not all(isinstance(v, (int, float)) and v == v
                           for v in (left, top, right, bottom)):
                    raise ValueError("invalid non-finite cell box")
                if not (0 <= left < right <= page_rect.width
                        and 0 <= top < bottom <= page_rect.height):
                    raise ValueError("cell box outside PDF page")
                matched = [w[4] for w in words
                           if left <= (w[0] + w[2]) / 2 <= right
                           and top <= (w[1] + w[3]) / 2 <= bottom]
                extracted = " ".join(matched)
                observed, expected = normalize(cell["text"]), normalize(extracted)
                if observed == expected:
                    status = "normalized_exact"
                elif observed and expected and (observed in expected or expected in observed):
                    status = "needs_review_overlap_or_omission"
                else:
                    status = "needs_review_disagreement"
                cases.append({**position, "status": status,
                              "docling_text": cell["text"], "pdf_text": extracted})

    counts = Counter(case["status"] for case in cases)
    matched = counts["normalized_exact"]
    total = len(cases)
    issues = [case for case in cases if case["status"] != "normalized_exact"]
    safe = {
        "audit_kind": "pdf_embedded_text_cell_geometry_crosscheck_not_human_gold",
        "source_sha256": source_hash,
        "table_artifact_sha256": file_sha256(tables_path),
        "cells_total": total,
        "normalized_exact": matched,
        "needs_review": total - matched,
        "match_fraction": matched / total if total else None,
        "counts_by_status": dict(sorted(counts.items())),
        "unresolved_positions": [
            {"page": case["page"], "table_ref": case["table_ref"],
             "row": case["row"], "column": case["column"], "status": case["status"]}
            for case in issues
        ],
        "independent_human_review_complete": False,
        "audit_passed_without_disagreements": total > 0 and matched == total,
    }
    return safe, {**safe, "cases": cases}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--tables", required=True, type=Path)
    parser.add_argument("--private-out", required=True, type=Path)
    parser.add_argument("--safe-out", required=True, type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()
    safe, detailed = audit_tables(args.pdf, args.tables)
    from jsonschema import Draft202012Validator
    schema_path = Path(__file__).with_name("contracts") / "table-text-audit.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text())).validate(safe)
    for path, data in ((args.private_out, detailed), (args.safe_out, safe)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(safe, indent=2))
    if args.require_exact and not safe["audit_passed_without_disagreements"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
