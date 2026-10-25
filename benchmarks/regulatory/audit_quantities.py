"""Conservative admission of engineering quantities in regulatory text.

GLiNER predicts candidate spans; a bare code-section number is not a voltage
or current. Preserve raw candidates and report withheld spans without
claiming that admitted candidates establish engineering correctness.
"""
from __future__ import annotations

from collections import Counter
import re

# Require an explicit electrical unit inside the predicted entity span.
# This is deliberately narrow: a missing unit causes review, not inference.
_NUMBER = r"(?:[0-9]+(?:\.[0-9]+)?)(?:\s*/\s*[0-9]+(?:\.[0-9]+)?)?"
UNITS = {
    "voltage rating": re.compile(
        rf"(?i)(?<![A-Za-z]){_NUMBER}\s*(?:kV|V|kilovolts?|volts?)\b"),
    "current rating": re.compile(
        rf"(?i)(?<![A-Za-z]){_NUMBER}\s*(?:kA|A|kiloamperes?|amperes?|amps?)\b"),
}


def audit_quantity_candidates(cases: list[dict], predictions: dict) -> dict:
    counts = {name: Counter() for name in UNITS}
    withheld = []
    for case in cases:
        entities = predictions[case["id"]]["prediction"]["entities"]
        text = case["text"]
        for label, pattern in UNITS.items():
            for entity in entities.get(label, []):
                start, end = entity["start"], entity["end"]
                if not (isinstance(start, int) and isinstance(end, int)
                        and 0 <= start < end <= len(text)
                        and text[start:end] == entity["text"]):
                    raise ValueError("candidate offsets do not match input text")
                counts[label]["raw"] += 1
                if pattern.search(entity["text"]):
                    counts[label]["admitted_unit_present"] += 1
                else:
                    counts[label]["withheld_unit_missing"] += 1
                    withheld.append({"case_id": case["id"], "page": case["page"],
                                     "label": label, "start": start, "end": end})
    return {
        "audit_kind": "conservative_explicit_unit_presence_not_gold_accuracy",
        "counts": {
            label: {name: counts[label][name] for name in
                    ("raw", "admitted_unit_present", "withheld_unit_missing")}
            for label in UNITS
        },
        "withheld_positions": withheld,
        "raw_predictions_unchanged": True,
        "independent_engineering_accuracy_assessed": False,
    }
