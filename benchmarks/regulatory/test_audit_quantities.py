"""Unit admission checks for code-section versus electrical rating candidates."""
import unittest
from audit_quantities import audit_quantity_candidates


class QuantityAdmissionTests(unittest.TestCase):
    def audit(self, text, label, value):
        start = text.index(value)
        cases = [{"id": "p0001-c0001", "page": 1, "text": text}]
        results = {"p0001-c0001": {"prediction": {
            "entities": {label: [{"text": value, "start": start,
                                  "end": start + len(value)}]}}}}
        return audit_quantity_candidates(cases, results)

    def test_nec_section_number_is_not_a_voltage(self):
        result = self.audit("See NEC 220.85 for conditions", "voltage rating", "220.85")
        self.assertEqual(result["counts"]["voltage rating"]["withheld_unit_missing"], 1)
        self.assertEqual(result["counts"]["voltage rating"]["admitted_unit_present"], 0)
        self.assertFalse(result["independent_engineering_accuracy_assessed"])

    def test_nec_section_number_is_not_current(self):
        result = self.audit("Section 240.4 applies", "current rating", "240.4")
        self.assertEqual(result["counts"]["current rating"]["withheld_unit_missing"], 1)

    def test_explicit_units_are_admitted(self):
        for label, value in [
            ("voltage rating", "480 V"),
            ("voltage rating", "13.8 kV"),
            ("voltage rating", "120/240 volts"),
            ("current rating", "20 A"),
            ("current rating", "65 kA"),
        ]:
            with self.subTest(value=value):
                result = self.audit("Rated " + value + " nominal", label, value)
                self.assertEqual(result["counts"][label]["admitted_unit_present"], 1)

    def test_percent_is_not_electrical_unit(self):
        result = self.audit("Demand factor 80%", "voltage rating", "80%")
        self.assertEqual(result["counts"]["voltage rating"]["withheld_unit_missing"], 1)

    def test_offsets_must_match_original_text(self):
        cases = [{"id": "case", "page": 1, "text": "Rated 480 V"}]
        predictions = {"case": {"prediction": {"entities": {
            "voltage rating": [{"text": "480 V", "start": 0, "end": 5}]}}}}
        with self.assertRaises(ValueError):
            audit_quantity_candidates(cases, predictions)


if __name__ == "__main__":
    unittest.main()
