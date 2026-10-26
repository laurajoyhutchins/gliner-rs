import unittest

from profile_stages import parse_frames


class StageTraceTests(unittest.TestCase):
    def test_reconstructs_complete_frames_and_sums_repeated_layers(self):
        trace = "\n".join([
            "server started",
            "GLINER_STAGE\tprepare\t10",
            "GLINER_STAGE\tdeberta_attention\t20",
            "GLINER_STAGE\tdeberta_attention\t30",
            "GLINER_STAGE\tencoder\t100",
            "GLINER_STAGE\tpost_encoder\t50",
            "GLINER_STAGE\textract_total\t170",
            "GLINER_STAGE\tprepare\t8",
            "GLINER_STAGE\tencoder\t200",
            "GLINER_STAGE\tpost_encoder\t80",
            "GLINER_STAGE\textract_total\t300",
        ])
        frames = parse_frames(trace)
        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[0]["deberta_attention"], 50)
        self.assertEqual(frames[1]["encoder"], 200)

    def test_rejects_incomplete_or_unterminated_traces(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            parse_frames("GLINER_STAGE\textract_total\t11")
        with self.assertRaisesRegex(ValueError, "unterminated"):
            parse_frames("GLINER_STAGE\tprepare\t11")

    def test_ignores_spoofed_or_malformed_events(self):
        trace = ("GLINER_STAGE\tprepare\t10\n"
                 "GLINER_STAGE\tencoder\tnan\n"
                 "GLINER_STAGE\tencoder\t20\n"
                 "GLINER_STAGE\tpost_encoder\t30\n"
                 "GLINER_STAGE\textract_total\t80\n")
        self.assertEqual(parse_frames(trace)[0]["encoder"], 20)


if __name__ == "__main__":
    unittest.main()
