import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


spec = importlib.util.spec_from_file_location(
    "mhealth_features", Path(__file__).with_name("wsp-mhealth-features.py"),
)
mhealth = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mhealth)


def sample(activity, axes=(1, 2, 3)):
    # Non-accelerometer fields must be ignored, including the chest ECG leads.
    columns = ["ignore"] * 24
    columns[23] = activity
    for start, values in [(0, axes), (5, (4, 5, 6)), (14, (7, 8, 9))]:
        columns[start:start + 3] = values
    return " ".join(map(str, columns))


class MhealthFeaturesTests(unittest.TestCase):
    def test_ordered_accelerometers_and_contiguous_window_boundaries(self):
        lines = (
            [sample(0)] * 10 + [sample(4)] * 250 + [sample(7)] * 99
            + [sample(4)] * 100
        )
        windows = list(mhealth.subject_windows(lines, 8))
        self.assertEqual(
            [(w["subject"], w["start"], w["activity"], w["ordinal"]) for w in windows],
            [(8, 10, 4, 0), (8, 110, 4, 1), (8, 359, 4, 0)],
        )
        self.assertEqual(windows[0]["chest"][:6], [1, 2, 3, 0, 0, 0])
        self.assertEqual(windows[0]["ankle"][:6], [4, 5, 6, 0, 0, 0])
        self.assertEqual(windows[0]["wrist"][:6], [7, 8, 9, 0, 0, 0])

    def test_statistic_order_and_population_formulas(self):
        self.assertEqual(
            mhealth.features([(0, 0, 0), (2, 0, 0)]),
            [1, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0],
        )

    def test_malformed_or_nonfinite_labeled_rows_fail(self):
        with self.assertRaisesRegex(ValueError, "24 columns"):
            list(mhealth.subject_windows(["1 2 3"], 2))
        with self.assertRaisesRegex(ValueError, "invalid activity"):
            list(mhealth.subject_windows([sample(13)], 2))
        with self.assertRaisesRegex(ValueError, "nonfinite acceleration"):
            list(mhealth.subject_windows([sample(3, (1, "nan", 3))], 2))

    def test_noncanonical_archive_rejected_before_opening_zip(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "not-original.zip"
            path.write_bytes(b"PK\x03\x04")
            with self.assertRaisesRegex(ValueError, "original"):
                list(mhealth.archive_windows(path))


if __name__ == "__main__":
    unittest.main()
