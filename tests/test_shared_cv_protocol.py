import hashlib
import json
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "shared_splits" / "orthodontic_5fold_192_48_60_seed42.json"


class SharedCVProtocolTest(unittest.TestCase):
    def test_manifest_is_complete_and_disjoint(self):
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        self.assertEqual(payload["n_total"], 300)
        self.assertEqual(payload["n_folds"], 5)
        self.assertEqual(len(payload["folds"]), 5)

        test_counts = Counter()
        for expected_fold, fold in enumerate(payload["folds"], start=1):
            self.assertEqual(fold["fold"], expected_fold)
            self.assertEqual(
                (len(fold["train"]), len(fold["val"]), len(fold["test"])),
                (192, 48, 60),
            )
            train, val, test = map(set, (fold["train"], fold["val"], fold["test"]))
            self.assertFalse(train & val)
            self.assertFalse(train & test)
            self.assertFalse(val & test)
            self.assertEqual(len(train | val | test), 300)
            self.assertEqual(set(fold["groups"]["test"].values()), {10})
            test_counts.update(test)

        self.assertEqual(len(test_counts), 300)
        self.assertEqual(set(test_counts.values()), {1})

    def test_manifest_hash_and_fold1_regression(self):
        payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
        expected_hash = payload.pop("manifest_sha256")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), expected_hash)
        self.assertEqual(
            payload["folds"][0]["test"][:5],
            ["Class1_M2", "Class1_M21", "Class1_M22", "Class1_M23", "Class1_M28"],
        )


if __name__ == "__main__":
    unittest.main()
