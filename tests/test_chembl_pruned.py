"""CPU regression tests: python -B tests/test_chembl_pruned.py."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rdkit import Chem

from fedsteer.data import ClientQuantiles, alpha_reference_from_json, build_clients
from fedsteer.metrics import (SCORERS, decoration_sensitivity_metrics,
                              failure_penalized_pct_err, summarize)
from fedsteer.molecules import (assemble_ligand, clogp_residual_deco,
                               clogp_residual_deco_pruned, clogp_residual_deco_strict)
from scripts.prune_chembl import build_dataset


class PrunedDecorationTests(unittest.TestCase):
    def setUp(self):
        self.rec = {"core_attached": "[1*]c1ccccc1", "scaffold": "c1ccccc1"}

    def test_extras_cannot_improve_parent_score(self):
        baseline = clogp_residual_deco("[1*]C", self.rec)
        for text in ("[1*]C", "[1*]C.Cl", "Cl.Cl.[1*]C", "[1*]C.[Br-]"):
            self.assertAlmostEqual(clogp_residual_deco_pruned(text, self.rec), baseline)
        self.assertNotAlmostEqual(clogp_residual_deco("[1*]C.[Br-]", self.rec), baseline)
        self.assertAlmostEqual(clogp_residual_deco_strict("[1*]C", self.rec), baseline)
        self.assertTrue(math.isnan(clogp_residual_deco_strict("[1*]C.Cl", self.rec)))

    def test_attachment_failures_are_not_repaired(self):
        for text in ("C", "[1*]C.[2*]O", "[1*]C.[1*]O", "[1*]", "[1*]C(=O)(=O)C"):
            for scorer in (clogp_residual_deco_pruned, clogp_residual_deco_strict):
                with self.subTest(text=text, scorer=scorer.__name__):
                    self.assertTrue(math.isnan(scorer(text, self.rec)))

    def test_multiple_attached_fragments_are_valid(self):
        mol, reason = assemble_ligand("[1*]c1ccc([2*])cc1", "[1*]C.[2*]O")
        self.assertIsNone(reason)
        self.assertEqual(len(Chem.GetMolFrags(mol)), 1)
        self.assertFalse(any(a.GetAtomicNum() == 0 for a in mol.GetAtoms()))
        # A charged parent is allowed; this policy is not charge neutralization.
        mol, reason = assemble_ligand(self.rec["core_attached"], "[1*][NH3+]")
        self.assertIsNone(reason)

    def test_dataset_filters_all_splits_and_rebuilds_reference(self):
        records = [
            dict(self.rec, url="c/train-clean", client="c", split="train", target="[1*]C", score=999, prompt="train"),
            dict(self.rec, url="c/train-salt", client="c", split="train", target="[1*]C.[Br-]", score=-999, prompt="train"),
            dict(core_attached="[1*]c1ccncc1", scaffold="c1ccncc1", url="c/dev", client="c",
                 split="dev", target="[1*]O", score=888, prompt="dev"),
            dict(core_attached="[1*]C1CCCCC1", scaffold="C1CCCCC1", url="c/test-salt", client="c",
                 split="test", target="[1*]C.Cl", score=777, prompt="test"),
        ]
        spec = {"clients": ["c"], "rotations": [{"participants": ["c"], "held_out": []}],
                "format": "deco", "median_residual": {"c": -123}, "skew": {"stale": True}}
        with tempfile.TemporaryDirectory() as tmp:
            source, dest = Path(tmp) / "source", Path(tmp) / "pruned"
            source.mkdir()
            (source / "data.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
            (source / "clients.json").write_text(json.dumps(spec))
            original = (source / "data.jsonl").read_bytes()
            report = build_dataset(source, dest)
            kept = [json.loads(l) for l in (dest / "data.jsonl").read_text().splitlines()]
            self.assertEqual([r["url"] for r in kept], ["c/train-clean", "c/dev"])
            self.assertEqual(report["dropped"], 2)
            self.assertEqual(report["reasons"], {"unattached_fragments": 2})
            self.assertEqual((source / "data.jsonl").read_bytes(), original)
            ref = alpha_reference_from_json(json.loads((dest / "alpha_reference.json").read_text()))
            self.assertEqual(ref.clients["c"].sorted_scores, [kept[0]["score"]])
            _, training_refs = build_clients(kept, ["c"], alpha_mode="global")
            self.assertEqual(ref.to_json(), training_refs["c"].to_json())
            new_spec = json.loads((dest / "clients.json").read_text())
            self.assertNotIn("skew", new_spec)
            self.assertEqual(new_spec["median_residual"]["c"], kept[0]["score"])
            for rec in kept:
                self.assertEqual(rec["score"], clogp_residual_deco_strict(rec["target"], rec))
            self.assertEqual(failure_penalized_pct_err([[float("nan")]], [.5], ref), 1)
            with self.assertRaises(FileExistsError):
                build_dataset(source, dest)

    def test_scorers_registered(self):
        self.assertIs(SCORERS["clogp_residual_deco_pruned"], clogp_residual_deco_pruned)
        self.assertIs(SCORERS["clogp_residual_deco_strict"], clogp_residual_deco_strict)

    def test_supplementary_metrics_keep_failures(self):
        q = ClientQuantiles.fit([-1, 0, 1, 2])
        clean = decoration_sensitivity_metrics([["[1*]C", "[1*]C"]], [self.rec], [0, 1], q)
        extra = decoration_sensitivity_metrics([["[1*]C.Cl", "[1*]C.Cl"]], [self.rec], [0, 1], q)
        self.assertEqual(clean["pct_calib_err_pruned"], extra["pct_calib_err_pruned"])
        self.assertEqual(extra["pct_calib_err_penalized_strict"], 1)
        self.assertIsNone(extra["pct_calib_err_strict"])
        self.assertEqual(summarize({"clean": clean, "extra": extra})[
            "pct_calib_err_penalized_strict"]["worst"], 1)

    def test_checkpoint_selection_minimizes_new_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "evals"
            path.mkdir()
            for rnd, error in [(20, .2), (40, .4)]:
                d = {"snapshot": str(Path(tmp) / f"round_{rnd:04d}.pt"),
                     "summary": {"pct_calib_err_penalized": {"mean": error, "worst": error}}}
                (path / f"eval_round_{rnd:04d}_dev__test.json").write_text(json.dumps(d))
            output = subprocess.check_output([sys.executable, "-B", "scripts/summarize_sweep.py",
                                              "--run", tmp, "--select", "pct_calib_err_penalized",
                                              "--print_best"], text=True)
            self.assertTrue(output.strip().endswith("round_0020.pt"))

    def test_full_evaluation_includes_supplementary_scores(self):
        from fedsteer.evaluate import evaluate_loaded_client
        q = ClientQuantiles.fit([-1, 0, 1, 2])
        texts = [["[1*]C", "[1*]C.Cl"]]
        grid = np.asarray([[clogp_residual_deco(t, self.rec) for t in texts[0]]])
        model = SimpleNamespace(steer_control=SimpleNamespace(
            gain=lambda: np.float32(1), o=None, warp=SimpleNamespace(kind="none")))
        with patch("fedsteer.evaluate.score_grid", return_value=(grid, texts)):
            result = evaluate_loaded_client(model, None, [self.rec], [0, 1], clogp_residual_deco, q)
        self.assertIn("pct_calib_err_penalized_pruned", result)
        self.assertIn("pct_calib_err_penalized_strict", result)
        self.assertEqual(result["unscorable_row_rate"], 0)
        self.assertEqual(result["unscorable_row_rate_strict"], 1)


if __name__ == "__main__":
    unittest.main()
