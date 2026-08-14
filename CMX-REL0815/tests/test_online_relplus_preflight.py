import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO = Path("/home/zhuzhaoziao/rel_exp/cmx_rel+")
PYTHON = Path("/data/zhuzhaoziao/cmx/envs/cmx-py38/bin/python")


class OnlineRelPlusPreflightTest(unittest.TestCase):
    def test_actual_online_config_and_dataset_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [str(PYTHON), str(REPO / "scripts/validate_relplus_online.py"),
                 "--run-dir", directory],
                cwd=str(REPO), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=dict(os.environ), check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout)
            report = json.loads((Path(directory) / "data_reports/online_relplus_validation.json").read_text())
            self.assertEqual(report["status"], "PASS_ONLINE_RELPLUS_PREFLIGHT")
            self.assertEqual(report["missing_file_count"], 0)
            self.assertEqual(report["channel_order"], ["ReD", "EGVIA", "LOA"])
            self.assertFalse(report["horizontal_flip"])


if __name__ == "__main__":
    unittest.main()
