from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class GateCProbabilisticProfileEntrypointTests(unittest.TestCase):
    def test_module_help_exposes_executable_cli_entrypoint(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "journal.scripts.evaluate_gate_c_probabilistic_profile",
                "--help",
            ],
            cwd=repo_root,
            text=True,
            capture_output=True,
        )
        output = (proc.stdout or "") + (proc.stderr or "")

        self.assertEqual(proc.returncode, 0, output)
        self.assertIn("usage:", output.lower())
        self.assertIn("--mode", output)
        self.assertIn("--gate-a-checkpoint", output)
        self.assertIn("--dataset", output)
        self.assertIn("--output-dir", output)


if __name__ == "__main__":
    unittest.main()
