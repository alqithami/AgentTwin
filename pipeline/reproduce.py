"""Single-command end-to-end reproduction.

This is a convenience wrapper around:
- `python -m train.training_pipeline ...`
- `python -m eval.generate_results ...`

If you prefer Make, use:
  make reproduce
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="AgentTwin end-to-end reproduction wrapper")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--artifacts_root", type=str, default="artifacts", help="Artifacts output directory")
    parser.add_argument("--results_dir", type=str, default="results", help="Results output directory")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "reproduce"],
        default="reproduce",
        help="What to run",
    )
    args = parser.parse_args()

    cfg = str(Path(args.config))
    artifacts = str(Path(args.artifacts_root))
    results = str(Path(args.results_dir))

    if args.mode in {"train", "reproduce"}:
        cmd = [sys.executable, "-m", "train.training_pipeline", "--config", cfg, "--out_root", artifacts]
        print("[pipeline] Running training:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    if args.mode in {"eval", "reproduce"}:
        cmd = [
            sys.executable,
            "-m",
            "eval.generate_results",
            "--config",
            cfg,
            "--artifacts_root",
            artifacts,
            "--results_dir",
            results,
        ]
        print("[pipeline] Running evaluation:", " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
