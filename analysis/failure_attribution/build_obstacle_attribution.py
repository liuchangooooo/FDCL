from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from DIVO.curriculum.attribution import (
    AttributionConfig,
    build_attribution as _build_attribution,
    build_summary_text,
    compute_attribution,
    read_jsonl,
    write_attribution_outputs,
)


def build_attribution(
    records: Iterable[Mapping[str, Any]],
    min_support: int = 10,
) -> Dict[str, Any]:
    """Backward-compatible dict API used by existing analysis callers."""
    return _build_attribution(records, min_support=min_support)


def write_outputs(result: Dict[str, Any], output_dir: Path) -> None:
    write_attribution_outputs(result, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build z-space obstacle failure attribution from rollout JSONL."
    )
    parser.add_argument("--rollouts", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--min-support", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.rollouts)
    config = AttributionConfig(min_support=args.min_support)
    result = compute_attribution(records, config)
    write_attribution_outputs(result, args.output_dir)
    print(f"Loaded {len(records)} episodes from {args.rollouts}")
    print(f"Saved attribution outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
