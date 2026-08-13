"""Convenience entrypoints for parsing failure-driven experiment runs.

This module wraps the core parser in ``DIVO.utils.failure_driven_parser`` and
adds small helper utilities for loading exported tables inside plotting scripts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from DIVO.utils.failure_driven_parser import ParsedRun, parse_run_directory


REQUIRED_PARSED_FILES = [
    "run_meta.json",
    "layout_snapshots.csv",
    "obstacle_points.csv",
    "batch_stats.csv",
    "evolve_rounds.csv",
]


def ensure_parsed_dir(
    run_dir: Optional[str] = None,
    parsed_dir: Optional[str] = None,
    export_dir: Optional[str] = None,
    force_reparse: bool = False,
) -> Path:
    """Return a parsed directory, generating it from ``run_dir`` if needed."""

    if parsed_dir is not None:
        parsed_path = Path(parsed_dir).expanduser().resolve()
        if not parsed_path.exists():
            raise FileNotFoundError(f"Parsed directory does not exist: {parsed_path}")
        return parsed_path

    if run_dir is None:
        raise ValueError("Either run_dir or parsed_dir must be provided")

    run_path = Path(run_dir).expanduser().resolve()
    default_export = Path(export_dir).expanduser().resolve() if export_dir is not None else run_path / "parsed"

    if not force_reparse and _has_required_parsed_files(default_export):
        return default_export

    parsed_run = parse_run_directory(str(run_path))
    parsed_run.export(default_export)
    return default_export


def load_parsed_artifacts(parsed_dir: str) -> Dict[str, Any]:
    parsed_path = Path(parsed_dir).expanduser().resolve()
    if not parsed_path.exists():
        raise FileNotFoundError(f"Parsed directory does not exist: {parsed_path}")

    return {
        "parsed_dir": parsed_path,
        "run_meta": load_run_meta(parsed_path),
        "layout_snapshots": load_csv_table(parsed_path / "layout_snapshots.csv"),
        "obstacle_points": load_csv_table(parsed_path / "obstacle_points.csv"),
        "batch_stats": load_csv_table(parsed_path / "batch_stats.csv"),
        "evolve_rounds": load_csv_table(parsed_path / "evolve_rounds.csv"),
        "media_index": load_csv_table(parsed_path / "media_index.csv"),
        "parse_warnings": load_csv_table(parsed_path / "parse_warnings.csv"),
    }


def load_run_meta(parsed_dir: Path) -> Dict[str, Any]:
    with (parsed_dir / "run_meta.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_csv_table(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: _coerce_scalar(value) for key, value in row.items()})
    return rows


def sample_stage_ids(stage_ids: Iterable[int], max_count: int = 9) -> List[int]:
    unique = sorted(set(int(stage_id) for stage_id in stage_ids if stage_id is not None))
    if len(unique) <= max_count:
        return unique

    # Evenly sample the stage range so the selected stages span early/mid/late.
    indices = []
    for idx in range(max_count):
        pos = round(idx * (len(unique) - 1) / max(1, max_count - 1))
        indices.append(pos)
    return [unique[index] for index in sorted(set(indices))]


def default_figure_dir(parsed_dir: Path) -> Path:
    figure_dir = parsed_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    return figure_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse one Push-T failure-driven run into structured tables.")
    parser.add_argument("--run-dir", required=True, help="Path to one experiment output directory.")
    parser.add_argument(
        "--export-dir",
        default=None,
        help="Optional export directory. Defaults to <run-dir>/parsed.",
    )
    parser.add_argument(
        "--force-reparse",
        action="store_true",
        help="Regenerate parsed tables even if they already exist.",
    )
    args = parser.parse_args()

    parsed_dir = ensure_parsed_dir(
        run_dir=args.run_dir,
        export_dir=args.export_dir,
        force_reparse=args.force_reparse,
    )
    artifacts = load_parsed_artifacts(str(parsed_dir))

    print(f"Parsed directory: {parsed_dir}")
    print(
        "Counts: "
        f"snapshots={len(artifacts['layout_snapshots'])}, "
        f"obstacles={len(artifacts['obstacle_points'])}, "
        f"batches={len(artifacts['batch_stats'])}, "
        f"evolve_rounds={len(artifacts['evolve_rounds'])}, "
        f"warnings={len(artifacts['parse_warnings'])}"
    )


def _has_required_parsed_files(parsed_dir: Path) -> bool:
    return parsed_dir.exists() and all((parsed_dir / file_name).exists() for file_name in REQUIRED_PARSED_FILES)


def _coerce_scalar(value: Optional[str]) -> Any:
    if value is None:
        return None

    text = value.strip()
    if text == "":
        return None
    if text == "True":
        return True
    if text == "False":
        return False

    try:
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        return float(text)
    except ValueError:
        return text


if __name__ == "__main__":
    main()
