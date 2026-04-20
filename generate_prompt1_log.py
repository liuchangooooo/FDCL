#!/usr/bin/env python3
"""Generate a simplified prompt log.

This script keeps only three parts from each evolve block:
1) The line that starts with "[EVOLVE PROMPT]"
2) Header metadata lines right after it (e.g. episode_total/reason/dominant)
3) The section "【批次统计（粗粒度）】" and its data lines
"""

from __future__ import annotations

from pathlib import Path

EVOLVE_PROMPT_PREFIX = "[EVOLVE PROMPT]"
COARSE_STATS_HEADER = "【批次统计（粗粒度）】"


def is_delimiter(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= 10 and set(stripped) == {"="}


def is_meta_separator(line: str) -> bool:
    stripped = line.strip()
    return len(stripped) >= 10 and set(stripped) == {"-"}


def extract_simple_prompt_blocks(text: str) -> tuple[str, int]:
    lines = text.splitlines(keepends=True)
    output: list[str] = []
    kept_blocks = 0

    i = 0
    while i < len(lines):
        current = lines[i]
        if current.strip().startswith(EVOLVE_PROMPT_PREFIX):
            kept_blocks += 1
            output.append("=" * 100 + "\n")
            output.append(current)

            # Keep evolve header metadata lines up to the dashed separator.
            header_idx = i + 1
            header_line_count = 0
            while (
                header_idx < len(lines)
                and not is_delimiter(lines[header_idx])
                and header_line_count < 10
            ):
                output.append(lines[header_idx])
                header_line_count += 1
                if is_meta_separator(lines[header_idx]):
                    header_idx += 1
                    break
                header_idx += 1

            j = header_idx
            while j < len(lines) and not is_delimiter(lines[j]):
                if lines[j].strip() == COARSE_STATS_HEADER:
                    output.append(lines[j])
                    j += 1

                    while j < len(lines) and not is_delimiter(lines[j]):
                        stripped = lines[j].strip()
                        if stripped.startswith("【") and stripped != COARSE_STATS_HEADER:
                            break
                        if stripped == "":
                            output.append(lines[j])
                            break
                        output.append(lines[j])
                        j += 1
                    break
                j += 1

            output.append("=" * 100 + "\n\n")

            i = j
            while i < len(lines) and not is_delimiter(lines[i]):
                i += 1
            if i < len(lines) and is_delimiter(lines[i]):
                i += 1
            continue

        i += 1

    return "".join(output), kept_blocks





def main() -> int:
    input_path = Path("data/outputs/2026.04.15/22.16.17_td3_pusht_llm_curriculum/evolve_prompt.log")
    output_path = Path("data/outputs/2026.04.15/22.16.17_td3_pusht_llm_curriculum/prompt1.log")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must be different")

    source = input_path.read_text(encoding="utf-8")
    cleaned, kept_blocks = extract_simple_prompt_blocks(source)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Kept evolve blocks: {kept_blocks}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
