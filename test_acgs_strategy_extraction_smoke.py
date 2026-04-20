#!/usr/bin/env python3
"""Smoke test for ACGS strategy extraction compatibility."""

import ast
import copy
import logging
import pathlib
import sys

ROOT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

TARGET_FILE = ROOT_DIR / "DIVO" / "workspace" / "rl_workspace" / "td3_curriculum_workspace.py"


def _load_extract_function_from_source():
    """Load _extract_clean_generate_obstacles directly from source via AST."""
    source = TARGET_FILE.read_text(encoding="utf-8")
    module = ast.parse(source)

    target = None
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "TD3CurriculumWorkspace":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_extract_clean_generate_obstacles":
                    target = copy.deepcopy(item)
                    break
        if target is not None:
            break

    if target is None:
        raise RuntimeError("Cannot find _extract_clean_generate_obstacles in source file")

    temp_module = ast.Module(body=[target], type_ignores=[])
    ast.fix_missing_locations(temp_module)

    namespace = {
        "ast": ast,
        "copy": copy,
        "LOGGER": logging.getLogger("acgs-strategy-extraction-smoke"),
    }
    exec(compile(temp_module, str(TARGET_FILE), "exec"), namespace)
    return namespace["_extract_clean_generate_obstacles"]


TOPOLOGY_CODE = """
def helper():
    # helper should never appear in extracted result
    return []


def generate_obstacles(tblock_pose, num_obstacles):
    \"\"\"docstring should be removed\"\"\"
    # full-line comment should be removed
    x = float(tblock_pose[0])  # inline comment should be removed
    y = 0.1
    return [{"x": x, "y": y}]
"""


def _assert_clean_result(result):
    assert result is not None, "extract result is None"
    assert "def generate_obstacles" in result, "missing generate_obstacles definition"
    assert "def helper" not in result, "unexpected helper function leaked"
    assert '"""' not in result, "docstring was not removed"
    assert "#" not in result, "comments were not removed"


def _run_one_case(simulate_no_unparse):
    extract_fn = _load_extract_function_from_source()
    dummy = type("DummyWorkspace", (), {"topology_generator_code": TOPOLOGY_CODE})()

    original_unparse = getattr(ast, "unparse", None)
    try:
        if simulate_no_unparse and hasattr(ast, "unparse"):
            delattr(ast, "unparse")

        result = extract_fn(dummy)
        _assert_clean_result(result)
        return result
    finally:
        if original_unparse is not None and not hasattr(ast, "unparse"):
            ast.unparse = original_unparse


def main():
    print("=" * 72)
    print("Smoke test: _extract_clean_generate_obstacles")
    print("=" * 72)

    result_normal = _run_one_case(simulate_no_unparse=False)
    print("[PASS] normal path (ast.unparse available)")
    print(result_normal)
    print("-" * 72)

    result_fallback = _run_one_case(simulate_no_unparse=True)
    print("[PASS] fallback path (ast.unparse unavailable)")
    print(result_fallback)
    print("-" * 72)

    extract_fn = _load_extract_function_from_source()
    dummy_missing = type("DummyWorkspace", (), {"topology_generator_code": "def x():\n    return 1\n"})()
    missing = extract_fn(dummy_missing)
    assert missing is None, "expected None when generate_obstacles is absent"
    print("[PASS] missing generate_obstacles returns None")

    print("=" * 72)
    print("All smoke checks passed")
    print("=" * 72)


if __name__ == "__main__":
    main()
