import inspect
import signal
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


OPEN_TOKENS = {"0", "c", "r", "g"}
RESET_TOKENS = {"r"}
GOAL_TOKENS = {"g"}
WALL_TOKENS = {"1"}


class MazeValidationError(ValueError):
    """Raised when a generated AntMaze layout violates hard constraints."""


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "numpy":
        return np
    raise ImportError(f"Import blocked in AntMaze generator sandbox: {name}")


def _normalize_cell(cell: Any) -> str:
    """Convert common Gymnasium-Robotics maze cells to a compact token."""
    if isinstance(cell, np.generic):
        cell = cell.item()

    if isinstance(cell, str):
        token = cell.strip().lower()
        if token in ("reset", "start", "agent"):
            return "r"
        if token in ("goal", "target"):
            return "g"
        if token in ("combined", "reset_goal", "start_goal"):
            return "c"
        if token in ("empty", "free", "open"):
            return "0"
        if token in ("wall", "block", "obstacle"):
            return "1"
        if token in OPEN_TOKENS or token in WALL_TOKENS:
            return token
        raise MazeValidationError(f"unsupported maze cell token: {cell!r}")

    if isinstance(cell, (int, float, bool)):
        value = int(cell)
        if value == 0:
            return "0"
        if value == 1:
            return "1"
        raise MazeValidationError(f"unsupported numeric maze cell: {cell!r}")

    raise MazeValidationError(f"unsupported maze cell type: {type(cell).__name__}")


def normalize_maze_map(maze_map: Sequence[Sequence[Any]]) -> List[List[Any]]:
    """
    Normalize a LLM-generated maze map into Gymnasium-Robotics compatible cells.

    The internal validation uses compact lowercase tokens. The returned map uses
    integers for walls/free cells and strings only for reset/goal markers so it
    can be passed directly as the `maze_map` argument.
    """
    if not isinstance(maze_map, (list, tuple)):
        raise MazeValidationError("maze_map must be a list/tuple of rows")
    if len(maze_map) == 0:
        raise MazeValidationError("maze_map is empty")

    normalized_tokens: List[List[str]] = []
    width: Optional[int] = None
    for row_idx, row in enumerate(maze_map):
        if not isinstance(row, (list, tuple)):
            raise MazeValidationError(f"maze row {row_idx} is not a list/tuple")
        if width is None:
            width = len(row)
            if width == 0:
                raise MazeValidationError("maze_map has an empty row")
        elif len(row) != width:
            raise MazeValidationError("maze_map must be rectangular")
        normalized_tokens.append([_normalize_cell(cell) for cell in row])

    return _tokens_to_gym_map(normalized_tokens)


def _tokens_to_gym_map(tokens: Sequence[Sequence[str]]) -> List[List[Any]]:
    gym_map: List[List[Any]] = []
    for row in tokens:
        gym_row: List[Any] = []
        for token in row:
            if token == "1":
                gym_row.append(1)
            elif token == "0":
                gym_row.append(0)
            elif token in ("r", "g", "c"):
                gym_row.append(token)
            else:
                raise MazeValidationError(f"unsupported normalized token: {token!r}")
        gym_map.append(gym_row)
    return gym_map


def _gym_map_to_tokens(maze_map: Sequence[Sequence[Any]]) -> List[List[str]]:
    return [[_normalize_cell(cell) for cell in row] for row in maze_map]


def _find_cells(tokens: Sequence[Sequence[str]], allowed: set) -> List[Tuple[int, int]]:
    cells: List[Tuple[int, int]] = []
    for r, row in enumerate(tokens):
        for c, token in enumerate(row):
            if token in allowed:
                cells.append((r, c))
    return cells


def _has_path(tokens: Sequence[Sequence[str]], starts: Sequence[Tuple[int, int]], goals: set) -> bool:
    height = len(tokens)
    width = len(tokens[0])
    queue = deque(starts)
    seen = set(starts)

    while queue:
        r, c = queue.popleft()
        if (r, c) in goals:
            return True
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if (nr, nc) in seen:
                continue
            if tokens[nr][nc] in WALL_TOKENS:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))
    return False


def validate_maze_map(
    maze_map: Sequence[Sequence[Any]],
    min_size: int = 3,
    max_size: int = 9,
    require_border_walls: bool = True,
    min_open_ratio: float = 0.25,
    max_open_ratio: float = 0.75,
) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validate an AntMaze layout before using it to instantiate MuJoCo.

    Returns:
        (is_valid, reason, metadata)
    """
    try:
        normalized = normalize_maze_map(maze_map)
        tokens = _gym_map_to_tokens(normalized)

        height = len(tokens)
        width = len(tokens[0])
        if height < min_size or width < min_size:
            raise MazeValidationError(f"maze too small: {height}x{width}")
        if height > max_size or width > max_size:
            raise MazeValidationError(f"maze too large: {height}x{width}, max={max_size}")

        if require_border_walls:
            for c in range(width):
                if tokens[0][c] not in WALL_TOKENS or tokens[height - 1][c] not in WALL_TOKENS:
                    raise MazeValidationError("top/bottom border must be walls")
            for r in range(height):
                if tokens[r][0] not in WALL_TOKENS or tokens[r][width - 1] not in WALL_TOKENS:
                    raise MazeValidationError("left/right border must be walls")

        reset_cells = _find_cells(tokens, RESET_TOKENS | {"c"})
        goal_cells = _find_cells(tokens, GOAL_TOKENS | {"c"})
        if len(reset_cells) == 0:
            raise MazeValidationError("maze must contain at least one reset/start cell")
        if len(goal_cells) == 0:
            raise MazeValidationError("maze must contain at least one goal cell")
        if set(reset_cells) == set(goal_cells) and len(reset_cells) == 1:
            raise MazeValidationError("reset and goal cannot be only the same cell")

        open_count = sum(1 for row in tokens for token in row if token in OPEN_TOKENS)
        total = height * width
        open_ratio = open_count / float(total)
        if open_ratio < min_open_ratio:
            raise MazeValidationError(f"not enough open cells: ratio={open_ratio:.2f}")
        if open_ratio > max_open_ratio:
            raise MazeValidationError(f"too few walls/obstacles: open ratio={open_ratio:.2f}")

        if not _has_path(tokens, reset_cells, set(goal_cells)):
            raise MazeValidationError("reset/start cell is not connected to goal cell")

        metadata = {
            "height": height,
            "width": width,
            "open_ratio": open_ratio,
            "reset_cells": reset_cells,
            "goal_cells": goal_cells,
            "normalized_map": normalized,
        }
        return True, "valid", metadata
    except Exception as exc:
        return False, str(exc), None


class MazeGeneratorExecutor:
    """
    Safe executor for LLM-generated AntMaze layout generators.

    Expected function:
        def generate_maze_map(seed: int = None) -> list:
            ...

    The executor intentionally validates only layout-level constraints. MuJoCo
    environment construction is left to later stages.
    """

    def __init__(
        self,
        min_size: int = 3,
        max_size: int = 9,
        require_border_walls: bool = True,
        min_open_ratio: float = 0.25,
        max_open_ratio: float = 0.75,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.require_border_walls = require_border_walls
        self.min_open_ratio = min_open_ratio
        self.max_open_ratio = max_open_ratio
        self.generate_maze_map = None
        self.generator_code: Optional[str] = None
        self._output_format_error_count = 0
        self.sandbox_globals = {
            "np": np,
            "numpy": np,
            "__builtins__": {
                "__import__": _safe_import,
                "range": range,
                "len": len,
                "abs": abs,
                "min": min,
                "max": max,
                "float": float,
                "int": int,
                "list": list,
                "dict": dict,
                "tuple": tuple,
                "enumerate": enumerate,
                "zip": zip,
                "sum": sum,
                "round": round,
            },
        }

    def load_generator_code(self, code: str) -> bool:
        try:
            local_globals = dict(self.sandbox_globals)
            exec(code, local_globals)
            if "generate_maze_map" not in local_globals:
                raise ValueError("code does not define generate_maze_map")
            fn = local_globals["generate_maze_map"]
            if not callable(fn):
                raise ValueError("generate_maze_map is not callable")

            sig = inspect.signature(fn)
            params = list(sig.parameters.values())
            has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
            has_varkw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
            if not (has_varargs or has_varkw) and len(params) > 1:
                raise ValueError(
                    "generate_maze_map should accept zero or one optional seed argument, "
                    f"got signature {sig}"
                )

            self.sandbox_globals = local_globals
            self.generate_maze_map = fn
            self.generator_code = code
            self._output_format_error_count = 0
            return True
        except Exception as exc:
            print(f"[AntMazeGenerator] load failed: {exc}")
            return False

    def load_generator_file(self, path: str) -> bool:
        with open(path, "r", encoding="utf-8") as handle:
            return self.load_generator_code(handle.read())

    def generate(self, seed: Optional[int] = None, timeout_sec: int = 5) -> List[List[Any]]:
        if self.generate_maze_map is None:
            raise RuntimeError("AntMaze generator has not been loaded")

        def _timeout_handler(signum, frame):
            raise TimeoutError("generate_maze_map call timed out")

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_sec)
        try:
            try:
                raw_map = self.generate_maze_map(seed)
            except TypeError:
                if seed is not None:
                    raw_map = self.generate_maze_map()
                else:
                    raise
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        normalized = normalize_maze_map(raw_map)
        valid, reason, _ = self.validate(normalized)
        if not valid:
            raise MazeValidationError(reason)
        return normalized

    def validate(self, maze_map: Sequence[Sequence[Any]]) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
        return validate_maze_map(
            maze_map,
            min_size=self.min_size,
            max_size=self.max_size,
            require_border_walls=self.require_border_walls,
            min_open_ratio=self.min_open_ratio,
            max_open_ratio=self.max_open_ratio,
        )

    def sanity_check(self, num_tests: int = 5, timeout_sec: int = 5) -> bool:
        for i in range(num_tests):
            seed = int(np.random.randint(0, 2**31 - 1))
            try:
                maze_map = self.generate(seed=seed, timeout_sec=timeout_sec)
                valid, reason, metadata = self.validate(maze_map)
                if not valid:
                    print(f"[AntMazeGenerator] sanity check {i + 1}/{num_tests} failed: {reason}")
                    return False
                if metadata is None:
                    print(f"[AntMazeGenerator] sanity check {i + 1}/{num_tests} failed: missing metadata")
                    return False
            except Exception as exc:
                print(f"[AntMazeGenerator] sanity check {i + 1}/{num_tests} failed: {exc}")
                return False
        print(f"[AntMazeGenerator] sanity check passed: {num_tests} tests")
        return True


def format_maze_map(maze_map: Sequence[Sequence[Any]]) -> str:
    """Pretty-print a maze map for logs and prompt debugging."""
    normalized = normalize_maze_map(maze_map)
    tokens = _gym_map_to_tokens(normalized)
    return "\n".join(" ".join(row) for row in tokens)
