import os
from typing import Any, Dict, Optional

from DIVO.env.antmaze import MazeGeneratorExecutor, format_maze_map


def load_antmaze_generator(generator_path: str, **executor_kwargs) -> MazeGeneratorExecutor:
    """Load a generate_maze_map program and validate that it can sample maps."""
    executor = MazeGeneratorExecutor(**executor_kwargs)
    if not executor.load_generator_file(generator_path):
        raise RuntimeError(f"Failed to load AntMaze generator: {generator_path}")
    if not executor.sanity_check(num_tests=1):
        raise RuntimeError(f"AntMaze generator failed sanity check: {generator_path}")
    return executor


def make_antmaze_env_kwargs_from_generator(
    generator_path: str,
    seed: Optional[int] = None,
    save_map_path: Optional[str] = None,
    **executor_kwargs,
) -> Dict[str, Any]:
    """
    Create gym.make kwargs for a generator-defined AntMaze layout.

    This keeps stage-2 integration lightweight: the environment still receives a
    normal `maze_map`, while the curriculum layer owns how that map is produced.
    """
    executor = load_antmaze_generator(generator_path, **executor_kwargs)
    maze_map = executor.generate(seed=seed)

    if save_map_path:
        os.makedirs(os.path.dirname(save_map_path), exist_ok=True)
        with open(save_map_path, "w", encoding="utf-8") as handle:
            handle.write(format_maze_map(maze_map))
            handle.write("\n")

    return {"maze_map": maze_map}
