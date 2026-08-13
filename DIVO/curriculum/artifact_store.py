from __future__ import annotations

from pathlib import Path


class GeneratorArtifactStore:
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.generator_dir = self.output_dir / "generators"
        self.generator_dir.mkdir(parents=True, exist_ok=True)

    def save_initial(self, code: str) -> str:
        return self._write(self.output_dir / "initial_generator.py", code)

    def save_current(self, code: str) -> str:
        return self._write(self.output_dir / "current_generator.py", code)

    def save_generator_version(self, code: str, generator_id: int) -> str:
        return self._write(self.generator_dir / f"generator_{generator_id:03d}.py", code)

    def save_evolved(self, code: str, evolve_count: int) -> str:
        return self._write(self.generator_dir / f"evolve_{evolve_count:03d}.py", code)

    def _write(self, path: Path, code: str) -> str:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")
        return str(path)
