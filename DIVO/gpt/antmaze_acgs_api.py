import logging
import os
from typing import Any, Dict, List, Optional

from DIVO.env.antmaze import MazeGeneratorExecutor
from DIVO.gpt.prompt_builder import PromptBuilder
from DIVO.gpt.utils import extract_code, get_client, llm_interaction

LOGGER = logging.getLogger(__name__)


def _resolve_prompt_dir(prompt_dir: str) -> str:
    if os.path.isdir(prompt_dir):
        return prompt_dir
    if os.path.isabs(prompt_dir):
        return prompt_dir

    # This file lives under <repo>/DIVO/gpt. Resolve relative prompt paths from
    # the repository root as a fallback, which is needed when running from
    # CurricuLLM/.
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidate = os.path.join(repo_root, prompt_dir)
    if os.path.isdir(candidate):
        return candidate
    return prompt_dir


class AntMazeACGS_API:
    """
    LLM-in-the-loop API for AntMaze maze-layout generators.

    The generated program must define:
        def generate_maze_map(seed: int = None) -> list
    """

    def __init__(
        self,
        task_name: str = "AntMaze",
        prompt_dir: str = "DIVO/gpt/prompt",
        api_type: str = "deepseek",
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        max_evolve_retries: int = 3,
        sanity_check_count: int = 5,
        executor_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.api_type = api_type
        self.api_key = api_key
        self.base_url = base_url
        self.client = None
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_builder = PromptBuilder(task_name=task_name, prompt_dir=_resolve_prompt_dir(prompt_dir))
        self.executor = MazeGeneratorExecutor(**(executor_kwargs or {}))
        self.generator_code: Optional[str] = None
        self.max_evolve_retries = int(max_evolve_retries)
        self.sanity_check_count = int(sanity_check_count)

    def _get_client(self):
        if self.client is None:
            self.client = get_client(
                api_type=self.api_type,
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self.client

    def load_generator_code(self, code: str) -> bool:
        if not code:
            return False
        if not self.executor.load_generator_code(code):
            return False
        self.generator_code = code
        return True

    def load_generator_file(self, path: str) -> bool:
        with open(path, "r", encoding="utf-8") as handle:
            return self.load_generator_code(handle.read())

    def export_generator_code(self) -> Optional[str]:
        return self.generator_code

    def init_generator(self) -> Optional[str]:
        system = self.prompt_builder.load_initial_system()
        user = self.prompt_builder.load_initial_user()
        LOGGER.info("[AntMazeACGS] Generating initial maze generator...")
        raw = llm_interaction(
            self._get_client(),
            self.model,
            system,
            user,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        code = extract_code(raw)
        if code is None:
            LOGGER.error("[AntMazeACGS] Initial code extraction failed")
            return None
        if not self.load_generator_code(code):
            LOGGER.error("[AntMazeACGS] Initial code loading failed")
            return None
        if not self.executor.sanity_check(num_tests=self.sanity_check_count):
            LOGGER.error("[AntMazeACGS] Initial generator sanity check failed")
            self.generator_code = None
            return None
        return code

    def evolve(
        self,
        batch_stats: Dict[str, int],
        fv_result: Optional[Dict],
        reason: str,
        current_generator_code: Optional[str] = None,
        feedback_mode: str = "coarse",
        attribution_result: Optional[Dict] = None,
        coverage_summary: Optional[Dict] = None,
        attribution_history: Optional[List[Dict]] = None,
        cfa_result: Optional[Dict] = None,
    ) -> Optional[str]:
        if current_generator_code is None:
            current_generator_code = self.generator_code

        system = self.prompt_builder.load_evolve_system()
        user = self.prompt_builder.build_evolve_user(
            batch_stats=batch_stats,
            fv_result=fv_result,
            reason=reason,
            current_generator_code=current_generator_code,
            feedback_mode=feedback_mode,
            attribution_result=attribution_result,
            coverage_summary=coverage_summary,
            attribution_history=attribution_history,
            cfa_result=cfa_result,
        )

        old_code = self.generator_code
        for attempt in range(self.max_evolve_retries):
            LOGGER.info("[AntMazeACGS] Evolve attempt %s/%s", attempt + 1, self.max_evolve_retries)
            raw = llm_interaction(
                self._get_client(),
                self.model,
                system,
                user,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            code = extract_code(raw)
            if code is None:
                LOGGER.warning("[AntMazeACGS] LLM returned no code")
                continue
            if not self.load_generator_code(code):
                LOGGER.warning("[AntMazeACGS] Candidate generator failed to load")
                continue
            if not self.executor.sanity_check(num_tests=self.sanity_check_count):
                LOGGER.warning("[AntMazeACGS] Candidate generator failed sanity check")
                if old_code:
                    self.load_generator_code(old_code)
                continue
            return code

        if old_code:
            self.load_generator_code(old_code)
        return None

    def generate_maze_map(self, seed: Optional[int] = None):
        return self.executor.generate(seed=seed)

    def get_prompt_text(
        self,
        batch_stats: Dict[str, int],
        fv_result: Optional[Dict],
        reason: str,
        current_generator_code: Optional[str] = None,
        feedback_mode: str = "coarse",
        attribution_result: Optional[Dict] = None,
        coverage_summary: Optional[Dict] = None,
        attribution_history: Optional[List[Dict]] = None,
        cfa_result: Optional[Dict] = None,
    ) -> tuple:
        system = self.prompt_builder.load_evolve_system()
        user = self.prompt_builder.build_evolve_user(
            batch_stats=batch_stats,
            fv_result=fv_result,
            reason=reason,
            current_generator_code=current_generator_code or self.generator_code,
            feedback_mode=feedback_mode,
            attribution_result=attribution_result,
            coverage_summary=coverage_summary,
            attribution_history=attribution_history,
            cfa_result=cfa_result,
        )
        return system, user

    @property
    def has_generator(self) -> bool:
        return self.generator_code is not None and self.executor.generate_maze_map is not None
