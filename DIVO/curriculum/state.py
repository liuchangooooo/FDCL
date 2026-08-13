from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def default_batch_stats() -> Dict[str, int]:
    return {
        'success': 0,
        'collision': 0,
        'timeout': 0,
        'fall': 0,
    }


@dataclass
class CurriculumRuntimeState:
    initial_generator_code: Optional[str] = None
    current_generator_code: Optional[str] = None
    evolve_count: int = 0
    total_episode_count: int = 0
    success_rate_history: List[float] = field(default_factory=list)
    warmup_completed: bool = False
    first_evolve_triggered: bool = False
    last_evolve_episode: int = 0
    pre_evolve_checkpoint_saved: bool = False
    pending_evolve_reason: Optional[str] = None

    batch_stats: Dict[str, int] = field(default_factory=default_batch_stats)
    batch_failure_counts: Dict[str, int] = field(default_factory=dict)
    batch_episode_count: int = 0
    batch_obstacle_rollout_records: List[Dict[str, Any]] = field(default_factory=list)

    failure_vector_history: List[Dict[str, Any]] = field(default_factory=list)
    evolve_history: List[Dict[str, Any]] = field(default_factory=list)
    attribution_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RLLoopState:
    updates: int = 0
    critic_update: int = 0
    global_step: int = 0
    epoch: int = 0
    num_timesteps: int = 0
    episode_count: int = 0
    num_episode: int = 0
    epsilon: float = 1.0
    max_test_score: float = -100.0
    last_validate_reward: float = -100.0
    recent_episode_rewards: List[float] = field(default_factory=list)
