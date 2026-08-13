"""Nav v4 training/evaluation provenance contract.

The manifest is deliberately small: it records only the facts needed to keep
legacy geometry/reward runs out of the fixed-two-Pillar B/M/U/D protocol.
"""
import hashlib
import json
import os


BENCHMARK_VERSION = "nav_bmud_v3_clearance34"
TRAINING_PROTOCOL_VERSION = "nav_train_v4_start45_goal39_pair34_fail2"
TRAINING_MANIFEST_NAME = "training_manifest.json"
MANIFEST_SCHEMA_VERSION = 1

# Single source for the fixed Nav v4 training contract.
TRAIN_NUM_PILLARS = 2
TRAIN_OBSTACLE_TYPE = "Pillar"
TRAIN_OBSTACLE_MOTION = "static"
TRAIN_FAILURE_HANDLING = {
    "collision_is_failure": True,
    "collision_penalty": 2.0,
    "oob_is_failure": True,
    "oob_penalty": 2.0,
}

# Exact formal B/M/U/D evaluation budget and seed contract.
FORMAL_EVALUATION_CONFIG = {
    "n_env": 20,
    "max_steps": 500,
    "base_seed": 2024,
}


class ProtocolError(RuntimeError):
    """A run/checkpoint does not satisfy the formal Nav v4 protocol."""


def build_training_manifest(trainer, seed, tag):
    """Return the canonical manifest written before a new training run."""
    return {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "task": "nav",
        "benchmark_version": BENCHMARK_VERSION,
        "training_protocol": {
            "version": TRAINING_PROTOCOL_VERSION,
            "obstacles": {
                "type": TRAIN_OBSTACLE_TYPE,
                "motion": TRAIN_OBSTACLE_MOTION,
                "count": TRAIN_NUM_PILLARS,
            },
            "failure_handling": dict(TRAIN_FAILURE_HANDLING),
        },
        "run": {
            "trainer": str(trainer),
            "seed": int(seed),
            "tag": str(tag),
        },
    }


def validate_training_manifest(manifest):
    """Validate the formal fields and return *manifest* on success."""
    if not isinstance(manifest, dict):
        raise ProtocolError("training manifest must be a JSON object")

    expected = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "task": "nav",
        "benchmark_version": BENCHMARK_VERSION,
    }
    errors = [
        f"{key}={manifest.get(key)!r} (expected {value!r})"
        for key, value in expected.items()
        if manifest.get(key) != value
    ]

    training = manifest.get("training_protocol")
    if not isinstance(training, dict):
        errors.append("training_protocol is missing or not an object")
    else:
        if training.get("version") != TRAINING_PROTOCOL_VERSION:
            errors.append(
                f"training_protocol.version={training.get('version')!r} "
                f"(expected {TRAINING_PROTOCOL_VERSION!r})"
            )
        obstacles = training.get("obstacles")
        expected_obstacles = {
            "type": TRAIN_OBSTACLE_TYPE,
            "motion": TRAIN_OBSTACLE_MOTION,
            "count": TRAIN_NUM_PILLARS,
        }
        if not isinstance(obstacles, dict):
            errors.append("training_protocol.obstacles is missing or not an object")
        else:
            errors.extend(
                f"training_protocol.obstacles.{key}={obstacles.get(key)!r} "
                f"(expected {value!r})"
                for key, value in expected_obstacles.items()
                if obstacles.get(key) != value
            )
        failure_handling = training.get("failure_handling")
        if not isinstance(failure_handling, dict):
            errors.append("training_protocol.failure_handling is missing or not an object")
        else:
            errors.extend(
                f"training_protocol.failure_handling.{key}={failure_handling.get(key)!r} "
                f"(expected {value!r})"
                for key, value in TRAIN_FAILURE_HANDLING.items()
                if (
                    type(failure_handling.get(key)) is not type(value)
                    or failure_handling.get(key) != value
                )
            )

    run = manifest.get("run")
    if not isinstance(run, dict):
        errors.append("run is missing or not an object")
    else:
        if not isinstance(run.get("trainer"), str) or not run["trainer"]:
            errors.append("run.trainer must be a non-empty string")
        if (
            not isinstance(run.get("seed"), int)
            or isinstance(run.get("seed"), bool)
        ):
            errors.append("run.seed must be an integer")
        if not isinstance(run.get("tag"), str) or not run["tag"]:
            errors.append("run.tag must be a non-empty string")

    if errors:
        raise ProtocolError("invalid Nav v4 training manifest: " + "; ".join(errors))
    return manifest


def training_manifest_path(run_dir):
    return os.path.join(os.path.abspath(run_dir), TRAINING_MANIFEST_NAME)


def load_training_manifest(run_dir):
    """Load and validate the manifest located directly in *run_dir*."""
    path = training_manifest_path(run_dir)
    if not os.path.isfile(path):
        raise ProtocolError(
            f"missing {TRAINING_MANIFEST_NAME} in run directory {os.path.dirname(path)!r}"
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read training manifest {path!r}: {exc}") from exc
    return validate_training_manifest(manifest)


def validate_checkpoint_training_protocol(ckpt):
    """Require the selected best checkpoint and a valid manifest in its directory."""
    ckpt = os.path.abspath(ckpt)
    if not os.path.isfile(ckpt):
        raise ProtocolError(f"checkpoint does not exist: {ckpt!r}")
    if os.path.basename(ckpt) != "best.pt":
        raise ProtocolError(
            f"formal Nav evaluation requires the validation-selected best.pt; got {ckpt!r}"
        )
    return load_training_manifest(os.path.dirname(ckpt))


def training_manifest_sha256(manifest):
    """Stable digest used to bind an evaluation result to its manifest."""
    validate_training_manifest(manifest)
    payload = json.dumps(
        manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_training_manifest(run_dir, trainer, seed, tag):
    """Create a manifest, refusing to certify an untracked legacy directory.

    A matching manifest permits an intentional same-run restart.  A non-empty
    directory without one is rejected so old checkpoints cannot be relabelled
    merely by launching the new trainer with an old tag.
    """
    run_dir = os.path.abspath(run_dir)
    os.makedirs(run_dir, exist_ok=True)
    path = training_manifest_path(run_dir)
    desired = build_training_manifest(trainer=trainer, seed=seed, tag=tag)

    if os.path.exists(path):
        existing = load_training_manifest(run_dir)
        if existing != desired:
            raise ProtocolError(
                f"existing manifest does not match this run: {path!r}; use a fresh v4 tag"
            )
        return path

    existing_files = os.listdir(run_dir)
    if existing_files:
        preview = ", ".join(sorted(existing_files)[:5])
        raise ProtocolError(
            f"refusing to certify non-empty run directory without a manifest: "
            f"{run_dir!r} (contains: {preview}); use a fresh v4 tag"
        )

    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(desired, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
    return path
