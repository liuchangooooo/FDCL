"""Nav v4 训练/评测 provenance 合同的纯文件系统回归。"""
import json
import contextlib
import io
import os
import tempfile

from nav.protocol import (
    BENCHMARK_VERSION,
    FORMAL_EVALUATION_CONFIG,
    TRAINING_PROTOCOL_VERSION,
    ProtocolError,
    build_training_manifest,
    load_training_manifest,
    training_manifest_sha256,
    validate_checkpoint_training_protocol,
    validate_training_manifest,
    write_training_manifest,
)


def _must_raise(fn, contains):
    try:
        fn()
    except ProtocolError as exc:
        assert contains in str(exc), str(exc)
    else:
        raise AssertionError(f"expected ProtocolError containing {contains!r}")


def main():
    manifest = build_training_manifest("nav.train_stage2", seed=0, tag="navv4_test")
    assert validate_training_manifest(manifest) is manifest
    assert manifest["benchmark_version"] == BENCHMARK_VERSION
    assert manifest["training_protocol"]["version"] == TRAINING_PROTOCOL_VERSION
    assert manifest["training_protocol"]["obstacles"] == {
        "type": "Pillar", "motion": "static", "count": 2,
    }
    assert manifest["training_protocol"]["failure_handling"] == {
        "collision_is_failure": True,
        "collision_penalty": 2.0,
        "oob_is_failure": True,
        "oob_penalty": 2.0,
    }
    assert FORMAL_EVALUATION_CONFIG == {"n_env": 20, "max_steps": 500, "base_seed": 2024}

    with tempfile.TemporaryDirectory() as root:
        run_dir = os.path.join(root, "navv4_test")
        write_training_manifest(run_dir, "nav.train_stage2", seed=0, tag="navv4_test")
        loaded = load_training_manifest(run_dir)
        assert training_manifest_sha256(loaded) == training_manifest_sha256(manifest)

        ckpt = os.path.join(run_dir, "best.pt")
        with open(ckpt, "wb") as f:
            f.write(b"protocol-test-placeholder")
        assert validate_checkpoint_training_protocol(ckpt) == loaded
        latest = os.path.join(run_dir, "latest.pt")
        with open(latest, "wb") as f:
            f.write(b"diagnostic-only-placeholder")
        _must_raise(
            lambda: validate_checkpoint_training_protocol(latest),
            "requires the validation-selected best.pt",
        )
        _must_raise(
            lambda: write_training_manifest(
                run_dir, "nav.train_stage2", seed=1, tag="navv4_test"
            ),
            "existing manifest does not match this run",
        )

        legacy = os.path.join(root, "legacy_three_pillar")
        os.makedirs(legacy)
        with open(os.path.join(legacy, "best.pt"), "wb") as f:
            f.write(b"legacy")
        _must_raise(
            lambda: write_training_manifest(
                legacy, "nav.train_stage2", seed=0, tag="legacy_three_pillar"
            ),
            "refusing to certify non-empty run directory",
        )
        _must_raise(
            lambda: validate_checkpoint_training_protocol(os.path.join(legacy, "best.pt")),
            "missing training_manifest.json",
        )

        invalid = json.loads(json.dumps(manifest))
        invalid["training_protocol"]["obstacles"]["count"] = 3
        _must_raise(lambda: validate_training_manifest(invalid), "count=3")

        missing_failure = json.loads(json.dumps(manifest))
        del missing_failure["training_protocol"]["failure_handling"]
        _must_raise(
            lambda: validate_training_manifest(missing_failure),
            "failure_handling is missing",
        )

        wrong_penalty = json.loads(json.dumps(manifest))
        wrong_penalty["training_protocol"]["failure_handling"]["collision_penalty"] = 10.0
        _must_raise(
            lambda: validate_training_manifest(wrong_penalty),
            "collision_penalty=10.0",
        )

        wrong_failure_type = json.loads(json.dumps(manifest))
        wrong_failure_type["training_protocol"]["failure_handling"][
            "collision_is_failure"
        ] = 1
        _must_raise(
            lambda: validate_training_manifest(wrong_failure_type),
            "collision_is_failure=1",
        )

        # 聚合器只能接收标准预算 + 合法 manifest/digest + 自洽标量 BMUD。
        from nav import compare_fourcell as compare

        bmud = {
            "B": 0.1, "M": 0.2, "U": 0.3, "D": 0.4,
            "D_static": 0.5, "D_dynamic": 0.4,
            "dynamic_drop": 0.1, "AVG": 0.25,
        }

        def make_result(tag, *, formal=True, digest_ok=True):
            directory = os.path.join(root, tag)
            write_training_manifest(directory, "nav.train_stage2", seed=0, tag=tag)
            checkpoint = os.path.join(directory, "best.pt")
            with open(checkpoint, "wb") as f:
                f.write(tag.encode("utf-8"))
            run_manifest = load_training_manifest(directory)
            result = {
                "benchmark_version": BENCHMARK_VERSION,
                "ckpt": checkpoint,
                "kind": "skill",
                "training_protocol_version": TRAINING_PROTOCOL_VERSION,
                "training_protocol_verified": True,
                "training_manifest_sha256": (
                    training_manifest_sha256(run_manifest) if digest_ok else "bad-digest"
                ),
                "diagnostic_override": not formal,
                "formal_aggregate_eligible": formal,
                "evaluation_config": (
                    FORMAL_EVALUATION_CONFIG if formal else {**FORMAL_EVALUATION_CONFIG, "n_env": 1}
                ),
                "BMUD": bmud,
            }
            with open(os.path.join(directory, "eval_bmud.json"), "w", encoding="utf-8") as f:
                json.dump(result, f)

        make_result("navv4_d_libcur_s0")
        make_result("navv4_diagnostic", formal=False)
        make_result("navv4_bad_digest", digest_ok=False)
        old_runs = compare.RUNS
        compare.RUNS = root
        try:
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                compare.aggregate()
        finally:
            compare.RUNS = old_runs
        output = capture.getvalue()
        assert "navv4_d_libcur_s0" in output
        assert "跨 seed 汇总" in output and "d (ours)" in output
        assert "navv4_diagnostic" not in output
        assert "navv4_bad_digest" not in output
        assert "evaluation config missing/non-standard" in output
        assert "training manifest digest mismatch" in output

    print("ALL PASS")


if __name__ == "__main__":
    main()
