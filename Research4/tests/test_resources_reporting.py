from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import dual_axis_transformer.resources as resource_module
from dual_axis_transformer.reporting import build_report, discover_final_records
from dual_axis_transformer.resources import (
    CPUInfo,
    GPUInfo,
    ResourcePolicy,
    ResourceSnapshot,
    RuntimeObservation,
    WorkloadSpec,
    plan_jobs,
    recommend_next_launch,
)
from dual_axis_transformer.scheduler import (
    ScheduledCommand,
    _job_environment,
    execute_schedule,
)
from dual_axis_transformer.verdict import _difference_gate


def fake_snapshot() -> ResourceSnapshot:
    return ResourceSnapshot(
        captured_at="2026-08-06T00:00:00+00:00",
        hostname="test-host",
        platform="test",
        python_version="3.13",
        torch_version="2.11",
        cuda_version="13.0",
        cpu=CPUInfo(
            logical_cores=32,
            physical_cores=16,
            total_memory_gb=128.0,
            available_memory_gb=100.0,
            utilization_percent=10.0,
        ),
        gpus=(
            GPUInfo(
                index=0,
                name="Small GPU",
                total_memory_gb=16.0,
                free_memory_gb=14.0,
                utilization_percent=0.0,
                memory_utilization_percent=0.0,
                compute_capability="8.6",
                bf16_supported=True,
                torch_usable=True,
                probe_error=None,
                tier="standard",
            ),
            GPUInfo(
                index=1,
                name="Large GPU",
                total_memory_gb=80.0,
                free_memory_gb=70.0,
                utilization_percent=0.0,
                memory_utilization_percent=0.0,
                compute_capability="9.0",
                bf16_supported=True,
                torch_usable=True,
                probe_error=None,
                tier="xlarge",
            ),
        ),
    )


def workload(name: str, *, profiling: bool = False) -> WorkloadSpec:
    return WorkloadSpec(
        name=name,
        global_batch_size=64,
        max_micro_batch_size=64,
        estimated_model_memory_gb=4.0,
        estimated_activation_memory_per_sample_gb=0.5,
        profiling=profiling,
    )


def test_visible_gpu_without_working_torch_cuda_falls_back_to_cpu(
    monkeypatch: object,
) -> None:
    monkeypatch.setattr(
        resource_module,
        "_nvidia_smi_rows",
        lambda: {
            0: {
                "name": "Visible but unusable GPU",
                "total_memory_gb": 24.0,
                "free_memory_gb": 23.0,
                "utilization_percent": 0.0,
                "memory_utilization_percent": 0.0,
            }
        },
    )
    monkeypatch.setattr(resource_module.torch.cuda, "is_available", lambda: False)

    snapshot = resource_module.detect_resources()
    assert len(snapshot.gpus) == 1
    assert snapshot.gpus[0].torch_usable is False
    assert snapshot.gpus[0].probe_error
    assert plan_jobs([workload("fallback")], snapshot)[0].device == "cpu"


def test_adaptive_planner_spreads_jobs_and_uses_large_gpu_capacity() -> None:
    policy = ResourcePolicy(allow_colocation=True, max_jobs_per_gpu=2)
    plans = plan_jobs(
        [workload("first"), workload("second"), workload("third")],
        fake_snapshot(),
        policy,
    )
    assert len(plans) == 3
    assert {plan.gpu_index for plan in plans[:2]} == {0, 1}
    large = next(plan for plan in plans if plan.gpu_index == 1)
    small = next(plan for plan in plans if plan.gpu_index == 0)
    assert large.micro_batch_size > small.micro_batch_size
    assert all(plan.effective_batch_size == 64 for plan in plans)
    assert all(plan.wave == 0 for plan in plans)


def test_profile_jobs_are_exclusive_and_move_to_next_wave() -> None:
    plans = plan_jobs(
        [
            workload("profile-1", profiling=True),
            workload("profile-2", profiling=True),
            workload("profile-3", profiling=True),
        ],
        fake_snapshot(),
        ResourcePolicy(allow_colocation=True, profile_exclusive=True),
    )
    assert [plan.wave for plan in plans].count(0) == 2
    assert [plan.wave for plan in plans].count(1) == 1
    assert all(plan.exclusive for plan in plans)


def test_required_micro_batch_is_never_silently_reduced() -> None:
    jobs = [
        WorkloadSpec(
            name=f"fair-{index}",
            global_batch_size=128,
            max_micro_batch_size=128,
            estimated_model_memory_gb=4.0,
            estimated_activation_memory_per_sample_gb=0.5,
            required_micro_batch_size=64,
        )
        for index in range(3)
    ]
    plans = plan_jobs(
        jobs,
        fake_snapshot(),
        ResourcePolicy(allow_colocation=False, max_jobs_per_gpu=1),
    )
    # Only the large fake GPU fits micro=64, so jobs must queue in waves.
    assert [plan.wave for plan in plans] == [0, 1, 2]
    assert {plan.gpu_index for plan in plans} == {1}
    assert all(plan.micro_batch_size == 64 for plan in plans)
    assert all(plan.gradient_accumulation_steps == 2 for plan in plans)
    for plan in plans:
        environment = _job_environment(plan)
        assert environment["CONCEPT_BUS_MICRO_BATCH"] == "64"
        assert environment["CONCEPT_BUS_GRAD_ACCUM"] == "2"


def test_three_fair_jobs_use_two_40g_gpus_in_two_waves() -> None:
    original = fake_snapshot()
    snapshot = replace(
        original,
        gpus=(
            replace(
                original.gpus[0],
                name="A100 40G #0",
                total_memory_gb=40.0,
                free_memory_gb=39.0,
                tier="large",
            ),
            replace(
                original.gpus[1],
                name="A100 40G #1",
                total_memory_gb=40.0,
                free_memory_gb=39.0,
                tier="large",
            ),
        ),
    )
    jobs = [
        WorkloadSpec(
            name=name,
            global_batch_size=128,
            max_micro_batch_size=128,
            estimated_model_memory_gb=4.0,
            estimated_activation_memory_per_sample_gb=0.12,
            required_micro_batch_size=64,
        )
        for name in ("standard", "projector", "v2")
    ]
    plans = plan_jobs(
        jobs,
        snapshot,
        ResourcePolicy(allow_colocation=False, max_jobs_per_gpu=1),
    )
    assert [plan.wave for plan in plans] == [0, 0, 1]
    assert {plan.gpu_index for plan in plans if plan.wave == 0} == {0, 1}
    assert all(plan.micro_batch_size == 64 for plan in plans)
    assert all(plan.gradient_accumulation_steps == 2 for plan in plans)


def test_runtime_recommendations_preserve_effective_batch() -> None:
    recommendation = recommend_next_launch(
        RuntimeObservation(
            gpu_utilization_percent=55.0,
            peak_memory_fraction=0.60,
            cpu_utilization_percent=50.0,
            available_ram_fraction=0.50,
            dataloader_wait_fraction=0.20,
        )
    )
    assert "increase_dataloader_workers" in recommendation["actions"]
    assert "increase_micro_batch_after_recalibration" in recommendation["actions"]
    assert recommendation["effective_batch_must_remain_constant"] is True


def write_final(
    output_root: Path,
    stage: str,
    method: str,
    seed: int,
    metrics: dict[str, float],
    *,
    run_hash: str = "hash",
    source: str | None = None,
    attempt: int = 1,
) -> None:
    target = (
        output_root
        / "runs"
        / stage
        / method
        / f"seed{seed}"
        / run_hash
        / f"attempt{attempt}"
        / "metrics"
    )
    target.mkdir(parents=True)
    (target / "final.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "method": method,
                "seed": seed,
                "metrics": metrics,
                "source_fingerprint": source,
            }
        ),
        encoding="utf-8",
    )


def test_record_discovery_filters_source_and_never_counts_duplicate_seed(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    write_final(
        output, "dual_tag", "standard", 11, {"loss": 9.0},
        run_hash="old", source="old-source",
    )
    write_final(
        output, "dual_tag", "standard", 11, {"loss": 2.0},
        run_hash="new-a", source="new-source",
    )
    write_final(
        output, "dual_tag", "standard", 11, {"loss": 1.0},
        run_hash="new-b", source="new-source", attempt=2,
    )
    records = discover_final_records(
        output, source_fingerprint="new-source"
    )
    assert len(records) == 1
    assert records[0]["seed"] == 11
    assert records[0]["metrics"]["loss"] == 1.0


def test_attention_increment_gate_requires_every_seed_to_improve() -> None:
    records = []
    for seed, treatment, reference in (
        (11, 0.90, 0.80),
        (22, 0.90, 0.80),
        (33, 0.79, 0.80),
    ):
        records.extend(
            [
                {
                    "stage": "dual_tag",
                    "method": "concept_bus_v2",
                    "seed": seed,
                    "metrics": {"f1": treatment},
                },
                {
                    "stage": "dual_tag",
                    "method": "concept_projector",
                    "seed": seed,
                    "metrics": {"f1": reference},
                },
            ]
        )
    gate = _difference_gate(
        records,
        name="attention_increment",
        stage="dual_tag",
        treatment="concept_bus_v2",
        reference="concept_projector",
        metric="f1",
        minimum_runs=3,
        threshold=0.01,
        require_positive_each_seed=True,
    )
    assert gate["mean_difference"] > 0.01
    assert gate["direction_consistent"] is False
    assert gate["status"] == "fail"


def test_report_centralizes_csv_svg_and_interactive_js(tmp_path: Path) -> None:
    output_root = tmp_path / "output"
    write_final(output_root, "dual_tag", "standard", 11, {"f1": 0.80})
    write_final(output_root, "dual_tag", "standard", 22, {"f1": 0.82})
    write_final(output_root, "dual_tag", "concept_bus_v2", 11, {"f1": 0.95})
    write_final(output_root, "dual_tag", "concept_bus_v2", 22, {"f1": 0.97})

    resource_dir = output_root / "resources"
    resource_dir.mkdir(parents=True)
    (resource_dir / "snapshot.json").write_text(
        json.dumps(fake_snapshot().to_dict()), encoding="utf-8"
    )

    report_dir = build_report(output_root, stage="dual_tag")
    assert (report_dir / "per_seed.csv").exists()
    assert (report_dir / "summary.csv").exists()
    assert (report_dir / "summary.json").exists()
    assert (report_dir / "figures" / "f1.svg").exists()
    assert (report_dir / "dashboard" / "index.html").exists()
    assert (report_dir / "dashboard" / "data.js").exists()
    assert (report_dir / "dashboard" / "report.js").exists()
    assert "concept_bus_v2" in (report_dir / "dashboard" / "data.js").read_text(
        encoding="utf-8"
    )


def test_scheduler_executes_wave_captures_and_forwards_logs(
    tmp_path: Path, capsys: object
) -> None:
    plan = plan_jobs(
        [
            WorkloadSpec(
                name="hello",
                global_batch_size=4,
                max_micro_batch_size=4,
                estimated_model_memory_gb=1.0,
                estimated_activation_memory_per_sample_gb=0.1,
            )
        ],
        ResourceSnapshot(
            captured_at="2026-08-06T00:00:00+00:00",
            hostname="cpu-only",
            platform="test",
            python_version="3.13",
            torch_version="2.11",
            cuda_version=None,
            cpu=CPUInfo(8, 4, 16.0, 12.0, 10.0),
            gpus=(),
        ),
    )
    run_dir = execute_schedule(
        plan,
        [
            ScheduledCommand(
                name="hello",
                command=(
                    sys.executable,
                    "-c",
                    "import sys; print('scheduled-ok 进度正常'); "
                    "print('scheduled-warning', file=sys.stderr)",
                ),
            )
        ],
        tmp_path / "output",
        monitor=False,
        monitor_interval_seconds=0.01,
    )
    statuses = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert statuses[0]["status"] == "completed"
    assert "scheduled-ok 进度正常" in (run_dir / "hello" / "stdout.log").read_text(
        encoding="utf-8"
    )
    assert "scheduled-warning" in (run_dir / "hello" / "stderr.log").read_text(
        encoding="utf-8"
    )
    captured = capsys.readouterr()
    assert "hello" in captured.out and "scheduled-ok 进度正常" in captured.out
    assert "hello" in captured.err and "scheduled-warning" in captured.err
    assert all(len(line) <= 120 for line in captured.out.splitlines())
    assert all(len(line) <= 120 for line in captured.err.splitlines())
