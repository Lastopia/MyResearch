from __future__ import annotations

from pathlib import Path

from cfg import CFG
from dual_axis_transformer.training_log import (
    CumulativeTrainingTimer,
    DeviceLogStatus,
    FixedWidthTrainingLogger,
    _display_width,
    checkpoint_due,
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_timer_total_survives_checkpoint_resume() -> None:
    first_clock = FakeClock()
    first = CumulativeTrainingTimer(first_clock)
    first.start()
    first_clock.now = 12.5
    assert first.mark_log_interval() == 12.5
    first_clock.now = 20.0
    state = first.state_dict()

    resumed_clock = FakeClock()
    resumed = CumulativeTrainingTimer(resumed_clock)
    resumed.load_state_dict(state)
    resumed.start()
    resumed_clock.now = 5.0

    assert resumed.elapsed_seconds == 25.0
    assert resumed.mark_log_interval() == 12.5


def test_fixed_width_rows_keep_separators_aligned(tmp_path: Path) -> None:
    clock = FakeClock()
    timer = CumulativeTrainingTimer(clock)
    timer.start()
    status = DeviceLogStatus("GPU", "Test GPU", 2, 4, 3.5, 24.0)
    path = tmp_path / "train.log"

    with FixedWidthTrainingLogger(
        path,
        timer=timer,
        device_provider=lambda: status,
        console_mode="quiet",
    ) as logger:
        clock.now = 1.25
        train = logger.log(
            "train", step=50, total_steps=1000, loss=1.25, seed=11, task="dual_tag"
        )
        clock.now = 2.0
        valid = logger.log(
            "valid", step=50, total_steps=1000, loss=1.1, seed=11, task="dual_tag"
        )
        checkpoint = logger.log(
            "checkpt", step=50, total_steps=1000, seed=11, task="dual_tag"
        )

    lines = path.read_text(encoding="utf-8").splitlines()
    separator_positions = [
        [index for index, character in enumerate(line) if character == "|"]
        for line in [lines[0], train, valid, checkpoint]
    ]
    assert all(positions == separator_positions[0] for positions in separator_positions)
    assert "GPU 2/4 Test GPU" in train
    assert "3.50/24.00 GiB" in train
    assert "1.25s/1.25s" in train
    assert "-/2.00s" in valid


def test_summary_console_hides_train_but_shows_validation(
    tmp_path: Path, capsys: object
) -> None:
    timer = CumulativeTrainingTimer(lambda: 0.0)
    status = DeviceLogStatus("CPU", "CPU", 0, 0, None, None)
    with FixedWidthTrainingLogger(
        tmp_path / "train.log",
        timer=timer,
        device_provider=lambda: status,
        console_mode="summary",
    ) as logger:
        logger.log("train", step=1, loss=1.0)
        logger.log("valid", step=1, loss=0.9)
    captured = capsys.readouterr().out
    assert "TRAIN" not in captured
    assert "VALID" in captured
    data_line = (tmp_path / "train.log").read_text(encoding="utf-8").splitlines()[2]
    assert data_line.split(" | ")[5].strip() == ""


def test_interval_console_shows_train_progress(
    tmp_path: Path, capsys: object
) -> None:
    timer = CumulativeTrainingTimer(lambda: 0.0)
    status = DeviceLogStatus("CPU", "CPU", 0, 0, None, None)
    with FixedWidthTrainingLogger(
        tmp_path / "train.log",
        timer=timer,
        device_provider=lambda: status,
        console_mode="interval",
    ) as logger:
        logger.log("train", step=100, total_steps=6000, loss=1.0)
    captured = capsys.readouterr().out
    assert "TRAIN" in captured
    assert "0100/6000" in captured
    assert max(_display_width(line) for line in captured.splitlines()) <= 120


def test_console_is_compact_but_file_keeps_full_values(
    tmp_path: Path, capsys: object
) -> None:
    timer = CumulativeTrainingTimer(lambda: 0.0)
    status = DeviceLogStatus("GPU", "NVIDIA L40S", 1, 2, 3.5, 44.6)
    path = tmp_path / "train.log"
    task = "formal_language_model/concept_bus_v2"
    with FixedWidthTrainingLogger(
        path,
        timer=timer,
        device_provider=lambda: status,
        console_mode="interval",
    ) as logger:
        logger.log(
            "train", step=100, total_steps=30518, loss=1.23456789,
            seed=11, task=task,
        )
    console = capsys.readouterr().out.strip()
    file_text = path.read_text(encoding="utf-8")
    assert _display_width(console) <= 120
    assert "G1/2 L40S" in console
    assert "lm/v2" in console
    assert "GPU 1/2 NVIDIA L40S" in file_text
    assert task in file_text


def test_training_intervals_are_centralized_in_cfg() -> None:
    assert CFG["train"]["log_interval_steps"] > 0
    assert CFG["train"]["eval_interval_steps"] > 0
    assert CFG["train"]["checkpoint_interval_minutes"] == 120
    assert CFG["logging"]["console_mode"] == "interval"


def test_checkpoint_policy_uses_accumulated_training_time() -> None:
    clock = FakeClock()
    timer = CumulativeTrainingTimer(clock)
    timer.start()
    clock.now = 7199.0
    assert not checkpoint_due(timer, 0.0, 120)
    clock.now = 7200.0
    assert checkpoint_due(timer, 0.0, 120)
    assert checkpoint_due(timer, 7200.0, 120, final_step=True)
