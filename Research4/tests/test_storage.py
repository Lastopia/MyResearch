from __future__ import annotations

import copy
from pathlib import Path

from cfg import CFG
from dual_axis_transformer.storage import (
    checkpoint_dir_for_run,
    consume_legacy_compatibility,
    legacy_compatibility_active,
    legacy_compatibility_pending,
    prepare_storage,
)


def test_legacy_data_and_checkpoints_move_out_of_output(tmp_path: Path) -> None:
    cfg = copy.deepcopy(CFG)
    output = tmp_path / "output"
    legacy_data = output / "data" / "dual_tag" / "dataset" / "train.jsonl"
    legacy_data.parent.mkdir(parents=True)
    legacy_data.write_text("example\n", encoding="utf-8")
    nested_data = output / "diagnostic" / "data" / "clutrr" / "archive.zip"
    nested_data.parent.mkdir(parents=True)
    nested_data.write_bytes(b"archive")

    run_dir = (
        output
        / "runs"
        / "dual_tag"
        / "concept_bus_v2"
        / "seed11"
        / "abcdef123456"
        / "attempt1"
    )
    legacy_checkpoint = run_dir / "checkpoints" / "latest.pt"
    legacy_checkpoint.parent.mkdir(parents=True)
    legacy_checkpoint.write_bytes(b"checkpoint")

    roots = prepare_storage(cfg, output)
    checkpoint_dir = checkpoint_dir_for_run(cfg, output, run_dir)

    assert roots.data == tmp_path / "data"
    assert roots.checkpoints == tmp_path / "ckpt"
    assert (roots.data / "dual_tag" / "dataset" / "train.jsonl").exists()
    assert (
        roots.data / "legacy_output" / "diagnostic" / "clutrr" / "archive.zip"
    ).exists()
    assert (checkpoint_dir / "latest.pt").read_bytes() == b"checkpoint"
    assert not (output / "data").exists()
    assert not (output / "diagnostic").exists()
    assert not (run_dir / "checkpoints").exists()
    assert (run_dir / "checkpoint_location.json").exists()
    assert legacy_compatibility_pending(output)
    consume_legacy_compatibility(output, "new-storage-source")
    assert not legacy_compatibility_pending(output)
    assert legacy_compatibility_active(output, "new-storage-source")
    assert not legacy_compatibility_active(output, "future-source")
