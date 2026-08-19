from __future__ import annotations

import math
from collections.abc import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableBatchSampler(Sampler[list[int]]):
    """Deterministic shuffled batches addressable by epoch and batch offset."""

    def __init__(
        self,
        dataset: Sized,
        *,
        batch_size: int,
        seed: int,
        epoch: int = 0,
        start_batch: int = 0,
        drop_last: bool = True,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset_size = len(dataset)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = int(epoch)
        self.start_batch = int(start_batch)
        if self.batches_per_epoch <= 0:
            raise ValueError("dataset is smaller than one batch")
        self.set_position(self.epoch, self.start_batch)

    @property
    def batches_per_epoch(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return math.ceil(self.dataset_size / self.batch_size)

    def set_position(self, epoch: int, start_batch: int) -> None:
        epoch = int(epoch)
        start_batch = int(start_batch)
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        if not 0 <= start_batch <= self.batches_per_epoch:
            raise ValueError("start_batch is outside the epoch")
        self.epoch = epoch
        self.start_batch = start_batch

    def __len__(self) -> int:
        return self.batches_per_epoch - self.start_batch

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(
            self.dataset_size,
            generator=generator,
        ).tolist()
        for batch_index in range(self.start_batch, self.batches_per_epoch):
            start = batch_index * self.batch_size
            batch = order[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch
