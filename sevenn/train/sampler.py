import math
from typing import Iterator, List, Optional

import numpy as np
import torch.utils.data.sampler
from torch_geometric.data import Dataset


class OrderedSampler(torch.utils.data.sampler.Sampler):
    """
    Deterministic sampler for DDP training with resume support.
    Work both for single / multi GPU.
    For single GPU, use world_size=1, rank=0 (default).
    """

    def __init__(
        self,
        dataset,
        sequence: Optional[List[int]] = None,
        shuffle: bool = False,
        seed: int = 777,
        world_size: int = 1,
        rank: int = 0,
    ):
        if sequence is None:
            self.sequence = np.arange(len(dataset))
        else:
            self.sequence = np.array(sequence)
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        assert world_size > 0 and rank < world_size
        self.world_size = world_size
        self.rank = rank

        self.total_samples_per_rank = math.ceil(
            len(self.sequence) / self.world_size
        )
        self.total_size = self.total_samples_per_rank * self.world_size

        self._start_index = 0

    def continue_from_data_progress(
        self,
        numpy_rng_state: dict,
        total_data_num: int = -1,
        current_data_index: int = 0,
    ):
        if numpy_rng_state is not None:
            self.rng.bit_generator.state = numpy_rng_state
        if total_data_num < 0:  # Nothing to continue
            return
        elif total_data_num != len(self.sequence):
            raise ValueError(
                'data_progress not compatible'
                + 'set reset_data_progress: True'
            )
        self._start_index = current_data_index

    def get_rng_state(self):
        return self.rng.bit_generator.state

    def permutate_sequence(self):
        return self.rng.permutation(self.sequence)

    def refresh_sequence(self):
        self._start_index = 0

    def __iter__(self) -> Iterator[int]:
        indices = self.sequence.copy()
        if self.shuffle:
            # deterministically shuffle based on numpy rng state
            indices = self.permutate_sequence()

        # add extra samples to make it evenly divisible
        padding_size = self.total_size - len(indices)
        if padding_size <= len(indices):
            padding_sequence = indices[:padding_size]
        else:
            rep_num = math.ceil(padding_size / len(indices))
            padding_sequence = np.tile(indices, rep_num)[:padding_size]
        indices = np.concatenate((indices, padding_sequence))
        assert len(indices) == self.total_size

        # subsample
        indices = indices[
            self._start_index + self.rank : self.total_size : self.world_size
        ]
        assert len(indices) == len(self)
        self.refresh_sequence()  # after one epoch, it initializes to 0.

        return iter(indices)

    def __len__(self) -> int:
        current_idx_per_rank = int(self._start_index / self.world_size)
        return self.total_samples_per_rank - current_idx_per_rank
