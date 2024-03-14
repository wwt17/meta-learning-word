from typing import Optional
import torch


class ConcatLMDataset(torch.utils.data.Dataset):
    def __init__(self, sequence, context_length: int, stride: Optional[int] = None, offset: int = 0):
        super().__init__()

        self.sequence = sequence
        self.context_length = context_length
        if stride is None:
            stride = context_length
        self.start_indices = range(offset, len(sequence) - context_length + 1, stride)

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, i):
        start = self.start_indices[i]
        end = start + self.context_length
        return self.sequence[start:end]