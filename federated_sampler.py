import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torchvision import transforms

import pandas as pd
import random
import numpy as np
from PIL import Image
import os
from copy import deepcopy

class FederatedBatchSampler:
    """Batch sampler for federated learning, sampling batches from multiple datasets."""
    def __init__(self, datasets, batch_size):
        self.datasets = datasets
        self.batch_size = batch_size
        self.dataloaders = [Data.DataLoader(ds, batch_size=self.batch_size, shuffle=True) for ds in self.datasets]
        self.iterators = self.reset_iterators()

    def reset_iterators(self):
        """Reset iterators for each dataloader."""
        return [iter(dl) for dl in self.dataloaders]

    def sample_batch_from_dataset(self, dataset_index):
        """
        Sample a batch from a specific dataset.
        """
        try:
            batch = next(self.iterators[dataset_index])
        except StopIteration:
            self.iterators[dataset_index] = iter(self.dataloaders[dataset_index])
            batch = next(self.iterators[dataset_index])
        return batch

    def sample_batches(self):
        """
        Sample batches from all datasets.
        """
        batches = []
        for i, iterator in enumerate(self.iterators):
            try:
                batch = next(iterator)
            except StopIteration:
                self.iterators[i] = iter(self.dataloaders[i])
                batch = next(self.iterators[i])
            batches.append(batch)
        return batches

    def start_new_epoch(self):
        """Start a new epoch by resetting all iterators."""
        self.iterators = self.reset_iterators()