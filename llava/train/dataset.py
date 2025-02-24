import io

from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index
import os
import random
import re
import json
from typing import Dict

import cv2
import imageio
import numpy as np
import torch
import torchvision.transforms as T
import transformers
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, WeightedRandomSampler
from torchvision.transforms.functional import InterpolationMode


class WeightedConcatDataset(ConcatDataset):
    def __init__(self, datasets, weights):
        super().__init__(datasets)
        self.weights = torch.DoubleTensor(weights)
        self.total_size = sum(len(d) for d in datasets)
        self.sampler = WeightedRandomSampler(weights=self.weights, num_samples=self.total_size, replacement=True)

    def __iter__(self):
        return iter(self.sampler)

    def __len__(self):
        return self.total_size




