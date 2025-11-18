#TODO testen met verschillende waarden, clean-up, ?zwart-wit?, ?nieuw model?, ptt
import os
import time
from collections import Counter

import torch as pt
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize,
    RandomHorizontalFlip, RandomRotation, ColorJitter
)
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# ====================== CONFIG ======================
device = pt.device("cuda" if pt.cuda.is_available() else "cpu")
print("Using device:", device)