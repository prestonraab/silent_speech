# in main.py
import torch
import torch.nn as nn
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
print(os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]) # Prints 1, assuming the variable is set

torch.manual_seed(0)
conv = nn.Conv1d(1, 65537, 3, padding=1)

x = torch.ones([1, 1, 3])
y_mps = conv.to("mps")(x.to("mps")) # Fails with NotImplementedError
