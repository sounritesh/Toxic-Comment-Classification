import torch
import torch_xla.core.xla_model as xm

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = xm.xla_device()