import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
device = torch.device(DEVICE)

if device.type == "cuda":
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")

BASE_STAMP = 1577836800 # 00:00:00 01-01-2020 GMT