from pyexpat import model
from src.models.mlp import BertClassifier
from transformers import AutoTokenizer
from src.config.config import MODEL_REGISTRY, device
import os
import torch

def load_artifacts(params):
    tokenizer = AutoTokenizer.from_pretrained(params["bert_path"], do_lower_case=True)

    model = BertClassifier(params)
    model.load_state_dict(torch.load(os.path.join(MODEL_REGISTRY, "model.bin"), map_location=device))
    model.eval()

    return model, tokenizer
