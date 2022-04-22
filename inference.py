import torch
import torch.nn as nn
import numpy as np
import time

import transformers

from src.models.mlp import BertClassifier

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--text", type=str, default="", help="Text to generate inference for")

args = parser.parse_args()

def load_model(params):
    tokenizer = transformers.AutoTokenizer.from_pretrained(params['bert_path'], do_lower_case=True)

    model = BertClassifier(params)
    model.load_state_dict(torch.load("model.bin", map_location=torch.device('cpu')))
    model.eval()

    return model, tokenizer

def run(params):
    s = time.time()
    print("Importing model and tokenizer...")
    model, tokenizer = load_model(params)
    print(f"Model and Tokenizer loaded: took {time.time()-s} seconds.")

    s = time.time()
    print("Starting inference...")
    inputs = tokenizer.encode_plus(
        args.text,
        add_special_tokens=True,
        max_length = 128,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt'
    )

    res = model(
        inputs['input_ids'],
        mask=inputs["attention_mask"],
        token_type_ids=inputs["token_type_ids"]
    )
    res = torch.nn.Sigmoid()(res).item()
    print(f"Inference received as {res}: took {time.time()-s} seconds")

def main():
    params = {
        'dropout': args.dropout,
        'lr': args.lr,
        'bert_path': args.bert_path,
        'input_size': 768,
        'ntargets': 1,
        'hidden_size': args.hidden_size
    }

    run(params)

if __name__ == "__main__":
    main()
