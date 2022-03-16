from pytest import param
import src.utils.config as config
import src.data.dataset as dataset
import src.utils.engine as engine
from src.utils.metrics import eval_perf
import torch
import pandas as pd
import torch.nn as nn
import numpy as np

import transformers

from src.models.mlp import BertClassifier
from sklearn import model_selection
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup

import optuna
import random
from tqdm.notebook import tqdm

import os

from argparse import ArgumentParser

parser = ArgumentParser(description="Train model on the dataset and evaluate results.")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--train_path", type=str)
parser.add_argument("--val_path", type=str)
parser.add_argument("--test_path", type=str)
parser.add_argument("--bert_path", type=str)

parser.add_argument("--output_dir", type=str)

parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--hidden_size", type=int, default=32)

parser.add_argument("--epochs", type=int, default=15)

parser.add_argument("--train_batch_size", type=int, default=128)
parser.add_argument("--val_batch_size", type=int, default=256)

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def run(params, save_model=True):
    df_train = pd.read_csv(args.train_path).sample(frac=1).reset_index(drop=True)
    df_val = pd.read_csv(args.val_path).sample(frac=1).reset_index(drop=True)
    df_test = pd.read_csv(args.test_path).sample(frac=1).reset_index(drop=True)

    tokenizer = transformers.AutoTokenizer.from_pretrained(params['bert_path'], do_lower_case=True)

    train_dataset = dataset.ToxicityDatasetBERT(
        df_train.comment_text.values, df_train.toxic.values, tokenizer, args.max_len
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size
    )

    valid_dataset = dataset.ToxicityDatasetBERT(
        df_val.comment_text.values, df_val.toxic.values, tokenizer, args.max_len
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.val_batch_size
    )

    device = torch.device(config.DEVICE)
    model = BertClassifier(params)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(df_train) / args.train_batch_size * args.epochs)
    optimizer = Adam(optimizer_parameters, lr=params['lr'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    early_stopping_iter = 3
    early_stopping_counter = 0

    best_roc_auc = 0
    for epoch in range(args.epochs):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        accuracy, precision, recall, fscore, roc_auc = eval_perf(targets, outputs, 0.5)
        print(f"Accuracy Score = {accuracy}")
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            if save_model:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'{epoch}_model.bin'))
        else:
            early_stopping_counter += 1

        if early_stopping_iter < early_stopping_counter:
            break

        print(f"EPOCH[{epoch}]: accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1-score: {fscore}, roc_auc: {roc_auc}")

    return best_roc_auc

def objective(trial):
    params = {
        'lstm_layers': trial.suggest_int('lstm_layers', 1, 3),
        # 'lstm_layers': 1,
        'mlp_layers': trial.suggest_int('mlp_layers', 1, 3),
        'lstm_hidden_size': trial.suggest_int('lstm_hidden_size', 18, 768),
        # 'lstm_hidden_size': 20,
        'mlp_hidden_size': trial.suggest_int('mlp_hidden_size', 18, 768),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.7),
        'lr': trial.suggest_loguniform('lr', 1e-5, 1e-2),
        'threshold': 0.5
    }
    return run(params, False)

def main():
    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=20)

    # trial_ = study.best_trial

    # print(f"\n Best Trial: {trial_.values}, Params: {trial_.params}")

    # score = run(trial_.params, True)

    # print(score)
    params = {
        'dropout': 0.3,
        'lr': 1e-3,
        'bert_path': args.bert_path,
        'input_size': 768,
        'ntargets': 1,
        'hidden_size': args.hidden_size
    }

    run(params)

if __name__ == "__main__":
    main()
