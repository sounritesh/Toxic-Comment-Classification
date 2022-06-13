import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import random
import spacy
import wandb
import optuna
import json

import transformers

from src.models.mlp import BertClassifier
from src.data.prepare_data import prepare_dataset
from src.data import dataset
from src.utils import engine
from src.config import config
from src.utils.metrics import eval_perf

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--bert_path", type=str, default="l3cube-pune/hing-bert")
parser.add_argument("--hidden_size", type=int, default=450)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--names_path", type=str)
parser.add_argument("--max_len", type=int, default=128)
parser.add_argument("--preprocess", action="store_true")
parser.add_argument("--tune", action="store_true")
parser.add_argument("--checkpoint", type=str, default="")

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

train_data_loader, valid_data_loader, test_data_loader = None, None, None

def load_model(params):
    tokenizer = transformers.AutoTokenizer.from_pretrained(params['bert_path'], do_lower_case=True)

    model = BertClassifier(params)
    model.load_state_dict(torch.load("model.bin", map_location=torch.device('cpu')))
    model.eval()

    return model, tokenizer

def preprocess_dataset(params):
    # df = pd.read_csv(args.data_path).sample(frac=1).reset_index(drop=True)
    df = prepare_dataset()
    df.blocked = df.blocked.astype(float)
    df.banned = df.banned.astype(float)
    df.body = df.body.astype(str)

    # blocked_df = df[df['blocked']==1]
    # df = pd.concat([ blocked_df, df[(df['blocked']==0) & (df['banned']==0)].sample(n=len(blocked_df)) ])

    df_train = df.sample(frac=0.8)
    df_rest = df.drop(df_train.index)
    df_val = df_rest.sample(frac=0.4)
    df_test = df_rest.drop(df_val.index)

    blocked_df = df_train[df_train['blocked']==1]
    df_train = pd.concat([ blocked_df, df_train[(df_train['blocked']==0) & (df_train['banned']==0)].sample(n=len(blocked_df)) ])

    print(f"Training, Development and Validation Split, \ntrain: {df_train['blocked'].value_counts()} \nval: {df_val['blocked'].value_counts()} \ntest: {df_test['blocked'].value_counts()}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(params['bert_path'], do_lower_case=True)

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf")

    names = pd.read_csv(args.names_path).name.values.tolist()

    valid_dataset = dataset.ToxicityDatasetBERT(
        df_val.body.values,
        df_val.blocked.values,
        tokenizer,
        args.max_len,
        args.preprocess,
        nlp,
        names
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.val_batch_size
    )

    # test_dataset = dataset.ToxicityDatasetBERT(
    #     df_test.body.values,
    #     df_test.blocked.values,
    #     tokenizer,
    #     args.max_len,
    #     args.preprocess,
    #     nlp,
    #     names
    # )
    # test_data_loader = torch.utils.data.DataLoader(
    #     test_dataset, batch_size=args.val_batch_size
    # )

    return None, valid_data_loader, None

def run(params, train_data_loader, valid_data_loader, test_data_loader, save_model=True):
    wandb.init(
        project="pacemaker-threshold-eval",
        entity="now-and-me",
        config=params
    )

    device = torch.device(config.DEVICE)

    model = BertClassifier(params)
    if args.checkpoint != "":
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    wandb.watch(model, log="all", log_freq=10, idx=None, log_graph=(True))

    s = time.time()
    print("Starting inference...")

    outputs, targets, _ = engine.eval_fn(valid_data_loader, model, device, True)
    accuracy, precision, recall, fscore, roc_auc, pr_auc = eval_perf(targets, outputs, params['threshold'])

    wandb.log({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": fscore,
        "roc-auc": roc_auc,
        "pr-auc": pr_auc,
    })

    wandb.summary['test_f1'] = fscore
    wandb.finish()

    print(f"VALID RESULTS accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1-score: {fscore}, roc_auc: {roc_auc}, pr_auc: {pr_auc}")

    print(f"Inference received: took {time.time()-s} seconds")

    return fscore

def objective(trial):
    params = {
        'hidden_size': args.hidden_size,
        'dropout': args.dropout,
        'lr': 0.0004,
        'bert_path': args.bert_path,
        'input_size': 768,
        'ntargets': 1,
        'threshold': trial.suggest_uniform('threshold', 0.1, 0.95)
    }
    return run(params, train_data_loader, valid_data_loader, test_data_loader, False)

def main():
    global train_data_loader
    global valid_data_loader
    global test_data_loader

    params = {
        'dropout': args.dropout,
        'lr': 0.0004,
        'bert_path': args.bert_path,
        'input_size': 768,
        'ntargets': 1,
        'hidden_size': args.hidden_size
    }

    train_data_loader, valid_data_loader, test_data_loader = preprocess_dataset(params)

    if args.tune:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)

        trial_ = study.best_trial
        print(f"\n Best Trial: {trial_.values}, Params: {trial_.params}")

        with open("src/config/params.json") as f:
            json.dump(trial_.params, f, indent=4)

        score = run(trial_.params, train_data_loader, valid_data_loader, test_data_loader, True)
        print(score)
    else:
        run(params, train_data_loader, valid_data_loader, test_data_loader)

if __name__ == "__main__":
    main()
