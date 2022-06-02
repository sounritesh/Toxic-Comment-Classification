import src.config.config as config
import src.data.dataset as dataset
import src.utils.engine as engine
from src.utils.metrics import eval_perf
from src.data.prepare_data import prepare_dataset

import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import wandb
import json
import spacy

import transformers

from src.models.mlp import BertClassifier
from torch.optim import Adam, lr_scheduler, SGD

import optuna
import random
from tqdm.notebook import tqdm
import os
from argparse import ArgumentParser

parser = ArgumentParser(description="Fetch, load and process data from Mongo client. Then train the model with optuna eager searched hyperparameters.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for all sampling purposes")
# parser.add_argument("--data_path", type=str, help="Path to training file")

parser.add_argument("--bert_path", default="unitary/toxic-bert", type=str, help="Path to base bert model")
parser.add_argument("--checkpoint", default="", type=str, help="Checkpoint model file to continue training from")

parser.add_argument("--lr", type=float, default=1e-4, help="Specifies the learning rate for optimizer")
parser.add_argument("--dropout", type=float, default=0.3, help="Specifies the dropout for BERT output")

parser.add_argument("--preprocess", action="store_true", help="To apply preprocessing step")
parser.add_argument("--tune", action="store_true", help="To tune model by trying different hyperparams")

parser.add_argument("--output_dir", type=str, help="Path to output directory for saving model checkpoints")
parser.add_argument("--names_path", type=str, help="Path to name csv")

parser.add_argument("--max_len", type=int, default=128, help="Specifies the maximum length of input sequence")
parser.add_argument("--hidden_size", type=int, default=32, help="Specifies the hidden size of fully connected layer")

parser.add_argument("--epochs", type=int, default=15, help="Specifies the number of training epochs")

parser.add_argument("--train_batch_size", type=int, default=64, help="Specifies the training batch size")
parser.add_argument("--val_batch_size", type=int, default=256, help="Specifies the validation and testing batch size")

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

train_data_loader, valid_data_loader, test_data_loader = None, None, None

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

    train_dataset = dataset.ToxicityDatasetBERT(
        df_train.body.values,
        df_train.blocked.values,
        tokenizer,
        args.max_len,
        args.preprocess,
        nlp,
        names
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size
    )

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

    test_dataset = dataset.ToxicityDatasetBERT(
        df_test.body.values,
        df_test.blocked.values,
        tokenizer,
        args.max_len,
        args.preprocess,
        nlp,
        names
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.val_batch_size
    )

    return train_data_loader, valid_data_loader, test_data_loader

def run(params, train_data_loader, valid_data_loader, test_data_loader, save_model=True):
    wandb.init(
        project="pacemaker-xl-val",
        entity="now-and-me",
        config=params
    )

    device = torch.device(config.DEVICE)
    model = BertClassifier(params)
    if args.checkpoint != "":
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device)
    wandb.watch(model, log="all", log_freq=10, idx=None, log_graph=(True))

    if "bert" in params['bert_path'].lower():
        bert_flag = True
    else:
        bert_flag = False

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

    optimizer = SGD(optimizer_parameters, lr=params['lr'])

    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=0,
    )

    early_stopping_iter = 5
    early_stopping_counter = 0

    best_roc_auc = 0
    for epoch in range(args.epochs):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, bert_flag)
        outputs, targets, val_loss = engine.eval_fn(valid_data_loader, model, device, bert_flag)
        accuracy, precision, recall, fscore, roc_auc = eval_perf(targets, outputs)
        print(f"Accuracy Score = {accuracy}")
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            if save_model:
                torch.save(model.state_dict(), os.path.join(args.output_dir, f'{epoch}_model.bin'))
        else:
            early_stopping_counter += 1

        if early_stopping_iter < early_stopping_counter:
            break

        scheduler.step(roc_auc)
        print(f"EPOCH[{epoch+1}]: train loss: {train_loss}, val loss: {val_loss}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1-score: {fscore}, roc_auc: {roc_auc}")
        wandb.log({
            "train loss": train_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": fscore,
            "roc-auc": roc_auc,
            "val_loss": val_loss
        })

    outputs, targets, _ = engine.eval_fn(test_data_loader, model, device, bert_flag)
    accuracy, precision, recall, fscore, roc_auc = eval_perf(targets, outputs)

    wandb.summary['test_f1'] = fscore
    wandb.finish()

    print(f"TEST RESULTS accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1-score: {fscore}, roc_auc: {roc_auc}")

    return best_roc_auc

def objective(trial):
    params = {
        'hidden_size': trial.suggest_int('hidden_size', 100, 600),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.6),
        'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
        'bert_path': args.bert_path,
        'input_size': 768,
        'ntargets': 1,
    }
    return run(params, train_data_loader, valid_data_loader, test_data_loader, False)

def main():
    global train_data_loader
    global valid_data_loader
    global test_data_loader

    if args.tune:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)

        trial_ = study.best_trial
        print(f"\n Best Trial: {trial_.values}, Params: {trial_.params}")
        train_data_loader, valid_data_loader, test_data_loader = preprocess_dataset(trial_.params)

        with open("src/config/params.json") as f:
            json.dump(trial_.params, f, indent=4)

        score = run(trial_.params, train_data_loader, valid_data_loader, test_data_loader, True)
        print(score)
    else:
        params = {
            'dropout': args.dropout,
            'lr': args.lr,
            'bert_path': args.bert_path,
            'input_size': 768,
            'ntargets': 1,
            'hidden_size': args.hidden_size
        }
        train_data_loader, valid_data_loader, test_data_loader = preprocess_dataset(params)
        run(params, train_data_loader, valid_data_loader, test_data_loader)

if __name__ == "__main__":
    main()
