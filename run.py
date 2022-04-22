import src.config.config as config
import src.data.dataset as dataset
import src.utils.engine as engine
from src.utils.metrics import eval_perf
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import wandb

import transformers

from src.models.mlp import BertClassifier
from torch.optim import Adam, lr_scheduler, SGD

import optuna
import random
from tqdm.notebook import tqdm

import os

from argparse import ArgumentParser

parser = ArgumentParser(description="Train model on the dataset and evaluate results.")
parser.add_argument("--seed", type=int, default=0, help="Random seed for all sampling purposes")
parser.add_argument("--data_path", type=str, help="Path to training file")
# parser.add_argument("--val_path", type=str, help="Path to validation file")
# parser.add_argument("--test_path", type=str, help="Path to testing file")
parser.add_argument("--bert_path", default="bert-base-multilingual-uncased", type=str, help="Path to base bert model")

parser.add_argument("--lr", type=float, default=1e-4, help="Specifies the learning rate for optimizer")
parser.add_argument("--dropout", type=float, default=0.3, help="Specifies the dropout for BERT output")

parser.add_argument("--preprocess", action="store_true", help="To apply preprocessing step")
parser.add_argument("--tune", action="store_true", help="To tune model by trying different hyperparams")
# parser.add_argument("--bert", action="store_true", help="To signify whether the model is bert based")

parser.add_argument("--output_dir", type=str, help="Path to output directory for saving model checkpoints")

parser.add_argument("--max_len", type=int, default=128, help="Specifies the maximum length of input sequence")
parser.add_argument("--hidden_size", type=int, default=32, help="Specifies the hidden size of fully connected layer")

parser.add_argument("--epochs", type=int, default=15, help="Specifies the number of training epochs")

parser.add_argument("--train_batch_size", type=int, default=64, help="Specifies the training batch size")
parser.add_argument("--val_batch_size", type=int, default=256, help="Specifies the validation and testing batch size")

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

def run(params, save_model=True):
    wandb.init(
        project="pacemaker",
        entity="now-and-me",
        config=params
    )

    df = pd.read_csv(args.data_path).sample(frac=1).reset_index(drop=True)
    df.blocked = df.blocked.astype(float)
    df.body = df.body.astype(str)

    blocked_df = df[df['blocked']==1]
    df = pd.concat([ blocked_df, df[(df['blocked']==0) & (df['banned']==0)].sample(n=len(blocked_df)) ])

    df_train = df.sample(frac=0.8)
    df_rest = df.drop(df_train.index)
    df_val = df_rest.sample(frac=0.4)
    df_test = df_rest.drop(df_val.index)

    print(f"Stratification Split, \ntrain: {df_train['blocked'].value_counts()} \nval: {df_val['blocked'].value_counts()} \ntest: {df_test['blocked'].value_counts()}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(params['bert_path'], do_lower_case=True)

    train_dataset = dataset.ToxicityDatasetBERT(
        df_train.body.values,
        df_train.blocked.values,
        tokenizer,
        args.max_len,
        args.preprocess
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size
    )

    valid_dataset = dataset.ToxicityDatasetBERT(
        df_val.body.values,
        df_val.blocked.values,
        tokenizer,
        args.max_len,
        args.preprocess
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.val_batch_size
    )

    test_dataset = dataset.ToxicityDatasetBERT(
        df_test.body.values,
        df_test.blocked.values,
        tokenizer,
        args.max_len,
        args.preprocess
    )
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.val_batch_size
    )

    device = torch.device(config.DEVICE)
    model = BertClassifier(params)
    model.to(device)
    wandb.watch(model, log="all", log_freq=10, idx=None, log_graph=(True))

    if params['bert_path'] == "unitary/toxic-bert":
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

    num_train_steps = int(len(df_train) / args.train_batch_size * args.epochs)
    optimizer = SGD(optimizer_parameters, lr=params['lr'])

    scheduler = lr_scheduler.StepLR(
        optimizer,
        step_size=5,
        gamma=0.5
    )

    early_stopping_iter = 3
    early_stopping_counter = 0

    best_roc_auc = 0
    for epoch in range(args.epochs):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, bert_flag)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device, bert_flag)
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

        scheduler.step()
        print(f"EPOCH[{epoch+1}]: train loss: {train_loss}, accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1-score: {fscore}, roc_auc: {roc_auc}")
        wandb.log({
            "train loss": train_loss,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1-score": fscore,
            "roc-auc": roc_auc
        })

    outputs, targets = engine.eval_fn(test_data_loader, model, device, bert_flag)
    accuracy, precision, recall, fscore, roc_auc = eval_perf(targets, outputs)

    wandb.summary['test_f1'] = fscore
    wandb.finish()

    print(f"TEST RESULTS accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1-score: {fscore}, roc_auc: {roc_auc}")

    return best_roc_auc

def objective(trial):
    params = {
        'hidden_size': trial.suggest_int('hidden_size', 100, 768),
        'dropout': trial.suggest_uniform('dropout', 0.1, 0.7),
        'lr': trial.suggest_loguniform('lr', 1e-4, 1e-2),
        'bert_path': trial.suggest_categorical("bert_path", ["unitary/unbiased-toxic-roberta", "unitary/multilingual-toxic-xlm-roberta"]),
        'input_size': 768,
        'ntargets': 1,
    }
    return run(params, False)

def main():
    if args.tune:
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=20)

        trial_ = study.best_trial
        print(f"\n Best Trial: {trial_.values}, Params: {trial_.params}")

        score = run(trial_.params, True)
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

        run(params)

if __name__ == "__main__":
    main()
