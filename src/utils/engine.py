import torch
import torch.nn as nn
from tqdm import tqdm
from src.config.config import DEVICE


def loss_fn(outputs, targets):
    '''
    Function to calculate the Binary Cross Entropy Loss between logits and expected targets

    Parameters:
    outputs (torch.Tensor): logits returned by the model.
    targets (torch.Tensor): expected labels.

    Returns:
    loss (torch.Tensor): tensor of shape [1] reporting the loss.
    '''
    return nn.BCEWithLogitsLoss().to(DEVICE)(outputs, targets.view(-1, 1))


def train_fn(data_loader, model, optimizer, device, bert_flag):
    '''
    Function to carry out training for all batches in an epoch

    Parameters:
    data_loader (torch.utils.data.Dataloader): data loader containing all training samples and labels.
    model (torch.nn.Module): classification model to be trained.
    optimizer (torch.optim.Optimizer): optimizer for fitting the model.
    device: device to load the model and tensors onto.

    Returns:
    epoch_loss (torch.Tensor): tensor of shape [1] reporting loss for the epoch.
    '''

    model.train()
    loss_tot = 0
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True):
        ids = d["input_ids"]
        mask = d["attention_mask"]
        targets = d["targets"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()

        if bert_flag:
            token_type_ids = d["token_type_ids"]
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        else:
            outputs = model(ids=ids, mask=mask, token_type_ids=None)

        loss = loss_fn(outputs, targets)
        loss.backward()
        loss_tot += loss.item()
        # print(loss.item())
        optimizer.step()

    epoch_loss =  loss_tot/len(data_loader)
    return epoch_loss


def eval_fn(data_loader, model, device, bert_flag):
    '''
    Function to carry out evaluation for all batches

    Parameters:
    data_loader (torch.utils.data.Dataloader): data loader containing all training samples and labels.
    model (torch.nn.Module): classification model to be trained.
    device: device to load the model and tensors onto.

    Returns:
    fin_outputs (torch.Tensor): output logits from the model.
    fin_targets (torch.Tensor): expected labels.
    '''

    model.eval()
    fin_targets = []
    fin_outputs = []
    loss_tot = 0
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True):
            ids = d["input_ids"]
            mask = d["attention_mask"]
            targets = d["targets"]

            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)
            ids = ids.to(device, dtype=torch.long)
            if bert_flag:
                token_type_ids = d["token_type_ids"]
                token_type_ids = token_type_ids.to(device, dtype=torch.long)
                outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
            else:
                outputs = model(ids=ids, mask=mask, token_type_ids=None)

            loss = loss_fn(outputs, targets)
            loss_tot += loss.item()

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    epoch_loss =  loss_tot/len(data_loader)
    return fin_outputs, fin_targets, epoch_loss