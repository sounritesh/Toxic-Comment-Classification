from src.utils.config import *
import transformers
import torch.nn as nn
import torch


class BertClassifier(nn.Module):
    def __init__(self, params):
        super(BertClassifier, self).__init__()
        self.bert = transformers.AutoModel.from_pretrained(params['bert_path'])
        self.bert_drop = nn.Dropout(params['dropout'])

        self.mlp = nn.Linear(params['input_size'], params['ntargets'])

    def forward(self, ids, mask, token_type_ids):

        o = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)  
        output = self.mlp(o['pooler_output'])

        return output