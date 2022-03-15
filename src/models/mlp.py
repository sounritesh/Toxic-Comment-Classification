from utils.config import *
import transformers
import torch.nn as nn
import torch


class BertClassifier(nn.Module):
    def __init__(self, args):
        super(BertClassifier, self).__init__()
        self.bert = transformers.AutoModel.from_pretrained(args['bert_path'])
        self.bert_drop = nn.Dropout(args['dropout'])

        self.mlp = nn.Linear(args['input_size'], args['ntargets'])

    def forward(self, ids, mask, token_type_ids):

        o = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)  
        output = self.mlp(o['pooler_output'])

        return output