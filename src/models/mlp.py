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

        o = self.bert(ids.squeeze(), attention_mask=mask.squeeze(), token_type_ids=token_type_ids.squeeze())  
        
        try:
            output = self.mlp(o['pooler_output'])
        except:
            print(o[0].shape, 0[1].shape)
            output = self.mlp(o[1])

        return output