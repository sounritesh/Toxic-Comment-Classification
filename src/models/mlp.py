from src.utils.config import *
import transformers
import torch.nn as nn
import torch


class BertClassifier(nn.Module):
    def __init__(self, params):
        super(BertClassifier, self).__init__()
        self.bert = transformers.AutoModel.from_pretrained(params['bert_path'])
        self.bert_drop = nn.Dropout(params['dropout'])

        self.fc1 = nn.Linear(params['input_size'], params['hidden_size'])
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(params['hidden_size'], params['ntargets'])

    def forward(self, ids, mask, token_type_ids):

        o = self.bert(ids.squeeze(), attention_mask=mask.squeeze(), token_type_ids=token_type_ids.squeeze())  
        
        try:
            output = self.relu(self.fc1(o['pooler_output']))
            output = self.bert_drop(output)
            output = self.mlp(output)
        except:
            output = self.relu(self.fc1(o[1]))
            output = self.bert_drop(output)
            output = self.mlp(output)

        return output