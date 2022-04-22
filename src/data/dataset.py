from torch.utils.data import Dataset
import torch
import re

class ToxicityDatasetBERT(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len, preprocess):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preprocess = preprocess

    def __len__(self):
        return len(self.targets)

    @staticmethod
    def clean_text(text):
        text = str(text)
        text = re.sub(r'[0-9"]', '', text) # number
        text = re.sub(r'#[\S]+\b', '', text) # hash
        text = re.sub(r'@[\S]+\b', '', text) # mention
        text = re.sub(r'https?\S+', '', text) # link
        text = re.sub(r'\s+', ' ', text) # multiple white spaces

        return text

    def __getitem__(self, index):
        text = self.texts[index]

        if self.preprocess:
            text = self.clean_text(text)

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length = self.max_len,
            padding = 'max_length',
            truncation = True,
            return_tensors = 'pt'
        )

        inputs['targets'] = torch.tensor(self.targets[index], dtype=torch.float)

        return inputs