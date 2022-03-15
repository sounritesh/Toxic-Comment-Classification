from torch.utils.data import Dataset
import torch

class ToxicityDatasetBERT(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        text = self.texts[index]

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