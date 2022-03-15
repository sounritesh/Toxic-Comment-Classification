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
        text = self.targets[index]

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length = self.max_len,
            pad_to_max_length = True,
            truncation_strategy = 'longest_first',
            return_tensors = 'pt'
        )

        inputs['targets'] = torch.tensor(self.targets[index], dtype=torch.float)

        return inputs