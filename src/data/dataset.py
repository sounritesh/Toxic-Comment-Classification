from torch.utils.data import Dataset
import torch
import re
import spacy

class ToxicityDatasetBERT(Dataset):
    def __init__(self, texts, targets, tokenizer, max_len, preprocess, nlp, name_list = []):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.preprocess = preprocess
        self.nlp = nlp
        self.names = name_list


        self.inputs = []
        for text in self.texts:
            if self.preprocess:
                text = self.preprocess_text(text).strip()

            inputs = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length = self.max_len,
                padding = 'max_length',
                truncation = True,
                return_tensors = 'pt'
            )

            self.inputs.append(inputs)

    def __len__(self):
        return len(self.targets)
    
    def mask_name(self, text):
        masked_text = ""
        for t in text.split():
            if t.lower() in self.names:
                masked_text += " <PERSON>"
            else:
                masked_text += " {}".format(t)
        return masked_text

    @staticmethod
    def clean_text(text):
        text = str(text)
        text = re.sub(r'[0-9"]', '', text) # number
        text = re.sub(r'#[\S]+\b', '', text) # hash
        text = re.sub(r'@[\S]+\b', '', text) # mention
        text = re.sub(r'https?\S+', '', text) # link
        text = re.sub(r'\s+', ' ', text) # multiple white spaces

        return text

    def mask_text(self, text):
        doc = self.nlp(text)
        filtered_string = ""
        for token in doc:
            if token.ent_type_ in ['GPE', 'DATE', 'TIME', 'FAC', 'LOC', 'MONEY', 'NORP', 'ORG', 'PERCENT', 'PRODUCT', 'QUANTITY']:
                new_token = " <{}>".format(token.ent_type_)
            elif token.pos_ == "PUNCT":
                new_token = " "
            else:
                new_token = " {}".format(token.text)
            filtered_string += new_token
        filtered_string = filtered_string[1:]
        return filtered_string

    def preprocess_text(self, text):
        text = self.mask_text(text)
        text = self.clean_text(text)
        text = self.mask_name(text)
        text = self.clean_text(text)

        return text

    def __getitem__(self, index):
        inputs = self.inputs[index]

        inputs['targets'] = torch.tensor(self.targets[index], dtype=torch.float)

        return inputs