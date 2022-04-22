from sre_parse import Tokenizer
import streamlit as st
import pandas as pd
import numpy as np
from src.models.mlp import BertClassifier
import time
from transformers import AutoTokenizer
import torch

st.title('Profanity Check')

@st.cache
def load_model():
    params = {
        'dropout': 0.2335,
        'lr': 0.008,
        'bert_path': "unitary/toxic-bert",
        'input_size': 768,
        'ntargets': 1,
        'hidden_size': 450
    }
    model = BertClassifier(params)
    model.load_state_dict(torch.load("D:/NowAndMe/moderation/model.bin", map_location=torch.device('cpu')))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(params["bert_path"])
    return model, tokenizer


# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading model...')
s = time.time()
# Load 10,000 rows of data into the dataframe.
model, tokenizer = load_model()
# Notify the reader that the data was successfully loaded.
data_load_state.text(f'Done! (Took {time.time()-s} seconds)')

text = st.text_area("Thought", value="", placeholder="Enter thought")

def show_res(text):
    s = time.time()
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length = 128,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt'
    )

    res = model(
        inputs['input_ids'],
        mask=inputs["attention_mask"],
        token_type_ids=inputs["token_type_ids"]
    )
    res = torch.nn.Sigmoid()(res).item()
    st.text("Block")
    st.progress(float(res))
    st.text("Approve")
    st.progress(float(1-res))

    st.text(f'Done! (Took {time.time()-s} seconds)')

st.button("Process", on_click=show_res(text))
