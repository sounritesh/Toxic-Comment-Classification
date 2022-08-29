import torch
from pydantic import BaseModel
import uvicorn
from http import HTTPStatus
import requests
from datetime import datetime
from functools import wraps
from fastapi import FastAPI, Request

from src.utils.utils import load_artifacts
from src.config.config  import logger
from src.config.config import CONFIG_DIR, BASE_DIR
import json
import os
import re
import pandas as pd


app = FastAPI(
    title="Pacemaker - Now&Me",
    description="Predict whether a thought is potentially inappropriate or not.",
    version="0.1",
)

# Helper functions
def mask_name(text):
    masked_text = ""
    for t in text.split():
        if t.lower() in names:
            masked_text += " <PERSON>"
        else:
            masked_text += " {}".format(t)
    return masked_text

def replace_offensive_words(text):
    masked_text = ""
    for t in text.split():
        if t.lower() in hinglish_off_words:
            masked_text += " {}".format(str(offensive_words[offensive_words['hinglish'] == t.lower()].english.values[0]))
        else:
            masked_text += " {}".format(t)
    return masked_text

def clean_text(text):
    text = str(text)
    text = re.sub(r'[0-9"]', '', text) # number
    text = re.sub(r'#[\S]+\b', '', text) # hash
    text = re.sub(r'@[\S]+\b', '', text) # mention
    text = re.sub(r'https?\S+', '', text) # link
    text = re.sub('.', ' ', text) # full stop
    text = re.sub(r'\s+', ' ', text) # multiple white spaces

    return text

class Thought(BaseModel):
    text: str

def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


@app.get("/")
@construct_response
def _index(request: Request):
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {},
    }
    return response


@app.on_event("startup")
def load_data_files():
    global names
    global offensive_words
    global hinglish_off_words

    logger.info("Loading data files...")
    names = list(pd.read_csv(os.path.join(BASE_DIR, "names.csv")).name.values)
    offensive_words = pd.read_csv(os.path.join(BASE_DIR, "hinglish_offensive_words.csv"))
    hinglish_off_words = list(offensive_words.hinglish.values)
    logger.info("Data files loaded.")


@app.on_event("startup")
def load_model_and_tokenizer():
    global artifacts
    with open(os.path.join(CONFIG_DIR, "params.json")) as f:
        params = json.load(f)
    logger.info(params)
    artifacts = load_artifacts(params)
    logger.info("Model ready for inference!")


@app.put("/inference/")
@construct_response
def _infer(request: Request, thought: Thought) -> dict:
    """Get inference from the pacemaker model."""

    text = clean_text(thought.text)
    text = mask_name(text)
    text = clean_text(text).strip()
    text = replace_offensive_words(text)

    inputs = artifacts[1].encode_plus(
        text,
        add_special_tokens=True,
        max_length = 128,
        padding = 'max_length',
        truncation = True,
        return_tensors = 'pt'
    )

    res = artifacts[0](
        inputs['input_ids'],
        mask=inputs["attention_mask"],
        token_type_ids=inputs["token_type_ids"]
    )
    res = torch.nn.Sigmoid()(res).item()
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "text": text,
            "score": res
        },
    }
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)