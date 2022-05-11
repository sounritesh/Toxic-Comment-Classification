
from fastapi import FastAPI, Request
from pydantic import BaseModel
import torch
import uvicorn
from http import HTTPStatus
import requests
from datetime import datetime
from functools import wraps

from src.utils.utils import load_artifacts
from src.config.config  import logger
from src.config.config import CONFIG_DIR
import json
import os


app = FastAPI(
    title="Pacemaker - Now&Me",
    description="Predict whether a thought is potentially inappropriate or not.",
    version="0.1",
)


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

    inputs = artifacts[1].encode_plus(
        thought.text,
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
            "text": thought.text,
            "score": res
        },
    }
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)