
from fastapi import FastAPI, Request
from http import HTTPStatus
import requests
from datetime import datetime
from functools import wraps

from src.utils.utils import load_artifacts
from src.config.config  import logger

app = FastAPI(
    title="TagIfAI - Made With ML",
    description="Predict relevant tags given a text input.",
    version="0.1",
)

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

# @app.on_event("startup")
# def load_artifacts():
#     global artifacts
#     artifacts = load_artifacts()
#     logger.info("Model ready for inference!")