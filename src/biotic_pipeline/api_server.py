"""
FastAPI server for biotic interaction classification.

Configuration is read from config.toml in the working directory.
"""

import re
from pathlib import Path
from typing import List

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from biotic_pipeline.config import get_config

# ── Bootstrap ─────────────────────────────────────────────────────────────

_cfg     = get_config()
_scfg    = _cfg.server
_mcfg    = _cfg.model

if _mcfg.model_dir == "":
    raise RuntimeError("[model] model_dir is not set in config.toml")

if _mcfg.device == "auto":
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    _device = torch.device(_mcfg.device)

print(f"Loading model from {_mcfg.model_dir} on {_device} ...", flush=True)
_tokenizer = AutoTokenizer.from_pretrained(_mcfg.model_dir, local_files_only=True)
_model     = AutoModelForSequenceClassification.from_pretrained(
    _mcfg.model_dir, local_files_only=True).to(_device).eval()
print(f"Model ready. Threshold={_mcfg.threshold}", flush=True)

# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Biotic Interaction Classifier",
    description=(
        "Sentence-level biotic interaction detector. "
        "Returns label=1 when a sentence describes an ecological/parasitic/symbiotic "
        "interaction between two species."
    ),
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Schemas ───────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str
    threshold: float = _mcfg.threshold

class PredictResponse(BaseModel):
    text: str
    label: int
    probability: float
    threshold_used: float

class BatchRequest(BaseModel):
    sentences: List[str]
    threshold: float = _mcfg.threshold

class BatchResponse(BaseModel):
    results: List[PredictResponse]
    n_positive: int
    n_total: int

# ── Inference ─────────────────────────────────────────────────────────────

def _predict(text: str, threshold: float) -> dict:
    processed = re.sub(r"\s+", " ", text.lower()).strip()
    enc = _tokenizer(
        processed, return_tensors="pt",
        truncation=True, max_length=_mcfg.max_length,
    ).to(_device)
    with torch.no_grad():
        prob = torch.softmax(_model(**enc).logits, dim=-1)[0, 1].item()
    return {
        "text": text,
        "label": int(prob >= threshold),
        "probability": round(prob, 4),
        "threshold_used": threshold,
    }

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_dir": _mcfg.model_dir,
        "device": str(_device),
        "default_threshold": _mcfg.threshold,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(422, "text cannot be empty")
    return _predict(req.text, req.threshold)


@app.post("/batch", response_model=BatchResponse)
def batch_predict(req: BatchRequest):
    if not req.sentences:
        raise HTTPException(422, "sentences list cannot be empty")
    if len(req.sentences) > 500:
        raise HTTPException(422, "max 500 sentences per batch")
    results = [_predict(s, req.threshold) for s in req.sentences]
    return {
        "results": results,
        "n_positive": sum(r["label"] for r in results),
        "n_total": len(results),
    }


# ── Entry point ───────────────────────────────────────────────────────────

def main():
    uvicorn.run(
        "biotic_pipeline.api_server:app",
        host=_scfg.host,
        port=_scfg.port,
        workers=_scfg.workers,
        log_level=_scfg.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
