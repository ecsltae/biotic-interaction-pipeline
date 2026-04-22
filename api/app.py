#!/usr/bin/env python3
"""
Biotic Interaction Classifier API
==================================
Detects whether a sentence describes a biotic interaction between two species.

Model: BiomedBERT-base fine-tuned via knowledge distillation
       (teacher: BiomedBERT cv_reg × FLAN-T5-base v12 ensemble)
Performance: EP-relax F1=0.808 @ threshold=0.25

Configuration via environment variables:
  MODEL_DIR          Path to the HuggingFace model directory  [required]
  PORT               Listening port                           [default: 8003]
  DEFAULT_THRESHOLD  Classification threshold                 [default: 0.25]
  DEVICE             cuda:0 | cpu | auto                      [default: auto]

Endpoints:
  GET  /health   model info + status
  POST /predict  single sentence → label + probability
  POST /batch    list[sentence] → list[result]   (max 500 per call)
"""

import os
import re
from pathlib import Path
from typing import List

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import uvicorn

# ── Config (all overridable via env) ─────────────────────────────────────

_model_dir = os.environ.get("MODEL_DIR", "")
if not _model_dir:
    raise RuntimeError("MODEL_DIR environment variable is required")
MODEL_DIR = Path(_model_dir)

PORT      = int(os.environ.get("PORT", "8003"))
THRESHOLD = float(os.environ.get("DEFAULT_THRESHOLD", "0.25"))
MAX_LEN   = int(os.environ.get("MAX_LENGTH", "256"))

_device_env = os.environ.get("DEVICE", "auto")
if _device_env == "auto":
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    DEVICE = torch.device(_device_env)

# ── Load model ────────────────────────────────────────────────────────────

print(f"Loading model from {MODEL_DIR} on {DEVICE} ...", flush=True)
_tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)
_model     = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIR), local_files_only=True).to(DEVICE).eval()
print(f"Model ready. Threshold={THRESHOLD}", flush=True)

# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Biotic Interaction Classifier",
    description=(
        "Sentence-level biotic interaction detector. "
        "Returns label=1 if the sentence describes an ecological/parasitic/symbiotic "
        "interaction between two species."
    ),
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Schemas ───────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    text: str = Field(examples=["Wolbachia pipientis infects Drosophila melanogaster."])
    threshold: float = Field(default=THRESHOLD)

class PredictResponse(BaseModel):
    text: str
    label: int
    probability: float
    threshold_used: float

class BatchRequest(BaseModel):
    sentences: List[str]
    threshold: float = Field(default=THRESHOLD)

class BatchResponse(BaseModel):
    results: List[PredictResponse]
    n_positive: int
    n_total: int

# ── Inference ─────────────────────────────────────────────────────────────

def _preprocess(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _predict(text: str, threshold: float) -> dict:
    enc = _tokenizer(
        _preprocess(text), return_tensors="pt",
        truncation=True, max_length=MAX_LEN,
    ).to(DEVICE)
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
        "model_dir": str(MODEL_DIR),
        "device": str(DEVICE),
        "default_threshold": THRESHOLD,
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


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
