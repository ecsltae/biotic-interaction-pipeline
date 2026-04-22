"""
FastAPI server for biotic interaction classification.

Supports two model types, auto-detected from model_dir:
  - MultiTaskBiomedBERT  (multitask_config.json present)
  - HF AutoModelForSequenceClassification  (standard distilled/fine-tuned)

Configuration is read from config.toml in the working directory.
"""

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

_cfg  = get_config()
_scfg = _cfg.server
_mcfg = _cfg.model

if _mcfg.model_dir == "":
    raise RuntimeError("[model] model_dir is not set in config.toml")

if _mcfg.device == "auto":
    _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
else:
    _device = torch.device(_mcfg.device)

_model_dir = Path(_mcfg.model_dir)
_is_multitask = (_model_dir / "multitask_config.json").exists()

print(f"Loading {'multitask' if _is_multitask else 'standard'} model from {_model_dir} on {_device} ...", flush=True)
_tokenizer = AutoTokenizer.from_pretrained(str(_model_dir), local_files_only=True)

if _is_multitask:
    from biotic_pipeline.multitask_model import MultiTaskBiomedBERT
    _model = MultiTaskBiomedBERT.load(str(_model_dir), device=str(_device))
else:
    _model = AutoModelForSequenceClassification.from_pretrained(
        str(_model_dir), local_files_only=True).to(_device)

_model.eval()
print(f"Model ready. Threshold={_mcfg.threshold}", flush=True)

# ── App ───────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Biotic Interaction Classifier",
    description=(
        "Sentence-level biotic interaction detector. "
        "Returns label=1 when a sentence describes an ecological/parasitic/symbiotic "
        "interaction between two species."
    ),
    version="2.0.0",
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

def _encode(texts: list[str]):
    return _tokenizer(
        texts, truncation=True, max_length=_mcfg.max_length,
        padding=True, return_tensors="pt",
    ).to(_device)


def _predict_batch(texts: list[str], threshold: float) -> list[dict]:
    enc = _encode(texts)
    with torch.no_grad():
        if _is_multitask:
            out = _model(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                token_type_ids=enc.get("token_type_ids"),
            )
            probs = torch.softmax(out["cls_logits"], dim=-1)[:, 1].cpu().tolist()
        else:
            probs = torch.softmax(_model(**enc).logits, dim=-1)[:, 1].cpu().tolist()
    return [
        {
            "text": text,
            "label": int(p >= threshold),
            "probability": round(p, 4),
            "threshold_used": threshold,
        }
        for text, p in zip(texts, probs)
    ]

# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_dir": _mcfg.model_dir,
        "model_type": "multitask" if _is_multitask else "standard",
        "device": str(_device),
        "default_threshold": _mcfg.threshold,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(422, "text cannot be empty")
    return _predict_batch([req.text], req.threshold)[0]


@app.post("/batch", response_model=BatchResponse)
def batch_predict(req: BatchRequest):
    if not req.sentences:
        raise HTTPException(422, "sentences list cannot be empty")
    if len(req.sentences) > 500:
        raise HTTPException(422, "max 500 sentences per batch")
    results = []
    batch_size = 64
    for i in range(0, len(req.sentences), batch_size):
        results.extend(_predict_batch(req.sentences[i:i + batch_size], req.threshold))
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
