"""
Programmatic interface for the biotic interaction classifier.

Usage:
    from biotic_pipeline import BioticClassifier

    clf = BioticClassifier("path/to/full_typed_a05_ner2")
    print(clf.classify("Wolbachia infects Drosophila melanogaster."))
    # {'text': '...', 'label': 1, 'probability': 0.94, 'threshold_used': 0.13}

    results = clf.classify_batch(["sent1", "sent2", ...])
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class BioticClassifier:
    """
    Sentence-level biotic interaction classifier.

    Loads full_typed_a05_ner2 (MultiTaskBiomedBERT) or any standard
    HF sequence-classification model. Model type is auto-detected from
    the presence of multitask_config.json in model_dir.

    Args:
        model_dir:  Path to the model checkpoint directory.
        threshold:  Classification threshold. Default 0.13 (optimised for
                    full_typed_a05_ner2 on EP-relax, F1=0.868). A higher
                    value increases precision at the cost of recall.
        device:     "auto" (GPU if available, else CPU), "cpu", or "cuda:0".
        batch_size: Sentences per forward pass (default 32).
    """

    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.13,
        device: str = "auto",
        batch_size: int = 32,
    ):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")

        if device == "auto":
            self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        self.threshold  = threshold
        self.batch_size = batch_size

        self._tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir), local_files_only=True
        )

        self._is_multitask = (model_dir / "multitask_config.json").exists()
        if self._is_multitask:
            from biotic_pipeline.multitask_model import MultiTaskBiomedBERT
            self._model = MultiTaskBiomedBERT.load(str(model_dir), device=str(self._device))
        else:
            self._model = AutoModelForSequenceClassification.from_pretrained(
                str(model_dir), local_files_only=True
            ).to(self._device)

        self._model.eval()

    # ── Public API ────────────────────────────────────────────────────────

    def classify(self, text: str, threshold: float | None = None) -> dict:
        """
        Classify a single sentence.

        Returns:
            {"text": str, "label": int, "probability": float, "threshold_used": float}
        """
        if not text.strip():
            raise ValueError("text cannot be empty")
        return self.classify_batch([text], threshold=threshold)[0]

    def classify_batch(
        self,
        sentences: List[str],
        threshold: float | None = None,
    ) -> List[dict]:
        """
        Classify a list of sentences.

        Returns a list of dicts, one per sentence:
            {"text": str, "label": int, "probability": float, "threshold_used": float}
        """
        if not sentences:
            raise ValueError("sentences list cannot be empty")

        t = threshold if threshold is not None else self.threshold
        results = []
        for i in range(0, len(sentences), self.batch_size):
            chunk = sentences[i : i + self.batch_size]
            results.extend(self._infer(chunk, t))
        return results

    # ── Internal ──────────────────────────────────────────────────────────

    def _infer(self, texts: list[str], threshold: float) -> list[dict]:
        enc = self._tokenizer(
            texts,
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        with torch.no_grad():
            if self._is_multitask:
                out = self._model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    token_type_ids=enc.get("token_type_ids"),
                )
                probs = torch.softmax(out["cls_logits"], dim=-1)[:, 1].cpu().tolist()
            else:
                probs = torch.softmax(
                    self._model(**enc).logits, dim=-1
                )[:, 1].cpu().tolist()

        return [
            {
                "text":           text,
                "label":          int(p >= threshold),
                "probability":    round(p, 4),
                "threshold_used": threshold,
            }
            for text, p in zip(texts, probs)
        ]
