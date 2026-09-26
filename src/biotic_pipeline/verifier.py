"""
Programmatic interface for the query-conditioned triple verifier.

Where BioticClassifier asks "does this sentence describe an interaction?",
this asks "do THESE TWO taxa interact, according to this passage, and which
one is the subject?". The decision unit is the candidate triple, not the
sentence, so it needs a candidate generator upstream -- see README.

Usage:
    from biotic_pipeline import TripleVerifier

    v = TripleVerifier("path/to/joint_a05_s1")
    print(v.verify("Wolbachia", "infects", "Drosophila melanogaster",
                   "Wolbachia infects Drosophila melanogaster."))
    # {'interacts': 1, 'p_interact': 0.99, 'direction': 'FORWARD', ...}

    results = v.verify_batch([(s1, rel, s2, passage), ...])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from biotic_pipeline import xenc_format as _X
from biotic_pipeline.joint_model import Student

# Loaded once, at import. A per-call try/except would swallow a missing lexicon and
# hand every relation the same default polarity, degrading direction to near-chance
# with no error.
from biotic_pipeline import polarity as _polarity_mod


class TripleVerifier:
    """
    Query-conditioned joint verifier: pair interaction + argument direction.

    The encoder receives the two taxa sorted alphabetically as segment A, and the
    passage with the alphabetically-first taxon marked @...@ and the second #...#
    as segment B. Swapping the two input taxa therefore produces a byte-identical
    input: `p_interact` is order-invariant by construction and the direction head is
    exactly antisymmetric. The relation string never reaches the encoder; it enters
    only at the direction head, as a 2-valued polarity embedding.

    Args:
        model_dir:   Path to the joint checkpoint (config.json, model.safetensors,
                     direction_head.pt, student_config.json).
        threshold:   Interaction threshold. Default 0.50. On the 437-row benchmark
                     that gives precision 0.852 / recall 0.959. Raise it to buy
                     precision: 0.88 -> 0.871/0.907, 0.95 -> 0.882/0.878.
        dir_abstain: Report a direction only above this confidence, else UNCERTAIN.
        device:      "auto", "cpu", or "cuda:0".
        batch_size:  Candidates per forward pass.
        threads:     Torch CPU threads (ignored on GPU).
    """

    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.50,
        dir_abstain: float = 0.71,
        device: str = "auto",
        batch_size: int = 16,
        threads: int = 8,
        max_length: int = 256,
    ):
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        cfg_path = model_dir / "student_config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{model_dir} has no student_config.json -- this does not look like a "
                f"joint verifier checkpoint. For the sentence-level model use "
                f"BioticClassifier instead."
            )
        self.config = json.loads(cfg_path.read_text())
        self.threshold = threshold
        self.dir_abstain = dir_abstain
        self.batch_size = batch_size
        self.max_length = max_length

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        if self._device.type == "cpu":
            torch.set_num_threads(threads)

        self._model = Student(enc=str(model_dir),
                              detach_dir=self.config.get("detach_dir", False))
        self._model.dir.load_state_dict(
            torch.load(model_dir / "direction_head.pt", map_location="cpu"))
        self._model = self._model.to(self._device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # ── Public ────────────────────────────────────────────────────────────

    def verify(self, species1: str, relation: str, species2: str,
               passage: str, threshold: float | None = None) -> dict:
        """Verify a single candidate triple against its passage."""
        if not passage.strip():
            raise ValueError("passage cannot be empty")
        return self.verify_batch([(species1, relation, species2, passage)],
                                 threshold=threshold)[0]

    def verify_batch(
        self,
        candidates: Sequence[Tuple[str, str, str, str]],
        threshold: float | None = None,
    ) -> List[dict]:
        """
        Verify a list of (species1, relation, species2, passage) tuples.

        Returns one dict per candidate:
            {"species1", "relation", "species2",
             "interacts": int, "p_interact": float,
             "direction": "FORWARD"|"REVERSE"|"UNCERTAIN",
             "p_species1_is_subject": float, "direction_confidence": float,
             "both_taxa_located": int, "unknown_polarity": int,
             "threshold_used": float}
        """
        if not candidates:
            raise ValueError("candidates list cannot be empty")
        t = threshold if threshold is not None else self.threshold
        out: List[dict] = []
        for i in range(0, len(candidates), self.batch_size):
            out.extend(self._infer(list(candidates[i : i + self.batch_size]), t))
        return out

    # ── Internal ──────────────────────────────────────────────────────────

    def _polarity(self, rel: str) -> Tuple[int, bool]:
        p, _src = _polarity_mod.polarity(str(rel))
        if p is None or p == 0:
            return 1, True            # unknown relation -> agent-side default
        return (1 if p > 0 else 0), False

    @torch.no_grad()
    def _infer(self, batch: list, threshold: float) -> list[dict]:
        s1 = [str(c[0]) for c in batch]
        rel = [str(c[1]) for c in batch]
        s2 = [str(c[2]) for c in batch]
        txt = [str(c[3]) for c in batch]

        a, b = _X.build_many("mark_canon", s1, rel, s2, txt)
        pol_flags = [self._polarity(r) for r in rel]
        pol = torch.tensor([p[0] for p in pol_flags], dtype=torch.long,
                           device=self._device)

        enc = self._tokenizer(a, b, truncation="only_second",
                              max_length=self.max_length, padding=True,
                              return_tensors="pt").to(self._device)
        logits, s, ok = self._model(enc["input_ids"], enc["attention_mask"],
                                    enc.get("token_type_ids"), pol=pol)
        P = torch.softmax(logits.float(), -1)[:, 1].cpu().numpy()
        D = s.float().cpu().numpy()
        OK = ok.float().cpu().numpy()

        at_is_s1 = np.array([x.lower() <= y.lower() for x, y in zip(s1, s2)])
        p_at = 1 / (1 + np.exp(-D))
        p_s1 = np.where(at_is_s1, p_at, 1 - p_at)
        conf = np.abs(p_s1 - 0.5) * 2

        res = []
        for i in range(len(batch)):
            located = bool(OK[i] > 0)
            if not located or conf[i] < self.dir_abstain:
                direction = "UNCERTAIN"
            else:
                direction = "FORWARD" if p_s1[i] >= 0.5 else "REVERSE"
            res.append({
                "species1": s1[i], "relation": rel[i], "species2": s2[i],
                "interacts": int(P[i] >= threshold),
                "p_interact": round(float(P[i]), 4),
                "direction": direction,
                "p_species1_is_subject": round(float(p_s1[i]), 4),
                "direction_confidence": round(float(conf[i]), 4),
                "both_taxa_located": int(located),
                "unknown_polarity": int(pol_flags[i][1]),
                "threshold_used": threshold,
            })
        return res
