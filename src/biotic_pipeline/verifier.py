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

Scoring follows the standalone handoff scorer (predict.py) step for step, so the two
return identical results for the same candidates, model and batch size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

from biotic_pipeline import candidate_rules as _rules
from biotic_pipeline import xenc_format as _X
from biotic_pipeline.joint_model import Student

# Loaded once, at import. A per-call try/except would swallow a missing lexicon and
# hand every relation the same default polarity, degrading direction to near-chance
# with no error.
from biotic_pipeline import polarity as _polarity_mod

#: The four values ``direction`` can take. BIDIRECTIONAL is a property of the relation
#: (the lexicon calls it mutual: mutualism, interacts with, co-occurs with, ...), not a
#: guess. UNCERTAIN means the model is not confident, a taxon was not found in the
#: passage, or the pair does not interact.
DIRECTIONS = ("FORWARD", "REVERSE", "BIDIRECTIONAL", "UNCERTAIN")


class TripleVerifier:
    """
    Query-conditioned joint verifier: pair interaction + argument direction.

    The encoder receives the two taxa sorted alphabetically as segment A, and the
    passage with the alphabetically-first taxon marked @...@ and the second #...#
    as segment B. Swapping the two input taxa therefore produces a byte-identical
    input: `p_interact` is order-invariant by construction and the direction head is
    exactly antisymmetric. The relation string never reaches the encoder; it enters
    only at the direction head, as a polarity embedding.

    Args:
        model_dir:   Path to the joint checkpoint (config.json, model.safetensors,
                     direction_head.pt, student_config.json, tokenizer files).
        threshold:   Interaction threshold. Default 0.50, fixed before evaluation.
                     Raise it to trade recall for precision (see README).
        dir_abstain: Report FORWARD/REVERSE only at or above this confidence,
                     else UNCERTAIN. Default 0.60.
        device:      "auto", "cpu", or "cuda:0".
        batch_size:  Candidates per forward pass.
        threads:     Torch CPU threads (ignored on GPU).
        max_length:  Wordpiece window. Default: the length the model was trained with.
        rules:       Apply the deterministic candidate rules (candidate_rules.py) that
                     reject candidates which cannot be an interaction between two
                     distinct organisms. Default True.
    """

    def __init__(
        self,
        model_dir: str,
        threshold: float = 0.50,
        dir_abstain: float = 0.60,
        device: str = "auto",
        batch_size: int = 16,
        threads: int = 8,
        max_length: Optional[int] = None,
        rules: bool = True,
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
        fmt = self.config.get("input_format", "mark_canon")
        if fmt != "mark_canon":
            raise ValueError(
                f"{model_dir} was trained with input_format={fmt!r}; this verifier builds "
                f"mark_canon inputs and would silently mis-score it."
            )
        self.threshold = threshold
        self.dir_abstain = dir_abstain
        self.batch_size = batch_size
        self.max_length = int(max_length or self.config.get("max_len", 256))
        self.rules = rules

        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        if self._device.type == "cpu":
            torch.set_num_threads(threads)

        dir_state = torch.load(model_dir / "direction_head.pt", map_location="cpu")
        self._model = Student(enc=str(model_dir),
                              detach_dir=self.config.get("detach_dir", False),
                              dir_state=dir_state)
        self._model = self._model.to(self._device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

    # ── Public ────────────────────────────────────────────────────────────

    def verify(self, species1: str, relation: str, species2: str, passage: str,
               threshold: Optional[float] = None, rules: Optional[bool] = None) -> dict:
        """Verify a single candidate triple against its passage."""
        return self.verify_batch([(species1, relation, species2, passage)],
                                 threshold=threshold, rules=rules)[0]

    def verify_batch(
        self,
        candidates: Sequence[Tuple[str, str, str, str]],
        threshold: Optional[float] = None,
        rules: Optional[bool] = None,
    ) -> List[dict]:
        """
        Verify a list of (species1, relation, species2, passage) tuples.

        Taxa should be given as they appear in the passage (surface forms), since the
        model marks them in the text.

        Returns one dict per candidate, in input order:
            species1, relation, species2   echoed back
            interacts              0/1, the decision (rules applied)
            p_interact             model score 0-1, before the rules
            rejected_by_rule       "" or the name of the rule that rejected the row
            direction              FORWARD | REVERSE | BIDIRECTIONAL | UNCERTAIN
            p_species1_is_subject  0-1, None for BIDIRECTIONAL relations
            direction_confidence   |p - 0.5| * 2, None for BIDIRECTIONAL relations
            symmetric_relation     1 if the lexicon calls the relation mutual
            both_taxa_located      0 if a taxon string was not found in the passage
            unknown_polarity       1 if the relation is outside the polarity lexicon
            truncated              1 if the passage exceeded the wordpiece window
            threshold_used         the interaction threshold applied
        """
        if not candidates:
            raise ValueError("candidates list cannot be empty")
        for i, c in enumerate(candidates):
            if len(c) != 4:
                raise ValueError(f"candidate {i} must be (species1, relation, species2, passage)")
            if any(v is None or not str(v).strip() or str(v) == "nan" for v in c):
                raise ValueError(f"candidate {i} has an empty or missing value: {tuple(c)!r}")
        t = self.threshold if threshold is None else threshold
        use_rules = self.rules if rules is None else rules
        s1 = [str(c[0]) for c in candidates]
        rel = [str(c[1]) for c in candidates]
        s2 = [str(c[2]) for c in candidates]
        txt = [str(c[3]) for c in candidates]
        return self._score(s1, rel, s2, txt, t, use_rules)

    # ── Internal ──────────────────────────────────────────────────────────

    @staticmethod
    def _polarity(rel: str, n_pol: int):
        """(head index, lexicon polarity, unknown?) -- the mapping training used."""
        p, _src = _polarity_mod.polarity(str(rel))
        return _polarity_mod.polarity_to_index(p, n_pol), p, p is None

    @torch.no_grad()
    def _score(self, s1, rel, s2, text, threshold, use_rules) -> List[dict]:
        a, b = _X.build_many("mark_canon", s1, rel, s2, text)
        n_pol = self._model.dir.n_pol
        _pl = [self._polarity(r, n_pol) for r in rel]
        pol = np.array([x[0] for x in _pl], dtype="int64")
        lex = [x[1] for x in _pl]
        unknown = np.array([x[2] for x in _pl])
        # how much of each passage survives the window: segment B is cut first
        full = [len(self._tokenizer(x, y, add_special_tokens=True)["input_ids"])
                for x, y in zip(a, b)]
        truncated = np.array([n > self.max_length for n in full])

        P, D, OK = [], [], []
        bs = self.batch_size
        for i in range(0, len(a), bs):
            e = self._tokenizer(a[i:i + bs], b[i:i + bs], truncation="only_second",
                                max_length=self.max_length, padding=True,
                                return_tensors="pt").to(self._device)
            pt = torch.tensor(pol[i:i + bs], dtype=torch.long, device=self._device)
            logits, s, ok, _u = self._model(e["input_ids"], e["attention_mask"],
                                            e.get("token_type_ids"), pol=pt)
            P.extend(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
            D.extend(s.float().cpu().numpy())
            OK.extend(ok.float().cpu().numpy())
        P, D, OK = np.array(P), np.array(D), np.array(OK)

        if use_rules:
            why = [_rules.reject_reason(t, x, r, y) or "" for t, x, r, y in zip(text, s1, rel, s2)]
        else:
            why = [""] * len(text)
        interacts = ((P >= threshold) & np.array([w == "" for w in why])).astype(int)

        # the @ taxon is the alphabetically first one; decode back to the caller's order
        at_is_s1 = np.array([x.lower() <= y.lower() for x, y in zip(s1, s2)])
        p_at_subject = 1 / (1 + np.exp(-D))
        p_s1_subject = np.where(at_is_s1, p_at_subject, 1 - p_at_subject)
        conf = np.abs(p_s1_subject - 0.5) * 2
        symmetric = np.array([_polarity_mod.is_symmetric(p) for p in lex])
        direction = np.where(conf < self.dir_abstain, "UNCERTAIN",
                             np.where(p_s1_subject >= 0.5, "FORWARD", "REVERSE"))
        direction = np.where(symmetric, "BIDIRECTIONAL", direction)
        direction = np.where(OK > 0, direction, "UNCERTAIN")          # taxon not found
        direction = np.where(interacts == 1, direction, "UNCERTAIN")  # no interaction

        out: List[dict] = []
        for i in range(len(text)):
            sym = bool(symmetric[i])
            out.append({
                "species1": s1[i], "relation": rel[i], "species2": s2[i],
                "interacts": int(interacts[i]),
                "p_interact": round(float(P[i]), 4),
                "rejected_by_rule": why[i],
                "direction": str(direction[i]),
                "p_species1_is_subject": None if sym else round(float(p_s1_subject[i]), 4),
                "direction_confidence": None if sym else round(float(conf[i]), 4),
                "symmetric_relation": int(sym),
                "both_taxa_located": int(OK[i] > 0),
                "unknown_polarity": int(unknown[i]),
                "truncated": int(truncated[i]),
                "threshold_used": threshold,
            })
        return out
