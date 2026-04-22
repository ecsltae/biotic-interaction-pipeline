#!/usr/bin/env python3
"""
Multi-task BiomedBERT: interaction classification + species NER, joint training.

Architecture:
    BiomedBERT encoder (shared weights)
        ├── Head 1: [CLS] → Linear(768, 2)  — interaction yes/no
        └── Head 2: each token → Linear(768, n_labels)  — NER (BIO scheme)

NER label set (configurable):
    "basic":  O, B-SP, I-SP               — any species mention
    "typed":  O, B-HOST, I-HOST, B-PATHOGEN, I-PATHOGEN, B-SPECIES, I-SPECIES

Loss: α * CE_classification + (1-α) * CE_NER
      NER loss is masked for tokens where no annotation is available (-100).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig
from typing import Optional


LABEL_SETS = {
    # Species only
    "basic":      ["O", "B-SP", "I-SP"],
    # Species with role distinction
    "typed":      ["O", "B-HOST", "I-HOST", "B-PATHOGEN", "I-PATHOGEN", "B-SPECIES", "I-SPECIES"],
    # Species + interaction verb/phrase (recommended)
    "full":       ["O", "B-SP", "I-SP", "B-INT", "I-INT"],
    # Species (typed) + interaction verb/phrase
    "full_typed": ["O", "B-HOST", "I-HOST", "B-PATHOGEN", "I-PATHOGEN", "B-SPECIES", "I-SPECIES", "B-INT", "I-INT"],
}


class MultiTaskBiomedBERT(nn.Module):
    def __init__(self, encoder_name: str, ner_scheme: str = "basic", alpha: float = 0.5):
        """
        Args:
            encoder_name: HuggingFace model ID or local path
            ner_scheme: "basic" or "typed"
            alpha: weight for classification loss (1-alpha goes to NER)
        """
        super().__init__()
        self.alpha = alpha
        self.ner_labels = LABEL_SETS[ner_scheme]
        self.n_ner = len(self.ner_labels)
        self.label2id = {l: i for i, l in enumerate(self.ner_labels)}

        self.encoder = AutoModel.from_pretrained(encoder_name)
        hidden = self.encoder.config.hidden_size  # 768 for base models

        self.cls_head = nn.Linear(hidden, 2)
        self.ner_head = nn.Linear(hidden, self.n_ner)
        self.dropout  = nn.Dropout(0.1)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        cls_labels:     Optional[torch.Tensor] = None,  # (batch,) — 0/1
        ner_labels:     Optional[torch.Tensor] = None,  # (batch, seq_len) — -100 = ignore
    ):
        kwargs = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.encoder(**kwargs)
        seq_out = self.dropout(outputs.last_hidden_state)   # (B, L, H)
        cls_out = self.dropout(outputs.last_hidden_state[:, 0])  # (B, H)

        cls_logits = self.cls_head(cls_out)   # (B, 2)
        ner_logits = self.ner_head(seq_out)   # (B, L, n_ner)

        loss = None
        cls_loss = None
        ner_loss = None

        if cls_labels is not None:
            cls_loss = nn.CrossEntropyLoss()(cls_logits, cls_labels)

        if ner_labels is not None:
            # -100 masks positions with no annotation
            ner_loss = nn.CrossEntropyLoss(ignore_index=-100)(
                ner_logits.view(-1, self.n_ner),
                ner_labels.view(-1),
            )

        if cls_loss is not None and ner_loss is not None:
            loss = self.alpha * cls_loss + (1 - self.alpha) * ner_loss
        elif cls_loss is not None:
            loss = cls_loss
        elif ner_loss is not None:
            loss = ner_loss

        return {
            "loss": loss,
            "cls_loss": cls_loss,
            "ner_loss": ner_loss,
            "cls_logits": cls_logits,
            "ner_logits": ner_logits,
        }

    def save(self, path: str):
        import os, json
        from transformers import AutoTokenizer
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), f"{path}/pytorch_model.bin")
        encoder_name = self.encoder.config._name_or_path
        cfg = {
            "encoder_name": encoder_name,
            "ner_scheme": [k for k, v in LABEL_SETS.items() if v == self.ner_labels][0],
            "alpha": self.alpha,
            "ner_labels": self.ner_labels,
        }
        with open(f"{path}/multitask_config.json", "w") as f:
            json.dump(cfg, f, indent=2)
        self.encoder.config.save_pretrained(path)
        # Save tokenizer so AutoTokenizer.from_pretrained(path) works without HF connection
        AutoTokenizer.from_pretrained(encoder_name).save_pretrained(path)

    @classmethod
    def load(cls, path: str, device="cpu"):
        import json
        cfg = json.load(open(f"{path}/multitask_config.json"))
        model = cls(cfg["encoder_name"], cfg["ner_scheme"], cfg["alpha"])
        model.load_state_dict(torch.load(f"{path}/pytorch_model.bin", map_location=device,
                                         weights_only=True))
        return model.to(device)
