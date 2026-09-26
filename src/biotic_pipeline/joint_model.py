"""Joint interaction + direction cross-encoder — inference definition.

Self-contained: no training code, no repo layout assumptions.

The encoder sees the two taxa sorted alphabetically as segment A, and the passage with the
alphabetically-first taxon wrapped in @...@ and the second in #...# as segment B. Swapping the
two input taxa therefore produces a byte-identical input: the interaction score is order-
invariant by construction, and the direction head is exactly antisymmetric, so
P(@ is subject) = 1 - P(# is subject) at every parameter setting.

The relation string never reaches the encoder. It enters only at the direction head, as a
2-valued polarity embedding.
"""
import torch, torch.nn as nn
from transformers import AutoModelForSequenceClassification

AT_ID, HASH_ID = 36, 7          # '@' and '#' in the BiomedBERT uncased vocab
ENC = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
# Note: Student(enc=<local model dir>) is what predict.py uses, so no network
# access is needed at inference time. ENC is only the default.


def span_pool(hidden, ids, marker):
    """Mean-pool the tokens strictly between the first two `marker` tokens of each row."""
    B, T, H = hidden.shape
    m = (ids == marker)
    out = hidden.new_zeros(B, H); ok = hidden.new_zeros(B)
    idx = [torch.nonzero(m[i], as_tuple=False).flatten() for i in range(B)]
    for i, p in enumerate(idx):
        if p.numel() >= 2:
            lo, hi = int(p[0]) + 1, int(p[1])
            if hi > lo:
                out[i] = hidden[i, lo:hi].mean(0); ok[i] = 1.0
        elif p.numel() == 1:                       # marker pair truncated away
            lo = int(p[0]) + 1
            if lo < T:
                out[i] = hidden[i, lo:min(lo + 8, T)].mean(0); ok[i] = 1.0
    return out, ok


class DirectionHead(nn.Module):
    """Exactly antisymmetric: s(hA,hB) = -s(hB,hA) for every parameter setting."""
    def __init__(self, hid=768, pol_dim=16, inner=256):
        super().__init__()
        self.pol_emb = nn.Embedding(2, pol_dim)
        self.g = nn.Sequential(nn.Linear(3 * hid + pol_dim, inner), nn.GELU(),
                               nn.Dropout(0.1), nn.Linear(inner, 1))
    def forward(self, hA, hB, cls, pol):
        pe = self.pol_emb(pol)
        return (self.g(torch.cat([hA, hB, cls, pe], -1))
                - self.g(torch.cat([hB, hA, cls, pe], -1))).squeeze(-1)
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class Student(nn.Module):
    def __init__(self, enc=ENC, detach_dir=False):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(enc, num_labels=2)
        self.dir = DirectionHead(self.bert.config.hidden_size)
        self.detach_dir = detach_dir
    def encode(self, input_ids, attention_mask, token_type_ids=None):
        kw = dict(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        if token_type_ids is not None: kw["token_type_ids"] = token_type_ids
        o = self.bert(**kw)
        return o.logits, o.hidden_states[-1]
    def forward(self, input_ids, attention_mask, token_type_ids=None, pol=None):
        logits, hid = self.encode(input_ids, attention_mask, token_type_ids)
        h = hid.detach() if self.detach_dir else hid
        hA, okA = span_pool(h, input_ids, AT_ID)
        hB, okB = span_pool(h, input_ids, HASH_ID)
        s = self.dir(hA, hB, h[:, 0], pol) if pol is not None else None
        return logits, s, (okA * okB)


