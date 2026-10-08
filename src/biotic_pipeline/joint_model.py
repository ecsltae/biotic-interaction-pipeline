"""Joint interaction + direction cross-encoder — inference definition.

Self-contained: no training code, no repo layout assumptions.

The encoder sees the two taxa sorted alphabetically as segment A, and the passage with the
alphabetically-first taxon wrapped in @...@ and the second in #...# as segment B. Swapping the
two input taxa therefore produces a byte-identical input: the interaction score is order-
invariant by construction, and the direction head is exactly antisymmetric, so
P(@ is subject) = 1 - P(# is subject) at every parameter setting.

The relation string never reaches the encoder. It enters only at the direction head, as a
polarity embedding (2- or 3-valued depending on the checkpoint).
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
    """Antisymmetric direction score, optionally with a symmetric directedness score.

    s_dir = g(hA, hB, cls, pol) - g(hB, hA, cls, pol)   exactly antisymmetric
    s_und = h(hA, hB, cls, pol) + h(hB, hA, cls, pol)   exactly symmetric (if trained)

    n_pol is 2 for checkpoints trained with polarity {patient, agent} and 3 for those trained
    with {patient, agent, symmetric}; `from_state_dict` reads both off the saved weights so a
    checkpoint can never be loaded with the wrong head.
    """
    def __init__(self, hid=768, pol_dim=16, inner=256, n_pol=2, with_und=False):
        super().__init__()
        self.n_pol = n_pol
        self.pol_emb = nn.Embedding(n_pol, pol_dim)
        mk = lambda: nn.Sequential(nn.Linear(3 * hid + pol_dim, inner), nn.GELU(),
                                   nn.Dropout(0.1), nn.Linear(inner, 1))
        self.g = mk()
        self.h = mk() if with_und else None
    @classmethod
    def from_state_dict(cls, sd, hid=768):
        n_pol, pol_dim = sd["pol_emb.weight"].shape
        inner = sd["g.0.weight"].shape[0]
        head = cls(hid, pol_dim=pol_dim, inner=inner, n_pol=n_pol, with_und="h.0.weight" in sd)
        head.load_state_dict(sd)
        return head
    def forward(self, hA, hB, cls, pol):
        pe = self.pol_emb(pol)
        ab = torch.cat([hA, hB, cls, pe], -1)
        ba = torch.cat([hB, hA, cls, pe], -1)
        s_dir = (self.g(ab) - self.g(ba)).squeeze(-1)
        s_und = (self.h(ab) + self.h(ba)).squeeze(-1) if self.h is not None else None
        return s_dir, s_und
    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class Student(nn.Module):
    def __init__(self, enc=ENC, detach_dir=False, dir_state=None):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(enc, num_labels=2)
        hid = self.bert.config.hidden_size
        self.dir = (DirectionHead.from_state_dict(dir_state, hid) if dir_state is not None
                    else DirectionHead(hid))
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
        s, u = self.dir(hA, hB, h[:, 0], pol) if pol is not None else (None, None)
        return logits, s, (okA * okB), u


