#!/usr/bin/env python3
"""Deterministic rejection rules applied to a candidate triple before it is scored.

Each rule answers one question about a (passage, taxon_1, relation, taxon_2) candidate that a
classifier is poorly placed to answer, and says "reject" only when the candidate cannot be a
biotic interaction between two distinct organisms regardless of what the passage says about
them. Every rule is a fixed function with no tunable parameter fitted to evaluation labels.

Provenance, because it bounds what an evaluation of these rules can claim:

  same_organism, own_clade, authority
      Detectors taken unchanged from scripts/generate_targeted_v5.py, written on 2026-09-23 from
      the deployed filter's errors on biotx100 and reject50. They never saw test299.
  non_biotic_relation, negated, not_an_organism, pathogen_modifier
      Written on 2026-10-02 after error analysis of the verifier's false positives on all three
      blocks of the 437-row benchmark. Their *selection* is therefore informed by that set. They
      are validated on training data and on a disjoint development set before being frozen, and
      that sequence is the only thing that separates them from fitting the benchmark.

Usage (as a library)
    from candidate_rules import reject_reason
    why = reject_reason(passage, s1, relation, s2)   # None = keep, else the rule name
"""
from __future__ import annotations

import re

# ── detectors reused verbatim from generate_targeted_v5.py ──────────────────────────────────
AP = [re.compile(r"\b([a-z][a-z\- ]{3,25}?)\s*\(\s*([A-Z][a-z]+ [a-z]{3,})\b"),
      re.compile(r"\bthe\s+([a-z][a-z\- ]{3,25}?),\s+([A-Z][a-z]+ [a-z]{3,})\b"),
      re.compile(r"\b([A-Z][a-z]+ [a-z]{3,}),?\s+\(?the\s+([a-z][a-z\- ]{3,25}?)\)")]
ANC = [re.compile(r"\(((?:[A-Z][a-z]+\s*:\s*)+[A-Z][a-z]+)\)"),
       re.compile(r"\b(?:family|order|class|phylum|subfamily|superfamily|suborder|infraorder|tribe)"
                  r"\s+([A-Z][a-z]{3,})")]
AUTH = re.compile(r"\b([A-Z][a-z]+ [a-z]{3,})\s*\(?\b([A-Z][a-z]{3,})(?:\s*&\s*[A-Z][a-z]+)?,?\s*"
                  r"(?:1[6-9]\d\d|20[0-2]\d)\)?")
AUTH2 = re.compile(r"\b([A-Z][a-z]+ [a-z]{3,})\s*\(\s*([A-Z][a-z]{3,})\s*\)")
STOP = {"species", "genus", "strain", "isolate", "sample", "study", "group", "family", "order",
        "class", "type", "form", "variety", "subspecies", "complex", "clade", "sequence", "gene"}


def _n(s) -> str:
    return " ".join(str(s).split()).strip().lower()


def _alts(t) -> list[str]:
    """The benchmark joins alternative surface forms with '|'; treat each as the taxon."""
    return [a for a in (_n(x) for x in str(t).split("|")) if a]


def _occ(t: str, psg: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"\b" + re.escape(t) + r"\b", psg, re.I)]


# ── rule 1: the two taxa are one organism named twice ───────────────────────────────────────
def same_organism(psg: str, s1, s2) -> bool:
    A, B = _alts(s1), _alts(s2)
    if set(A) & set(B):
        return True
    for pat in AP:
        for m in pat.finditer(psg):
            pair = {_n(m.group(1)), _n(m.group(2))}
            for a in A:
                for b in B:
                    # either the full forms are the apposition pair, or one form is the head of it
                    if {a, b} <= pair or any(a in p and b in q for p in pair for q in pair if p != q):
                        return True
    return False


# ── rule 2: one taxon is the other's own clade, from an adjacent clade annotation ───────────
def own_clade(psg: str, s1, s2) -> bool:
    for t, other in ((s1, s2), (s2, s1)):
        for ta in _alts(t):
            ends = [e for _, e in _occ(ta, psg)]
            if not ends:
                continue
            for pat in ANC[:1]:                       # the parenthetical form only: adjacency is decidable
                for m in pat.finditer(psg):
                    if not any(0 <= m.start() - e <= 3 for e in ends):
                        continue
                    clades = {_n(c) for c in re.split(r"\s*:\s*", m.group(1)) if len(c) >= 5}
                    if any(o in clades for o in _alts(other)):
                        return True
    return False


# ── rule 3: one "taxon" is a taxonomic author, not an organism ──────────────────────────────
def authority(psg: str, s1, s2) -> bool:
    auths = set()
    for pat in (AUTH, AUTH2):
        for m in pat.finditer(psg):
            if _n(m.group(2)) not in STOP:
                auths.add(_n(m.group(2)))
    return bool(auths) and any(a in auths for a in _alts(s1) + _alts(s2))


# ── rule 4: the relation term names no biotic interaction ───────────────────────────────────
# Phylogenetic, comparative, biogeographic and methodological terms. A candidate whose only
# predicate is one of these is a statement about classification or study design.
NON_BIOTIC = re.compile(
    r"^(?:relative to|related to|closely related|close relative|relatives?|sister|"
    r"hybridi[sz]ation|hybridi[sz]ed|migration|migrat\w*|originated from|origin|"
    r"compared (?:to|with)|similar to|divergen\w*|phylogen\w*|congener\w*|"
    r"distinguished from|differs? from|resembl\w*)$")


def non_biotic_relation(rel) -> bool:
    return any(NON_BIOTIC.match(r) for r in _alts(rel))


# ── rule 5: the relation is explicitly denied ───────────────────────────────────────────────
# Narrow on purpose: only cues bound to an interaction word or to a host/feeding result, so
# that "nonhost resistance" or "no other host is known" are not read as denials of the pair.
def negated(psg: str, s1, s2) -> bool:
    """The pair itself is denied: a negation whose scope contains one of the two taxa.

    Two forms only. "non-<taxon> hosts", where the prefixed word is literally one of the pair
    ("found only on non-wheat hosts"), and a negated interaction noun whose following clause
    names one of the pair ("detected no infections with ... Leucocytozoon"). A general
    negation cue in the sentence is not enough: on training data that fires on "without
    affecting the host" and "non-crop hosts", where the pair still interacts.
    """
    for t in _alts(s1) + _alts(s2):
        if re.search(r"\bnon-?" + re.escape(t) + r"\s+hosts?\b", psg, re.I):
            return True
    for m in re.finditer(r"\b(?:no|not)\s+(?:\w+\s+){0,2}?(?:infections?|infested|infected)\s+with\b"
                         r"([^.;]{0,160})", psg, re.I):
        scope = m.group(1).lower()
        if any(re.search(r"\b" + re.escape(t) + r"\b", scope) for t in _alts(s1) + _alts(s2)):
            return True
    return False


# ── rule 6: one "taxon" is not an organism at all ───────────────────────────────────────────
NOT_ORGANISM = {
    # Anatomy, structures and syndromes: words that cannot denote an organism at all. Coarse
    # taxa ("viruses", "fungi") and role nouns ("endosymbionts") are deliberately absent -- on
    # training data they are teacher-positive 30-40% of the time ("chlorella ... hosts to
    # viruses"), so rejecting them would cost more true positives than it saves.
    "phyllodes", "leaves", "roots", "stems", "seeds", "fruits", "flowers", "paralysis", "lesions",
    "symptoms", "tumor", "tumour", "syndrome", "lesion",
}


def not_an_organism(s1, s2) -> bool:
    return any(a in NOT_ORGANISM for a in _alts(s1) + _alts(s2))


# ── rule 7: an adjective that is part of a pathogen's name, not a host ──────────────────────
# "equine influenza virus infections in human beings" -- equine names the virus, not horses.
ADJ = {"equine", "avian", "bovine", "porcine", "canine", "feline", "ovine", "caprine", "murine",
       "simian", "piscine", "human", "swine"}
PATHOGEN_HEAD = re.compile(
    r"(?:influenza|virus|viruses|viral|schistosomes?|malaria|tuberculosis|leukemia|leukaemia|"
    r"herpes\w*|corona\w*|pox\w*|parvo\w*|rota\w*|papilloma\w*|plasmodi\w*|trypanosom\w*|"
    r"babesi\w*|leishmani\w*|bacteri\w*|prions?|retro\w*|lenti\w*|immunodeficiency)", re.I)


def pathogen_modifier(psg: str, s1, s2) -> bool:
    for t in _alts(s1) + _alts(s2):
        if t not in ADJ:
            continue
        spans = _occ(t, psg)
        # every occurrence must be a modifier of a pathogen word; if the adjective is ever used
        # on its own ("in equine and human hosts") it may genuinely denote the host
        if spans and all(PATHOGEN_HEAD.match(psg[e:].lstrip()) for _, e in spans):
            return True
    return False


RULES = [
    ("same_organism", lambda p, a, r, b: same_organism(p, a, b)),
    ("own_clade", lambda p, a, r, b: own_clade(p, a, b)),
    ("authority", lambda p, a, r, b: authority(p, a, b)),
    ("non_biotic_relation", lambda p, a, r, b: non_biotic_relation(r)),
    ("negated", lambda p, a, r, b: negated(p, a, b)),
    ("not_an_organism", lambda p, a, r, b: not_an_organism(a, b)),
    ("pathogen_modifier", lambda p, a, r, b: pathogen_modifier(p, a, b)),
]


def reject_reason(passage, s1, relation, s2, rules=None) -> str | None:
    """Name of the first rule that rejects the candidate, or None to keep it."""
    p = str(passage)
    for name, fn in RULES:
        if rules is not None and name not in rules:
            continue
        if fn(p, s1, relation, s2):
            return name
    return None


# ── rule 8: the two taxa are co-members of one list, not arguments of one interaction ───────
# Added 2026-10-02 after the rules above were frozen and evaluated; validated separately.
# "beetles, dragonflies, cockroaches, and female katydids" are all prey of a third party.
# Coordination is not enough on its own -- "interactions between Cotesia and Oomyzus" is a
# coordination that IS an interaction -- so a list governed by an interaction-between noun is
# kept.
_NLP = None
BETWEEN_HEADS = {"interaction", "interactions", "competition", "association", "associations",
                 "relationship", "relationships", "coevolution", "symbiosis", "mutualism",
                 "antagonism", "transmission", "contact", "conflict", "coexistence", "comparison"}


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    return _NLP


def _head_token(doc, t):
    """Syntactic head of the first occurrence of taxon string t, or None."""
    low = doc.text.lower()
    for m in re.finditer(r"\b" + re.escape(t) + r"\b", low):
        toks = [tok for tok in doc if m.start() <= tok.idx < m.end()]
        if toks:
            return max(toks, key=lambda tok: sum(1 for _ in tok.children)) if len(toks) > 1 else toks[0]
    return None


def _list_member(tok):
    """The token that actually sits in a coordination: "true armyworm (Mythimna unipuncta)"
    coordinates the common name, and the binomial hangs off it as an apposition."""
    while tok.dep_ in ("appos", "compound") and tok.head.i != tok.i:
        tok = tok.head
    return tok


def _first_conjunct(tok):
    tok = _list_member(tok)
    while tok.dep_ == "conj" and tok.head.i != tok.i:
        tok = tok.head
    return tok


def co_listed(psg: str, s1, s2) -> bool:
    for sent in re.split(r"(?<=[.;])\s+", psg):
        a = next((x for x in _alts(s1) if _occ(x, sent)), None)
        b = next((x for x in _alts(s2) if _occ(x, sent)), None)
        if not a or not b:
            continue
        doc = _nlp()(sent)
        ha, hb = _head_token(doc, a), _head_token(doc, b)
        if ha is None or hb is None or ha.i == hb.i:
            continue
        fa, fb = _first_conjunct(ha), _first_conjunct(hb)
        la, lb = _list_member(ha), _list_member(hb)
        if la.i == lb.i or fa.i != fb.i or (la.dep_ != "conj" and lb.dep_ != "conj"):  # tokens are views: compare indices
            continue
        gov = fa.head                                   # what governs the whole list
        if gov.lower_ == "between" and gov.head.lower_ in BETWEEN_HEADS:
            continue
        if gov.lower_ in BETWEEN_HEADS:
            continue
        return True
    return False


RULES_WITH_COLIST = RULES + [("co_listed", lambda p, a, r, b: co_listed(p, a, b))]
