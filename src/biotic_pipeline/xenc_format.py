"""Shared input-format builders for the triple cross-encoder.

One place so training and evaluation can never drift. A model records its format
in student_config.json under "input_format"; eval_unified.py reads it back, so
older checkpoints (no key) keep the original triple-query behaviour.

Formats
  triple       A = "s1 [SEP] relation [SEP] s2"   B = passage          (original)
  pair         A = "s1 [SEP] s2"                  B = passage
  triple_mark  A = "s1 [SEP] relation [SEP] s2"   B = marked passage
  pair_mark    A = "s1 [SEP] s2"                  B = marked passage
  mark_only    A = marked passage                 B = None (single segment)

Marking follows the punctuation "typed entity marker" of Zhou & Chen (2021):
the first taxon is wrapped in @ ... @, the second in # ... #, in place, so the
encoder sees which spans the question is about rather than having to align them.
"""
import re

FORMATS = ("triple", "pair", "triple_mark", "pair_mark", "mark_only",
           "pair_canon", "mark_canon")
M1O, M1C, M2O, M2C = "@", "@", "#", "#"


def _locate(sp, txt):
    """Character span of taxon `sp` in `txt`, with graceful fallbacks. None if absent."""
    sp = str(sp).strip()
    if not sp:
        return None
    for cand in (sp,):
        m = re.search(re.escape(cand), txt, re.I)
        if m:
            return m.span()
    toks = sp.split()
    if len(toks) > 1:                      # binomial -> genus, then epithet/head noun
        for cand in (toks[0], toks[-1]):
            m = re.search(re.escape(cand), txt, re.I)
            if m:
                return m.span()
    for cand in (sp + "s", sp.rstrip("s")):  # trivial plural / singular
        if cand and cand != sp:
            m = re.search(re.escape(cand), txt, re.I)
            if m:
                return m.span()
    return None


def mark_passage(text, s1, s2):
    """Wrap the two taxon mentions in place. Overlapping or missing spans are skipped."""
    txt = str(text)
    spans = []
    a, b = _locate(s1, txt), _locate(s2, txt)
    if a:
        spans.append((a[0], a[1], M1O, M1C))
    if b and not (a and not (b[1] <= a[0] or b[0] >= a[1])):   # drop overlap with the first
        spans.append((b[0], b[1], M2O, M2C))
    out, prev = [], 0
    for st, en, o, c in sorted(spans):
        out.append(txt[prev:st]); out.append(f"{o} {txt[st:en]} {c}"); prev = en
    out.append(txt[prev:])
    return "".join(out)


def build(fmt, s1, rel, s2, text):
    """Return (segment_a, segment_b); segment_b is None for single-segment formats."""
    s1, s2, rel, text = str(s1), str(s2), str(rel), str(text)
    if fmt == "triple":
        return f"{s1} [SEP] {rel} [SEP] {s2}", text
    if fmt == "pair":
        return f"{s1} [SEP] {s2}", text
    if fmt == "triple_mark":
        return f"{s1} [SEP] {rel} [SEP] {s2}", mark_passage(text, s1, s2)
    if fmt == "pair_mark":
        return f"{s1} [SEP] {s2}", mark_passage(text, s1, s2)
    if fmt == "mark_only":
        return mark_passage(text, s1, s2), None
    if fmt in ("pair_canon", "mark_canon"):
        # the evaluated label is symmetric ("do these two taxa interact?"), so make the
        # representation order-invariant instead of asking the model to learn the symmetry
        a, b = sorted([s1, s2], key=str.lower)
        if fmt == "pair_canon":
            return f"{a} [SEP] {b}", text
        return f"{a} [SEP] {b}", mark_passage(text, a, b)
    raise ValueError(f"unknown input format {fmt!r}")


def build_many(fmt, s1s, rels, s2s, texts):
    ab = [build(fmt, a, r, b, t) for a, r, b, t in zip(s1s, rels, s2s, texts)]
    A = [x[0] for x in ab]
    B = None if ab and ab[0][1] is None else [x[1] for x in ab]
    return A, B
