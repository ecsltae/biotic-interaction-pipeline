"""
GloBI pre-filter — fast sentence-level gating before the neural classifier.

Two-stage OR logic (zero false negatives, ~12% precision on biomedical text):
  1. Interaction vocabulary regex (GloBI terms + curated biomedical stems)
  2. Binomial species name lookup via Aho-Corasick automaton (~0.5ms/sentence)

A sentence passes if EITHER condition holds. The classifier downstream handles
precision — the filter only removes sentences with zero interaction signal.

Usage:
    from filter import build_filter, has_interaction_signal

    filt = build_filter(
        interaction_dict_csv="data/interaction_dict.csv",   # GloBI terms
        species_dict_csv="data/species_dict.csv",           # 4.2M binomials
    )
    if filt(sentence):
        # send to classifier
"""

import re
from pathlib import Path
from typing import Optional


# ── Interaction regex ─────────────────────────────────────────────────────

_BIOMEDICAL = (
    r'infect|parasit|\bhost\b|pathogen|\bvector\b|zoonot|symbiont|symbioti|'
    r'endophyt|mycorrhiz|\bnodule|nematod|fungal|\bfungi\b|bacteri|viral|\bvirus\b|protozoa|'
    r'transmit|reservoir|definitive host|intermediate host|'
    r'harbour|harbor|coloniz|colonise|life cycle|'
    r'\bprey\b|predat|pollina|feed on|feeds on|\beats\b|ingest|'
    r'herbivory|herbivore|mutuali|commensali|kleptoparasit|'
    r'\bHIV\b|\bAIDS\b|\bSARS\b|\bMERS\b|\bCOVID\b|'
    r'chickenpox|smallpox|monkeypox|\bmeasles\b|\bmumps\b|\brubella\b|'
    r'\bmalaria\b|\bdengue\b|\bebola\b|\brabies\b|\btuberculosis\b|\bTB\b|'
    r'influenza|\bflu\b|\bplague\b|\bcholera\b|\btyphus\b|\btyphoid\b|'
    r'\bsyphilis\b|\bleprosy\b|\banthrax\b|\bbotulism\b|\btetanus\b|'
    r'leishmanian|\btrypanosomia|schistosom|\btoxoplasm|\bcryptospor'
)


def _build_interaction_pattern(interaction_dict_csv: Optional[Path] = None) -> re.Pattern:
    """Build combined regex from GloBI terms file + curated biomedical vocabulary."""
    globi_terms = []
    if interaction_dict_csv and Path(interaction_dict_csv).exists():
        import csv
        with open(interaction_dict_csv) as f:
            for row in csv.DictReader(f):
                t = row.get("interaction", "").strip()
                if len(t) >= 3:
                    globi_terms.append(re.escape(t.lower()))

    if globi_terms:
        alternation = "|".join(globi_terms)
        return re.compile(
            r"(?:(?<!\w)(?:" + alternation + r")(?!\w))|(?:" + _BIOMEDICAL + r")",
            re.IGNORECASE,
        )
    return re.compile(_BIOMEDICAL, re.IGNORECASE)


# ── Species automaton ─────────────────────────────────────────────────────

def _build_species_automaton(species_dict_csv: Optional[Path] = None):
    """Build Aho-Corasick automaton from CSV of binomial species names."""
    if species_dict_csv is None or not Path(species_dict_csv).exists():
        return None
    try:
        import ahocorasick
    except ImportError:
        return None

    binomial_re = re.compile(r'^[A-Z][a-z]+ [a-z]+$')
    A = ahocorasick.Automaton()
    with open(species_dict_csv) as f:
        next(f)  # skip header
        for line in f:
            name = line.strip()
            if binomial_re.match(name):
                key = name.lower()
                if key not in A:
                    A.add_word(key, name)
    if len(A) == 0:
        return None
    A.make_automaton()
    return A


def _has_species_mention(sentence: str, automaton) -> bool:
    if automaton is None:
        return False
    text_lower = sentence.lower()
    for end_idx, value in automaton.iter(text_lower):
        key = value.lower()
        start = end_idx - len(key) + 1
        end   = end_idx + 1
        if (start == 0 or not text_lower[start - 1].isalpha()) and \
           (end >= len(text_lower) or not text_lower[end].isalpha()):
            return True
    return False


# ── Public API ────────────────────────────────────────────────────────────

def build_filter(
    interaction_dict_csv: Optional[str] = None,
    species_dict_csv: Optional[str] = None,
) -> "callable":
    """
    Build and return a filter function: sentence -> bool.

    Args:
        interaction_dict_csv: Path to CSV with 'interaction' column (GloBI terms).
                              Falls back to curated biomedical vocabulary only.
        species_dict_csv:     Path to CSV of binomial species names (one per line,
                              header row expected). Falls back to interaction-only.

    Returns:
        Callable[str, bool] — True if sentence should be sent to the classifier.
    """
    interaction_csv = Path(interaction_dict_csv) if interaction_dict_csv else None
    species_csv     = Path(species_dict_csv)     if species_dict_csv     else None

    pattern   = _build_interaction_pattern(interaction_csv)
    automaton = _build_species_automaton(species_csv)

    n_terms   = len(pattern.pattern.split("|"))
    n_species = len(automaton) if automaton else 0
    print(f"[filter] interaction pattern built ({n_terms} alternations)", flush=True)
    print(f"[filter] species automaton: {n_species:,} binomial names", flush=True)

    def _filter(sentence: str) -> bool:
        if pattern.search(sentence):
            return True
        return _has_species_mention(sentence, automaton)

    return _filter


# ── Sentence splitter ─────────────────────────────────────────────────────

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
_XML_TAG    = re.compile(r'<[^>]+>')


def split_sentences(text: str) -> list[str]:
    """Sentence splitter for biomedical text; strips XML/HTML tags first."""
    text = _XML_TAG.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return [s.strip() for s in _SENT_SPLIT.split(text) if len(s.strip()) > 20]
