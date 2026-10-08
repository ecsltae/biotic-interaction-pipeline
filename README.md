# biotic-interaction-pipeline

Biotic interaction detection for bulk article processing: a pair-conditioned triple verifier
that also returns direction (recommended), and the legacy sentence-level classifier.

Detects sentences describing ecological/parasitic/symbiotic interactions between two species
(e.g. "Wolbachia pipientis infects Drosophila melanogaster").

## Two paths, two decision units

This package ships two models, and they answer different questions.

```
                                        ┌─► TripleVerifier  (v3.1, recommended)
text                                    │     asks: do THESE TWO taxa interact,
 └─► sentence splitter                  │           and which is the subject?
       └─► GloBI pre-filter             │     input: (taxon, relation, taxon) + passage
             └─► candidate triples ─────┘
             │
             └─► BioticClassifier  (v2, legacy)
                   asks: does this sentence describe an interaction?
                   input: the sentence alone
```

**Which to use.** `TripleVerifier` unless you have a reason not to. On a 437-row
expert-graded benchmark (labels revised 2026-10-06), at the default threshold with the
candidate rules on:

| | precision | recall |
|---|---|---|
| **TripleVerifier** @ 0.50 | **0.888** | **0.948** |
| BioticClassifier (legacy, its recorded decisions) | 0.863 | 0.781 |

More precise, with seventeen points more recall. It also returns the argument direction
for free (same model, same speed, ~32 candidates/s on 8 CPU threads, ~1.5 GB RAM).

**The catch, stated plainly.** `TripleVerifier` takes a *candidate triple*, not a
sentence. This package does not generate candidates: the `text → sentences →
pre-filter` stages produce sentences, and something must turn those into
(taxon, relation, taxon) tuples before the verifier can score them. If you already
have a rule layer that emits candidates — two taxon surface forms and an interaction
surface form per candidate — feed it straight in. If you do not, `BioticClassifier`
remains the end-to-end path.

Swapping the checkpoint path in `config.toml` will **not** upgrade you: the decision
unit is different, and a sentence alone cannot be scored by the verifier.

### TripleVerifier

The weights are not in this repository (409 MB zipped). Unzip `joint_a05_s1.zip` and
pass the folder:

```python
from biotic_pipeline import TripleVerifier

v = TripleVerifier("path/to/joint_a05_s1")      # CPU by default if no GPU; device="cuda:0" to force
v.verify("Haemophilus influenzae", "pathogen of", "human",
         "Haemophilus influenzae is a major pathogen of humans.")
# {'interacts': 1, 'p_interact': 0.999, 'rejected_by_rule': '', 'direction': 'FORWARD',
#  'p_species1_is_subject': 0.9997, 'direction_confidence': 0.9994, 'symmetric_relation': 0,
#  'both_taxa_located': 1, 'unknown_polarity': 0, 'truncated': 0, 'threshold_used': 0.5, ...}

v.verify_batch([(s1, rel, s2, passage), ...])     # list of dicts, same order
v.verify_batch(candidates, threshold=0.95)        # buy precision (table below)
```

Give the taxa **as they appear in the passage** (surface forms such as "rabbits", not
canonical names): the model marks them in the text.

| key | meaning |
|---|---|
| `interacts` | 0/1, **the decision** — replaces the sentence classifier's verdict |
| `p_interact` | model score 0–1, before the rules; use it for your own cutoff |
| `rejected_by_rule` | `""`, or the name of the candidate rule that rejected the row |
| `direction` | `FORWARD` (species1 acts on species2), `REVERSE`, `BIDIRECTIONAL` (a mutual relation: symbiosis, *interacts with*, *co-occurs with*), or `UNCERTAIN` (not confident, a taxon not found, or no interaction) |
| `p_species1_is_subject`, `direction_confidence` | what `direction` is thresholded from (`None` for mutual relations) |
| `both_taxa_located` | 0 if a taxon string was not found in the passage — treat the row as unreviewed |
| `truncated` | 1 if the passage exceeded the 256-wordpiece window; split long abstracts into sentences |
| `unknown_polarity` | 1 if the relation is outside the polarity lexicon |

`FORWARD`/`REVERSE` are judged against the **canonical** relation, so
`Leptodora --prey--> Bosmina` returns `REVERSE`: the canonical relation is *preyed upon
by*, and *Leptodora* is the predator. Order does not matter: give the same pair the
other way round and `p_interact` is bit-identical while `FORWARD`/`REVERSE` swap. That is
architectural, not learned.

**Candidate rules** (`rules=True`, the default) reject, before the model's verdict,
candidates that cannot be an interaction between two distinct organisms: the same
organism named twice, a taxon and its own clade, a taxonomic author parsed as a taxon
(*Pterostichus melanarius (Illiger)*), a non-biotic relation term, an explicit negation
of the pair, an organ or syndrome parsed as a taxon, and an adjective inside a pathogen's
name (*equine* influenza virus). Pass `rules=False` to turn them off.

**Thresholds.** 0.50 is the default, fixed before evaluation rather than tuned. Raising it
buys precision (same benchmark, rules on):

| threshold | precision | recall |
|---|---|---|
| 0.50 | 0.888 | 0.948 |
| 0.70 | 0.890 | 0.932 |
| 0.90 | 0.907 | 0.892 |
| 0.95 | 0.920 | 0.869 |
| 0.99 | 0.969 | 0.745 |

**When to distrust it.** The main residual error is co-occurrence in a shared host: two
organisms both related to a *third* one can read as interacting (*Acanthamoeba* and
*Pseudomonas* both infecting a horse). Upstream entity errors are not fixed here: if the
rule layer hands it the wrong species, it will verify the wrong species. Direction is
weakest on bare relational nouns (*host*, *pathogen*, *infection*) and strongest where
the relation word carries direction (*pathogen of*). `BIDIRECTIONAL` comes from the
relation lexicon, not the model, so it is only as complete as the lexicon's list of
mutual relations.

## Architecture (legacy sentence path)

```
text
 └─► sentence splitter
       └─► GloBI pre-filter  (~1ms/sentence, pure Python, near-zero FN)
             └─► multi-task BiomedBERT classifier  (~14ms/sentence on GPU)
                   └─► results.csv
```

**Pre-filter** passes a sentence if it contains either:
1. A known interaction term (GloBI vocabulary + curated biomedical stems: *infect, parasit, host, pathogen, malaria, HIV, …*)
2. A binomial species name (Aho-Corasick lookup over 4.2M names)

**Classifier** is a multi-task BiomedBERT model jointly trained on interaction classification
and species/role NER (HOST / PATHOGEN distinction). It outperforms both the single-task
distilled model and the two-model ensemble at a third of the inference cost.

| Model | EP-relax F1 | Threshold |
|-------|-------------|-----------|
| distilled BiomedBERT v2 | 0.808 | 0.25 |
| BiomedBERT × FLAN-T5 ensemble | 0.857 | — |
| **multi-task BiomedBERT (this model)** | **0.868** | **0.13** |

Source for model training: [biotic-interaction-classifier](https://github.com/ecsltae/biotic-interaction-classifier)

## Local setup

```bash
# Install with uv
uv sync

# Edit config.toml — set model.model_dir
cp config.toml config.toml.local
$EDITOR config.toml.local

# Start the API
uv run biotic-api
# → http://localhost:8003/docs

# Process a folder of articles
uv run python process_articles.py \
  --input  /data/articles/ \
  --output results.csv
```

## Programmatic use

```python
from biotic_pipeline import BioticClassifier

clf = BioticClassifier("path/to/full_typed_a05_ner2")

# Single sentence
result = clf.classify("Wolbachia pipientis infects Drosophila melanogaster.")
# → {'text': '...', 'label': 1, 'probability': 0.94, 'threshold_used': 0.13}

# Batch
results = clf.classify_batch(["sent1", "sent2", ...])
```

## Configuration

All settings live in `config.toml`. In production, Ansible deploys this from
`deploy/templates/config.toml.j2` filled with values from `deploy/group_vars/all.yml`.

```toml
[server]
host = "0.0.0.0"
port = 8003

[model]
model_dir = "/opt/biotic-pipeline/model"   # path to full_typed_a05_ner2 checkpoint
device = "auto"
threshold = 0.13                           # optimised on EP-relax (F1=0.868)

[data]
interaction_dict = "data/interaction_dict.csv"   # GloBI terms (~30KB)
species_dict = "data/species_dict.csv"           # 4.2M binomials (~150MB, optional)
```

## API reference

```
GET  /health          model info, device, default threshold
POST /predict         {"text": "...", "threshold": 0.13}
POST /batch           {"sentences": [...], "threshold": 0.13}   (max 500)
```

Response format:

```json
{
  "text": "Wolbachia pipientis infects Drosophila melanogaster.",
  "label": 1,
  "probability": 0.9412,
  "threshold_used": 0.13
}
```

## Deployment

See [`deploy/`](deploy/) for the Ansible playbook.

```bash
# Edit inventory and group_vars
cp deploy/inventory.yml deploy/inventory.local.yml
$EDITOR deploy/inventory.local.yml
$EDITOR deploy/group_vars/all.yml

ansible-playbook -i deploy/inventory.local.yml deploy/playbook.yml
```

The playbook:
1. Installs `uv` on the target host
2. Pulls and installs this package from GitHub (`uv add git+…`)
3. Downloads `interaction_dict.csv` from the classifier repo
4. Deploys `config.toml` from the Jinja2 template
5. Installs and starts a systemd service (`biotic-pipeline.service`) running `uv run biotic-api`

Model weights must be copied separately (set `biotic_copy_model: true` and `biotic_model_source`
in `group_vars/all.yml`, or symlink an existing directory with `biotic_link_model: true`).
The model checkpoint is `classifier/models/multitask/full_typed_a05_ner2`.
