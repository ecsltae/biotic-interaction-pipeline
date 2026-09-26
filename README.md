# biotic-interaction-pipeline

Sentence-level biotic interaction detector for bulk article processing.

Detects sentences describing ecological/parasitic/symbiotic interactions between two species
(e.g. "Wolbachia pipientis infects Drosophila melanogaster").

## Two paths, two decision units

This package now ships two models, and they answer different questions.

```
                                        ┌─► TripleVerifier  (v3, recommended)
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
expert-graded benchmark, against the sentence-level model's own recorded decisions:

| | precision | recall | F1 |
|---|---|---|---|
| **TripleVerifier** @ 0.50 | **0.852** | **0.959** | **0.902** |
| BioticClassifier (legacy) | 0.850 | 0.785 | 0.816 |

Same precision, seventeen points more recall, McNemar p = 2.3e-04. It also returns the
argument direction for free — same model, same speed, ~34 candidates/s on 8 CPU threads.

**The catch, stated plainly.** `TripleVerifier` takes a *candidate triple*, not a
sentence. This package does not yet generate candidates: the `text → sentences →
pre-filter` stages produce sentences, and something must turn those into
(taxon, relation, taxon) tuples before the verifier can score them. If you already
have a rule layer that emits candidates — two taxon surface forms and an interaction
surface form per candidate — feed it straight in. If you do not, `BioticClassifier`
remains the end-to-end path.

Swapping the checkpoint path in `config.toml` will **not** upgrade you: the decision
unit is different, and a sentence alone cannot be scored by the verifier.

### TripleVerifier

```python
from biotic_pipeline import TripleVerifier

v = TripleVerifier("path/to/joint_a05_s1")
v.verify("Wolbachia", "infects", "Drosophila melanogaster",
         "Wolbachia pipientis infects Drosophila melanogaster.")
# {'interacts': 1, 'p_interact': 0.9992, 'direction': 'FORWARD',
#  'p_species1_is_subject': 0.9368, 'both_taxa_located': 1, ...}

v.verify_batch([(s1, rel, s2, passage), ...], threshold=0.88)
```

Order does not matter: give the same pair the other way round and `p_interact` is
bit-identical while `direction` flips. That is architectural, not learned.

`direction` is `FORWARD` (species1 is the subject), `REVERSE`, or `UNCERTAIN`, and is
judged against the **canonical** relation — so `Leptodora --prey--> Bosmina` returns
`REVERSE`, because the canonical relation is *preyed upon by*.

**Thresholds.** 0.50 is the default and is validated: three independent held-out dev
splits each chose ≈0.5 for maximum F1. To buy precision:

| threshold | precision | recall | F1 |
|---|---|---|---|
| 0.50 | 0.852 | 0.959 | 0.902 |
| 0.88 | 0.871 | 0.907 | 0.888 |
| 0.95 | 0.882 | 0.878 | 0.880 |
| 0.99 | 0.924 | 0.744 | 0.824 |

**When to distrust it.** `both_taxa_located = 0` means a taxon string was not found in
the passage and the answer is much weaker. `unknown_polarity = 1` means the relation is
outside the polarity lexicon and the direction fell back to a default. The main residual
error is co-occurrence in a shared host — two organisms both related to a third can read
as interacting. Upstream entity errors are not fixed here.

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
