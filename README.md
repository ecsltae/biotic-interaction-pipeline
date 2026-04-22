# biotic-interaction-pipeline

Sentence-level biotic interaction detector for bulk article processing.

Detects sentences describing ecological/parasitic/symbiotic interactions between two species
(e.g. "Wolbachia pipientis infects Drosophila melanogaster").

## Architecture

```
text
 └─► sentence splitter
       └─► GloBI pre-filter  (~1ms/sentence, pure Python, near-zero FN)
             └─► BiomedBERT classifier API  (~14ms/sentence on GPU)
                   └─► results.csv
```

**Pre-filter** passes a sentence if it contains either:
1. A known interaction term (GloBI vocabulary + curated biomedical stems: *infect, parasit, host, pathogen, malaria, HIV, …*)
2. A binomial species name (Aho-Corasick lookup over 4.2M names)

**Classifier** is a BiomedBERT-base model distilled from an ensemble teacher (EP-relax F1=0.808 at threshold=0.25).

Source for model training: [biotic-interaction-classifier](https://github.com/ecsltae/biotic-interaction-classifier)

## Local setup

```bash
# Install with uv
uv sync

# Edit config.toml — set model.model_dir
cp config.toml config.toml.local
$EDITOR config.toml

# Start the API
uv run biotic-api
# → http://localhost:8003/docs

# Process a folder of articles
uv run python process_articles.py \
  --input  /data/articles/ \
  --output results.csv
```

## Configuration

All settings live in `config.toml`. In production, Ansible deploys this from
`deploy/templates/config.toml.j2` filled with values from `deploy/group_vars/all.yml`.

```toml
[server]
host = "0.0.0.0"
port = 8003

[model]
model_dir = "/opt/biotic-pipeline/model"   # path to HuggingFace weights
device = "auto"
threshold = 0.25

[data]
interaction_dict = "data/interaction_dict.csv"   # GloBI terms (~30KB)
species_dict = "data/species_dict.csv"           # 4.2M binomials (~150MB, optional)
```

## API reference

```
GET  /health          model info + device
POST /predict         {"text": "...", "threshold": 0.25}
POST /batch           {"sentences": [...], "threshold": 0.25}   (max 500)
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
