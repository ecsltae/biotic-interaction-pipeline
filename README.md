# biotic-interaction-pipeline

Sentence-level biotic interaction detector for bulk article processing.

Detects sentences describing ecological/parasitic/symbiotic interactions between two species (e.g. "Wolbachia pipientis infects Drosophila melanogaster").

## Architecture

Two-stage pipeline:

```
text
 └─► sentence splitter
       └─► GloBI pre-filter  (~1ms/sentence, pure Python, near-zero FN)
             └─► BiomedBERT classifier  (~14ms/sentence on GPU)
                   └─► results.csv
```

**Pre-filter** passes a sentence if it contains either:
1. A known interaction term (GloBI vocabulary + curated biomedical stems: *infect, parasit, host, pathogen, malaria, HIV, …*)
2. A binomial species name (Aho-Corasick lookup over 4.2M names)

**Classifier** is a BiomedBERT-base model distilled from an ensemble teacher (EP-relax F1=0.808 at threshold=0.25).

## Setup

```bash
pip install -r requirements.txt
```

PyAhocorasick is optional but strongly recommended for the species-name fallback.

## Running the API

```bash
export MODEL_DIR=/path/to/distilled_BiomedBERT_v2
python api/app.py
# → http://localhost:8003/docs
```

Or with Docker:
```bash
cp .env.example .env          # edit MODEL_DIR
docker build -t biotic-pipeline .
docker run --gpus all --env-file .env \
  -v /host/models:/models \
  -p 8003:8003 biotic-pipeline
```

## Processing articles

```bash
# Folder of .txt files
python process_articles.py \
  --input  /data/articles/ \
  --output results.csv \
  --api    http://localhost:8003

# CSV with a text column
python process_articles.py \
  --input    abstracts.csv \
  --text-col abstract \
  --output   results.csv

# With custom GloBI/species dictionaries (improves filter recall)
python process_articles.py \
  --input            articles/ \
  --output           results.csv \
  --interaction-dict data/interaction_dict.csv \
  --species-dict     data/species_dict.csv
```

Output CSV columns: `article_id, sentence, label, probability, threshold_used`

## Model

The classifier is not included in this repo (binary weights are too large for git).
Download or copy the model directory and set `MODEL_DIR` accordingly.

Source: [biotic-interaction-classifier](https://github.com/ecsltae/biotic-interaction-classifier)
Relevant model: `models/distilled_BiomedBERT_v2`

## API reference

```
GET  /health          model info + device
POST /predict         {"text": "...", "threshold": 0.25}
POST /batch           {"sentences": [...], "threshold": 0.25}  (max 500)
```

## Deployment (Ansible/Jenkins)

Key variables for your inventory:
- `model_dir`: path on target host where model weights are placed
- `port`: API port (default 8003)
- `device`: `cuda:0` or `cpu`

The API process should be managed by systemd or a process supervisor.
A `/health` endpoint is available for readiness probes.
