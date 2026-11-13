# Perturb Engine — Demo

Interactive demo of a rule-based engine that rewrites education questions without changing their meaning.
Each rewrite is checked before it is kept: hard checks (numbers, negations, protected text), sentence-embedding
similarity (SBERT) and natural-language inference (DeBERTa-v3-large). Runs locally and offline after setup.

| Axis | Example |
|---|---|
| Lexical | *How do enzymes **affect** the rate…* → *How do enzymes **influence** the rate…* |
| Syntactic | *Can a doctor treat a patient… **if it is an emergency**?* → ***If it is an emergency**, can a doctor…?* |
| Pragmatic | neutral opener, typo, instruction prefix, distracting sentence |

## Requirements

Windows 10/11, Python 3.12, 8 GB RAM, 4 GB free disk. Internet is needed once, for setup.

## Setup

```
git clone https://github.com/utk1college/Perturb-Engine-Demo.git
cd Perturb-Engine-Demo
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl
.venv\Scripts\python download_models.py
```

`download_models.py` fetches the two checking models (about 2 GB) at the exact versions the engine uses.

## Run

```
.venv\Scripts\python demo_ui\app.py
```

Wait for `Ready ... http://127.0.0.1:8765` (about 30 s), then open **http://127.0.0.1:8765**.
Preset questions are precomputed about a minute after startup and then load instantly; typed questions take
5–15 s on CPU. Stop with `Ctrl + C`. Options: `--port 8000`, `--device cuda`.

## Troubleshooting

| Problem | Fix |
|---|---|
| `py` is not recognised | Install Python 3.12 from python.org with "Add to PATH" ticked |
| `Can't find model 'en_core_web_lg'` | Run the spaCy install line from Setup again |
| Missing-model or offline error at startup | Run `.venv\Scripts\python download_models.py` with internet |
| Port already in use | `.venv\Scripts\python demo_ui\app.py --port 8000` |

## Structure

| Path | Contents |
|---|---|
| `perturb_v3/engine.py` | rewording rules |
| `perturb_v3/checks.py` | hard checks, SBERT and NLI scoring |
| `perturb_v3/config.json` | thresholds and pinned model versions |
| `src/pragmatic_perturbation.py` | pragmatic strategies (typos, prefixes, distractors) |
| `demo_ui/app.py` | local web server |
| `demo_ui/demo_engine.py`, `demo_rules.py`, `phase1_bridge.py` | connect the page to the engine |
| `demo_ui/static/` | the page and bundled fonts (SIL Open Font License) |

Full project: [utk1college/Perturb-Engine](https://github.com/utk1college/Perturb-Engine).
