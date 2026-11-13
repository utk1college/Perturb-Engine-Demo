"""Download the two checking models at the versions pinned in perturb_v3/config.json (run once, needs internet)."""
import json
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

cfg = json.loads((Path(__file__).parent / "perturb_v3" / "config.json").read_text(encoding="utf-8"))["models"]

print("Downloading SBERT:", cfg["sbert"]["name"])
SentenceTransformer(cfg["sbert"]["name"], revision=cfg["sbert"]["revision"])

print("Downloading NLI:", cfg["nli"]["name"])
AutoTokenizer.from_pretrained(cfg["nli"]["name"], revision=cfg["nli"]["revision"])
AutoModelForSequenceClassification.from_pretrained(cfg["nli"]["name"], revision=cfg["nli"]["revision"])

print("Done. The demo now runs offline.")
