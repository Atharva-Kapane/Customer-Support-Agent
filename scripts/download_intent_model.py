"""Download the DistilBERT intent classifier 
into models/intent_classifier_distilbert."""

from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "models" / "intent_classifier_distilbert"
REPO = "Atharva-Kapane/twcs-distilbert-intent-classifier"


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {REPO} -> {DEST}")
    snapshot_download(repo_id=REPO, local_dir=str(DEST))
    print("Done.")


if __name__ == "__main__":
    main()