"""
utils.py — Shared helpers: label maps, dataset class, compute metrics, and configuration.
"""

import torch
from sklearn.metrics import accuracy_score, f1_score


# ──────────────────────────────────────────────
# Configuration / Hyperparameters
# ──────────────────────────────────────────────
MODEL_NAME = "distilbert-base-cased"
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHED_MODEL_DIR = "distilbert-reviews-genres"

# Training hyperparameters
NUM_EPOCHS = 3
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 3e-5
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
LOGGING_STEPS = 50
SAMPLE_SIZE = 2000   # reviews to sample per genre from raw data
REVIEWS_PER_GENRE = 1000  # reviews to use per genre after sampling
TRAIN_SPLIT = 800    # out of REVIEWS_PER_GENRE

# W&B configuration
WANDB_PROJECT = "mlops-assignment2"
WANDB_RUN_NAME = "distilbert-run-1"

# Genre URLs for UCSD Goodreads dataset
GENRE_URL_DICT = {
    "poetry": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_poetry.json.gz",
    "children": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_children.json.gz",
    "comics_graphic": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_comics_graphic.json.gz",
    "fantasy_paranormal": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_fantasy_paranormal.json.gz",
    "history_biography": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_history_biography.json.gz",
    "mystery_thriller_crime": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_mystery_thriller_crime.json.gz",
    "romance": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_romance.json.gz",
    "young_adult": "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_reviews_young_adult.json.gz",
}


# ──────────────────────────────────────────────
# Label mapping helpers
# ──────────────────────────────────────────────
def build_label_maps(labels):
    """Build label2id and id2label dictionaries from a list of labels."""
    unique_labels = sorted(set(labels))
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


# ──────────────────────────────────────────────
# Custom PyTorch Dataset
# ──────────────────────────────────────────────
class ReviewDataset(torch.utils.data.Dataset):
    """Custom PyTorch dataset for tokenized reviews."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# ──────────────────────────────────────────────
# Compute metrics for Trainer
# ──────────────────────────────────────────────
def compute_metrics(pred):
    """Return accuracy and weighted F1 for HuggingFace Trainer."""
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }
