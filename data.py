"""
data.py — Data loading, sampling, train/test split, and encoding.
"""

import gzip
import json
import pickle
import random

import requests
from transformers import DistilBertTokenizerFast

from utils import (
    GENRE_URL_DICT,
    MAX_LENGTH,
    MODEL_NAME,
    REVIEWS_PER_GENRE,
    SAMPLE_SIZE,
    TRAIN_SPLIT,
    ReviewDataset,
    build_label_maps,
)


# ──────────────────────────────────────────────
# Download & sample reviews
# ──────────────────────────────────────────────
def load_reviews(url, head=10000, sample_size=2000):
    """Stream reviews from a gzipped JSON URL and return a random sample."""
    reviews = []
    count = 0

    response = requests.get(url, stream=True)
    print(f"  HTTP {response.status_code}")
    with gzip.open(response.raw, "rt", encoding="utf-8") as file:
        for line in file:
            d = json.loads(line)
            reviews.append(d["review_text"])
            count += 1
            if head is not None and count >= head:
                break

    return random.sample(reviews, min(sample_size, len(reviews)))


def download_all_genres(genre_url_dict=GENRE_URL_DICT, sample_size=SAMPLE_SIZE):
    """Download and sample reviews for every genre. Returns a dict {genre: [reviews]}."""
    genre_reviews = {}
    for genre, url in genre_url_dict.items():
        print(f"Loading reviews for genre: {genre}")
        genre_reviews[genre] = load_reviews(url, head=10000, sample_size=sample_size)
    return genre_reviews


# ──────────────────────────────────────────────
# Train / test split
# ──────────────────────────────────────────────
def split_data(genre_reviews_dict, reviews_per_genre=REVIEWS_PER_GENRE, train_split=TRAIN_SPLIT):
    """Split genre_reviews_dict into train/test texts and labels."""
    train_texts, train_labels = [], []
    test_texts, test_labels = [], []

    for genre, reviews in genre_reviews_dict.items():
        sampled = random.sample(reviews, min(reviews_per_genre, len(reviews)))
        for review in sampled[:train_split]:
            train_texts.append(review)
            train_labels.append(genre)
        for review in sampled[train_split:]:
            test_texts.append(review)
            test_labels.append(genre)

    print(f"Train: {len(train_texts)} samples | Test: {len(test_texts)} samples")
    return train_texts, train_labels, test_texts, test_labels


# ──────────────────────────────────────────────
# Tokenization & dataset creation
# ──────────────────────────────────────────────
def encode_data(train_texts, train_labels, test_texts, test_labels, model_name=MODEL_NAME, max_length=MAX_LENGTH):
    """Tokenize texts, encode labels, and return PyTorch datasets + label maps."""
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

    print("Tokenizing training data...")
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    print("Tokenizing test data...")
    test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=max_length)

    label2id, id2label = build_label_maps(train_labels)

    train_labels_enc = [label2id[y] for y in train_labels]
    test_labels_enc = [label2id[y] for y in test_labels]

    train_dataset = ReviewDataset(train_encodings, train_labels_enc)
    test_dataset = ReviewDataset(test_encodings, test_labels_enc)

    return tokenizer, train_dataset, test_dataset, label2id, id2label


# ──────────────────────────────────────────────
# Main (run standalone to download & prepare data)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Download reviews
    genre_reviews = download_all_genres()

    # Save raw data as pickle for reuse
    pickle.dump(genre_reviews, open("genre_reviews_dict.pickle", "wb"))
    print("Saved genre_reviews_dict.pickle")

    # 2. Split
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)

    # 3. Encode & build datasets
    tokenizer, train_dataset, test_dataset, label2id, id2label = encode_data(
        train_texts, train_labels, test_texts, test_labels
    )

    print(f"Label map: {label2id}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size:  {len(test_dataset)}")
    print("Data preparation complete!")
