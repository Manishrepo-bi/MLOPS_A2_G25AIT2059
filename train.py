"""
train.py — Model loading, W&B integration, Trainer setup, and training loop.
"""

import pickle

import wandb
from transformers import (
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
)

from utils import (
    CACHED_MODEL_DIR,
    DEVICE,
    EVAL_BATCH_SIZE,
    LEARNING_RATE,
    LOGGING_STEPS,
    MAX_LENGTH,
    MODEL_NAME,
    NUM_EPOCHS,
    TRAIN_BATCH_SIZE,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
    WARMUP_STEPS,
    WEIGHT_DECAY,
    compute_metrics,
)
from data import download_all_genres, split_data, encode_data


def train():
    """Full training pipeline: data → model → W&B → Trainer → train → save."""

    # ── 1. Prepare data ──────────────────────────────
    print("=" * 60)
    print("Step 1: Preparing data...")
    print("=" * 60)

    # Try loading cached data first
    try:
        genre_reviews = pickle.load(open("genre_reviews_dict.pickle", "rb"))
        print("Loaded cached genre_reviews_dict.pickle")
    except FileNotFoundError:
        print("No cached data found. Downloading...")
        genre_reviews = download_all_genres()
        pickle.dump(genre_reviews, open("genre_reviews_dict.pickle", "wb"))

    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)
    tokenizer, train_dataset, test_dataset, label2id, id2label = encode_data(
        train_texts, train_labels, test_texts, test_labels
    )

    # ── 2. Initialise W&B ────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Initialising Weights & Biases...")
    print("=" * 60)

    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "model": MODEL_NAME,
            "epochs": NUM_EPOCHS,
            "batch_size": TRAIN_BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_length": MAX_LENGTH,
            "dataset": "UCSD Goodreads",
        },
    )

    # ── 3. Load pre-trained model ────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Loading pre-trained model...")
    print("=" * 60)

    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    ).to(DEVICE)

    print(f"Model loaded on device: {DEVICE}")

    # ── 4. Training arguments ────────────────────────
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_steps=LOGGING_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        run_name=WANDB_RUN_NAME,
    )

    # ── 5. Create Trainer & train ────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Training...")
    print("=" * 60)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    # ── 6. Save model & tokenizer ────────────────────
    print("\n" + "=" * 60)
    print("Step 5: Saving model...")
    print("=" * 60)

    trainer.save_model(CACHED_MODEL_DIR)
    tokenizer.save_pretrained(CACHED_MODEL_DIR)
    print(f"Model and tokenizer saved to: {CACHED_MODEL_DIR}")

    # ── 7. End W&B run ───────────────────────────────
    wandb.finish()

    print("\nTraining complete!")
    return trainer, model, tokenizer, test_dataset, test_labels, id2label


if __name__ == "__main__":
    train()
