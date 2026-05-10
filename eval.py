"""
eval.py — Evaluation, metrics, saving results, and W&B artifact upload.
"""

import json
import pickle

import wandb
from sklearn.metrics import classification_report
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

from utils import (
    CACHED_MODEL_DIR,
    DEVICE,
    EVAL_BATCH_SIZE,
    MODEL_NAME,
    WANDB_PROJECT,
    WANDB_RUN_NAME,
    compute_metrics,
)
from data import split_data, encode_data


def evaluate():
    """Load saved model, evaluate on test set, log metrics & artifact to W&B."""

    # ── 1. Prepare data ──────────────────────────────
    print("=" * 60)
    print("Step 1: Preparing test data...")
    print("=" * 60)

    genre_reviews = pickle.load(open("genre_reviews_dict.pickle", "rb"))
    train_texts, train_labels, test_texts, test_labels = split_data(genre_reviews)
    tokenizer, train_dataset, test_dataset, label2id, id2label = encode_data(
        train_texts, train_labels, test_texts, test_labels
    )

    # ── 2. Load saved model ──────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Loading saved model...")
    print("=" * 60)

    model = DistilBertForSequenceClassification.from_pretrained(CACHED_MODEL_DIR).to(DEVICE)
    print(f"Model loaded from: {CACHED_MODEL_DIR} on device: {DEVICE}")

    # ── 3. Initialise W&B ────────────────────────────
    wandb.init(
        project=WANDB_PROJECT,
        name=f"{WANDB_RUN_NAME}-eval",
        config={"model": MODEL_NAME, "phase": "evaluation"},
    )

    # ── 4. Evaluate ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Running evaluation...")
    print("=" * 60)

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate()
    print(f"\nEvaluation results: {eval_results}")

    # ── 5. Log final metrics to W&B ──────────────────
    wandb.log({
        "final/loss": eval_results["eval_loss"],
        "final/accuracy": eval_results["eval_accuracy"],
        "final/f1": eval_results["eval_f1"],
    })

    # ── 6. Classification report ─────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Generating classification report...")
    print("=" * 60)

    preds = trainer.predict(test_dataset).predictions.argmax(-1)
    pred_labels = [id2label[l] for l in preds.flatten().tolist()]

    report = classification_report(
        test_labels, pred_labels, target_names=list(id2label.values()), output_dict=True
    )
    print(classification_report(test_labels, pred_labels, target_names=list(id2label.values())))

    # Save report to JSON
    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved eval_report.json")

    # ── 7. Upload as W&B Artifact ────────────────────
    artifact = wandb.Artifact("eval-report", type="evaluation")
    artifact.add_file("eval_report.json")
    wandb.log_artifact(artifact)
    print("Uploaded eval_report.json as W&B artifact")

    # ── 8. Print summary ─────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Accuracy : {eval_results['eval_accuracy']:.4f}")
    print(f"  F1 Score : {eval_results['eval_f1']:.4f}")
    print(f"  Eval Loss: {eval_results['eval_loss']:.4f}")

    wandb.finish()
    print("\nEvaluation complete!")

    return eval_results


if __name__ == "__main__":
    evaluate()
