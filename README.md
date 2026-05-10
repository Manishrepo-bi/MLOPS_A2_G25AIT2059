# MLOps Assignment 2 — DistilBERT Goodreads Genre Classifier

Fine-tuning a DistilBERT model to classify Goodreads book reviews into genres, with full experiment tracking via Weights & Biases and model publishing to Hugging Face Hub.

**Course:** MLOps | PGD AI Programme | IIT Jodhpur

---

## Project Overview

This project builds a complete MLOps workflow around a text classification task. Given a Goodreads book review, the model predicts which of 7 genres the reviewed book belongs to — poetry, comics & graphic, fantasy & paranormal, history & biography, mystery/thriller/crime, romance, or young adult.

The focus is not on achieving perfect accuracy, but on operationalizing the model: tracking experiments, versioning artifacts, and publishing a reproducible, publicly accessible model.

---

## Repository Structure

```
├── data.py            # Data loading, sampling, train/test split, encoding
├── train.py           # Model loading, W&B init, Trainer setup, training loop, HF push
├── eval.py            # Evaluation, metrics, classification report, W&B artifact upload
├── utils.py           # Shared helpers: label maps, Dataset class, compute_metrics
├── requirements.txt   # Python dependencies
└── README.md
```

---

## Model Selection

We use **DistilBERT** (`distilbert-base-cased`) from Hugging Face. DistilBERT is a distilled (compressed) version of the original BERT model — it is 40% smaller and 60% faster while retaining ~97% of BERT's language understanding capability. For this assignment it is an ideal choice: it fine-tunes quickly on a free GPU (Kaggle P100), handles long review text up to 512 tokens, and is well-documented with strong community support. The cased variant was chosen because book reviews contain meaningful capitalization (author names, proper nouns, genre-specific terminology) that the model can use as a signal.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/your-username/mlops-assignment2.git
cd mlops-assignment2
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

You need a [Weights & Biases](https://wandb.ai) account and a [Hugging Face](https://huggingface.co) account (with a Write token).

```bash
export WANDB_API_KEY=your_wandb_api_key
export HF_TOKEN=your_huggingface_write_token
```

On **Kaggle**, add these as Secrets (Settings → Add-ons → Secrets) and load them at the top of your notebook:

```python
from kaggle_secrets import UserSecretsClient
import os

secrets = UserSecretsClient()
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
os.environ["HF_TOKEN"]      = secrets.get_secret("HF_TOKEN")
```

> **Note:** Kaggle requires Internet to be enabled in notebook Settings for data download and model push to work.

### 4. Update your Hugging Face repo name

In `train.py`, replace line 20 with your actual HF username:

```python
HF_REPO_NAME = 'your-username/distilbert-goodreads-genres'
```

### 5. Run training

```bash
python train.py
```

This will: download the data → encode it → train the model → log everything to W&B → push the model to Hugging Face Hub.

### 6. Run evaluation (optional standalone)

```bash
python eval.py
```

---

## Results

| Metric     | Score |
|------------|-------|
| Accuracy   | 0.XX  |
| F1 Score   | 0.XX  |
| Eval Loss  | 0.XX  |

> Fill in your actual scores after training completes.

---

## Links

- **Hugging Face model:** https://huggingface.co/your-username/distilbert-goodreads-genres
- **W&B project dashboard:** https://wandb.ai/your-username/mlops-assignment2

> Replace the placeholder links above with your actual URLs after completing training.

---

## Training Summary

The model was fine-tuned for 3 epochs using the Hugging Face `Trainer` API with the following key hyperparameters:

| Parameter              | Value   |
|------------------------|---------|
| Model                  | distilbert-base-cased |
| Epochs                 | 3       |
| Train batch size       | 16      |
| Eval batch size        | 32      |
| Learning rate          | 3e-5    |
| Warmup steps           | 100     |
| Weight decay           | 0.01    |
| Max token length       | 512     |
| Samples per genre      | 1000    |
| Train / test split     | 800 / 200 per genre |

All runs are fully tracked on W&B — training loss, validation loss, accuracy, F1, learning rate schedule, and GPU utilization are logged automatically.

---

## Dataset

Reviews sourced from the [UCSD Book Graph](https://mengtingwan.github.io/data/goodreads.html) — a large dataset of Goodreads reviews organized by genre. 1000 reviews per genre are sampled (800 train, 200 test), covering 7 genres for a total of 5600 training and 1400 test examples.
