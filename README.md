# GeoVisionists--Land-Classification

Project Overview
This repository contains the code LSTM-based satellite imagery forecasting and land-cover change detection on the BigEarthNet-S2 Portugal subset. Our model learns from 11-year sequences of multispectral feature vectors (per-band means) and predicts land-cover classes one year ahead.

Repository Contents
scripts/ – Training, evaluation, data-loading, and filtering scripts

model.py – LSTMClassifier LightningModule definition

data_loader.py – BigEarthNetDataset and collate function

filter_portugal.py – Utility to filter the global metadata to Portugal

train.py – Entry point for training (with subset, hyperparameters, checkpointing)

evaluate.py – Runs evaluation on saved checkpoints

metadata.parquet – Portugal metadata (patch IDs, labels)

requirements.txt – Python dependencies

GeoVisionists.pdf – Final report in PDF

CODE_OF_CONDUCT.md, LICENSE (if included)

Dependencies
Tested on Python 3.8+
Key libraries (see requirements.txt for full list):

torch

torchvision

pytorch_lightning

pandas

rasterio

numpy

Install via:

bash
Copy
Edit
pip install -r requirements.txt
Data Preparation
Download Sentinel-2 patches into BigEarthNet-S2/ (not included in this repo).

Ensure metadata.parquet sits in the project root alongside scripts/.

Usage
1. Filter Portugal (optional)
bash
Copy
Edit
python -m scripts.filter_portugal \
  --metadata global_metadata.parquet \
  --output metadata.parquet \
  --region PORTUGAL
2. Train
Train on a random subset (e.g., 5 000 samples), with custom hyperparameters:

bash
Copy
Edit
python -m scripts.train \
  --data_root ./BigEarthNet-S2 \
  --metadata ./metadata.parquet \
  --subset_size 5000 \
  --batch_size 16 \
  --num_workers 4 \
  --max_epochs 10 \
  --hidden_dim 64 \
  --num_layers 2 \
  --dropout 0.3 \
  --lr 1e-3
Outputs checkpoints and CSV logs under lightning_logs/.

3. Evaluate
Point at your best checkpoint to compute metrics on the held-out validation set:

bash
Copy
Edit
python -m scripts.evaluate \
  --checkpoint lightning_logs/version_0/checkpoints/best.ckpt \
  --data_root ./BigEarthNet-S2 \
  --metadata ./metadata.parquet
Code Structure & Highlights
BigEarthNetDataset reads per-patch multispectral TIFFs, resamples bands to a common grid, computes per-band means.

LSTMClassifier subclasses pl.LightningModule:

Two-layer nn.LSTM (hidden_size=--hidden_dim, layers=--num_layers, dropout)

Linear head for 19 land-cover logits

BCEWithLogitsLoss for multi-label forecasting

AdamW(lr=--lr, weight_decay=1e-4) + ReduceLROnPlateau scheduler

Training uses mixed-precision (precision="16-mixed"), CPU/GPU auto-selection, train/val split 90/10.

Citation & Acknowledgments
BigEarthNet-S2 dataset – Sentinel-2 satellite imagery → https://bigearth.net

Urban Growth in Jakarta – NASA Earth Observatory

Generalizable Satellite+ML – ResearchGate

LSTM Architecture – ResearchGate

Sequence Modeling Benchmarks – Bai et al. (2018)

Please see GeoVisionists.pdf for full references and detailed results.
