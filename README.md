# GeoVisionists – Land Classification

## Project Overview
This repository contains the code for an LSTM‐based satellite imagery forecasting and land‐cover change detection pipeline, applied to the BigEarthNet-S2 Portugal subset.  
Our model learns from 11-year sequences of multispectral feature vectors (per-band means) and predicts land-cover classes one year ahead.

---

## Repository Contents

- **scripts/**  
  - `data_loader.py` – `BigEarthNetDataset` and collate function  
  - `filter_portugal.py` – Utility to filter global metadata to Portugal  
  - `train.py` – Training entry point (supports subset, hyperparameters, checkpointing)  
  - `evaluate.py` – Evaluation on saved checkpoints  
- **model.py** – `LSTMClassifier` LightningModule definition  
- **metadata.parquet** – Portugal metadata (patch IDs, labels)  
- **requirements.txt** – Python dependencies  
- **GeoVisionists.pdf** – Final report in PDF  
- **CODE_OF_CONDUCT.md**, **LICENSE** (if included)  

---

## Dependencies

Tested on Python 3.8+. See `requirements.txt` for the full list, including:

- `torch`  
- `torchvision`  
- `pytorch_lightning`  
- `pandas`  
- `rasterio`  
- `numpy`

Install with:

```bash
pip install -r requirements.txt


Data Preparation
Download Sentinel-2 patches into BigEarthNet-S2/ (not included in this repo).

Place metadata.parquet in the project root (alongside scripts/).

Optional) Filter Portugal
python -m scripts.filter_portugal \
  --metadata global_metadata.parquet \
  --output metadata.parquet \
  --region PORTUGAL


Train
Train on a random subset (e.g., 5 000 samples) with custom hyperparameters:
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

Evaluate
Compute metrics on the held-out validation set:
python -m scripts.evaluate \
  --checkpoint lightning_logs/version_0/checkpoints/best.ckpt \
  --data_root ./BigEarthNet-S2 \
  --metadata ./metadata.parquet
