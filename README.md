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
