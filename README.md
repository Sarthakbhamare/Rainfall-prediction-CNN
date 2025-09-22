# Rainfall Prediction using Convolutional Neural Networks (CNN)

Authors:  
- Sarthak Bhamare  
- Prasad Kumbhar

Repository: [Rainfall-prediction-CNN](https://github.com/Sarthakbhamare/Rainfall-prediction-CNN)  
Date: 22.09.2025

---

## 1) Executive Summary

This project builds a Convolutional Neural Network (CNN) model to predict rainfall from meteorological data. It organizes data, experimentation, and model training within a reproducible workflow based on a Jupyter notebook.

- Objective: Predict rainfall (amount/intensity or rain/no-rain) using spatial patterns captured by CNNs.
- Approach: Preprocess and explore the dataset, train a CNN in a notebook workflow, and evaluate using suitable metrics (regression or classification).
- Outcomes: A compact, notebook-driven pipeline that can be extended to full scripts and MLOps tooling.

---

## 2) Overall Repository Structure

At the time of review, the repository contains the following top-level items:
- Dataset/ — top-level folder intended for data
- cnn_on_rainfall_forcasting.ipynb — Jupyter notebook implementing the end‑to‑end workflow

Recommended organization (you can adopt incrementally):
- data/
  - raw/ — original source data
  - processed/ — cleaned, aligned, split datasets
- notebooks/
  - EDA.ipynb — exploratory analysis and data checks
  - Train_CNN.ipynb — model training and evaluation
- src/
  - data_loader.py — loading, cleaning, feature engineering
  - model.py — CNN architecture definition
  - train.py — training loop, checkpoints, early stopping
  - eval.py — metrics, error analysis, plots
- models/
  - checkpoints/ — intermediate checkpoints
  - best_model.[h5|pt] — best-performing model artifact
- reports/
  - figures/ — loss curves, confusion matrix, example predictions
  - results.md — experiment summaries
- requirements.txt or environment.yml — dependency management
- README.md — quick start, data notes, how to run

---

## 3) Technology Stack

The project is notebook-centric and Python-based.

- Language and Environment:
  - Python (version as per your environment)
  - Jupyter Notebook for interactive experimentation
- Core Libraries (typical for CNN rainfall workflows; confirm in the notebook):
  - Data/Science: NumPy, Pandas, (Xarray/netCDF4 if gridded), Matplotlib/Seaborn
  - Deep Learning: TensorFlow/Keras or PyTorch
  - Utilities: Scikit‑learn for metrics/splits; possibly Rasterio/GeoPandas if geospatial rasters are used
- Hardware:
  - CPU or GPU (CUDA-enabled GPU recommended for faster training)

---

## 4) Data Explanation

Describe your dataset clearly so others can reproduce results and assess limitations.

- Source and Coverage:
  - Spatial: [Region/extent and resolution, e.g., 0.1° grid or H×W pixels]
  - Temporal: [Date range and cadence, e.g., daily/hourly; train/val/test windows]
- Inputs (features):
  - Example: Multi-channel satellite/radar images, meteorological fields (temperature, humidity, wind), past rainfall lags
- Targets (labels):
  - Regression: Rainfall amount (mm)
  - Classification: Rain/no-rain or rainfall classes/bins
- Data Quality and Handling:
  - Missingness: [masking/interpolation/exclusion strategy]
  - Imbalance: [class distribution or skew; re-weighting or sampling if used]
  - Normalization: [per-channel standardization/min-max]

Tip: Document exact file formats placed under `Dataset/` (e.g., CSV/NetCDF/GeoTIFF) and any folder hierarchy (e.g., train/val/test).

---

## 5) Preprocessing and EDA

A clear preprocessing pipeline is essential for consistent training.

- Ingestion: Load raw files from `Dataset/` (paths documented in the notebook).
- Cleaning:
  - Handle missing values (impute, mask, or drop)
  - Remove outliers if justified
- Alignment:
  - Spatial: resample/reproject to a common grid if needed
  - Temporal: align timestamps across data sources
- Feature Engineering:
  - Channel stacking, lag features, derived indices
- Scaling:
  - Normalize inputs; consider log-transform or clipping for heavy-tailed rainfall
- Splitting:
  - Train/Validation/Test by time (preferred for forecasting) or by geography to reduce leakage
- EDA:
  - Plot distributions, correlations, seasonal cycles, and class balance

---

## 6) Implemented Model Used

The implemented model is a Convolutional Neural Network (CNN) defined and trained inside `cnn_on_rainfall_forcasting.ipynb`.

- Problem Formulation:
  - Regression (forecast amount) or Classification (rain event/category)
- Loss and Metrics:
  - Regression: MSE/MAE/Huber; RMSE, MAE, R², correlation
  - Classification: BCE/CE/Focal; accuracy, precision/recall/F1, ROC AUC, PR AUC
- Optimization:
  - Optimizer: Adam/AdamW with an initial learning rate (e.g., 1e‑3)
  - Schedulers: ReduceLROnPlateau/Cosine decay (optional)
  - Regularization: Dropout, L2 weight decay, early stopping

---

## 7) CNN Architecture

A typical and effective CNN stack for gridded meteorological inputs is outlined below. Align these elements with the notebook’s exact implementation.

- Input:
  - Tensor of shape [H × W × C] (e.g., 128×128×3 for 3 channels)
- Convolutional Backbone:
  - Block 1: Conv2D(32, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  - Block 2: Conv2D(64, 3×3) → BatchNorm → ReLU → MaxPool(2×2)
  - Block 3: Conv2D(128, 3×3) → BatchNorm → ReLU → [Dropout optional]
- Head:
  - GlobalAveragePooling2D
  - Dense(128) → ReLU → Dropout(0.3–0.5)
  - Output:
    - Regression: Dense(1), linear
    - Classification: Dense(K), softmax or sigmoid
- Rationale:
  - CNNs capture local spatial features relevant for convective structures and precipitation patterns.

If temporal sequences are involved, consider:
- 3D CNNs on [T × H × W × C], or
- CNN feature extractor per time step + LSTM/Temporal Convolution.

---

## 8) Training Configuration

- Hyperparameters:
  - Batch size: [e.g., 16–64]
  - Epochs: [e.g., 50–200 with early stopping]
  - Learning rate: [initial value and scheduler]
  - Regularization: [dropout rate, weight decay]
- Reproducibility:
  - Set random seeds (Python, NumPy, DL framework)
  - Freeze versions in `requirements.txt` or `environment.yml`
- Hardware and Runtime:
  - Device: [CPU/GPU model]
  - Training time: [per epoch and total]

---

## 9) Evaluation, Results, and Visuals

Report what matters to users and stakeholders.

- Validation Strategy:
  - Time-based or spatially stratified splits to avoid leakage
  - Optional cross-validation (chronological folds)
- Baselines:
  - Persistence (last value), climatology mean/quantiles, simple linear/RF baseline
- Key Metrics:
  - Regression: RMSE, MAE, R², correlation
  - Classification: F1, ROC AUC, PR AUC; confusion matrix
- Visualizations:
  - Loss/metric curves (train vs. val)
  - Predictions vs. ground truth (maps or charts)
  - Error distributions and spatial error maps
- Findings:
  - [Summarize strengths and typical failure modes, e.g., extremes underestimation]

Insert figures under `reports/figures/` and link them here for portability.

---

## 10) Inference and Usage

- Inputs and Format:
  - Expected shape and normalization of input tensors/images
- Running Inference:
  - From notebook cells (provide example cell)
  - [Optional] `src/infer.py` for batch inference
- Outputs:
  - Predicted rainfall value or class with optional confidence
- Performance:
  - Latency per sample and throughput on typical hardware

---

## 11) Ethical, Safety, and Operational Considerations

- Responsible Use:
  - Communicate uncertainty, especially for high-impact decisions (flood alerts)
- Bias/Equity:
  - Data sparsity or sensor coverage can bias regional performance
- Robustness:
  - Monitor for distribution shifts (seasonal regime changes, sensor outages)
- Governance:
  - Version models, store configs and seeds, maintain experiment logs

---

## 12) Future Work

- Temporal modeling (3D CNNs or CNN+LSTM/TemporalConv)
- Multimodal fusion (topography, land use, atmospheric predictors)
- Probabilistic forecasting (quantile loss, calibration)
- Resolution improvements (super-resolution or multi-scale tiling)
- MLOps (data pipelines, artifact tracking, CI for evaluation)

---

## 13) How to Run

- Environment:
  - Create and activate environment; install dependencies (e.g., `pip install -r requirements.txt`)
- Data:
  - Place raw files under `Dataset/` (document folder layout)
- Notebook:
  - Open `cnn_on_rainfall_forcasting.ipynb`
  - Run preprocessing, training, and evaluation cells in sequence
- Artifacts:
  - Save best model weights to `models/` and figures to `reports/figures/`

---

## 14) Collaborator Setup (Add Prasad Kumbhar and Sarthak Bhamare)

Add collaborators through the repository settings (you must be the repo owner or have admin rights).

- Steps:
  - Go to the repo on GitHub → Settings → Collaborators (or “Collaborators and teams”)
  - Click “Add people”
  - Search and add:
    - “Prasad Kumbhar” (use exact GitHub username if different; confirm identity from profile)
    - “SarthakBhamare” (owner typically already has access; adding is not required if you are the owner)
  - Choose the appropriate role:
    - Write: push to branches
    - Maintain or Admin: broader control (use sparingly)
  - Send invitations and wait for acceptance

Note: If “Prasad Kumbhar” uses a different GitHub handle, search by username, profile URL, or email.

---

## Appendix A: Repository Snapshot (Detected)

- Dataset/  
- cnn_on_rainfall_forcasting.ipynb

(If additional files are added—requirements.txt, scripts, or README—list them here to keep this report in sync.)

---

## Appendix B: Configuration Checklist

- Random seeds set and recorded
- Exact data version and time window recorded
- Model hyperparameters saved with run
- Metric definitions and evaluation split documented
- Environment file committed for reproducibility
