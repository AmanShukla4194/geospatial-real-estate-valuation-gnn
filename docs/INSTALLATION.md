# Installation Guide

## Geospatial Real Estate Valuation using Graph Neural Networks

Author: Aman Shukla

Version: 1.0.0

---

# Overview

This document explains how to set up and execute the Geospatial Real Estate Valuation project from a clean environment.

The project predicts residential property prices using both traditional Machine Learning and Graph Neural Networks while incorporating geospatial relationships between neighbouring properties.

---

# System Requirements

## Operating System

- Windows 10 / Windows 11
- Ubuntu 22.04+
- macOS 13+

---

## Python

Python 3.13 or later

Verify your installation:

```bash
python --version
```

---

## Git

Verify Git installation:

```bash
git --version
```

---

# Clone the Repository

```bash
git clone https://github.com/AmanShukla4194/geospatial-real-estate-valuation-gnn.git
```

Move into the project directory:

```bash
cd geospatial-real-estate-valuation-gnn
```

---

# Create a Virtual Environment

Windows

```powershell
python -m venv .venv
```

Linux/macOS

```bash
python3 -m venv .venv
```

---

# Activate the Virtual Environment

Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows CMD

```cmd
.venv\Scripts\activate.bat
```

Linux/macOS

```bash
source .venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Dataset

The repository does not include the dataset because large datasets are excluded from Git version control.

Create the following directory if it does not exist:

```
data/raw/
```

Place the King County Housing Dataset inside:

```
data/raw/kc_house_data.csv
```

---

# Project Execution Workflow

Execute the project modules in the following order.

---

## 1. Data Cleaning

```bash
python -m src.preprocessing.data_cleaning
```

---

## 2. Feature Engineering

```bash
python -m src.features.feature_engineering
```

---

## 3. Baseline XGBoost Training

```bash
python -m src.training.train_baseline_model
```

---

## 4. Baseline Model Evaluation

```bash
python -m src.training.model_evaluation
```

---

## 5. Spatial Embedding Generation

```bash
python -m src.embeddings.spatial_embeddings
```

---

## 6. Graph Neural Network Training

```bash
python -m src.models.train_gnn
```

---

## 7. Graph Neural Network Evaluation

```bash
python -m src.models.evaluate_gnn
```

---

## 8. Launch the Dashboard

```bash
streamlit run app/dashboard.py
```

The application will be available at:

```
http://localhost:8501
```

---

# Expected Project Structure

After successful execution the following directories should exist.

```
models/

reports/

graph/

data/processed/
```

Generated files will be created automatically during execution.

---

# Troubleshooting

## ModuleNotFoundError

Verify that the virtual environment is activated.

---

## Missing Dataset

Ensure the dataset exists at:

```
data/raw/kc_house_data.csv
```

---

## Streamlit Not Found

Install project dependencies again.

```bash
pip install -r requirements.txt
```

---

## PyTorch Errors

Verify the installation.

```bash
python -c "import torch"
```

---

# Verification Commands

```bash
python -m src.features.feature_engineering

python -m src.training.train_baseline_model

python -m src.training.model_evaluation

python -m src.models.train_gnn

python -m src.models.evaluate_gnn

streamlit run app/dashboard.py
```

---

# Additional Notes

- Use a dedicated virtual environment for the project.
- Do not commit datasets, trained models, or generated artifacts.
- Follow semantic Git commit messages linked to GitHub Issues.
- Execute the workflow sequentially to generate all intermediate outputs.