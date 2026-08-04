# Deployment Guide

## Geospatial Real Estate Valuation using Graph Neural Networks

**Author:** Aman Shukla

**Version:** 1.0.0

---

# Overview

This document describes how to deploy and execute the Geospatial Real Estate Valuation project in a local development environment.

The project consists of a complete machine learning pipeline for predicting residential property prices using both traditional Machine Learning and Graph Neural Networks (GNNs), along with an interactive Streamlit dashboard for visualization.

---

# Deployment Requirements

Before deploying the project, ensure the following software is installed.

## Operating System

- Windows 10 / Windows 11
- Ubuntu 22.04+
- macOS 13+

---

## Python

Python 3.13 or later

Verify installation:

```bash
python --version
```

---

## Git

```bash
git --version
```

---

## Required Python Packages

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

# Clone Repository

Clone the repository from GitHub.

```bash
git clone https://github.com/AmanShukla4194/geospatial-real-estate-valuation-gnn.git
```

Navigate into the project.

```bash
cd geospatial-real-estate-valuation-gnn
```

---

# Create Virtual Environment

Windows

```powershell
python -m venv .venv
```

Linux/macOS

```bash
python3 -m venv .venv
```

---

# Activate Virtual Environment

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

# Dataset Placement

The dataset is intentionally excluded from Git version control.

Create the directory if it does not already exist.

```
data/raw/
```

Copy the King County Housing Dataset into:

```
data/raw/kc_house_data.csv
```

---

# Complete Project Execution

Execute each module in sequence.

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

## 3. Train Baseline Model

```bash
python -m src.training.train_baseline_model
```

---

## 4. Evaluate Baseline Model

```bash
python -m src.training.model_evaluation
```

---

## 5. Generate Spatial Embeddings

```bash
python -m src.embeddings.spatial_embeddings
```

---

## 6. Train Graph Neural Network

```bash
python -m src.models.train_gnn
```

---

## 7. Evaluate Graph Neural Network

```bash
python -m src.models.evaluate_gnn
```

---

# Launch Dashboard

Start the Streamlit application.

```bash
streamlit run app/dashboard.py
```

Open the browser and navigate to:

```
http://localhost:8501
```

---

# Generated Outputs

Running the project produces several generated artifacts.

## Processed Data

```
data/processed/
```

Contains cleaned and feature-engineered datasets.

---

## Models

```
models/
```

Contains trained baseline and Graph Neural Network models.

---

## Reports

```
reports/
```

Contains generated Markdown summaries and analysis reports.

---

## Graph Artifacts

```
graph/
```

Contains graph structures and intermediate files generated during graph construction.

---

# Deployment Notes

The following directories are intentionally excluded from Git version control:

- data/
- models/
- graph/
- reports/figures/

These directories are generated locally when executing the pipeline.

---

# Updating Dependencies

If new packages are installed, regenerate the dependency list.

```bash
pip freeze > requirements.txt
```

Review the file before committing to ensure only required packages are included.

---

# Troubleshooting

## ModuleNotFoundError

Verify the virtual environment is activated.

---

## Missing Dataset

Confirm the dataset exists at:

```
data/raw/kc_house_data.csv
```

---

## Missing Python Packages

Install project dependencies again.

```bash
pip install -r requirements.txt
```

---

## Streamlit Launch Issues

Verify Streamlit installation.

```bash
streamlit --version
```

---

## PyTorch Verification

```bash
python -c "import torch; print(torch.__version__)"
```

---

## PyTorch Geometric Verification

```bash
python -c "import torch_geometric; print(torch_geometric.__version__)"
```

---

# Recommended Deployment Workflow

```
Clone Repository
        │
        ▼
Create Virtual Environment
        │
        ▼
Install Dependencies
        │
        ▼
Place Dataset
        │
        ▼
Run Data Cleaning
        │
        ▼
Run Feature Engineering
        │
        ▼
Train Baseline Model
        │
        ▼
Evaluate Baseline Model
        │
        ▼
Generate Spatial Embeddings
        │
        ▼
Train Graph Neural Network
        │
        ▼
Evaluate Graph Neural Network
        │
        ▼
Launch Streamlit Dashboard
```

---

# Future Deployment Options

The project can be extended for deployment using platforms such as:

- Streamlit Community Cloud
- Docker
- Microsoft Azure
- Amazon Web Services (AWS)
- Google Cloud Platform (GCP)

Additional configuration may be required depending on the deployment platform.

---

# Summary

The project is designed to be reproducible from a clean environment. By following the steps in this guide, users can clone the repository, install dependencies, generate all intermediate artifacts, train both machine learning models, evaluate their performance, and launch the interactive dashboard without modifying the source code.