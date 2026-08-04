# Project Structure

## Geospatial Real Estate Valuation using Graph Neural Networks

**Author:** Aman Shukla

**Version:** 1.0.0

---

# Overview

This document describes the overall repository structure, software architecture, and module responsibilities for the Geospatial Real Estate Valuation project.

The project follows a modular architecture where data preprocessing, feature engineering, graph construction, machine learning, graph neural networks, evaluation, and visualization are separated into independent components.

---

# Repository Structure

```
geospatial-real-estate-valuation-gnn/
│
├── app/
├── data/
├── docs/
├── reports/
├── src/
├── CHANGELOG.md
├── LICENSE
├── README.md
├── requirements.txt
└── .gitignore
```

---

# Root Directory

## app/

Contains the Streamlit application responsible for interactive visualization and user interaction.

```
app/
│
├── dashboard.py
├── components/
└── services/
```

### dashboard.py

Main entry point for the Streamlit application.

Responsibilities:

- Configure Streamlit
- Build dashboard layout
- Load visualization components
- Display project information

---

### components/

Contains reusable dashboard components.

Current components include:

- Dataset overview
- Performance metrics
- Interactive housing map
- Property prediction analysis
- Project summary

Each component is implemented independently to improve readability and maintainability.

---

### services/

Contains helper functions used by the dashboard.

Responsibilities include:

- Dataset loading
- Metric computation
- Map generation
- Prediction processing

Separating business logic from the user interface keeps the dashboard modular and easier to extend.

---

# data/

Stores datasets used by the project.

```
data/
│
├── raw/
└── processed/
```

### raw/

Contains the original King County Housing Dataset.

This directory is excluded from Git version control.

---

### processed/

Contains datasets generated during preprocessing and feature engineering.

Examples include cleaned datasets and engineered feature datasets.

This directory is excluded from Git version control.

---

# docs/

Contains project documentation.

```
docs/
│
├── INSTALLATION.md
├── PROJECT_STRUCTURE.md
└── DEPLOYMENT.md
```

These documents explain project setup, architecture, and deployment.

---

# reports/

Stores generated project reports.

Examples include:

- Exploratory Data Analysis summary
- Feature Engineering summary
- Graph Construction summary
- Spatial Embedding summary
- Model Comparison summary
- GNN Training summary

Reports are generated in Markdown format for easy review.

---

# src/

Contains the complete source code for the project.

```
src/
│
├── preprocessing/
├── features/
├── graph/
├── embeddings/
├── gnn/
├── training/
├── evaluation/
├── models/
├── utils/
├── config.py
└── project_info.py
```

---

## preprocessing/

Responsible for preparing the raw dataset.

Modules include:

- Dataset loading
- Exploratory Data Analysis
- Data cleaning
- Dataset validation

Output:

Cleaned housing dataset.

---

## features/

Responsible for feature engineering.

Generated features include:

- House age
- Sale year
- Sale month
- Renovation indicators
- Years since renovation
- Distance to Seattle city centre

Output:

Feature-engineered dataset.

---

## graph/

Responsible for geospatial processing.

Responsibilities include:

- Geographic distance calculation
- Spatial neighbour identification
- Graph construction

Output:

Graph representation of the housing dataset.

---

## embeddings/

Responsible for spatial feature embeddings.

Responsibilities include:

- Generate spatial embeddings
- Validate embeddings
- Generate embedding reports

These embeddings provide spatial context for the Graph Neural Network.

---

## gnn/

Responsible for preparing graph datasets compatible with PyTorch Geometric.

Responsibilities include:

- Build graph dataset
- Validate graph dataset
- Generate dataset reports

---

## training/

Contains baseline machine learning training.

Responsibilities include:

- Train XGBoost baseline model
- Evaluate baseline model

---

## evaluation/

Responsible for project evaluation.

Includes:

- Baseline report generation
- Model comparison
- Feature importance analysis
- Performance summaries

---

## models/

Contains Graph Neural Network implementation.

Responsibilities include:

- GNN architecture
- GNN training
- GNN evaluation

---

## utils/

Utility modules shared across multiple packages.

Examples include:

- Visualization
- Statistical functions
- Map utilities
- Spatial helper functions

---

## config.py

Central configuration module.

Stores:

- Project paths
- Dataset configuration
- Graph configuration
- Model configuration
- Dashboard configuration
- Embedding configuration

---

## project_info.py

Stores reusable project metadata including:

- Project name
- Author
- Version
- Dataset information
- Framework information

---

# Project Workflow

The complete pipeline follows the sequence below.

```
Raw Dataset
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Baseline XGBoost
      │
      ▼
Graph Construction
      │
      ▼
Spatial Embeddings
      │
      ▼
Graph Dataset
      │
      ▼
Graph Neural Network
      │
      ▼
Model Evaluation
      │
      ▼
Interactive Dashboard
```

---

# Software Design Principles

The project follows several software engineering best practices:

- Modular architecture
- Separation of concerns
- Reusable components
- Centralized configuration
- Independent evaluation modules
- Maintainable folder hierarchy
- Clear documentation
- GitHub issue-based development workflow

---

# Summary

The repository is organized into independent modules that separate preprocessing, feature engineering, graph processing, model training, evaluation, and visualization.

This modular design improves readability, maintainability, extensibility, and reproducibility while providing a clear workflow from raw data to interactive model visualization.