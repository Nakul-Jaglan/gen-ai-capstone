# Japan Real Estate Price Predictor

> **GenAI Capstone — Milestone 1**  
> ML-based property valuation system trained on 50,000+ Japanese real-estate transactions with an interactive Streamlit dashboard.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Business Context](#business-context)
3. [System Architecture](#system-architecture)
4. [Dataset](#dataset)
5. [ML Pipeline](#ml-pipeline)
6. [Feature Engineering](#feature-engineering)
7. [Model Evaluation](#model-evaluation)
8. [Tech Stack](#tech-stack)
9. [Project Structure](#project-structure)
10. [Setup & Installation](#setup--installation)
11. [Running the App](#running-the-app)
12. [Deployment](#deployment)

---

## Project Overview

This project implements a **Random Forest Regressor** to predict real-estate trade prices across Japanese prefectures. The end product is a fully interactive web application where a user inputs property attributes and receives an instant price estimate alongside market analytics and feature importance visualizations.

---

## Business Context

Real-estate pricing in Japan is influenced by a layered set of factors — location proximity to transit, zoning regulations, building age, structural composition, and regional supply-demand dynamics. Manual appraisal is time-consuming and inconsistent. An ML-driven valuation tool provides:

- **Buyers** — an independent benchmark before negotiating.
- **Sellers / Agents** — data-backed listing price guidance.
- **Analysts** — regional trend exploration and market comparisons.

---

## System Architecture

```
┌──────────────────────────────────────────────────────┐
│                   User (Browser)                     │
└───────────────────────┬──────────────────────────────┘
                        │  HTTP
┌───────────────────────▼──────────────────────────────┐
│              Streamlit Frontend (app.py)              │
│  ┌──────────────┐  ┌────────────┐  ┌──────────────┐  │
│  │  Input Form  │  │  Predict   │  │   Analytics  │  │
│  │  (Sidebar)   │  │   Card     │  │     Tabs     │  │
│  └──────┬───────┘  └─────┬──────┘  └──────────────┘  │
└─────────┼────────────────┼─────────────────────────  ┘
          │                │
┌─────────▼────────────────▼──────────────────────────┐
│             Preprocessing Pipeline                   │
│   Label Encoding → MinMax Scaling → Feature Eng.    │
└─────────────────────────┬───────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────┐
│         Random Forest Regressor Model                │
│              (rf_model_new.joblib)                   │
└─────────────────────────────────────────────────────┘
```

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | Japanese Ministry of Land real-estate transaction records |
| **File** | `02.csv` |
| **Records** | ~52,400 transactions |
| **Target** | `TradePrice` (JPY) |
| **Regions** | Aomori Prefecture and surrounding municipalities |
| **Years** | 2006 – 2019 |

### Columns Used After Cleaning

| Category | Features |
|---|---|
| **Categorical (10)** | Type, Region, Municipality, DistrictName, NearestStation, LandShape, Structure, Classification, CityPlanning, Direction |
| **Numerical (8)** | Frontage, TotalFloorArea, BuildingYear, Breadth, CoverageRatio, FloorAreaRatio, MinTimeToNearestStation, Area |
| **Engineered (1)** | AgeOfBuilding (Year − BuildingYear) |
| **Dropped** | 16 columns — redundant identifiers, flags, free-text fields |

---

## ML Pipeline

```
Raw CSV (52,408 rows)
        │
        ▼
1. Drop irrelevant columns (16 cols)
        │
        ▼
2. Remove Agricultural Land rows
        │
        ▼
3. Drop rows with NaN in required columns
        │
        ▼
4. Label Encode categorical columns (10 cols)
        │
        ▼
5. MinMaxScaler on numeric columns (8 cols)
        │
        ▼
6. Feature engineering: AgeOfBuilding = Year − BuildingYear
        │
        ▼
7. MinMaxScaler on extended numeric set (9 cols incl. AgeOfBuilding)
        │
        ▼
8. log1p transform on TradePrice (target)
        │
        ▼
9. Train / Test split (80 / 20, random_state=42)
        │
        ▼
10. RandomForestRegressor(random_state=42)
        │
        ▼
11. expm1 to invert log on predictions
```

---

## Feature Engineering

| Feature | Derivation | Rationale |
|---|---|---|
| `AgeOfBuilding` | `Year - BuildingYear` | Older buildings trade at lower prices; captures depreciation |
| Log-transform on `TradePrice` | `log1p(price)` | Reduces right-skew in the price distribution for stable regression |

---

## Model Evaluation

Two models were evaluated:

### Baseline — Linear Regression

| Metric | Train | Test |
|---|---|---|
| MSE | — | — |
| RMSE | — | — |
| R² | — | — |

### Final — Random Forest Regressor

| Metric | Train | Test |
|---|---|---|
| MSE | — | — |
| RMSE | — | — |
| R² | — | — |

> Metric values populate after running `GenAI_Capstone_V2.ipynb` end-to-end.

**Random Forest was chosen** as the final model due to its superior test R² and robustness to the mixed feature types in this dataset.

---

## Tech Stack

| Component | Technology |
|---|---|
| **ML** | scikit-learn (Random Forest, Linear Regression, MinMaxScaler, LabelEncoder) |
| **Data Processing** | Pandas, NumPy |
| **Frontend** | Streamlit |
| **Visualizations** | Plotly |
| **Model Persistence** | joblib |
| **Language** | Python 3.14 |

---

## Project Structure

```
gen-ai-capstone/
├── app.py                    # Streamlit web application
├── 02.csv                    # Raw dataset
├── rf_model_new.joblib       # Trained Random Forest model
├── minmaxscaler.joblib       # Saved MinMaxScaler (reference artifact)
├── notebooks/
│   └── GenAI_Capstone_V2.ipynb   # EDA, model training & evaluation notebook
├── assets/
│   └── report/
│       └── report.tex            # LaTeX report
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup & Installation

### Prerequisites
- Python 3.10 or higher
- `pip`

### 1. Clone the repository

```bash
git clone https://github.com/Nakul-Jaglan/gen-ai-capstone.git
cd gen-ai-capstone
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** by default.

### Usage
1. Use the **sidebar** to configure all property attributes (type, location, dimensions, building details, zoning).
2. Click **Predict Property Price** to get the estimated trade price.
3. Explore **Market Insights**, **Feature Analysis**, and **About** tabs for data-driven charts.

---

## Deployment

The app is designed to deploy on:

| Platform | Notes |
|---|---|
| **Streamlit Community Cloud** | Connect GitHub repo → set `app.py` as entrypoint → deploy |
| **Hugging Face Spaces** | Use `Streamlit` SDK, upload files, set `app.py` as main |
| **Render** | Add `streamlit run app.py --server.port $PORT --server.headless true` as start command |

> Make sure `02.csv`, `rf_model_new.joblib`, and `minmaxscaler.joblib` are committed to the repo or uploaded to the hosting platform — the app requires them at runtime.
