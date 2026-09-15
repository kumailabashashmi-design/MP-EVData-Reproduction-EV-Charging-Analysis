# MP-EVData Reproduction: EV Charging Load Profile Analysis

## Overview

This repository provides a reproducible analysis framework based on the MP-EVData dataset:

"MP-EVData: An AI-Augmented Dataset of Multi-Prototype Electric Vehicle Charging Load Profiles in China"

This project analyzes EV charging load characteristics, time-of-use (TOU) response behavior, charging station clustering patterns, and machine learning-based load prediction.

---

# Dataset Description

The analysis is based on the MP-EVData dataset.

The dataset contains electric vehicle charging load information collected from 10 charging stations in China with high-resolution temporal measurements.

Original dataset files include:
```text
dataset/

├── charging_session.xlsx
├── station-level_load_Profile_15min.xlsx
├── station-level_load_Profile_1h.xlsx
└── price.xlsx
```

### Dataset Components

The dataset provides different levels of electric vehicle charging information:

- **Charging Session Data**
  - Contains individual EV charging event records.
  - Used to analyze charging behavior and temporal charging patterns.

- **15-Minute Load Profiles**
  - Provides high-resolution station-level charging load measurements.
  - Used for daily load profile analysis and short-term variation analysis.

- **Hourly Load Profiles**
  - Provides aggregated hourly charging load information.
  - Used for long-term load pattern analysis.

- **Time-of-Use (TOU) Price Data**
  - Contains electricity price information under different pricing periods.
  - Used to evaluate charging response behavior and peak-load avoidance.

```
## Analysis Workflow

The complete analysis pipeline follows a structured workflow to transform raw EV charging data into meaningful insights.

### 1. Data Preprocessing

Raw charging datasets are processed and cleaned before analysis.

The preprocessing steps include:

- Loading raw charging session and station-level datasets.
- Handling missing values and checking data consistency.
- Aggregating charging loads at station level.
- Generating processed datasets for further analysis.

Processed outputs include:

- Station-level charging features.
- Time-series load profiles.
- Aggregated statistical characteristics.

---

### 2. EV Charging Load Profile Analysis

The charging load characteristics are analyzed using:

- 15-minute interval load profiles.
- Daily charging demand patterns.
- Peak and valley load identification.
- Average load and load factor calculation.

The analysis helps identify charging demand fluctuations and station-level differences.

---

### 3. Time-of-Use (TOU) Response Analysis

The impact of electricity pricing periods on EV charging behavior is evaluated.

The analysis considers:

- Peak charging periods.
- Off-peak charging periods.
- Shoulder periods.

Key indicators include:

- Average charging load under different TOU periods.
- Peak avoidance ratio.
- Charging behavior response to electricity pricing signals.

---

### 4. Charging Station Clustering Analysis

Machine learning-based clustering is applied to identify charging station groups with similar load characteristics.

The clustering process includes:

- Feature selection from station-level load statistics.
- Feature standardization.
- K-Means clustering.
- Principal Component Analysis (PCA) visualization.

The identified clusters represent different charging demand patterns and operational characteristics.

---

### 5. Machine Learning-Based Load Prediction

A machine learning model is developed to predict EV charging load behavior.

The prediction workflow includes:

- Feature preparation.
- Model training and testing.
- Performance evaluation using statistical metrics.
- Feature importance analysis.

Evaluation metrics include:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score
---

# Results Summary

The implemented analysis framework provides insights into EV charging demand characteristics, station-level differences, pricing response behavior, and predictive modeling performance.

## 1. Load Profile Characteristics

The load profile analysis reveals significant variations among charging stations.

Key observations:

- Charging demand shows clear daily variation patterns.
- Most stations experience higher charging activity during specific demand periods.
- Peak and valley load differences indicate different operational characteristics among stations.
- Load factor analysis identifies stations with higher utilization efficiency.

---

## 2. Time-of-Use (TOU) Response Findings

The TOU analysis demonstrates the influence of electricity pricing signals on charging behavior.

Main findings:

- Charging demand is redistributed across different pricing periods.
- Off-peak periods generally contain higher charging activity.
- Some stations show strong peak avoidance behavior under TOU pricing.
- The peak avoidance ratio indicates the effectiveness of demand-shifting strategies.

---

## 3. Charging Station Clustering Results

K-Means clustering identifies four major charging station groups based on load characteristics.

The clusters represent:

- **High-demand stations**
  - Large peak load and high average charging demand.

- **Medium-demand stations**
  - Moderate charging activity with stable load patterns.

- **Low-demand stations**
  - Limited charging demand and lower utilization.

- **Special-pattern stations**
  - Stations showing unique load characteristics compared with other groups.

PCA visualization is used to illustrate the separation of charging station clusters.

---

## 4. Machine Learning Prediction Results

The machine learning model evaluates the capability of predicting EV charging load patterns.

The analysis includes:

- Model training using station-level charging features.
- Testing using unseen data samples.
- Quantitative performance evaluation.

Performance indicators:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score

Feature importance analysis highlights the contribution of different charging characteristics to load prediction.
---

# Project Structure

The repository is organized as follows:

```text
MP-EVData-Reproduction/

│
├── dataset/
│   ├── charging_session.xlsx
│   ├── station-level_load_Profile_15min.xlsx
│   ├── station-level_load_Profile_1h.xlsx
│   └── price.xlsx
│
├── processed_data/
│   ├── station_features.csv
│   ├── tou_analysis_results.csv
│   ├── kmeans_cluster_results.csv
│   └── cluster_characteristics.csv
│
├── notebooks/
│   ├── 01_Data_Exploration_Cleaning.ipynb
│   ├── 02_Load_Profile_Analysis.ipynb
│   ├── 03_TOU_Response_Analysis.ipynb
│   ├── 04_Clustering_Analysis.ipynb
│   └── 05_Machine_Learning_Analysis.ipynb
│
├── figures/
│   ├── load_profile_figures/
│   ├── tou_analysis_figures/
│   ├── clustering_figures/
│   └── prediction_figures/
│
├── scripts/
│   ├── data_process_charging.py
│   └── calculate_stats.py
│
└── README.md
```

---

# Software Requirements

The analysis was performed using Python-based scientific computing tools.

Recommended environment:

- Python >= 3.9

Required packages:

```text
pandas
numpy
matplotlib
scikit-learn
openpyxl
jupyter
seaborn
```

Install dependencies using:

```bash
pip install pandas numpy matplotlib scikit-learn openpyxl jupyter seaborn
```

---

# How to Run the Analysis

## Step 1: Clone the Repository

```bash
git clone <repository-url>
cd MP-EVData-Reproduction
```

---

## Step 2: Prepare Dataset

Place the original MP-EVData files inside:

```text
dataset/
```

The required files are:

- charging_session.xlsx
- station-level_load_Profile_15min.xlsx
- station-level_load_Profile_1h.xlsx
- price.xlsx

---

## Step 3: Data Processing

Run the preprocessing scripts:

```bash
python scripts/data_process_charging.py
```

The processed datasets will be generated in:

```text
processed_data/
```

---

## Step 4: Run Analysis Notebooks

Execute notebooks sequentially:

1. Data exploration and cleaning

```
01_Data_Exploration_Cleaning.ipynb
```

2. Load profile analysis

```
02_Load_Profile_Analysis.ipynb
```

3. TOU response analysis

```
03_TOU_Response_Analysis.ipynb
```

4. Charging station clustering

```
04_Clustering_Analysis.ipynb
```

5. Machine learning prediction

```
05_Machine_Learning_Analysis.ipynb
```

---

# Reproducibility

All analysis procedures, statistical calculations, visualization methods, and machine learning workflows are provided to enable reproducible research based on the MP-EVData dataset.

The repository allows researchers to reproduce:

- EV charging load profile analysis.
- TOU pricing response evaluation.
- Charging station clustering.
- Machine learning-based load prediction.

---

# Citation

If you use this dataset or analysis framework in your research, please cite the original MP-EVData publication:

```text
MP-EVData: An AI-Augmented Dataset of Multi-Prototype Electric Vehicle Charging Load Profiles in China.

Please refer to the original publication for complete citation details.
```

---

# License

This repository is intended for academic research and educational purposes.

Users are encouraged to properly acknowledge the original dataset source and publication when using the provided materials.

---

# Contact

For questions regarding the dataset, analysis framework, or reproduction results, please refer to the corresponding authors of the original publication.
