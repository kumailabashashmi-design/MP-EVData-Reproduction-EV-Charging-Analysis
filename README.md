# MP-EVData Reproduction: EV Charging Load Profile Analysis

## Overview

This repository provides a reproducible Python-based framework for analyzing electric vehicle (EV) charging load characteristics using the **MP-EVData dataset**.

The project aims to reproduce important empirical analyses related to EV charging behavior, including charging session characterization, station-level load analysis, time-of-use (TOU) response behavior, clustering of charging patterns, and machine learning-based station classification.

The workflow follows a complete data analysis pipeline:

- Data acquisition and documentation
- Data cleaning and preprocessing
- Charging session feature extraction
- Load profile visualization
- Peak-valley difference and load factor analysis
- Time-of-use electricity price response analysis
- Charging pattern clustering
- Machine learning-based classification


---

# Dataset Description

The analysis is based on the **MP-EVData dataset**, which contains multi-prototype electric vehicle charging information collected from multiple charging stations.

The dataset includes:

- Individual charging session records
- Station-level charging load profiles
- Time-series charging demand information
- Electricity price information for TOU analysis


## Dataset Components

```text
dataset

├── charging session.xlsx
│   └── Individual EV charging events

├── station-level load Profile 15min.xlsx
│   └── High-resolution station charging load measurements

├── station-level load Profile 1h.xlsx
│   └── Hourly aggregated charging load profiles

└── price.xlsx
    └── Time-of-use electricity pricing information
```

---

# Project Structure

```text
MP-EVData-Reproduction-EV-Charging-Analysis

├── dataset
│   ├── charging session.xlsx
│   ├── station-level load Profile 15min.xlsx
│   ├── station-level load Profile 1h.xlsx
│   └── price.xlsx
│
├── notebooks
│   ├── 01_Data_Exploration_Cleaning.ipynb
│   ├── 02_Load_Profile_Analysis.ipynb
│   ├── 03_TOU_Response_Analysis.ipynb
│   ├── 04_Clustering_Analysis.ipynb
│   └── 05_Machine_Learning_Analysis.ipynb
│
├── code
│   ├── data_process_charging.py
│   ├── calculate_stats.py
│   ├── plot_profile.py
│   ├── plot_daily.py
│   └── visualization scripts
│
├── figures
│   └── Generated analysis figures
│
├── processed_data
│   └── Cleaned datasets and analysis outputs
│
├── requirements.txt
│
└── README.md
```

---

# Methodology

The analysis workflow consists of the following stages:


## 1. Data Exploration and Cleaning

The raw charging datasets are examined and processed.

Tasks include:

- Checking data consistency
- Handling missing values
- Converting timestamps
- Extracting charging duration
- Calculating charging energy characteristics


---

## 2. Charging Load Profile Analysis

Station-level charging demand patterns are analyzed.

The analysis includes:

- Daily charging load curves
- Monthly charging frequency patterns
- Station-level comparison
- Charging demand variation analysis


Key indicators:

- Average charging load
- Peak load
- Valley load
- Peak-valley difference
- Load factor


---

## 3. Time-of-Use (TOU) Response Analysis

The relationship between electricity pricing periods and EV charging behavior is investigated.

The analysis considers:

- Peak pricing periods
- Off-peak pricing periods
- Charging avoidance behavior
- Charging demand shifting


---

## 4. Charging Pattern Clustering

Machine learning clustering methods are applied to identify stations with similar charging characteristics.

Methods:

- Feature extraction
- Feature standardization
- K-Means clustering
- Cluster visualization


The clustering analysis reveals different charging demand patterns among stations.


---

## 5. Machine Learning Classification

Machine learning models are developed to classify charging station characteristics.

The workflow includes:

- Feature preparation
- Model training
- Model testing
- Performance evaluation
- Feature importance interpretation


Evaluation metrics:

- Mean Absolute Error (MAE)
- Root Mean Square Error (RMSE)
- R² Score


---

# Visualization Outputs

The repository generates various figures including:

- 3D charging session scatter plots
- Daily charging load curves
- Monthly charging frequency visualization
- Station comparison plots
- Clustering visualization
- Machine learning evaluation results


Example:

```text
figures/

├── A1_daily_profile.png
├── peak_avoidance_ratio.png
├── cluster_visualization.png
└── other analysis figures
```

---

# Technologies Used

Programming language:

- Python


Libraries:

- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook


---

# Installation

Clone the repository:

```bash
git clone https://github.com/kumailabashashmi-design/MP-EVData-Reproduction-EV-Charging-Analysis.git
```

Install required packages:

```bash
pip install -r requirements.txt
```


---

# Reproducibility

Run notebooks sequentially:

```text
01_Data_Exploration_Cleaning.ipynb

↓

02_Load_Profile_Analysis.ipynb

↓

03_TOU_Response_Analysis.ipynb

↓

04_Clustering_Analysis.ipynb

↓

05_Machine_Learning_Analysis.ipynb
```

All analysis steps, visualizations, and outputs can be reproduced using the provided notebooks and scripts.


---

# Research Objective

The main objective of this project is to understand EV charging demand characteristics and explore how data-driven methods can support:

- Charging infrastructure planning
- Load management
- Demand response strategies
- Smart charging optimization


---

# Author

**Kumail Abbas Hashmi**

Research Project:  
**Reproduction and Analysis of EV Charging Load Profiles Using Python, Clustering and Machine Learning**
