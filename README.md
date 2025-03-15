# AlzEEG-Classifier

## Overview
AlzEEG-Classifier is a machine learning pipeline designed to classify EEG signals into three groups:

- Alzheimer's Disease (A)
- Frontotemporal Dementia (FTD)
- Healthy Controls (HC)

This project processes EEG data, extracts features, and applies machine learning models to classify subjects based on their EEG recordings.

---

## Dataset

Access the data on [Kaggle](https://www.kaggle.com/datasets/yosftag/open-nuro-dataset/data)

---

## Recent Updates

1. Feature Selection Enhancements
We've refined feature selection by implementing:

- ANOVA F-test: Selects features with the highest discriminative power.
- Random Forest Importance: Ranks feature importance based on decision tree splits.
- Recursive Feature Elimination (RFE): Uses logistic regression to iteratively eliminate less relevant features.
- Selected Features

The following features were chosen based on their statistical and predictive importance:
- EEG band power features (alpha, beta, gamma) from specific channels
- Power ratio features (e.g., alpha/beta ratios)
- Common Spatial Patterns (CSP) features

See `selected_features.txt` for the full list of features selected.

---

2. Data Processing Updates

- Filtering Improvements: Applied a 1-45 Hz bandpass filter and 60 Hz notch filter to remove noise.
- Epoch Extraction: Adjusted epoching strategy to ensure balanced classes across folds.
- Label Correction: Ensured proper mapping of subject groups (Alzheimer’s, FTD, Healthy) in y_train.

---

3. Model Evaluation & Visualization

- Cross-validation Accuracy Comparisons:
    - Band Power Features: Evaluated with Random Forest and ANOVA selection.
    - Power Ratios: Tested separately to analyze their contribution.
    - CSP Features: Extracted and assessed for classification improvement.
    - Selected 20 Features: The top-performing 20 features were compared against all available features.
- Visualization Improvements:
    - Feature importance graphs using bar charts.
    - Accuracy comparison plots across different feature types.

---

## How to Use

```python
# To zip the dataset
python data.py --zip --dataset "dataset" --output "dataset.zip"

# To create k-fold splits
python data.py --split --k 5

# To do both operations
python data.py --zip --split

# To use custom paths
python data.py --dataset "my_dataset_folder" --output "my_dataset.zip" --k 10
```