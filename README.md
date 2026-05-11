# CSE422 Lab Project — Disease Classification

**BRAC University | Machine Learning Lab**

A multi-class disease classification project that applies supervised and unsupervised machine learning models to predict patient disease type from clinical and demographic features.

---

## Overview

This project builds and compares multiple machine learning models that classify a patient's disease as **Type A**, **Type B**, or **Type C** based on eight input features. The pipeline covers the full ML workflow: exploratory data analysis, preprocessing, model training, and evaluation using standard metrics.

---

## Dataset

| Property | Details |
|---|---|
| File | `disease_classification_dataset.csv` |
| Rows | 1,800 |
| Input features | 8 |
| Target classes | 3 (Type A, Type B, Type C) |
| Class balance | ~589 / 634 / 577 (approximately balanced) |

**Features:**

| Feature | Type |
|---|---|
| Age | Quantitative |
| BMI | Quantitative |
| Blood Pressure | Quantitative |
| Cholesterol | Quantitative |
| Heart Rate | Quantitative |
| Smoking Habit | Categorical |
| Physical Activity Level | Categorical |
| Family History | Categorical |

---

## Project Structure

```
CSE422_Disease_Classification.ipynb   ← Main notebook (all sections below)
disease_classification_dataset.csv    ← Input dataset
01_correlation_heatmap.png
02_class_distribution.png
03a_eda_numeric.png
03b_eda_categorical.png
03c_eda_boxplots.png
04_scaling.png
05a_kmeans_elbow.png
05b_kmeans_clusters.png
06_accuracy_bar.png
07_precision_recall.png
08_confusion_matrices.png
09_roc_curves.png
10_auc_comparison.png
11_feature_importance.png
nn_loss_curve.png
README.md
```

---

## Notebook Sections

### Section 1 — Introduction
Defines the problem (multi-class classification), motivation (automated clinical diagnosis), and the overall approach.

### Section 2 — Dataset Description & EDA
- Loads the dataset and inspects shape, data types, and missing values.
- Generates a **correlation heatmap** — near-zero feature-target correlations are noted, indicating likely synthetic data.
- Checks class balance — confirmed balanced, so no SMOTE or resampling is needed.
- Produces distribution histograms, grouped bar charts, and boxplots to explore feature-class separability.

### Section 3 — Preprocessing

Three faults are identified and corrected:

| Fault | Solution |
|---|---|
| Missing values (BMI, Cholesterol, Smoking_Habit — ~5% each) | Median imputation for numeric; mode imputation for categorical |
| Categorical string features | Label Encoding |
| Differing feature scales | StandardScaler (Z-score normalisation), fit on training set only |

### Section 4 — Dataset Splitting
- **Stratified 80/20 split** (1,440 train / 360 test)
- Stratification ensures class proportions are preserved in both subsets.

### Section 5 — Model Training

| Model | Key Hyperparameters |
|---|---|
| Decision Tree | `max_depth=8` |
| K-Nearest Neighbors (KNN) | `k=7` |
| Neural Network (MLP) | Hidden layers: (128, 64), ReLU, Adam, early stopping |
| K-Means (unsupervised) | `k=3`, visualised via PCA 2D projection |

The MLP architecture is: **Input(8) → Dense(128, ReLU) → Dense(64, ReLU) → Output(3, Softmax)**

K-Means uses the **elbow method** to select k, then projects clusters into 2D with PCA.

### Section 6 — Evaluation & Comparison

Each supervised model is evaluated on:
- **Accuracy**
- **Weighted Precision**
- **Weighted Recall**
- **AUC (One-vs-Rest)**
- **Confusion Matrix**
- **ROC Curves** (per class)

Decision Tree feature importances are also plotted. A final summary table ranks all three models.

---

## Requirements

```
Python >= 3.8
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## How to Run

1. Clone or download this repository.
2. Place `disease_classification_dataset.csv` in `/content/sample_data/` (if using Google Colab) or update the file path in Section 2.
3. Open `CSE422_Disease_Classification.ipynb` in Jupyter Notebook or Google Colab.
4. Run all cells from top to bottom (`Runtime > Run all` in Colab).

---

## Key Findings

- All feature-target correlations are near zero, suggesting the dataset is synthetically generated with no strong discriminative signal.
- Feature distributions overlap heavily across all three disease types.
- K-Means clusters show significant overlap in PCA space, confirming the lack of natural groupings.
- All three supervised models perform near the random baseline (~33.3%), which is expected given the weak feature-label relationships in this dataset.
- Decision Tree feature importances are nearly uniform — no single feature dominates.

---

## Course Information

| Field | Details |
|---|---|
| Course | CSE422 — Machine Learning Lab |
| Institution | BRAC University |
| Task | Supervised + Unsupervised Classification |
