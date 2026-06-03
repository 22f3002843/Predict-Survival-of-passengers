# Survivor Detection Challenge - Predict survival of the passengers

Welcome to the **Survivor Detection** classification project. This repository contains data exploration, visualization, feature engineering, and model optimization to predict passenger survival outcomes from a historical maritime incident.

The dataset has been specifically prepared with real-world complexities—including missing values, feature obfuscation, and artificial noise—to simulate challenging industry placement assessments.

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset Description](#dataset-description)
3. [Local Execution Setup](#local-execution-setup)
4. [Exploratory Data Analysis (EDA) & Visualizations](#exploratory-data-analysis-eda--visualizations)
5. [Data Preprocessing & Feature Engineering](#data-preprocessing--feature-engineering)
6. [Model Selection & Evaluation](#model-selection--evaluation)
7. [Feature Importance Analysis](#feature-importance-analysis)
8. [Future Optimizations & Improvements](#future-optimizations--improvements)
9. [Inference & Submission](#inference--submission)

---

## Project Overview

The core objective is to build a binary classification model that predicts whether a passenger survived (`1`) or perished (`0`). The models are trained on historical passenger profiles and evaluated using **Categorization Accuracy** on a validation set:

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

---

## Dataset Description

The project consists of three primary data files:
- `maritime_train.csv`: Training dataset containing the features and target variable (`Outcome`).
- `maritime_test.csv`: Test dataset containing features only; used to generate final predictions.
- `maritime_sample_submission.csv`: Template file indicating the submission format.

### Data Dictionary

| Column Name | Type | Description |
| :--- | :--- | :--- |
| **PassengerName** | String | Unique identifier representing the passenger's name. |
| **TicketTier** | Integer | Proxy for socio-economic status ($1 = \text{Upper}$, $2 = \text{Middle}$, $3 = \text{Lower}$). |
| **Gender** | Categorical | Sex of the passenger (`male` or `female`). |
| **Age** | Float | Age of the passenger in years (contains missing values). |
| **RelativesAboard** | Float | Number of siblings or spouses traveling with the passenger. |
| **ParentsChildren** | Float | Number of parents or children traveling with the passenger. |
| **TicketCost** | Float | Ticket price paid (inclusive of artificial noise). |
| **BoardingPort** | Categorical | Port of embarkation ($C = \text{Cherbourg}$, $Q = \text{Queenstown}$, $S = \text{Southampton}$). |
| **CLass** | String | Raw ticket identifier string. |
| **Berth** | String | Cabin or section number (highly sparse column). |
| **Singleton** | Integer | Binary flag: `1` if traveling alone, `0` if with family. |
| **FarePerPerson** | Float | TicketCost divided by total family size. |
| **Title** | Categorical | Extracted social title (e.g., `Mr.`, `Mrs.`, `Miss.`, `Master.`, `Rare`). |
| **Outcome** | Integer | **Target variable**: `1` for survived, `0` for deceased (present in train only). |

---

## Local Execution Setup

When running this notebook locally rather than inside a Kaggle session, make sure the CSV files are placed in the same directory as the notebook and load them using the local filenames:

```python
import pandas as pd

# Load the local datasets
train_df = pd.read_csv('maritime_train.csv')
test_df = pd.read_csv('maritime_test.csv')
```

Ensure the following packages are installed in your Python environment:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost lightgbm
```

---

## Exploratory Data Analysis (EDA) & Visualizations

We utilized several charts to understand feature correlations with survival rate:

1. **Survival Probability by Gender** (Bar Plot): Reveals a strong survival bias toward female passengers.
2. **Age Distribution by Outcome Status** (Histogram with KDE): Visualizes the survival density across age cohorts, showing higher survival rates for young children.
3. **Survival Rate by Boarding Port** (Bar Plot): Examines variations in survival rate depending on the passenger's embarkation point.
4. **TicketCost Distribution by Outcome Status** (Log-scaled Box Plot): Shows that passengers with higher-cost tickets had a significantly better chance of survival.
5. **Survival Probability vs. Family Size** (Point Plot): Demonstrates that passengers traveling in family groups of size 2-4 had the highest survival rates.
6. **PCA Scatter Plot** (2D Scatter Plot): Projects standardized features onto the first two Principal Components to visualize class separability in reduced dimensions.
7. **Confusion Matrix Heatmap**: Visualizes true/false positives and negatives for model predictions.
8. **Model Accuracy Comparison** (Horizontal Bar Plot): Compares all tested classifiers side-by-side.

---

## Data Preprocessing & Feature Engineering

- **Imputation**:
  - `Age` missing values were filled with the median age of the passenger's specific `Title` group.
  - `BoardingPort` missing values were filled with the dataset's port mode.
- **Dimensionality Reduction**: The highly sparse column `Berth` (>75% missing values) was dropped.
- **Categorical Encodings**:
  - `TicketTier` and `CLass` were encoded using `OrdinalEncoder`.
  - `Gender`, `BoardingPort`, and `Title` were encoded using `LabelEncoder`.
- **Scaling**: Numerical features were standardized using `StandardScaler` for the distance-based/linear models.
- **Validation Split**: Splitting the processed training data into an 80% train set and a 20% validation set to prevent overfitting.

### Title-Based Age Imputation Details
The `Age` feature contains 197 missing values in the training set and 49 in the test set. Instead of using a simple global median (which would ignore age discrepancies between children and adults), we leveraged the passenger's `Title` (e.g., *Mr., Miss., Mrs., Master.*) to calculate subgroup-specific median age values:

```python
# Impute missing Age based on the median age of their Title subgroup
train_df['Age'] = train_df['Age'].fillna(train_df.groupby('Title')['Age'].transform('median'))
test_df['Age'] = test_df['Age'].fillna(test_df.groupby('Title')['Age'].transform('median'))
```

This ensures that a passenger with the title `Master.` (typically young boys) is imputed with a child's median age, whereas a passenger with the title `Mr.` is imputed with an adult's median age, preserving the semantic validity of the feature.

---

## Model Selection & Evaluation

Five distinct machine learning algorithms were trained and evaluated on the validation dataset:

| Model | Validation Accuracy | F1-Score (Survived - 1) | F1-Score (Deceased - 0) |
| :--- | :---: | :---: | :---: |
| **LightGBM** | **83.92%** | **0.79** | **0.87** |
| **Random Forest** | 82.52% | 0.77 | 0.86 |
| **XGBoost** | 82.52% | 0.77 | 0.86 |
| **Gradient Boosting** | 81.82% | 0.75 | 0.86 |
| **Logistic Regression** | 79.72% | 0.74 | 0.83 |

*Note: Gradient Boosting's performance is further diagnosed in the notebook using a Confusion Matrix, showing 77 True Negatives, 40 True Positives, 10 False Positives, and 16 False Negatives.*

---

## Feature Importance Analysis

Tree-based ensemble models (LightGBM, Random Forest, and XGBoost) calculate feature importance based on split frequency or mean decrease in impurity. Across our best-performing models, the key features driving the predictions are:

1. **Gender**: Due to the "women and children first" maritime evacuation protocol, gender is the single most predictive feature. Females had a significantly higher survival rate than males.
2. **TicketCost & FarePerPerson**: Higher fare prices are highly correlated with upper socio-economic classes (e.g., TicketTier 1), which enjoyed priority access to lifeboats.
3. **Age & Title**: Children (Title: `Master.`) had higher survival priority, whereas young-to-middle-aged adult males (Title: `Mr.`) had the lowest survival rate.
4. **FamilySize & RelativesAboard**: Single passengers or very large families had lower survival probability, while small-to-medium families (size 2-4) showed higher resilience.

---

## Future Optimizations & Improvements

To push the prediction performance beyond the current **83.92% validation accuracy**, the following strategies could be pursued:

1. **Hyperparameter Tuning**:
   - Implement Automated Hyperparameter Search using **Optuna** or `GridSearchCV` / `RandomizedSearchCV` on LightGBM and XGBoost parameters (e.g., `learning_rate`, `max_depth`, `num_leaves`, `min_child_samples`, and `subsample`).
2. **Ensemble & Blending**:
   - Create a `VotingClassifier` or a Stacking ensemble using a combination of **LightGBM**, **Random Forest**, and **XGBoost** models to average out individual model variances and capitalize on their diverse decision boundaries.
3. **Advanced Feature Engineering**:
   - **Ticket Grouping**: Extract passengers sharing the same ticket string prefix or suffix, as group/ticket companions often shared outcomes.
   - **Age Bins**: Discretize the age variable into categorical cohorts (e.g., infant, child, teenager, adult, elderly) to help tree models partition nonlinear relationships easier.
   - **Interaction Features**: Create combinations like `Gender_Class` or `Age_Class` to highlight high-priority subsets (e.g., first-class females).

---

## Inference & Submission

The highest-performing model, **LightGBM (83.92% accuracy)**, was selected to generate the final predictions on the test dataset.

1. Test features were preprocessed using the identical encoding and imputation rules.
2. Predictions were output to `submission.csv` mapping `PassengerName` to the predicted survival `Outcome` binary status.
