# pickles-vehicle-price-prediction

Predicting vehicle sale prices (`Sold_Amount`) using machine learning, developed as a data science take-home assessment.

## Project Overview

This project follows a complete machine learning workflow, including data understanding, exploratory data analysis, data cleaning, feature engineering, categorical encoding, model experimentation, model diagnostics, and evaluation on an independent test dataset.

Five regression models were evaluated:

- Linear Regression
- Ridge Regression
- Decision Tree
- Random Forest
- XGBoost

A reusable `ModelTrainer` class was used to train models, generate predictions, calculate evaluation metrics, and record experiment results.

**Final model:** XGBoost

**Independent test performance:**

- **MAE:** $1,995.35
- **RMSE:** $3,443.55
- **R²:** 0.9079

The full reasoning, diagnostics, and observations are documented in the notebook:

`Vehicle_Price_Prediction.ipynb`

## Data

The project uses two datasets provided for the assessment:

- `DatiumTrain.rpt`
- `DatiumTest.rpt`

The raw data files are not included in this repository because they were provided directly as part of the assessment and are not redistributed.

To run the project, place both files inside the `data/` folder before running the notebook.

### Restricted Features

The following fields were excluded from modelling according to the assessment instructions:

- `AvgWholesale`
- `AvgRetail`
- `GoodWholesale`
- `GoodRetail`
- `TradeMin`
- `TradeMax`
- `PrivateMax`

These features were removed from the training and test datasets before modelling.

## Project Workflow

### 1. Data Understanding and EDA

The dataset was examined to understand:

- Dataset dimensions and data types
- Numerical feature distributions
- Categorical feature distributions
- Missing values
- Duplicate records
- Potential outliers
- Correlations with the target variable

### 2. Data Preprocessing

The preprocessing stage included:

- Removing restricted and unsuitable features
- Handling missing values
- Removing records with a missing target value
- Encoding categorical features
- Converting date information into useful features
- Aligning training and independent test features

### 3. Feature Engineering

Date features were transformed into useful components, including:

- `Sold_Year`
- `Sold_Month`
- `Compliance_Year`
- `Compliance_Month`

The existing `Age_Comp_Months` feature was retained because it already represented vehicle age at the time of sale.

### 4. Model Development

Five regression approaches were evaluated.

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Ridge Regression | $2,592.82 | $4,364.93 | 0.8209 |
| Decision Tree | $2,440.50 | $3,900.66 | 0.8570 |
| Random Forest | $1,682.54 | $2,853.37 | 0.9235 |
| XGBoost | **$1,630.96** | $3,150.46 | 0.9067 |
| Linear Regression | Unstable | Unstable | Unstable |

Linear Regression produced extremely large errors due to numerical instability associated with the feature matrix. Ridge Regression was therefore used as the linear baseline.

Random Forest and XGBoost provided substantially stronger predictive performance than the linear and single-tree approaches.

### Model Selection

XGBoost was selected as the final model based on **MAE**, which was the primary metric used for the vehicle pricing task.

Although Random Forest achieved better RMSE and R² on the validation set, XGBoost achieved the lowest MAE among the evaluated models.

## Independent Test Evaluation

After model selection, the final XGBoost model was retrained using the complete processed training dataset and evaluated on the independent test dataset.

The final evaluation produced:

- **MAE:** $1,995.35
- **RMSE:** $3,443.55
- **R²:** 0.9079

The model maintained strong performance on the independent test data, with an R² close to the validation result.

The independent test dataset contained 11,488 records initially. After preprocessing and target-related row filtering, 11,460 records remained for final evaluation.

## Output

The final predictions are saved to:

```text
outputs/test_predictions.csv
```

The prediction file contains 11,460 predictions for the processed independent test records.

The project also generates an experiment log:

```text
outputs/experiment_log.csv
```

This file contains the recorded results from the model experiments.

## Model Training Class

`src/model_trainer.py` contains the reusable `ModelTrainer` class used across the regression experiments.

The class provides a consistent workflow for:

- Model training
- Prediction
- Evaluation using MAE, RMSE, and R²
- Experiment result logging

The experiment results are exported to:

```text
outputs/experiment_log.csv
```

## Setup

### 1. Clone the repository

Clone the repository and move into the project folder.

### 2. Create a virtual environment

Creating a virtual environment is recommended.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the assessment datasets

Place the following files inside the `data/` folder:

```text
data/
├── DatiumTrain.rpt
└── DatiumTest.rpt
```

These files are intentionally excluded from Git.

### 5. Launch Jupyter

```bash
jupyter lab
```

For a clean run, use:

**Kernel → Restart Kernel and Run All Cells**

## Project Structure

```text
pickles-vehicle-price-prediction/
│
├── data/
│   └── Raw assessment datasets
│
├── outputs/
│   ├── experiment_log.csv
│   └── test_predictions.csv
│
├── src/
│   └── model_trainer.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── Vehicle_Price_Prediction.ipynb
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- JupyterLab

