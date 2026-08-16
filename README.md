# Loan Prediction – Feature Engineering & Data Preprocessing for Machine Learning

**AnalystLab Africa | Data Science Internship Programme**
**Week 2 Project – Wilson Moses**

## Business Scenario

Following Week 1's exploratory data analysis, this project prepares the Loan Prediction dataset for machine learning. The goal is to transform raw loan application data into a clean, encoded, scaled, machine-learning-ready dataset while documenting every preprocessing decision — supporting a future model that predicts whether a loan application should be approved.

## Business Questions Answered

1. Which features are most relevant to predicting loan approval?
2. Which variables require encoding, and which require scaling?
3. Are there redundant or highly correlated features?
4. How were missing values and outliers handled?
5. What preprocessing techniques improved dataset quality?
6. Is the dataset ready for machine learning?

## Dataset

- **Source:** [Loan Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset)
- **Records:** 614 loan applications, 13 original columns
- **Target variable:** `Loan_Status` (Y = approved, N = not approved)

## Project Workflow

### 1. Data Inspection
- Loaded the dataset and reviewed structure, data types, shape (614 rows × 13 columns), duplicates (0 found), and summary statistics.
- Identified missing values across 7 columns: `Gender`, `Married`, `Dependents`, `Self_Employed`, `LoanAmount`, `Loan_Amount_Term`, `Credit_History`.

### 2. Data Cleaning
- **Categorical missing values** (`Gender`, `Married`, `Dependents`, `Self_Employed`) → imputed with **mode**, since each represents a small share of observations and no natural "unknown" category exists.
- **`LoanAmount`** → imputed with **median**, chosen over mean because the distribution contains extreme high-income/high-loan outliers.
- **`Loan_Amount_Term`** → imputed with **mode**, since loan term is a discrete, repeatable value (360 months dominates) rather than a continuous average.
- **`Credit_History`** → imputed with **mode**, treated as a binary categorical flag.
- Verified zero missing values and zero duplicates after cleaning.

### 3. Feature Engineering
- Dropped `Loan_ID` (unique identifier, no predictive value).
- Created **`TotalIncome`** = `ApplicantIncome` + `CoapplicantIncome`, to capture full household earning capacity.
- Created **`LoanIncomeRatio`** = `LoanAmount` / `TotalIncome`, to capture affordability relative to income rather than loan size alone.
- Renamed all columns to `snake_case` for consistency and readability.

### 4. Feature Encoding
| Variable | Technique | Rationale |
|---|---|---|
| `gender`, `married`, `education`, `self_employed` | Binary (label) encoding | Naturally binary categories |
| `credit_history` | Type correction (float → int) | Already binary, needed dtype cleanup |
| `dependents` | Custom numerical encoding (`3+` → `3`) | Ordinal count, `3` treated as "3 or more" |
| `property_area` | One-hot encoding (drop-first) | Nominal category with 3 levels; `Rural` set as reference category |
| `loan_status` (target) | Binary encoding | Y → 1, N → 0 |

### 5. Feature Scaling
- Applied **StandardScaler** to all continuous numerical features (`applicant_income`, `coapplicant_income`, `loan_amount`, `loan_amount_term`, `total_income`, `loan_income_ratio`).
- Chosen because these features have very different scales and ranges (e.g., income in the thousands vs. ratios below 1), and several downstream models (e.g., logistic regression) are sensitive to feature magnitude.

### 6. Outlier Detection
- Used **boxplots** and the **IQR method** on scaled numerical features.
- Outliers were **retained** rather than removed — they reflect genuine high-income or high-loan applicants, not data errors, and are informative for loan approval prediction.

| Feature | Outliers detected (IQR) |
|---|---|
| `applicant_income` | 50 |
| `total_income` | 50 |
| `loan_amount` | 41 |
| `loan_income_ratio` | 25 |
| `coapplicant_income` | 18 |
| `loan_amount_term` | Not informative (IQR = 0) |

### 7. Feature Selection
- **Correlation heatmap** across all engineered/encoded features.
- Identified one high-correlation pair (≥ 0.80): `applicant_income` ↔ `total_income` (0.89) — expected, since `total_income` is derived from `applicant_income`. **`applicant_income` was dropped from the ML-ready dataset** to avoid redundancy.
- **Target correlation:** `credit_history` (0.54) is by far the strongest linear correlate of loan approval, followed by `property_area_Semiurban`, `married`, and `loan_income_ratio`.
- **Random Forest feature importance** confirms `credit_history` as the top predictor, followed by `loan_income_ratio`, `total_income`, `applicant_income`, and `loan_amount`.

### 8. Final Datasets Produced
| Dataset | Rows | Columns | Missing values | Duplicates |
|---|---|---|---|---|
| `loan_prediction_cleaned.csv` | 614 | 14 | 0 | 0 |
| `loan_prediction_ml_ready.csv` | 614 | 13 | 0 | 0 |

The ML-ready dataset is fully numeric, encoded, scaled, and free of redundant features — ready for model training.

## Repository Structure

```
├── README.md
├── Load_Prediction_Data_Inspection.ipynb      # Full preprocessing notebook
├── data/
│   ├── Prediction_Loan_Train.csv              # Original dataset
│   ├── loan_prediction_cleaned.csv            # Cleaned dataset
│   └── loan_prediction_ml_ready.csv           # Machine-learning-ready dataset
├── reports/
│   ├── ABC_LoanPrediction_BusinessUnderstanding_WilsonMoses.docx
│   └── ABC_LoanPrediction_DataPreprocessing_WilsonMoses.docx
└── ABC_LoanPrediction_Presentation_WilsonMoses.pptx
```

## Tools & Libraries

- **Environment:** Python 3, Jupyter Notebook (PyCharm)
- **Libraries:** pandas, NumPy, matplotlib, seaborn, scikit-learn (`StandardScaler`, `RandomForestClassifier`, `train_test_split`)

## Author

**Wilson Moses** — Junior Data Scientist, Data Science Internship Programme, AnalystLab Africa Consulting
