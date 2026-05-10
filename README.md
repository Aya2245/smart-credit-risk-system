# 💳 Credit Risk Predictor

> **Real-time credit risk classification powered by a full ML pipeline — from raw financial data to instant High / Low risk decisions.**

Built with **Streamlit** and **Scikit-learn**, this application predicts whether a client is a **High Risk** or **Low Risk** borrower by analysing payment history, credit utilisation, and behavioural financial patterns.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🎛️ Interactive UI | Clean Streamlit interface with user-friendly financial inputs |
| ⚡ Real-time Prediction | Instant risk classification on form submission |
| 🔁 Full ML Pipeline | Preprocessing → Feature Engineering → Scaling → PCA → SVM, all in one pipeline |
| 🧮 Feature Engineering | Derived ratios and behavioural signals computed automatically |
| 🏷️ Risk Classification | Clear **High Risk** / **Low Risk** output with confidence context |

---

## 🧠 Machine Learning Pipeline

The model is trained end-to-end and serialised as a single `model.pkl` pipeline object:

```
Raw Input
    │
    ▼
① Data Preprocessing     — missing values, invalid category fixes, outlier capping (IQR)
    │
    ▼
② Feature Engineering    — UTILIZATION_RATIO, AVG_BILL, PAY_RATIO_1, INCOME_UTIL_RATIO
    │
    ▼
③ Encoding               — binary remapping, one-hot encoding
    │
    ▼
④ Standard Scaling       — StandardScaler (numerical cols only)
    │
    ▼
⑤ PCA                    — dimensionality reduction (95% variance retained)
    │
    ▼
⑥ SVM Classifier         — RBF kernel, class_weight='balanced', tuned via GridSearchCV
    │
    ▼
Risk Prediction  →  🔴 High Risk  |  🟢 Low Risk
```

---

## 📊 Input Features

### 👤 Customer Profile
| Feature | Type | Description |
|---|---|---|
| `LIMIT_BAL` | Continuous | Credit limit in NT dollars |
| `AGE` | Continuous | Client age |
| `SEX` | Binary | Gender (encoded) |
| `EDUCATION` | Ordinal | Education level (1–4) |
| `INCOME` | Continuous | Estimated monthly income |

### 💰 Financial Behaviour
| Feature | Type | Description |
|---|---|---|
| `AVG_BILL` | Engineered | Average bill amount over 6 months |
| `AVG_PAY_AMT` | Engineered | Average payment amount over 6 months |
| `UTILIZATION_RATIO` | Engineered | `BILL_AMT1 / LIMIT_BAL` |
| `PAY_RATIO_1` | Engineered | `PAY_AMT1 / BILL_AMT2` |
| `INCOME_UTIL_RATIO` | Engineered | `INCOME / LIMIT_BAL` |
| `DEFAULT_FLAG` | Engineered | Binary indicator of prior default signals |

### 📅 Payment History (raw)
| Group | Columns |
|---|---|
| Repayment status | `PAY_0`, `PAY_2` … `PAY_6` |
| Bill amounts | `BILL_AMT1` … `BILL_AMT6` |
| Payment amounts | `PAY_AMT1` … `PAY_AMT6` |

> Repayment status codes: `-1` = paid on time, `1`–`8` = months overdue.

---

## 🛠️ Tech Stack

| Layer | Library |
|---|---|
| Web UI | `streamlit` |
| Data Handling | `pandas`, `numpy` |
| ML Pipeline | `scikit-learn` |
| Model Serialisation | `pickle` |
| Language | Python 3.10+ |

---

## 📂 Project Structure

```
credit-risk-predictor/
│
├── app.py                  # Streamlit application entry point
├── model.pkl               # Serialised ML pipeline (scaler + PCA + SVM)
├── requirements.txt        # Python dependencies
└── README.md               # You are here
```

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/credit-risk-predictor.git
cd credit-risk-predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 🗂️ Data Source

This project uses the **[Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)** dataset from the UCI Machine Learning Repository.

- **30,000** client records × **24** features + 1 target
- Target: `default payment next month` (binary: 0 = no default, 1 = default)
- Class imbalance: ~78% no-default / ~22% default → handled with `class_weight='balanced'`

---

