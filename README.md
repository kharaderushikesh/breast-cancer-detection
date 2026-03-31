# Logistic Regression — Breast Cancer Classification

A complete, beginner-friendly machine learning project using Logistic Regression to classify breast cancer tumors as malignant or benign.

---

## 📁 Project Structure


## 📁 Project Structure

logistic_regression_project/  
│  
├── logistic_regression.py   ← Main ML script (run this)  
├── requirements.txt         ← Python dependencies  
├── README.md                ← You are here  
│ 
├── [BREAST_CANCER_DETECTION.html](./BREAST_CANCER_DETECTION.html) ← Click to open HTML report

└── outputs/                 ← Auto-created when you run the script  
├── dataset_head.csv         First 5 rows of the dataset  
├── metrics.csv              Accuracy, Precision, Recall, F1  
├── confusion_matrix.png     Confusion matrix heatmap  
├── metrics_chart.png        Bar chart of all 4 metrics  
├── precision_recall_curve.png  
├── roc_curve.png  
└── feature_importance.png   Top 15 features by coefficient  

---

## ⚙️ Setup & Run

### Step 1 — Install Python
Python 3.8 or higher is required.

Check version:

python --version


---

### Step 2 — Create Virtual Environment (Recommended)

**Windows:**

python -m venv venv
venv\Scripts\activate


**Mac / Linux:**

python3 -m venv venv
source venv/bin/activate


---

### Step 3 — Install Dependencies

pip install -r requirements.txt


---

### Step 4 — Run the Project

python logistic_regression.py


All output files will be saved in the `outputs/` folder automatically.

---

## 🧠 What the Script Does

| Step | Action |
|------|--------|
| A | Load the Breast Cancer Wisconsin dataset (569 samples, 30 features) |
| B | Explore and preview the data |
| C | Split into 80% train / 20% test (stratified) |
| D | Standardize features using StandardScaler |
| E | Train a Logistic Regression model |
| F | Predict on the test set |
| G | Calculate Accuracy, Precision, Recall, F1-Score |
| H | Plot and save confusion matrix |
| I | Plot and save metrics bar chart |
| J | Plot and save Precision-Recall curve |
| K | Plot and save ROC curve |
| L | Plot and save feature importances |

---

## 📊 Expected Output (Console)

============================================================
LOGISTIC REGRESSION — BREAST CANCER CLASSIFICATION

[1] Dataset Loaded
Samples : 569
Features : 30
Classes : ['malignant', 'benign']

[6] Evaluation Metrics
────────────────────────────────────
Accuracy : 0.9825 (98.25%)
Precision : 0.9861 (98.61%)
Recall : 0.9861 (98.61%)
F1-Score : 0.9861 (98.61%)
────────────────────────────────────


---

## 📌 Metric Quick Reference

| Metric | Formula | Best Use Case |
|--------|---------|--------------|
| **Accuracy** | (TP + TN) / Total | Balanced datasets |
| **Precision** | TP / (TP + FP) | Reduce false positives |
| **Recall** | TP / (TP + FN) | Reduce false negatives |
| **F1-Score** | 2 × (P × R) / (P + R) | Imbalanced datasets |

---

## 📂 Dataset Info

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569
- **Features:** 30
- **Classes:** Malignant (0), Benign (1)
- **Note:** No download required (built into scikit-learn)

---

## ⚠️ Troubleshooting

**ModuleNotFoundError**

pip install -r requirements.txt


**Python not found**  
Download from: https://www.python.org/downloads/

**pip not found**

python -m pip install -r requirements.txt


---

## 🚀 Future Improvements

- Add Cross-Validation  
- Hyperparameter Tuning (GridSearchCV)  
- Build Streamlit Web App  
- Deploy on cloud (Render / Hugging Face / AWS)  

---

## 👨‍💻 Author

Rushikesh Kharade
(B.Tech Student | Aspiring Data Analyst)
