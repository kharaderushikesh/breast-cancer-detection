# ================================================================
# Logistic Regression Classification — Breast Cancer Detection
# Dataset   : sklearn Breast Cancer Wisconsin
# Libraries : scikit-learn, pandas, numpy, matplotlib, seaborn
# ================================================================

# ── A. IMPORTS ────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os

from sklearn.datasets        import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.metrics         import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc
)

# ── OUTPUT FOLDER ──────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

# ── B. LOAD DATASET ───────────────────────────────────────────
print("=" * 60)
print("  LOGISTIC REGRESSION — BREAST CANCER CLASSIFICATION")
print("=" * 60)

data = load_breast_cancer()
X    = data.data          # shape: (569, 30)
y    = data.target        # 0 = malignant, 1 = benign

df = pd.DataFrame(X, columns=data.feature_names)
df['target'] = y
df['diagnosis'] = df['target'].map({0: 'malignant', 1: 'benign'})

print(f"\n[1] Dataset Loaded")
print(f"    Samples  : {X.shape[0]}")
print(f"    Features : {X.shape[1]}")
print(f"    Classes  : {list(data.target_names)}")
print(f"\n    Class distribution:")
print(f"    {df['diagnosis'].value_counts().to_string()}")
print(f"\n    First 5 rows saved → outputs/dataset_head.csv")
df.head().to_csv("outputs/dataset_head.csv", index=False)

# ── C. TRAIN-TEST SPLIT (80-20, stratified) ───────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y      # preserves class distribution in both splits
)

print(f"\n[2] Train-Test Split (80/20 stratified)")
print(f"    Train size : {len(X_train)}")
print(f"    Test size  : {len(X_test)}")

# ── D. FEATURE SCALING ────────────────────────────────────────
# Fit ONLY on train data to prevent data leakage
scaler      = StandardScaler()
X_train_sc  = scaler.fit_transform(X_train)
X_test_sc   = scaler.transform(X_test)   # use train stats only!

print(f"\n[3] Feature Scaling Applied (StandardScaler)")
print(f"    Mean (train, feature 0): {scaler.mean_[0]:.4f}")
print(f"    Std  (train, feature 0): {scaler.scale_[0]:.4f}")

# ── E. TRAIN MODEL ────────────────────────────────────────────
model = LogisticRegression(
    max_iter=1000,           # ensure convergence
    random_state=42,
    class_weight='balanced'  # handles class imbalance gracefully
)
model.fit(X_train_sc, y_train)

print(f"\n[4] Model Trained: LogisticRegression")
print(f"    Solver     : {model.solver}")
print(f"    Max iter   : {model.max_iter}")
print(f"    Converged  : {model.n_iter_[0]} iterations")

# ── F. PREDICT ────────────────────────────────────────────────
y_pred = model.predict(X_test_sc)
y_prob = model.predict_proba(X_test_sc)[:, 1]  # probability of benign

print(f"\n[5] Predictions made on test set ({len(y_test)} samples)")

# ── G. EVALUATION METRICS ─────────────────────────────────────
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec  = recall_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred)
cm   = confusion_matrix(y_test, y_pred)

print(f"\n[6] Evaluation Metrics")
print(f"    {'─' * 36}")
print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
print(f"    Precision : {prec:.4f}  ({prec*100:.2f}%)")
print(f"    Recall    : {rec:.4f}  ({rec*100:.2f}%)")
print(f"    F1-Score  : {f1:.4f}  ({f1*100:.2f}%)")
print(f"    {'─' * 36}")

print(f"\n[7] Confusion Matrix")
print(f"    [[TN={cm[0,0]}  FP={cm[0,1]}]")
print(f"     [FN={cm[1,0]}  TP={cm[1,1]}]]")

print(f"\n[8] Full Classification Report")
print(f"    {'─' * 50}")
print(classification_report(y_test, y_pred, target_names=data.target_names))

# Save metrics to CSV
metrics_df = pd.DataFrame({
    'Metric':    ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Score':     [acc, prec, rec, f1],
    'Pct':       [f"{v*100:.2f}%" for v in [acc, prec, rec, f1]]
})
metrics_df.to_csv("outputs/metrics.csv", index=False)
print("    Metrics saved → outputs/metrics.csv")

# ── H. PLOT 1: CONFUSION MATRIX ───────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Greys',
    xticklabels=data.target_names,
    yticklabels=data.target_names,
    linewidths=0.8, linecolor='white',
    annot_kws={"size": 18, "weight": "bold"},
    ax=ax
)
ax.set_xlabel('Predicted Label', fontsize=12, labelpad=10)
ax.set_ylabel('Actual Label',    fontsize=12, labelpad=10)
ax.set_title('Confusion Matrix — Logistic Regression', fontsize=13, pad=14)
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.close()
print("\n[9] Plots saved:")
print("    → outputs/confusion_matrix.png")

# ── I. PLOT 2: METRICS BAR CHART ──────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
metrics_names  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [acc, prec, rec, f1]
colors = ['#1a1a1a', '#3a3a3a', '#5a5a5a', '#8a8a8a']

bars = ax.barh(metrics_names, metrics_values, color=colors, height=0.5)
ax.set_xlim(0.95, 1.005)
ax.set_xlabel('Score', fontsize=11)
ax.set_title('Model Performance Metrics', fontsize=13)
ax.bar_label(bars, labels=[f"{v:.4f}" for v in metrics_values],
             padding=4, fontsize=11, fontweight='bold')
ax.spines[['top','right']].set_visible(False)
ax.tick_params(axis='y', labelsize=11)
plt.tight_layout()
plt.savefig("outputs/metrics_chart.png", dpi=150, bbox_inches='tight')
plt.close()
print("    → outputs/metrics_chart.png")

# ── J. PLOT 3: PRECISION-RECALL CURVE ────────────────────────
p_curve, r_curve, thresholds_pr = precision_recall_curve(y_test, y_prob)
pr_auc = auc(r_curve, p_curve)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(r_curve, p_curve, color='black', lw=2, label=f'PR Curve (AUC = {pr_auc:.3f})')
ax.fill_between(r_curve, p_curve, alpha=0.08, color='black')
ax.set_xlabel('Recall',    fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=13)
ax.legend(fontsize=11)
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/precision_recall_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("    → outputs/precision_recall_curve.png")

# ── K. PLOT 4: ROC CURVE ──────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color='black', lw=2, label=f'ROC Curve (AUC = {roc_auc:.3f})')
ax.plot([0,1],[0,1], '--', color='#aaa', lw=1, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.07, color='black')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate',  fontsize=12)
ax.set_title('ROC Curve', fontsize=13)
ax.legend(fontsize=11)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("    → outputs/roc_curve.png")

# ── L. PLOT 5: FEATURE IMPORTANCE (TOP 15) ───────────────────
feature_importance = pd.DataFrame({
    'Feature':    data.feature_names,
    'Coefficient': np.abs(model.coef_[0])
}).sort_values('Coefficient', ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.barh(feature_importance['Feature'], feature_importance['Coefficient'],
               color='#1a1a1a', height=0.6)
ax.set_xlabel('|Coefficient| (Feature Importance)', fontsize=11)
ax.set_title('Top 15 Feature Importances — Logistic Regression', fontsize=12)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
print("    → outputs/feature_importance.png")

# ── DONE ──────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  ALL DONE — check the outputs/ folder for all files")
print(f"{'=' * 60}\n")
