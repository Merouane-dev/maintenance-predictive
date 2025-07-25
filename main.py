#!/usr/bin/env python3
# ============================================================
#  Maintenance Predictive Demo
#  Random Forest (supervisé) + Isolation Forest (non supervisé)
#  Dataset : AI4I 2020 Predictive Maintenance (local CSV)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_curve, auc)

# ------------------------------------------------------------
# 0. Paramètres généraux
# ------------------------------------------------------------
DATA_FILE = "ai4i2020.csv"      # ton fichier local
RANDOM_STATE = 42
FIG_DPI = 300

# ------------------------------------------------------------
# 1. Chargement des données
# ------------------------------------------------------------
assert Path(DATA_FILE).exists(), f"Fichier {DATA_FILE} introuvable."

df = pd.read_csv(DATA_FILE)

# ------------------------------------------------------------
# 2. Préparation / Nettoyage
# ------------------------------------------------------------
# Colonnes d'identifiants : on les supprime si elles existent
cols_to_drop = ["UDI", "Product ID", "Failure Type", "Target"]  # 'Target' n'existe pas ici mais on ignore si absente
df = df.drop(columns=cols_to_drop, errors="ignore")

# Détection automatique du nom de la cible
if "Machine failure" in df.columns:
    target_col = "Machine failure"
elif "Target" in df.columns:
    target_col = "Target"
else:
    raise ValueError("Impossible de trouver la colonne cible ('Machine failure' ou 'Target').")

# Variables explicatives / cible
X = df.drop(columns=[target_col])
y = df[target_col]

# Détection des colonnes catégorielles
cat_cols = [c for c in X.columns if X[c].dtype == "object"]
num_cols = [c for c in X.columns if c not in cat_cols]

# Pipeline de prétraitement
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# ------------------------------------------------------------
# 3. Split train/test
# ------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

# ------------------------------------------------------------
# 4. RANDOM FOREST (supervisé)
# ------------------------------------------------------------
rf_clf = Pipeline(
    steps=[
        ("prep", preprocess),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=RANDOM_STATE)
         )
    ]
)
rf_clf.fit(X_train, y_train)

# Prédictions & rapport
y_pred = rf_clf.predict(X_test)
y_proba = rf_clf.predict_proba(X_test)[:, 1]

report_txt = classification_report(y_test, y_pred, digits=3)
print("\n===== Random Forest – Rapport de classification =====")
print(report_txt)

# Sauvegarde du rapport
with open("report_random_forest.txt", "w", encoding="utf-8") as f:
    f.write(report_txt)

# Matrice de confusion
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No fail", "Fail"])
disp.plot(cmap="Blues")
plt.title("Random Forest – Matrice de confusion")
plt.savefig("confusion_matrix_rf.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()

# Courbe ROC
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
plt.xlabel("Taux de faux positifs (FPR)")
plt.ylabel("Taux de vrais positifs (TPR)")
plt.title("Random Forest – Courbe ROC")
plt.legend()
plt.grid(alpha=.3)
plt.savefig("roc_curve_rf.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()

# Importance des variables
rf_model = rf_clf.named_steps["rf"]
# Récupérer les noms de features après One-Hot
ohe = rf_clf.named_steps["prep"].named_transformers_["cat"]
ohe_features = ohe.get_feature_names_out(cat_cols) if len(cat_cols) else np.array([])
feature_names = list(ohe_features) + num_cols

importances = rf_model.feature_importances_
order = np.argsort(importances)[::-1]

top_n = 10 if len(feature_names) >= 10 else len(feature_names)
plt.barh(np.array(feature_names)[order][:top_n][::-1],
         importances[order][:top_n][::-1])
plt.xlabel("Importance")
plt.title("Random Forest – Top features")
plt.tight_layout()
plt.savefig("feature_importance_rf.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()

# ------------------------------------------------------------
# 5. ISOLATION FOREST (non supervisé)
# ------------------------------------------------------------
iso_model = Pipeline(
    steps=[
        ("prep", preprocess),
        ("iso", IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=RANDOM_STATE)
         )
    ]
)

# On entraîne sur les données "normales" (sans panne)
X_train_norm = X_train[y_train == 0]
iso_model.fit(X_train_norm)

# Prédiction (-1 = anomalie)
iso_raw = iso_model.predict(X_test)
y_pred_iso = (iso_raw == -1).astype(int)

cm_iso = confusion_matrix(y_test, y_pred_iso)
print("\n===== Isolation Forest – Matrice de confusion =====")
print(cm_iso)

# Scores d'anomalie
scores = iso_model.named_steps["iso"].decision_function(
    iso_model.named_steps["prep"].transform(X_test)
)

plt.hist(scores, bins=50)
plt.axvline(0, color="red", linestyle="--")
plt.title("Isolation Forest – Distribution des scores")
plt.xlabel("Score (plus petit = plus anormal)")
plt.ylabel("Fréquence")
plt.savefig("scores_isolation_forest.png", dpi=FIG_DPI, bbox_inches="tight")
plt.close()

# Sauvegarder un mini rapport Isolation Forest
rep_iso = classification_report(y_test, y_pred_iso, digits=3)
with open("report_isolation_forest.txt", "w", encoding="utf-8") as f:
    f.write(rep_iso)

print("\n===== Isolation Forest – Rapport =====")
print(rep_iso)

# ------------------------------------------------------------
# 6. Fin
# ------------------------------------------------------------
print("\n⏩ Figures enregistrées :")
print("  - confusion_matrix_rf.png")
print("  - roc_curve_rf.png")
print("  - feature_importance_rf.png")
print("  - scores_isolation_forest.png")
print("⏩ Rapports enregistrés :")
print("  - report_random_forest.txt")
print("  - report_isolation_forest.txt")
print("✅ Script terminé sans erreur.")
