import os, sys, json, numpy as np, pandas as pd, glob
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
FIGS = ROOT / "figs"; FIGS.mkdir(parents=True, exist_ok=True)
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)

TXT = FIGS / "secom_rf_report.txt"
ROC_PNG = FIGS / "secom_roc.png"
PR_PNG  = FIGS / "secom_pr.png"
CM_PNG  = FIGS / "secom_confusion_matrix.png"
IMP_CSV = FIGS / "secom_feat_importances.csv"
META    = FIGS / "secom_meta.json"

# ---------- 1) Load SECOM (LOCAL FIRST; robust to filename variants) ----------
def find_file(patterns):
    for pat in patterns:
        matches = glob.glob(str(DATA / pat))
        if matches:
            return Path(matches[0])
    return None

def load_secom_local():
    fX = find_file(["secom.data", "SECOM.data", "secom*.data*", "SECOM*.data*"])
    fY = find_file(["secom_labels.data", "SECOM_labels.data", "secom*labels*.data*", "SECOM*labels*.data*"])
    if not fX or not fY:
        raise FileNotFoundError("Could not find local SECOM files under ~/bneck/data/. Expected secom.data and secom_labels.data")
    # UCI files are whitespace-separated; labels file has 2 cols: label, timestamp
    X = pd.read_csv(fX, sep=r"\s+", header=None, engine="python", na_values=["NaN","nan","NULL","?"])
    y_raw = pd.read_csv(fY, sep=r"\s+", header=None, engine="python")
    y = y_raw.iloc[:,0].map({-1:0, 1:1}).astype(int)
    X.columns = [f"V{i+1}" for i in range(X.shape[1])]
    return X, y

def load_secom():
    # Try local first
    try:
        return load_secom_local()
    except Exception as e:
        print("[WARN] Local load failed:", e)
    # Fallback: OpenML → UCI HTTP
    try:
        from sklearn.datasets import fetch_openml
        dat = fetch_openml('secom', version=1, as_frame=True)
        X = dat.data
        y = pd.to_numeric(dat.target, errors='coerce').map({-1:0, 1:1}).astype(int)
        if X.columns.isnull().any():
            X.columns = [f"V{i+1}" for i in range(X.shape[1])]
        return X, y
    except Exception as e:
        print("[WARN] OpenML fetch failed:", e)
    try:
        uciX = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom.data"
        uciY = "https://archive.ics.uci.edu/ml/machine-learning-databases/secom/secom_labels.data"
        X = pd.read_csv(uciX, sep=r"\s+", header=None, engine="python")
        y = pd.read_csv(uciY, sep=r"\s+", header=None, engine="python").iloc[:,0].map({-1:0,1:1}).astype(int)
        X.columns = [f"V{i+1}" for i in range(X.shape[1])]
        return X, y
    except Exception as e:
        raise RuntimeError("Could not load SECOM (local/OpenML/UCI). Please ensure secom.data and secom_labels.data are in ~/bneck/data/.") from e

X, y = load_secom()
print("SECOM loaded. Shape:", X.shape, "  Pos class rate:", y.mean().round(3))

# ---------- 2) Preprocess ----------
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
imp = SimpleImputer(strategy="median").fit(Xtr)
Xtr = imp.transform(Xtr); Xte = imp.transform(Xte)
sc  = StandardScaler().fit(Xtr)
Xtr = sc.transform(Xtr); Xte = sc.transform(Xte)

# ---------- 3) Train baseline ----------
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced"
).fit(Xtr, ytr)

proba = rf.predict_proba(Xte)[:,1]

from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, average_precision_score, precision_recall_curve, roc_curve
pred05 = (proba >= 0.5).astype(int)
report = classification_report(yte, pred05, digits=4)
auc = roc_auc_score(yte, proba)
prec, rec, thr = precision_recall_curve(yte, proba)
ap = average_precision_score(yte, proba)

# Threshold tuned for max F1
f1s = [0.0] + [2*p*r/(p+r) if (p+r)>0 else 0.0 for p,r in zip(prec[1:], rec[1:])]
best_i = int(np.argmax(f1s))
best_thr = thr[best_i-1] if best_i>0 else 0.5
pred_best = (proba >= best_thr).astype(int)
report_best = classification_report(yte, pred_best, digits=4)

# ---------- 4) Save text + meta ----------
with open(TXT, "w") as f:
    f.write("=== SECOM RandomForest (balanced) ===\n")
    f.write(report + "\n")
    f.write(f"ROC-AUC: {auc:.4f}\n")
    f.write(f"PR-AUC (Average Precision): {ap:.4f}\n\n")
    f.write(f"=== Threshold tuned for max F1 (thr={best_thr:.3f}) ===\n")
    f.write(report_best + "\n")

META.write_text(json.dumps({
    "X_shape": list(X.shape),
    "class_balance": float(y.mean()),
    "test_auc": float(auc),
    "test_ap": float(ap),
    "best_threshold": float(best_thr),
}, indent=2))

# ---------- 5) Plots ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ROC
fpr, tpr, _ = roc_curve(yte, proba)
plt.figure(figsize=(4,3))
plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1], ls="--")
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("SECOM RF — ROC"); plt.legend()
plt.tight_layout(); plt.savefig(ROC_PNG, dpi=150); plt.close()

# PR
plt.figure(figsize=(4,3))
plt.plot(rec, prec, lw=2, label=f"AP={ap:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("SECOM RF — Precision-Recall"); plt.legend()
plt.tight_layout(); plt.savefig(PR_PNG, dpi=150); plt.close()

# Confusion matrix at best threshold
cm = confusion_matrix(yte, pred_best)
plt.figure(figsize=(3.6,3.2))
plt.imshow(cm, interpolation="nearest")
plt.title(f"Confusion (thr={best_thr:.2f})"); plt.xlabel("Predicted"); plt.ylabel("True")
for (i,j),v in np.ndenumerate(cm):
    plt.text(j, i, str(v), ha="center", va="center")
plt.tight_layout(); plt.savefig(CM_PNG, dpi=150); plt.close()

# ---------- 6) Feature importances ----------
imp_vals = rf.feature_importances_
feat_names = np.array(X.columns)
top_idx = np.argsort(imp_vals)[-40:][::-1]
pd.DataFrame({"feature": feat_names[top_idx], "importance": imp_vals[top_idx]}).to_csv(IMP_CSV, index=False)

print("\n[SAVED]")
for p in [TXT, ROC_PNG, PR_PNG, CM_PNG, IMP_CSV, META]:
    print(p)
