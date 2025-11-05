import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/"data"; FIGS = ROOT/"figs"; FIGS.mkdir(exist_ok=True)
META = FIGS/"secom_gru_meta.json"

# ---------------- Load SECOM (local) ----------------
X = pd.read_csv(DATA/"secom.data", sep=r"\s+", header=None, engine="python")
L = pd.read_csv(DATA/"secom_labels.data", sep=r"\s+", header=None, engine="python")
y = L.iloc[:,0].map({-1:0, 1:1}).astype(int).values
ts = None
if L.shape[1] >= 2:
    try:
        ts = pd.to_datetime(L.iloc[:,1], errors="coerce")
    except Exception:
        ts = None

X.columns = [f"V{i+1}" for i in range(X.shape[1])]

# sort by time if available
order = np.arange(len(X))
if ts is not None and ts.notna().any():
    order = np.argsort(ts.values)
X = X.iloc[order].reset_index(drop=True)
y = y[order]

# ---------------- Impute + Scale ----------------
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
imp = SimpleImputer(strategy="median").fit(X)
Ximp = imp.transform(X)
sc  = StandardScaler().fit(Ximp)
Xsc = sc.transform(Ximp).astype("float32")

# ---------------- Make sequences ----------------
T = 10  # window length
def make_seq(X2, y2, T):
    # label: last step in window
    N = X2.shape[0] - T + 1
    Xs = np.zeros((N, T, X2.shape[1]), dtype="float32")
    ys = np.zeros((N,), dtype="int32")
    for i in range(N):
        Xs[i] = X2[i:i+T]
        ys[i] = y2[i+T-1]
    return Xs, ys

Xs, ys = make_seq(Xsc, y, T)

# ---------------- Chronological split (70/15/15) ----------------
n = len(ys)
i1 = int(0.70*n); i2 = int(0.85*n)
Xtr, ytr = Xs[:i1], ys[:i1]
Xva, yva = Xs[i1:i2], ys[i1:i2]
Xte, yte = Xs[i2:], ys[i2:]

# ---------------- Build & train GRU ----------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L
from sklearn.utils.class_weight import compute_class_weight

inp = L.Input(shape=(T, Xtr.shape[-1]))
x = L.GRU(64, return_sequences=False)(inp)
x = L.Dropout(0.3)(x)
out = L.Dense(1, activation="sigmoid")(x)
model = keras.Model(inp, out)
model.compile(optimizer="adam", loss="binary_crossentropy")

cls_w = compute_class_weight(class_weight="balanced", classes=np.array([0,1]), y=ytr)
class_weight = {0: float(cls_w[0]), 1: float(cls_w[1])}

cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
hist = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=32, class_weight=class_weight, verbose=0, callbacks=cb)

# ---------------- Evaluate ----------------
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, classification_report, confusion_matrix
proba = model.predict(Xte, verbose=0).ravel()
auc = roc_auc_score(yte, proba)
ap  = average_precision_score(yte, proba)

# best-F1 threshold
prec, rec, thr = precision_recall_curve(yte, proba)
f1s = [0.0] + [2*p*r/(p+r) if (p+r)>0 else 0.0 for p,r in zip(prec[1:], rec[1:])]
best_i = int(np.argmax(f1s)); best_thr = float(thr[best_i-1]) if best_i>0 else 0.5
pred = (proba >= best_thr).astype(int)

# text report
rep = classification_report(yte, pred, digits=4)
cm = confusion_matrix(yte, pred)

# ---------------- Plots ----------------
# training curve
plt.figure(figsize=(4,3))
plt.plot(hist.history["loss"], label="train")
plt.plot(hist.history["val_loss"], label="val")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("GRU training")
plt.legend(); plt.tight_layout(); plt.savefig(FIGS/"secom_gru_train.png", dpi=150); plt.close()

# ROC
fpr, tpr, _ = roc_curve(yte, proba)
plt.figure(figsize=(4,3))
plt.plot(fpr, tpr, lw=2, label=f"AUC={auc:.3f}")
plt.plot([0,1],[0,1],"--")
plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("SECOM GRU — ROC")
plt.legend(); plt.tight_layout(); plt.savefig(FIGS/"secom_gru_roc.png", dpi=150); plt.close()

# PR
plt.figure(figsize=(4,3))
plt.plot(rec, prec, lw=2, label=f"AP={ap:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("SECOM GRU — PR")
plt.legend(); plt.tight_layout(); plt.savefig(FIGS/"secom_gru_pr.png", dpi=150); plt.close()

# Confusion
plt.figure(figsize=(3.6,3.2))
plt.imshow(cm, interpolation="nearest")
plt.title(f"Confusion (thr={best_thr:.2f})"); plt.xlabel("Predicted"); plt.ylabel("True")
for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha="center",va="center")
plt.tight_layout(); plt.savefig(FIGS/"secom_gru_cm.png", dpi=150); plt.close()

# ---------------- Save meta ----------------
json.dump({
    "seq_len": T,
    "splits": {"train": len(ytr), "val": len(yva), "test": len(yte)},
    "test_auc": float(auc),
    "test_ap": float(ap),
    "best_threshold": best_thr,
    "class_weight": class_weight
}, open(META,"w"), indent=2)

print("[DONE] AUC=%.3f  AP=%.3f  thr=%.2f" % (auc, ap, best_thr))
print("Saved:", META, str(FIGS/'secom_gru_roc.png'), str(FIGS/'secom_gru_pr.png'), str(FIGS/'secom_gru_cm.png'))
