import os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SIM = ROOT/"sim"; FIGS = ROOT/"figs"; FIGS.mkdir(exist_ok=True)

def load_queue_ts(tag):
    df = pd.read_csv(SIM/f"queue_ts_{tag}.csv")
    # columns: time_min, S1_queue, S2_queue, S3_queue
    return df[["S1_queue","S2_queue","S3_queue"]].values.astype("float32")

runs = []
for tag in ["baseline","risk","mitigate"]:
    p = SIM/f"queue_ts_{tag}.csv"
    if p.exists():
        runs.append((tag, load_queue_ts(tag)))

if not runs:
    raise SystemExit("No queue time-series found. Run Phase 4 sims first.")

# ---- Build dataset: windows over time across runs ----
T = 20  # window length
Xs = []  # (N, T, N_nodes, F_node=1)
ys = []  # class 0/1/2 for S1/S2/S3 bottleneck
for tag, arr in runs:
    # arr shape: (time, 3)
    for i in range(len(arr)-T):
        win = arr[i:i+T]              # (T, 3)
        Xs.append(win[:, :, None])    # add feature dim -> (T, 3, 1)
        # label = bottleneck at final step = argmax queue at time i+T-1
        ys.append(int(np.argmax(arr[i+T-1])))
Xs = np.array(Xs, dtype="float32")    # (N, T, 3, 1)
ys = np.array(ys, dtype="int32")

# ---- Graph adjacency for S1-S2-S3 line ----
A = np.array([[1,1,0],
              [1,1,1],
              [0,1,1]], dtype="float32")  # self-loops included
# normalize A_hat
D = np.diag(1.0/np.sqrt(A.sum(axis=1)))
A_hat = D @ A @ D

# ---- Train/test split (chronological across concatenated windows) ----
n = len(ys); i1 = int(0.8*n)
Xtr, ytr = Xs[:i1], ys[:i1]
Xte, yte = Xs[i1:], ys[i1:]

# ---- Keras model: TimeDistributed(GraphConv) -> GRU -> Dense(3) ----
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers as L

A_const = tf.constant(A_hat, dtype=tf.float32)

class GraphConv(L.Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        fin = int(input_shape[-1])              # input feature dim
        self.W = self.add_weight(
            shape=(fin, self.units),
            initializer="glorot_uniform",
            trainable=True,
            name="W"
        )
        super().build(input_shape)

    def call(self, X):
        # X: (batch, nodes, fin)
        AX = tf.einsum('ij,bjf->bif', A_const, X)     # (b, nodes, fin)
        return tf.einsum('bif,fu->biu', AX, self.W)   # (b, nodes, units)

    def compute_output_shape(self, input_shape):
        # -> (batch, nodes, units)
        return (input_shape[0], input_shape[1], self.units)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


inp = L.Input(shape=(T, 3, 1))        # (time, nodes, features)
x = L.TimeDistributed(GraphConv(8))(inp)    # -> (T, 3, 8)
x = L.TimeDistributed(L.ReLU())(x)
x = L.TimeDistributed(L.Flatten())(x)       # -> (T, 3*8)
x = L.GRU(32)(x)                             # temporal aggregation
out = L.Dense(3, activation="softmax")(x)    # class per station

model = keras.Model(inp, out)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

cb = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
hist = model.fit(Xtr, ytr, validation_split=0.2, epochs=40, batch_size=64, verbose=0, callbacks=cb)

# ---- Evaluate ----
proba = model.predict(Xte, verbose=0)
pred = proba.argmax(axis=1)

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
acc = accuracy_score(yte, pred)
f1m = f1_score(yte, pred, average="macro")
cm = confusion_matrix(yte, pred)

# ---- Plots ----
plt.figure(figsize=(4,3))
plt.plot(hist.history["loss"], label="train")
plt.plot(hist.history["val_loss"], label="val")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("ST-GNN-lite training")
plt.legend(); plt.tight_layout(); plt.savefig(FIGS/"stgnn_train.png", dpi=150); plt.close()

plt.figure(figsize=(3.6,3.2))
plt.imshow(cm, interpolation="nearest")
plt.title("ST-GNN-lite Confusion"); plt.xlabel("Predicted"); plt.ylabel("True")
for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha="center",va="center")
plt.tight_layout(); plt.savefig(FIGS/"stgnn_confusion.png", dpi=150); plt.close()

json.dump({
    "seq_len": T,
    "acc": float(acc),
    "macro_f1": float(f1m),
    "classes": ["S1","S2","S3"]
}, open(FIGS/"stgnn_meta.json","w"), indent=2)

print(f"[DONE] ST-GNN-lite  acc={acc:.3f}  macroF1={f1m:.3f}")
print("Saved: figs/stgnn_meta.json  figs/stgnn_confusion.png")

model.save(FIGS/"stgnn_model.keras")
np.savetxt("sim/stgnn_preds_windows.csv", pred, fmt="%d")  # 0=S1,1=S2,2=S3 per window

