"""AB-DNN for Semantic-Importance-Aware Scheduling in IoT Networks — web demo.
Run:  pip install -r requirements.txt   then   python app.py   and open http://127.0.0.1:5000
"""
import os, io, time
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import joblib
from flask import Flask, jsonify, request, render_template, send_from_directory
import tensorflow as tf

import pipeline as P

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# ---------------- load trained models + data once at start-up ----------------
print("Loading trained models ...")
BUNDLE = joblib.load(os.path.join(BASE, "models", "preprocessing.joblib"))
N_FEAT, N_CLS = len(BUNDLE["features"]), len(BUNDLE["classes"])
L = tf.keras.layers


def build_dnn():          # same architecture as notebook Section 6
    inp = L.Input(shape=(N_FEAT,))
    x = L.Dense(128, activation="relu")(inp); x = L.Dropout(0.3)(x)
    x = L.Dense(64, activation="relu")(x)
    return tf.keras.Model(inp, L.Dense(N_CLS, activation="softmax")(x))


def build_abdnn():
    inp = L.Input(shape=(N_FEAT,))
    alpha = L.Dense(N_FEAT, activation="softmax", name="attention")(L.Dense(64, activation="tanh", name="att_score")(inp))
    xw = L.Rescaling(float(N_FEAT), name="scale")(L.Multiply(name="reweight")([inp, alpha]))
    x = L.Dense(128, activation="relu")(xw); x = L.Dropout(0.3)(x)
    x = L.Dense(64, activation="relu")(x)
    return tf.keras.Model(inp, L.Dense(N_CLS, activation="softmax")(x))


def load(name, builder):
    path = os.path.join(BASE, "models", name)
    try:
        return tf.keras.models.load_model(path + ".keras")
    except Exception as e:  # different Keras version on this laptop -> rebuild and load weights
        print(f"  {name}.keras could not be loaded ({type(e).__name__}); using {name}.weights.h5")
        m = builder(); m.load_weights(path + ".weights.h5"); return m


AB = load("ab_dnn", build_abdnn)
DNN = load("baseline_dnn", build_dnn)
ALPHA = tf.keras.Model(AB.input, AB.get_layer("attention").output)
FEATURES = BUNDLE["features"]

T = np.load(os.path.join(BASE, "data", "test_set.npz"))
X_TEST, Y_TEST, BITS_TEST = T["X"], T["y"].astype(int), T["bits"]
PRED = {
    "Baseline DNN": DNN.predict(X_TEST, verbose=0).argmax(1),
    "AB-DNN": AB.predict(X_TEST, verbose=0).argmax(1),
}
TEST_ACC = {k: float((v == Y_TEST).mean()) for k, v in PRED.items()}
print("Test-set accuracy:", {k: round(v, 4) for k, v in TEST_ACC.items()})

RES = os.path.join(BASE, "data", "results")


def read_table(name):
    df = pd.read_csv(os.path.join(RES, name), index_col=0)
    return {"columns": list(df.columns), "rows": [[i] + [None if pd.isna(v) else float(v) for v in r]
                                                    for i, r in zip(df.index, df.values)]}


@app.route("/")
def index():
    charts = sorted(f for f in os.listdir(os.path.join(BASE, "static", "charts")) if f.endswith(".png"))
    return render_template("index.html", charts=charts)


@app.route("/api/summary")
def summary():
    return jsonify({
        "network": read_table("network_kpis.csv"),
        "classification": read_table("classification_metrics.csv"),
        "improvement": read_table("improvement_percentages.csv"),
        "test_accuracy_loaded_models": TEST_ACC,
        "n_test": int(len(Y_TEST)),
        "n_features": len(FEATURES),
    })


@app.route("/api/simulate", methods=["POST"])
def simulate():
    body = request.get_json(force=True, silent=True) or {}
    load = float(np.clip(float(body.get("load", 0.9)), 0.3, 1.1))
    snr = float(np.clip(float(body.get("snr", 20)), 0, 35))
    seed = int(body.get("seed", 42))
    t0 = time.time()
    out = {}
    for name, pred in [("FIFO (no ML)", None), ("Baseline DNN", PRED["Baseline DNN"]),
                       ("AB-DNN", PRED["AB-DNN"]), ("Oracle", Y_TEST)]:
        out[name] = P.simulate(Y_TEST if pred is None else pred, Y_TEST, BITS_TEST, load, snr,
                               seed=seed, fifo=pred is None)
    return jsonify({"load": load, "snr": snr, "results": out, "seconds": round(time.time() - t0, 2)})


@app.route("/api/classify", methods=["POST"])
def classify():
    try:
        return _classify()
    except Exception as e:                       # show a readable message in the page
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 200


def _classify():
    if "file" in request.files and request.files["file"].filename:
        f = request.files["file"]
        data = f.read()
        if f.filename.lower().endswith((".xlsx", ".xls")):
            raw = pd.read_excel(io.BytesIO(data))
        else:
            raw = pd.read_csv(io.BytesIO(data), low_memory=False)
        source = f.filename
    else:
        raw = pd.read_csv(os.path.join(BASE, "data", "sample_ton_iot_records.csv"), low_memory=False)
        source = "built-in sample (25 records of each TON_IoT type)"
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.head(5000).reset_index(drop=True)
    X = P.transform(raw, BUNDLE)
    p_ab, p_dnn, alpha = AB.predict(X, verbose=0), DNN.predict(X, verbose=0), ALPHA.predict(X, verbose=0)

    type_col = next((c for c in raw.columns if c.lower() == "type"), None)
    truth = P.importance_of(raw[type_col]).tolist() if type_col else None
    rows = []
    for i in range(len(raw)):
        top = np.argsort(alpha[i])[::-1][:3]
        ab_cls = P.CLASS_NAMES[int(p_ab[i].argmax())]
        rows.append({
            "type": str(raw[type_col].iloc[i]) if type_col else "",
            "true": truth[i] if truth else "",
            "dnn": P.CLASS_NAMES[int(p_dnn[i].argmax())],
            "ab": ab_cls,
            "conf": float(p_ab[i].max()),
            "budget": P.DEADLINE_MS[ab_cls],
            "top": [FEATURES[j] for j in top],
        })
    resp = {"source": source, "n": len(rows), "rows": rows[:500],
            "counts": {c: sum(r["ab"] == c for r in rows) for c in P.CLASS_NAMES}}
    if truth:
        resp["accuracy"] = {"AB-DNN": float(np.mean([r["ab"] == r["true"] for r in rows])),
                            "Baseline DNN": float(np.mean([r["dnn"] == r["true"] for r in rows]))}
    return jsonify(resp)


@app.route("/download/sample")
def download_sample():
    return send_from_directory(os.path.join(BASE, "data"), "sample_ton_iot_records.csv", as_attachment=True)


if __name__ == "__main__":
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=False)
