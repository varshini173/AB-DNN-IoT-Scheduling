"""Shared preprocessing, labels and wireless simulation for the AB-DNN web app.
Mirrors Sections 3, 4, 9 and 10 of the notebook exactly."""
import collections
import numpy as np
import pandas as pd
from scipy.special import erfc

CLASS_NAMES = ["LOW", "MEDIUM", "HIGH"]
IMPORTANCE_MAP = {
    "normal": "LOW",
    "scanning": "MEDIUM", "password": "MEDIUM", "xss": "MEDIUM", "mitm": "MEDIUM",
    "dos": "HIGH", "ddos": "HIGH", "injection": "HIGH", "backdoor": "HIGH", "ransomware": "HIGH",
}
DEADLINE_MS = {"LOW": 200, "MEDIUM": 50, "HIGH": 10}


def importance_of(type_series):
    t = type_series.astype(str).str.strip().str.lower()
    return t.map(IMPORTANCE_MAP).fillna("MEDIUM")


def packet_bits(raw):
    cols = [c for c in ["src_bytes", "dst_bytes"] if c in raw.columns]
    if not cols:
        return np.full(len(raw), 1024.0)
    payload = raw[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)
    return (payload * 8).clip(256, 4096).astype(float).to_numpy()


def transform(raw, bundle):
    """Raw TON_IoT network rows -> scaled feature matrix the trained models expect."""
    work = raw.copy()
    work.columns = [str(c).strip() for c in work.columns]
    work = work.replace("-", np.nan)
    parts = []
    for c in bundle["num_cols"]:
        col = pd.to_numeric(work[c], errors="coerce") if c in work.columns else pd.Series(np.nan, index=work.index)
        col = col.fillna(bundle["medians"][c]).astype(float)
        if c in bundle["log_cols"]:
            col = np.log1p(col.clip(lower=0))
        parts.append(col.rename(c))
    num = pd.concat(parts, axis=1)
    present = [c for c in bundle["cat_cols"] if c in work.columns]
    cat = pd.get_dummies(work[present].astype(str), prefix=present, dtype=float) if present else pd.DataFrame(index=work.index)
    X = pd.concat([num, cat], axis=1).reindex(columns=bundle["features"], fill_value=0.0)
    X = X.fillna(0.0).to_numpy(dtype=float)
    return np.clip(bundle["scaler"].transform(X), -10, 10).astype("float32")


# ---------------- wireless system model (same as notebook Section 9-10) ----------------
B_HZ, SLOT_S, BITS_PER_HZ = 20e6, 1e-3, 2
SLOT_CAPACITY = int(B_HZ * SLOT_S * BITS_PER_HZ)
MAX_TX, MAX_AGE = 4, 500
DEADLINE_SLOTS = np.array([DEADLINE_MS[c] for c in CLASS_NAMES])


def ber_qpsk(snr_lin):
    return 0.5 * erfc(np.sqrt(snr_lin))


def per_from_ber(ber, bits):
    return 1 - (1 - ber) ** bits


def simulate(pred_cls, true_cls, bits, load, snr_db, n_slots=3000, seed=42, fifo=False):
    rng = np.random.default_rng(seed)
    nc = len(CLASS_NAMES)
    lam = load * SLOT_CAPACITY / bits.mean()
    snr_lin = 10 ** (snr_db / 10)
    queues = [collections.deque()] if fifo else [collections.deque() for _ in range(nc)]
    lat = [[] for _ in range(nc)]
    sent = good = useful = 0
    on_time, generated = np.zeros(nc), np.zeros(nc)
    tx_att = tx_fail = 0
    for t in range(n_slots):
        for _ in range(rng.poisson(lam)):
            i = rng.integers(len(true_cls))
            pkt = [t, int(true_cls[i]), int(bits[i]), 0]
            generated[pkt[1]] += 1
            queues[0 if fifo else int(pred_cls[i])].append(pkt)
        cap = SLOT_CAPACITY
        order = [0] if fifo else list(range(nc - 1, -1, -1))
        for q in (queues[k] for k in order):
            retry = []
            while q and cap >= q[0][2]:
                pkt = q.popleft()
                if t - pkt[0] > MAX_AGE:
                    continue
                cap -= pkt[2]; sent += pkt[2]; pkt[3] += 1; tx_att += 1
                gamma = snr_lin * rng.exponential(1.0)          # fading draw first (same order as notebook)
                if rng.random() < per_from_ber(ber_qpsk(gamma), pkt[2]):
                    tx_fail += 1
                    if pkt[3] < MAX_TX:
                        retry.append(pkt)
                    continue
                d = t - pkt[0] + 1
                lat[pkt[1]].append(d); good += pkt[2]
                if d <= DEADLINE_SLOTS[pkt[1]]:
                    on_time[pkt[1]] += 1; useful += pkt[2]
            q.extendleft(reversed(retry))
    T = n_slots * SLOT_S
    hi = nc - 1
    return {
        "high_mean_latency_ms": float(np.mean(lat[hi])) if lat[hi] else None,
        "high_p95_latency_ms": float(np.percentile(lat[hi], 95)) if lat[hi] else None,
        "high_deadline_miss": float(1 - on_time[hi] / max(generated[hi], 1)),
        "all_deadline_miss": float(1 - on_time.sum() / max(generated.sum(), 1)),
        "throughput_mbps": sent / T / 1e6,
        "goodput_mbps": good / T / 1e6,
        "semantic_goodput_mbps": useful / T / 1e6,
        "spectral_eff": good / T / B_HZ,
        "per": tx_fail / max(tx_att, 1),
    }
