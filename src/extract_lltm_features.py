#!/usr/bin/env python3
"""Extract LLTM features from passage text and compute calibrated difficulty.

Usage: python src/extract_lltm_features.py

Reads data/passages.csv, computes 5 features, writes updated CSV.
LLTM weights (fixed from Kang 2026, EFSL):
  w = [1.5, 0.8, 0.6, 1.2, 0.3]
Features:
  f_sent_var:  sentence-length variance (std/mean of words per sentence)
  f_long_word: fraction of words with >8 characters
  f_mean_sent: mean sentence length / 20
  f_inv_ttr:   1 - (unique words / total words)
  f_ctx_dep:   fraction of context-dependency words (the, this, however, etc.)

Calibration to theta scale:
  b_hat = alpha + beta * lltm_raw
  where alpha, beta are fitted to map lltm_raw range to true_difficulty range.
"""
import csv, re, numpy as np
from pathlib import Path

WEIGHTS = np.array([1.5, 0.8, 0.6, 1.2, 0.3])
CTX_WORDS = {'the','this','that','these','those','it','its','they','them','their',
             'which','who','however','therefore','moreover','thus','hence','whereas'}

def extract(text):
    words = text.split()
    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not words or not sents: return [0]*5
    sl = [len(s.split()) for s in sents]
    return [
        np.std(sl)/(np.mean(sl)+1e-8),
        sum(1 for w in words if len(w)>8)/len(words),
        np.mean(sl)/20.0,
        1.0 - len(set(w.lower() for w in words))/len(words),
        sum(1 for w in words if w.lower() in CTX_WORDS)/len(words),
    ]

if __name__ == "__main__":
    path = Path(__file__).resolve().parent.parent / "data" / "passages.csv"
    rows = list(csv.DictReader(open(path)))
    true_d = np.array([float(r['true_difficulty']) for r in rows])
    lltm_raw = []
    for r in rows:
        f = extract(r['text'])
        r['f_sent_var'], r['f_long_word'], r['f_mean_sent'] = round(f[0],4), round(f[1],4), round(f[2],4)
        r['f_inv_ttr'], r['f_ctx_dep'] = round(f[3],4), round(f[4],4)
        r['lltm_difficulty'] = round(float(np.dot(f, WEIGHTS)), 4)
        lltm_raw.append(r['lltm_difficulty'])
    lltm_raw = np.array(lltm_raw)
    beta = np.std(true_d) / np.std(lltm_raw)
    alpha = np.mean(true_d) - beta * np.mean(lltm_raw)
    for r, raw in zip(rows, lltm_raw):
        r['lltm_calibrated'] = round(float(alpha + beta * raw), 4)
    fields = list(rows[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    from scipy.stats import spearmanr
    rho, _ = spearmanr(true_d, [float(r['lltm_calibrated']) for r in rows])
    print(f"Features extracted. LLTM rho = {rho:.3f}. Calibration: b = {alpha:.3f} + {beta:.3f} * raw")
