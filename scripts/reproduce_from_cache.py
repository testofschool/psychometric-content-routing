#!/usr/bin/env python3
"""
Reproducibility check: try to rebuild the reported Phase 3 / Phase 4 numbers
from the committed cache (cache/api_cache.json) WITHOUT calling any API.

This script does not modify src/run_experiment.py. It runs two attempts:

  A. Key-based lookup. Rebuild every request exactly as src/run_experiment.py
     does, derive its cache key (md5, as in run_experiment.py, and sha256 of
     the same string, since committed keys are 64 hex chars) and count hits.

  B. Order-based reconstruction (fallback, NOT key-verified). Entries in the
     committed cache are stored in call order (file order == `cached_at`
     order). The entries that carry a `model` field form one complete run
     (24 generations, 36 pairwise, 72 absolute). We assign them to
     (pair/strategy, user, repeat) by assuming run_experiment.py's loop order.
     This assumption cannot be checked against the keys, so its output is a
     reconstruction, not a reproduction of the reported run.

Usage: python scripts/reproduce_from_cache.py
"""
import csv, hashlib, json, re
from math import comb
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = json.load(open(ROOT / "cache" / "api_cache.json"))
PROMPTS = ROOT / "prompts"
MODEL = "llama-3.3-70b-versatile"; JUDGE = "llama-3.1-8b-instant"
STRATS = ["corpus_prefix", "random", "difficulty_only", "pcr"]
PAIRS = [("pcr", "difficulty_only"), ("pcr", "random")]
REPS = 3

corpus = list(csv.DictReader(open(ROOT / "data" / "passages.csv")))
users = list(csv.DictReader(open(ROOT / "data" / "users.csv")))


def keystr(model, system, um, temp, salt=""):
    return f"{model}:{system}:{um}:{temp}:{salt}"


def lookup_hits(strings):
    md5 = sum(hashlib.md5(s.encode()).hexdigest() in CACHE for s in strings)
    sha = sum(hashlib.sha256(s.encode()).hexdigest() in CACHE for s in strings)
    return md5, sha


def attempt_a():
    # Phase 1 requests are fully determined by committed files.
    p1_sys = (PROMPTS / "phase1_difficulty_rating.txt").read_text().split("USER:")[0].replace("SYSTEM:", "").strip()
    p1 = [keystr(MODEL, p1_sys, f"Rate this passage:\n\n{c['text']}", 0.3) for c in corpus]
    # Phase 2 requests depend on routing; only the system prompt + budget-4
    # selection matter. Rebuild with the same routing functions.
    from scipy.special import expit
    def fv(t, b, a=1.5):
        p = expit(a * (t - b)); return p * (1 - p)
    d = [float(c["true_difficulty"]) for c in corpus]
    def route(sn, theta):
        if sn == "corpus_prefix": return list(range(4))
        if sn == "random": return sorted(np.random.RandomState(42).choice(len(corpus), 4, replace=False).tolist())
        if sn == "difficulty_only": return sorted(np.argsort(d)[-4:].tolist())
        return sorted(np.argsort([fv(theta, b) for b in d])[-4:].tolist())
    tmpl = (PROMPTS / "phase2_generation.txt").read_text()
    p2 = []
    for u in users:
        sp = tmpl.split("USER:")[0].replace("SYSTEM:", "").strip().replace("{user_label}", u["label"])
        for sn in STRATS:
            ct = "\n\n".join(f"Passage {j+1}: {corpus[i]['text']}" for j, i in enumerate(route(sn, float(u["theta"]))))
            p2.append(keystr(MODEL, sp, f"Please explain:\n\n{ct}", 0.3))
    return {"phase1": (len(p1), *lookup_hits(p1)), "phase2": (len(p2), *lookup_hits(p2))}


def winner(text):
    try: return json.loads(re.sub(r"```json|```", "", text).strip()).get("winner", "?")
    except Exception: return "?"


def score(text):
    try: return json.loads(re.sub(r"```json|```", "", text).strip()).get("score")
    except Exception:
        n = re.findall(r"\d+\.?\d*", text[:50]); return float(n[0]) if n else None


def binom_one_sided(k, n):
    return sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n


def attempt_b():
    run = [v for v in CACHE.values() if v.get("model")]
    gen = [v for v in run if v["model"] == MODEL]
    judge = [v for v in run if v["model"] == JUDGE]
    p3 = [v for v in judge if '"winner"' in v["content"]]
    p4 = [v for v in judge if '"winner"' not in v["content"]]
    out = {"n_entries_with_model": len(run), "n_gen": len(gen), "n_p3": len(p3), "n_p4": len(p4)}
    assert len(p3) == len(PAIRS) * len(users) * REPS and len(p4) == len(STRATS) * len(users) * REPS
    it = iter(p3)
    for pa, pb in PAIRS:
        wins = 0; user_wins = 0
        for u in users:
            uw = 0
            for rep in range(REPS):
                w = winner(next(it)["content"])
                if rep % 2 == 1: w = "B" if w == "A" else "A" if w == "B" else "?"
                uw += (w == "A")
            wins += uw; user_wins += (uw >= 2)
        n = len(users) * REPS
        out[f"{pa}_vs_{pb}"] = {"pcr_wins": wins, "total": n, "binom_one_sided_p": round(binom_one_sided(wins, n), 4),
                                "users_majority_pcr": f"{user_wins}/{len(users)}"}
    it = iter(p4); sc = {s: [] for s in STRATS}
    for u in users:
        for s in STRATS:
            for rep in range(REPS):
                v = score(next(it)["content"])
                if v is not None: sc[s].append(float(v))
    out["phase4"] = {s: {"n": len(v), "mean": round(float(np.mean(v)), 2), "std": round(float(np.std(v)), 2)} for s, v in sc.items()}
    a = np.array(sc["pcr"])
    for o in ("difficulty_only", "random", "corpus_prefix"):
        b = np.array(sc[o])
        out["phase4"][f"cohens_d_pcr_vs_{o}"] = round(float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)), 2)
    return out


if __name__ == "__main__":
    print(f"cache entries: {len(CACHE)}; key lengths: {sorted({len(k) for k in CACHE})}")
    print("A. key-based lookup (requests, md5 hits, sha256 hits):")
    for k, v in attempt_a().items(): print(f"   {k}: {v}")
    print("B. order-based reconstruction (assumes run_experiment.py loop order; not key-verified):")
    print(json.dumps(attempt_b(), indent=2))
