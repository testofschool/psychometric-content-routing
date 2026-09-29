# Psychometric Content Routing (PCR)

**Frontier-Value Selection for User-Conditioned LLM Processing**

Jung Min Kang (2026)

**Paper / archive:** Zenodo, DOI [10.5281/zenodo.20242296](https://doi.org/10.5281/zenodo.20242296)

## Key Result

Under equal chunk counts (4 of 20), PCR achieves:
- **6.06 ± 1.39** vs 3.67 ± 2.36 (difficulty-only), Cohen's d = 1.23 — *reconstructed from cache by call order only, not key-verified; see [Reproducibility status](#reproducibility-status)*
- 15/18 judge-repeat wins (p = 0.004); 5/6 user-level trend (p ≈ 0.109) — *NOT reproducible from the committed cache; see [Reproducibility status](#reproducibility-status)*
- Trends above corpus-prefix (5.67) and fixed-random (4.72) — *reconstructed from cache by call order only, not key-verified*

The pairwise PCR vs fixed-random result reported in the paper (12/18) is likewise *NOT reproducible from the committed cache*.

## Reproducibility status

The reported figures above (and in `results/reported_run/summary_stats.json`, `main.tex`) **could not be reproduced from the committed cache** by re-running the code.

**Why.** `src/run_experiment.py` (`call_groq`) builds cache keys as `md5("{model}:{system}:{user_msg}:{temperature}:{salt}")` (32 hex chars) and writes the token count as `tokens`. The committed `cache/api_cache.json` has 197 entries with 64-hex-char keys and a `tokens_used` field, so it is not readable by the committed code. Which code revision, prompts and labels produced it cannot be determined from the repository (for example, cached judge replies say "early elementary students" while `data/users.csv` says "Elementary student", which may or may not reflect a prompt difference). Re-running `run_experiment.py` therefore gets zero cache hits and would issue fresh API calls. The original run logs and `results/experiment_results.json` were not committed.

**Reconstruction attempt.** `python scripts/reproduce_from_cache.py` (no API calls):

- Key-based lookup: rebuilding the Phase 1 (20) and Phase 2 (24) requests exactly as `run_experiment.py` does gives **0 hits** under both md5 and sha256 of the same key string.
- Order-based fallback: cache entries are stored in call order; the 132 entries carrying a `model` field form one complete run (24 generations, 36 pairwise, 72 absolute). Assigning them by `run_experiment.py`'s loop order (an assumption that cannot be checked against the keys) gives:

| Figure | Reported | Order-based reconstruction |
|---|---|---|
| Phase 4 mean ± std: PCR / difficulty-only / fixed-random / corpus-prefix | 6.06±1.39 / 3.67±2.36 / 4.72±0.80 / 5.67±1.11 | 6.06±1.39 / 3.67±2.36 / 4.72±0.80 / 5.67±1.11 (matches) |
| Cohen's d, PCR vs difficulty-only | 1.23 | 1.23 (matches) |
| Pairwise PCR vs difficulty-only | 15/18, p = 0.004 | 9/18, one-sided binomial p = 0.5927 |
| Pairwise PCR vs fixed-random | 12/18 | 10/18, one-sided binomial p = 0.4073 |
| Users with PCR majority (vs difficulty-only) | 5/6 | 3/6 |

So the absolute-score (Phase 4) figures are consistent with the cached run under the call-order assumption, while the pairwise (Phase 3) figures and their p-values are not. The reported numbers are left unchanged here; the reconstruction is **not** a replacement result.


## Method

PCR uses the **Frontier Value Function** F(θ,b) = P(1-P) from Item Response Theory to select which content an LLM should process for a specific user. Content at the user's learning frontier (θ ≈ b) gets priority; content that is too easy or too hard is deprioritized.

## Quick Start

```bash
pip install -r requirements.txt
export GROQ_API_KEY=your_key_here

# Run experiment (author-assigned difficulty, reported results)
python src/run_experiment.py

# Run with LLTM-estimated difficulty (experimental)
python src/run_experiment.py --lltm

# Regenerate simulation figures (1 and 6)
python src/make_figures.py

# Extract LLTM features from passage text
python src/extract_lltm_features.py
```

## Repository Structure

```
main.tex                                 LaTeX source
figures/                                 6 figures (fig1–fig6)
data/
  passages.csv                           20 passages: text + 5 LLTM features + difficulty estimates
  users.csv                              6 user ability profiles (θ = -2 to +3)
prompts/
  phase1_difficulty_rating.txt           LLM difficulty rating prompt
  phase2_generation.txt                  Explanation generation prompt
  phase3_pairwise_judge.txt              Pairwise comparison judge prompt
  phase4_absolute_judge.txt              Absolute scoring judge prompt
src/
  run_experiment.py                      Full Groq experiment (supports --lltm flag)
  make_figures.py                        Simulation figure generation (Figs 1, 6)
  extract_lltm_features.py              LLTM feature extraction from text
results/
  reported_run/summary_stats.json        Reported-run summary statistics
cache/
  api_cache.json                         Cached Groq API responses (197 entries; keys not compatible with src/, see Reproducibility status)
scripts/
  reproduce_from_cache.py                Cache-only reproducibility check (no API calls)
```

## Models

- **Generation:** Llama 3.3 70B Versatile (via Groq)
- **Judge:** Llama 3.1 8B Instant (via Groq)
- API limits at time of experiment may differ from current limits

## LLTM Feature Weights

| Feature | Weight | Description |
|---|---|---|
| Sentence-length variance | 1.5 | Std/mean of words per sentence |
| Long-word ratio | 0.8 | Fraction of words > 8 characters |
| Mean sentence length | 0.6 | Average sentence length / 20 |
| Inverse type-token ratio | 1.2 | 1 - (unique words / total words) |
| Context-word frequency | 0.3 | Fraction of context-dependency words |

Raw LLTM scores are linearly calibrated to the θ scale before routing. Correlation with author-assigned difficulty: ρ = 0.50.

## Related Papers

- [The Scaling Law of Evaluation Failure](https://arxiv.org/abs/2605.11205) (EFSL) — Kang, 2026
- [Isomorph-Eval](https://github.com/testofschool/isomorph-eval) — Kang, 2026

## Citation

```bibtex
@article{kang2026pcr,
  title={Psychometric Content Routing: Frontier-Value Selection for User-Conditioned LLM Processing},
  author={Kang, Jung Min},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.20242296}
}
```

## License

MIT
