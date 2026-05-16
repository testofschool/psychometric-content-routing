# Psychometric Content Routing (PCR)

**Frontier-Value Selection for User-Conditioned LLM Processing**

Jung Min Kang (2026)

**Paper:** [arXiv link pending]

## Key Result

Under equal chunk counts (4 of 20), PCR achieves:
- **6.06 ± 1.39** vs 3.67 ± 2.36 (difficulty-only), Cohen's d = 1.23
- 15/18 judge-repeat wins (p = 0.004); 5/6 user-level trend (p ≈ 0.109)
- Trends above corpus-prefix (5.67) and fixed-random (4.72)

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
  api_cache.json                         Cached Groq API responses (197 entries)
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
  note={arXiv preprint (pending)}
}
```

## License

MIT
