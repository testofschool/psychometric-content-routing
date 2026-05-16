#!/usr/bin/env python3
"""Regenerate all PCR paper figures from reported results."""
import json, sys
import numpy as np
from scipy.special import expit as sigmoid
from pathlib import Path
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"

def fig1():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b = np.linspace(-4, 4, 300)
    for tv, c, ls, lbl in [(-2,'#4a86c8','--','θ=-2'), (0,'#2ca02c','-','θ=0'), (2,'#d94444','-.','θ=+2')]:
        p = sigmoid(1.5*(tv-b)); ax.plot(b, p*(1-p), color=c, lw=2.5, ls=ls, label=f'${lbl}$')
    ax.set_xlabel('Content Difficulty (b)'); ax.set_ylabel('F(θ,b)=P(1-P)')
    ax.set_title('Frontier Value Function'); ax.legend()
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG/'fig1_frontier_value.png', dpi=200, bbox_inches='tight'); plt.close()

def fig6():
    np.random.seed(42); tw = np.array([1.5,0.8,0.6,1.2,0.3]); ratios = []
    for sd in [1,2,3,4,5,6,7,8]:
        gp, gu = [], []
        for s in range(10):
            rng=np.random.RandomState(s+100); K=200; bi=rng.randn(K,5)*(sd/3.0)@tw; th=rng.randn(50)*1.5
            for t in th:
                fv=sigmoid(1.5*(t-bi))*(1-sigmoid(1.5*(t-bi))); bud=int(K*0.3)
                gp.append(fv[np.argsort(-fv)[:bud]].sum()); gu.append(fv[rng.choice(K,bud,replace=False)].sum())
        ratios.append(np.mean(gp)/np.mean(gu))
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot([1,2,3,4,5,6,7,8], ratios, '-o', color='#2ecc71', lw=2.5, ms=8)
    ax.axhline(y=1, color='gray', ls='--', alpha=0.4)
    ax.set_xlabel('σ_b'); ax.set_ylabel('PCR / Uniform'); ax.set_title('PCR Advantage Scales with Diversity')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(FIG/'fig6_scaling.png', dpi=200, bbox_inches='tight'); plt.close()

if __name__ == "__main__":
    fig1(); fig6(); print("Figures 1 and 6 regenerated. Figures 2-5 require reported-run data.")
