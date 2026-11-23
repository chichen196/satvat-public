#!/usr/bin/env python3
"""fig3_threshold_content.py - Figure 3: threshold VAT that the reference model does not call visceral fat
(CHAOS, n = 40, original images), against each of the two reference models.

usage: fig3_threshold_content.py ARM_CASES_VIBE.csv ARM_CASES_TS.csv [OUTDIR]

ARM_CASES_*.csv are chaos_mri_arm.py's arm_cases.csv for --ref vibe and --ref ts (rows with variant == raw are used).
Rows: (a, b) against VIBESegmentator, (c, d) against TotalSegmentator MRI. Left: share of the 2006 method's VAT
that the model does not label visceral fat, against the model's VAT volume; dashed lines divide the subjects into
quarters of that model's VAT. Right: make-up of that share by model label, as a share of the 2006 method's VAT,
mean per quarter (totals above the bars; segments of 5% or more labelled).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
                     'font.size': 7, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'pdf.fonttype': 42})
DOT, GRID, INK2 = '#2a78d6', '#c9c8c3', '#52514e'
LABELS = [('rej_intestine_pct', 'bowel', '#2a78d6'), ('rej_organ_pct', 'organ', '#eb6834'),
          ('rej_muscle_pct', 'muscle', '#1baf7a'), ('rej_none_pct', 'unlabelled', '#eda100'),
          ('rej_model_sat_pct', 'model SAT', '#e87ba4'), ('rej_bone_pct', 'bone', '#008300')]
QN = ['Q1', 'Q2', 'Q3', 'Q4']


def load(path):
    d = pd.read_csv(path)
    d = d[d['variant'] == 'raw'].copy()
    assert len(d) == 40, f'{path}: {len(d)} raw rows, expected 40'
    d['vat_l'] = d['model_vat_ml'] / 1000.0
    d['q'] = pd.qcut(d['model_vat_ml'], 4, labels=QN)
    return d


def scatter(ax, d, name, letter):
    rho = spearmanr(d['vat_l'], d['meth_vat_rejected_pct'])[0]
    ax.scatter(d['vat_l'], d['meth_vat_rejected_pct'], s=16, c=DOT, edgecolors='white', linewidths=0.5, zorder=3)
    s = d.sort_values('vat_l')
    groups = [s.loc[s['q'] == q, 'vat_l'] for q in QN]
    edges = [(groups[i].max() + groups[i + 1].min()) / 2 for i in range(3)]
    for e in edges:
        ax.axvline(e, color=GRID, lw=0.7, ls=(0, (3, 2)), zorder=1)
    xmax = 5.5
    bounds = [0] + edges + [xmax]
    for i, q in enumerate(QN):
        ax.text((bounds[i] + bounds[i + 1]) / 2, 77, q, ha='center', va='center', color=INK2, fontsize=7)
    ax.set_xlim(0, xmax); ax.set_ylim(0, 80)
    ax.set_xlabel('Model VAT, abdominal slices (L)')
    ax.set_ylabel('2006-method VAT not called VAT\nby the model (% of 2006 VAT)')
    ax.text(0.97, 0.05, f'Spearman ρ = {rho:.2f}'.replace('-', '−'), transform=ax.transAxes, ha='right', va='bottom')
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    return rho


def bars(ax, d, name, letter, legend):
    comp = d[[c for c, _, _ in LABELS]].mul(d['meth_vat_rejected_pct'], axis=0) / 100.0
    mean = comp.groupby(d['q'], observed=True).mean().reindex(QN)
    x = np.arange(4)
    bottom = np.zeros(4)
    handles = []
    for c, lab, col in LABELS:
        v = mean[c].values
        if np.allclose(v, 0):
            continue
        h = ax.bar(x, v, 0.62, bottom=bottom, color=col, edgecolor='white', linewidth=0.6, label=lab)
        handles.append(h)
        for xi, (b, vi) in enumerate(zip(bottom, v)):
            if vi >= 5:
                ax.text(xi, b + vi / 2, f'{vi:.0f}', ha='center', va='center', fontsize=6.5,
                        color='white' if col in ('#2a78d6', '#eb6834', '#008300') else '#0b0b0b')
        bottom += v
    for xi, t in enumerate(bottom):
        ax.text(xi, t + 1.0, f'{t:.0f}%', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x, QN); ax.set_ylim(0, 52); ax.set_xlim(-0.6, 3.6)
    ax.set_xlabel('Quarter of model VAT (Q1 leanest)')
    ax.set_ylabel('% of 2006 VAT (mean)')
    ax.text(-0.17, 1.04, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    ax.legend(handles[::-1], [h.get_label() for h in handles[::-1]], title=legend, frameon=False,
              loc='upper left', bbox_to_anchor=(1.0, 1.02), fontsize=6.5, title_fontsize=7, handlelength=1.0,
              handleheight=1.0, borderaxespad=0.3, alignment='left')
    return mean.sum(axis=1)


def main():
    vibe, ts = load(sys.argv[1]), load(sys.argv[2])
    out = Path(sys.argv[3] if len(sys.argv) > 3 else '.')
    W, H = 174 / 25.4, 140 / 25.4
    fig, axs = plt.subplots(2, 2, figsize=(W, H), gridspec_kw=dict(width_ratios=[1, 0.86]))
    fig.subplots_adjust(left=0.1, right=0.86, top=0.95, bottom=0.08, wspace=0.42, hspace=0.5)
    for row, (d, name, full) in enumerate(((vibe, 'model', 'VIBESegmentator'), (ts, 'model', 'TotalSegmentator MRI'))):
        rho = scatter(axs[row, 0], d, name, 'ac'[row])
        tot = bars(axs[row, 1], d, name, 'bd'[row], 'model label')
        axs[row, 0].text(-0.2 + 0.075, 1.04, full, transform=axs[row, 0].transAxes, fontsize=8, fontweight='bold',
                         va='bottom')
        med = d.groupby('q', observed=True)['meth_vat_rejected_pct'].median().reindex(QN)
        print(f'{full}: rho {rho:.2f}; quarter means {np.round(tot.values, 1)}; medians {np.round(med.values, 1)}')
    for ext, kw in (('pdf', {}), ('png', dict(dpi=300)), ('tif', dict(dpi=600, pil_kwargs={'compression': 'tiff_lzw'}))):
        fig.savefig(out / f'fig3_threshold_content.{ext}', facecolor='white', **kw)
    from PIL import Image
    im = Image.open(out / 'fig3_threshold_content.tif')
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.split()[-1])
        bg.save(out / 'fig3_threshold_content.tif', compression='tiff_lzw', dpi=(600, 600))
    print('fig3_threshold_content written')


if __name__ == '__main__':
    main()
