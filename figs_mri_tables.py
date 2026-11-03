#!/usr/bin/env python3
"""figs_mri_tables.py - Figures 3 and 4 of the manuscript from arm_cases_mri3.csv (Oct 2026).

  Figure 3  threshold VAT that the model does not call VAT: (a) its share of the 2006
            method's VAT against the model's VAT volume, (b) its make-up by model label,
            by quarter of model VAT (means, so the stacks add up)
  Figure 4  Bland-Altman of threshold VAT against the model's VAT: (a) the 2006 method,
            (b) the wall rule with its MRI setting; bias, 95% limits of agreement and the
            regression of the difference on the mean (proportional bias)

    python3 figs_mri_tables.py --cases arm_cases_mri3.csv --out figs   [--variant raw|n4]

Writes fig3_threshold_content.{pdf,png,tif}, fig4_bland_altman.{pdf,png,tif} and
fig34_numbers.txt (every number quoted in the captions and Results).
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument('--cases', required=True)
ap.add_argument('--out', default='figs')
ap.add_argument('--variant', default='raw', choices=['raw', 'n4'])
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'Inter', 'DejaVu Sans'],
                     'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'legend.fontsize': 7, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                     'axes.spines.top': False, 'axes.spines.right': False})
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d9d8d4'
BLUE, ORANGE = '#2a78d6', '#eb6834'
# categorical slots 1-6 in fixed order (validated: adjacent CVD dE >= 9.1, normal >= 19.6)
GROUPS = [('intestine', 'bowel', '#2a78d6'), ('organ', 'organ', '#eb6834'), ('muscle', 'muscle', '#1baf7a'),
          ('none', 'unlabelled', '#eda100'), ('model_sat', 'model SAT', '#e87ba4'), ('bone', 'bone', '#008300')]

d = pd.read_csv(a.cases)
r = d[(d.variant == a.variant) & (d.status == 'ok')].copy().reset_index(drop=True)
n = len(r)
lines = [f'{a.cases}, variant {a.variant}, n = {n}']


def _rgb_tif(path):
    """matplotlib writes RGBA TIFFs; journals want RGB on white."""
    from PIL import Image
    im = Image.open(path)
    if im.mode != 'RGB':
        bg = Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.split()[-1] if im.mode == 'RGBA' else None)
        bg.save(path, compression='tiff_lzw', dpi=im.info.get('dpi', (600, 600)))


def save(fig, stem):
    for ext, kw in (('pdf', {}), ('png', dict(dpi=300)), ('tif', dict(dpi=600, pil_kwargs={'compression': 'tiff_lzw'}))):
        fig.savefig(out / f'{stem}.{ext}', bbox_inches='tight', **kw)
    plt.close(fig)
    _rgb_tif(out / f'{stem}.tif')


# ---------------------------------------------------------------- Figure 3
x = r.model_vat_ml.values / 1000.0
y = r.meth_vat_rejected_pct.values
rho, p = stats.spearmanr(x, y)
q = pd.qcut(r.model_vat_ml, 4, labels=False).values
edges = [r.model_vat_ml[q == k].max() / 1000.0 for k in range(3)]
lines.append(f'Fig 3a: share of 2006 VAT not model VAT, median {np.median(y):.1f}% (IQR {np.percentile(y,25):.1f}-{np.percentile(y,75):.1f}); '
             f'Spearman rho vs model VAT {rho:.2f} (p = {p:.1g})')
comp = np.zeros((4, len(GROUPS)))
for k in range(4):
    rk = r[q == k]
    for g, (col, _, _) in enumerate(GROUPS):
        comp[k, g] = np.mean(rk.meth_vat_rejected_pct * rk[f'rej_{col}_pct'] / 100.0)
    lines.append(f'  quarter {k+1} (model VAT {rk.model_vat_ml.min():.0f}-{rk.model_vat_ml.max():.0f} ml, n {len(rk)}): '
                 f'median {rk.meth_vat_rejected_pct.median():.1f}%, mean {rk.meth_vat_rejected_pct.mean():.1f}%; '
                 + ', '.join(f'{lab} {comp[k, g]:.1f}' for g, (_, lab, _) in enumerate(GROUPS)))
lines.append('  make-up of the rejected VAT, median over subjects (% of it): '
             + ', '.join(f'{lab} {r[f"rej_{col}_pct"].median():.1f}' for col, lab, _ in GROUPS))

fig, ax = plt.subplots(1, 2, figsize=(174 / 25.4, 70 / 25.4), gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.32))
A = ax[0]
for e in edges:
    A.axvline(e, color=GRID, lw=0.6, ls=(0, (3, 2)), zorder=0)
bounds = [x.min()] + edges + [x.max()]
for k in range(4):
    A.text((bounds[k] + bounds[k + 1]) / 2, 74, f'Q{k + 1}', ha='center', va='bottom', color=INK2, fontsize=7)
A.scatter(x, y, s=16, color=BLUE, edgecolor='white', linewidth=0.5, zorder=3)
A.set_xlabel('Model VAT, abdominal slices (L)')
A.set_ylabel("2006-method VAT not called VAT\nby the model (% of 2006 VAT)")
A.set_ylim(0, 80); A.set_xlim(0, max(x) * 1.05)
A.text(0.97, 0.06, f'Spearman ρ = {rho:.2f}'.replace('-', '−'), transform=A.transAxes, ha='right', color=INK, fontsize=7)
A.set_title('a', loc='left', fontweight='bold')
B = ax[1]
left = np.zeros(4); xs = np.arange(4)
for g, (_, lab, col) in enumerate(GROUPS):
    B.bar(xs, comp[:, g], bottom=left, width=0.62, color=col, edgecolor='white', linewidth=1.0, label=lab, zorder=2)
    for k in range(4):
        if comp[k, g] >= 4.0:
            B.text(xs[k], left[k] + comp[k, g] / 2, f'{comp[k, g]:.0f}', ha='center', va='center', fontsize=6.5,
                   color='white' if g in (0, 1, 5) else INK)
    left += comp[:, g]
for k in range(4):
    B.text(xs[k], left[k] + 1.0, f'{left[k]:.0f}%', ha='center', va='bottom', fontsize=7, color=INK)
B.set_xticks(xs, [f'Q{k + 1}' for k in range(4)])
B.set_xlabel('Quarter of model VAT (Q1 leanest)')
B.set_ylabel('% of 2006 VAT (mean)')
B.set_ylim(0, 52)
B.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, title='model label', title_fontsize=7,
         handlelength=1.0, handleheight=1.0, reverse=True)
B.set_title('b', loc='left', fontweight='bold')
save(fig, 'fig3_threshold_content')

# ---------------------------------------------------------------- Figure 4
panels = [('meth', '2006 method'), ('wall', 'Wall rule, MRI setting')]
fig, ax = plt.subplots(1, 2, figsize=(174 / 25.4, 72 / 25.4), sharey=True, gridspec_kw=dict(wspace=0.12))
allm, alld = [], []
for m, _ in panels:
    t, mo = r[f'{m}_vat_ml'].values, r.model_vat_ml.values
    allm.append((t + mo) / 2); alld.append(t - mo)
ylo, yhi = min(np.min(v) for v in alld), max(np.max(v) for v in alld)
for i, ((m, title), mean, diff) in enumerate(zip(panels, allm, alld)):
    A = ax[i]
    bias, sd = diff.mean(), diff.std(ddof=1)
    lo, hi = bias - 1.96 * sd, bias + 1.96 * sd
    reg = stats.linregress(mean, diff)
    ci = 1.96 * reg.stderr
    pct = 100 * diff / mean
    preg = stats.linregress(mean, pct)
    lines.append(f'Fig 4{"ab"[i]} {title}: bias {bias:+.0f} ml (SD {sd:.0f}), 95% LoA {lo:+.0f} to {hi:+.0f} ml; '
                 f'difference on mean slope {reg.slope:+.3f} (95% CI {reg.slope - ci:+.3f} to {reg.slope + ci:+.3f}), p = {reg.pvalue:.2g}; '
                 f'r(threshold, model) {np.corrcoef(r[f"{m}_vat_ml"], r.model_vat_ml)[0, 1]:.3f}; '
                 f'as % of the mean: bias {pct.mean():+.1f}%, LoA {pct.mean() - 1.96 * pct.std(ddof=1):+.1f} to '
                 f'{pct.mean() + 1.96 * pct.std(ddof=1):+.1f}%, slope {preg.slope * 1000:+.2f} %/L (p = {preg.pvalue:.2g})')
    A.axhline(0, color=GRID, lw=0.6, zorder=0)
    for yv, ls, lab in ((bias, '-', f'bias {bias:+.0f}'), (hi, (0, (4, 2)), f'+1.96 SD {hi:+.0f}'),
                        (lo, (0, (4, 2)), f'−1.96 SD {lo:+.0f}')):
        A.axhline(yv, color=INK2, lw=0.8, ls=ls, zorder=1)
        A.annotate(lab.replace('-', '−') + ' ml', xy=(0.995, yv), xycoords=A.get_yaxis_transform(), xytext=(0, 1.5),
                   textcoords='offset points', va='bottom', ha='right', fontsize=6.5, color=INK2, zorder=4)
    xx = np.linspace(mean.min(), mean.max(), 50)
    A.plot(xx, reg.intercept + reg.slope * xx, color=ORANGE, lw=1.2, zorder=2,
           label=f'regression: slope {reg.slope:.2f}, p = {reg.pvalue:.2g}'.replace('-', '−'))
    A.scatter(mean, diff, s=16, color=BLUE, edgecolor='white', linewidth=0.5, zorder=3)
    A.set_xlabel('Mean of method and model VAT (ml)')
    A.set_title(f'{"ab"[i]}   {title}', loc='left', fontweight='bold')
    A.legend(loc='upper left', frameon=False, handlelength=1.4)
    A.set_ylim(ylo - 250, yhi + 350)
ax[0].set_ylabel('Method − model VAT (ml)')
save(fig, 'fig4_bland_altman')

(out / 'fig34_numbers.txt').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))
