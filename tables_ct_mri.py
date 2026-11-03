#!/usr/bin/env python3
"""tables_ct_mri.py - Tables 3 (CT) and 4 (MRI) of the paper from the per-slice / per-case run tables.

    python3 tables_ct_mri.py table3 --sep results/ct/sep_slices_settingB.csv --unet results/ct/sep_slices_unet_f2m.csv
    python3 tables_ct_mri.py table4 --mri results/mri/arm_cases_mri.csv --ct results/mri/arm_cases_ct_setting.csv

Table 3: per subject (slices summed) VAT and SAT error against the labels, %RMSE across subjects,
Pearson r for VAT, median VAT error, subjects with |VAT error| > 5%; misassigned shares are pooled
over all slices as a share of the fat mask F.  Held-out half = even-numbered subjects.
Table 4: medians (IQR) over the 40 CHAOS cases; misassignment within the threshold fat; the
"32 subjects" row leaves out the eight reviewed during development (--dev).
"""
import argparse, re, sys
import numpy as np, pandas as pd

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('table', choices=['table3', 'table4'])
ap.add_argument('--sep'); ap.add_argument('--unet'); ap.add_argument('--mri'); ap.add_argument('--ct')
ap.add_argument('--dev', default='0001,0011,0026,0027,0030,0033,0036,0037',
                help='CHAOS cases reviewed while the MRI wall setting was developed')
a = ap.parse_args()
M = lambda s: s.replace('-', '−')


def table3():
    s = pd.read_csv(a.sep); u = pd.read_csv(a.unet)
    u = u.rename(columns=lambda c: c.replace('unet_', 'unet_'))
    d = s.merge(u[['subject', 'slice'] + [c for c in u.columns if c.startswith('unet_')]], on=['subject', 'slice'])
    d['n'] = d.subject.map(lambda x: int(re.findall(r'\d+', x)[-1]))
    meths = [('meth', '2006 method'), ('zone', '2006 zone'), ('wall', 'Wall (setting B)'), ('unet', 'U-Net (f2m)')]
    out = {}
    for half, sel in (('even', d.n % 2 == 0), ('all', d.n >= 0)):
        x = d[sel]
        g = x.groupby('subject').sum(numeric_only=True)
        for t, _ in meths:
            ev = 100 * (g[f'{t}_vat'] - g.ref_vat) / g.ref_vat
            es = 100 * (g[f'{t}_sat'] - g.ref_sat) / g.ref_sat
            F = x.F_px.sum()
            out[(half, t)] = {
                'n': len(g), 'VAT %RMSE': np.sqrt((ev ** 2).mean()), 'SAT %RMSE': np.sqrt((es ** 2).mean()),
                'r (VAT)': np.corrcoef(g[f'{t}_vat'], g.ref_vat)[0, 1], 'Median VAT error (%)': ev.median(),
                'F misassigned (%)': 100 * (x[f'{t}_vat2sat'].sum() + x[f'{t}_sat2vat'].sum()) / F,
                'SAT called VAT (% of F)': 100 * x[f'{t}_sat2vat'].sum() / F,
                'VAT called SAT (% of F)': 100 * x[f'{t}_vat2sat'].sum() / F,
                'Subjects with VAT error > 5%': int((ev.abs() > 5).sum())}
    fmt = {'VAT %RMSE': '{:.1f}', 'SAT %RMSE': '{:.1f}', 'r (VAT)': '{:.3f}', 'Median VAT error (%)': '{:+.1f}',
           'F misassigned (%)': '{:.2f}', 'SAT called VAT (% of F)': '{:.2f}', 'VAT called SAT (% of F)': '{:.2f}',
           'Subjects with VAT error > 5%': '{:d}'}
    unet_fmt = {'VAT %RMSE': '{:.2f}', 'SAT %RMSE': '{:.2f}', 'F misassigned (%)': '{:.3f}',
                'SAT called VAT (% of F)': '{:.3f}', 'VAT called SAT (% of F)': '{:.3f}'}
    print(f'Measure (even half, n = {out[("even", "meth")]["n"]})'.ljust(34) + ''.join(f'{lab:>18}' for _, lab in meths))
    for k in fmt:
        print(k.ljust(34) + ''.join(f'{M((unet_fmt.get(k, fmt[k]) if t == "unet" else fmt[k]).format(out[("even", t)][k])):>18}' for t, _ in meths))
    for k in ('VAT %RMSE', 'F misassigned (%)'):
        print(f'{k}, all {out[("all", "meth")]["n"]}'.ljust(34)
              + ''.join(f'{M((unet_fmt.get(k, fmt[k]) if t == "unet" else fmt[k]).format(out[("all", t)][k])):>18}' for t, _ in meths))


def table4():
    m = pd.read_csv(a.mri); c = pd.read_csv(a.ct)
    raw = m[(m.variant == 'raw') & (m.status == 'ok')].set_index('case')
    n4 = m[(m.variant == 'n4') & (m.status == 'ok')].set_index('case')
    ct = c[(c.variant == 'raw') & (c.status == 'ok')].set_index('case')
    dev = {f'chaos_{int(x):04d}' for x in a.dev.split(',') if x.strip()}
    cols = [('2006 method', raw, 'meth'), ('2006 zone', raw, 'zone'), ('Wall, CT settings', ct, 'wall'),
            ('Wall, MRI settings', raw, 'wall'), ('Wall, MRI settings, N4', n4, 'wall')]
    mi = lambda s, d=1, sg=True: M((f'{s.median():+.{d}f} ({s.quantile(.25):+.{d}f} to {s.quantile(.75):+.{d}f})' if sg
                                    else f'{s.median():.{d}f} ({s.quantile(.25):.{d}f}–{s.quantile(.75):.{d}f})'))
    rows = []
    for lab, df, t in cols:
        S = df[f'{t}_sat_to_vat_pct'] + df[f'{t}_vat_to_sat_pct']
        keep = [k for k in df.index if k not in dev]
        has_f = f'{t}_misassigned_pctF' in df
        rows.append([lab, mi(df[f'{t}_vat_diff_pct']), mi(df[f'{t}_sat_diff_pct']), f'{df[f"{t}_vat_dice"].median():.3f}',
                     f'{df[f"{t}_sat_to_vat_pct"].median():.1f}', f'{df[f"{t}_vat_to_sat_pct"].median():.1f}',
                     mi(S, sg=False), str(int((S > 10).sum())), mi(S.loc[keep], sg=False) + f' [n = {len(keep)}]',
                     mi(df[f'{t}_misassigned_pctF'], sg=False) if has_f else '—',
                     str(int((df[f'{t}_misassigned_pctF'] > 5).sum())) if has_f else '—'])
    names = ['', 'VAT difference (%)', 'SAT difference (%)', 'VAT Dice', 'Model SAT called VAT (% of model SAT fat)',
             'Model VAT called SAT (% of model VAT fat)', 'Sum of the two (%)', 'Subjects with sum > 10%',
             'Sum, subjects not used in development', 'F misassigned (% of all threshold fat)', 'Subjects with F misassigned > 5%']
    for i, name in enumerate(names):
        if i == 0:
            print('Measure (n = 40; median, IQR)'.ljust(44) + ''.join(f'{r[0]:>26}' for r in rows)); continue
        print(name.ljust(44) + ''.join(f'{r[i]:>26}' for r in rows))


table3() if a.table == 'table3' else table4()
