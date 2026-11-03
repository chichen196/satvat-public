#!/usr/bin/env python3
"""fig1_pipeline.py - Figure 1, the study design (three parts), drawn to journal width (174 mm)."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'Inter', 'DejaVu Sans'],
                     'font.size': 6.5, 'pdf.fonttype': 42})
INK, INK2 = '#0b0b0b', '#52514e'
DATA, STEP, LINE = '#dbe9fb', '#f0efec', '#8a8984'
W, H = 174.0, 128.0
fig = plt.figure(figsize=(W / 25.4, H / 25.4))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis('off')


def box(x, y, w, h, title, body='', kind='step', align='center'):
    fc = {'data': DATA, 'step': STEP, 'method': 'white', 'out': 'white'}[kind]
    ec = {'data': '#86b6ef', 'step': '#c9c8c3', 'method': INK2, 'out': INK}[kind]
    lw = {'data': 0.6, 'step': 0.6, 'method': 0.7, 'out': 1.1}[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=1.2', fc=fc, ec=ec, lw=lw))
    ax.text(x + w / 2, y + h - 1.6, title, ha='center', va='top', fontsize=6.8, fontweight='bold', color=INK)
    if body:
        bx = x + w / 2 if align == 'center' else x + 3.0
        ax.text(bx, y + h - 5.0, body, ha=align, va='top', fontsize=6.0, color=INK2, linespacing=1.25)
    return (x, y, w, h)


def arrow(b1, b2, y1=None, y2=None):
    x1, x2 = b1[0] + b1[2], b2[0]
    y1 = b1[1] + b1[3] / 2 if y1 is None else y1
    y2 = b2[1] + b2[3] / 2 if y2 is None else y2
    ax.add_patch(FancyArrowPatch((x1 + 0.4, y1), (x2 - 0.4, y2), arrowstyle='-|>', mutation_scale=6,
                                 lw=0.7, color=LINE, shrinkA=0, shrinkB=0))


def band(y, h, letter, title):
    ax.add_patch(FancyBboxPatch((1.5, y), W - 3, h, boxstyle='round,pad=0,rounding_size=1.8', fc='none',
                                ec='#e2e1dc', lw=0.6))
    ax.text(3.5, y + h - 2.2, letter, fontsize=9, fontweight='bold', va='top', color=INK)
    ax.text(9.0, y + h - 2.5, title, fontsize=7.2, fontweight='bold', va='top', color=INK)


# ---- A: the port
band(97, 29, 'a', 'The 2006 method, reimplemented from its IDL source')
a1 = box(5, 100, 27, 16, 'IDL source', 'estfatvolmriv2.pro\n(2003–2007) and the\nthreshold program', 'data')
a2 = box(37, 100, 27, 16, 'Python port', 'function by function,\ncompared on\nsynthetic phantoms')
a3 = box(69, 99, 71, 18.5, 'Four departures, corrected',
         '1  threshold: Ridler–Calvard, not the IDL integer-bin-width loop\n'
         '2  threshold program: each slice’s own outline, not slice 0’s\n'
         '3  visceral ellipse fit: the subcutaneous pass’s jump test added\n'
         '4  orientation from the image header, not from the non-fat test\n'
         'port only: non-converged fits rejected (IDL’s AMOEBA would stop)', align='left')
a4 = box(145, 100, 25, 16, 'Corrected method', 'effect of each\ncorrection, phantoms\nand CHAOS: Table 2', 'out')
for p, q in ((a1, a2), (a2, a3), (a3, a4)):
    arrow(p, q)

# ---- B: CT arm
band(52.5, 42.5, 'b', 'CT arm: three boundaries on one fat mask')
b1 = box(5, 63.5, 30, 18, 'AATCT-IDS', '300 subjects\n3,213 slices, 5 mm\nexpert SAT and VAT\nlabels', 'data')
b2 = box(41, 63.5, 30, 18, 'Common fat mask F', 'body, grey ≤ 79, and\nlabelled SAT or VAT;\nthe threshold cannot\nrun on 8-bit JPEGs')
m1 = box(78, 78, 50, 10, '2006 method (corrected)', 'ellipse boundary: own split and visceral zone', 'method')
m2 = box(78, 67.5, 50, 10, 'Wall-following rule', 'muscle ring closed and filled; setting B', 'method')
m3 = box(78, 57, 50, 10, 'U-Net (f2m)', 'moves the boundary within F; 5-fold CV by subject', 'method')
b4 = box(135, 61.5, 35, 21, 'Scored against labels', 'misassigned share of F,\nVAT and SAT %RMSE;\nsettings on odd subjects,\neven half held out (n = 155)\nTable 3, Figure 2', 'out')
arrow(b1, b2)
for m in (m1, m2, m3):
    arrow(b2, m); arrow(m, b4)

# ---- C: MRI arm
band(3, 46, 'c', 'MRI arm: threshold VAT against an anatomy-aware model')
c1 = box(5, 22, 26, 18, 'CHAOS', '40 subjects, 1.5 T\nT1 in-phase GRE\n805 abdominal\nslices', 'data')
c2 = box(35, 22, 27, 18, 'Trunk', 'arms cut at their\nnotches; slices below\nthe lungs; edge-cut\nslices left out')
c3 = box(66, 28, 24, 12, 'Threshold fat', 'Ridler–Calvard;\noriginal and N4')
n1 = box(95, 33.5, 38, 10, '2006 method (corrected)', 'own split and visceral zone', 'method')
n2 = box(95, 22, 38, 10, 'Wall-following rule', 'CT setting; MRI setting', 'method')
n3 = box(66, 7, 67, 10, 'VIBESegmentator (pretrained, not retrained)',
         'on the original image: SAT minus the skin band; inner fat = VAT', 'method')
c5 = box(139, 12.5, 31, 23, 'Compared with model', 'misassignment within\nthe threshold fat;\nmake-up of threshold\nVAT the model rejects;\ndim fat missed\nTable 4, Figures 3–5', 'out')
arrow(c1, c2); arrow(c2, c3)
arrow(c3, n1); arrow(c3, n2)
arrow(c2, n3, y1=25, y2=12)
for m in (n1, n2, n3):
    arrow(m, c5)
for ext, kw in (('pdf', {}), ('png', dict(dpi=300)), ('tif', dict(dpi=600, pil_kwargs={'compression': 'tiff_lzw'}))):
    fig.savefig(out / f'fig1_design.{ext}', **kw)
from PIL import Image
im = Image.open(out / 'fig1_design.tif')
if im.mode != 'RGB':
    bg = Image.new('RGB', im.size, 'white'); bg.paste(im, mask=im.split()[-1]); bg.save(out / 'fig1_design.tif', compression='tiff_lzw', dpi=(600, 600))
print('fig1_design written')
