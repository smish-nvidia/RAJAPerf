#!/usr/bin/env python3
# Generates the SVG figures of femsweep-report/. Usage: make_figures.py <outdir>
import sys, os, math

OUT = sys.argv[1]

CSS = """<style>
svg{--bg:#fcfcfb;--fg:#0b0b0b;--fg2:#52514e;--fg3:#7a7975;--grid:#e6e5e1;--box:#efeeea;--line:#b9b8b2;
--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--hi:#e34948}
@media (prefers-color-scheme:dark){svg{--bg:#1a1a19;--fg:#f0efec;--fg2:#c3c2b7;--fg3:#9a998f;--grid:#33322f;--box:#292927;--line:#5a5954;
--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--hi:#e66767}}
text{font-family:-apple-system,"Segoe UI",Helvetica,Arial,sans-serif;fill:var(--fg);font-size:13px}
.t2{fill:var(--fg2)}.t3{fill:var(--fg3)}.title{font-size:16px;font-weight:600}.sub{font-size:13px;fill:var(--fg2)}
.lab{font-size:12px}.small{font-size:11px}.mono{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}.b{font-weight:600}
.bg{fill:var(--bg)}.box{fill:var(--box)}.grid{stroke:var(--grid);stroke-width:1}.ax{stroke:var(--line);stroke-width:1}
.f1{fill:var(--s1)}.f2{fill:var(--s2)}.f3{fill:var(--s3)}.f4{fill:var(--s4)}.fhi{fill:var(--hi)}.ffg2{fill:var(--fg2)}
.w1{fill:var(--s1);fill-opacity:.16}.w2{fill:var(--s2);fill-opacity:.16}.w3{fill:var(--s3);fill-opacity:.16}.w4{fill:var(--s4);fill-opacity:.16}
.s1{stroke:var(--s1)}.s2{stroke:var(--s2)}.s3{stroke:var(--s3)}.s4{stroke:var(--s4)}.shi{stroke:var(--hi)}
.ln{fill:none;stroke-width:2;stroke-linecap:round;stroke-linejoin:round}
.edge{fill:none;stroke:var(--line);stroke-width:1}.edge2{fill:none;stroke:var(--fg2);stroke-width:1.5}
.ring{stroke:var(--bg);stroke-width:2}.hollow{fill:var(--bg);stroke-width:2}
.arrow{stroke:var(--fg2);stroke-width:1.5;fill:none;marker-end:url(#ah)}
.arrowhi{stroke:var(--hi);stroke-width:1.5;fill:none;marker-end:url(#ahhi)}
</style>
<defs>
<marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" class="ffg2"/></marker>
<marker id="ahhi" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L8,4 L0,8 z" class="fhi"/></marker>
</defs>"""


def esc(s):
    return s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


class Svg:
    def __init__(self, w, h, title_text):
        self.w, self.h = w, h
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{esc(title_text)}">',
                      f'<title>{esc(title_text)}</title>', CSS, f'<rect class="bg" x="0" y="0" width="{w}" height="{h}" rx="8"/>']

    def add(self, s):
        self.parts.append(s)

    def rect(self, x, y, w, h, cls, rx=0, extra=''):
        self.add(f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" {extra}/>')

    def text(self, x, y, s, cls='', anchor='start', extra=''):
        self.add(f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" xml:space="preserve" {extra}>{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, cls):
        self.add(f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>')

    def path(self, d, cls):
        self.add(f'<path class="{cls}" d="{d}"/>')

    def circle(self, x, y, r, cls):
        self.add(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>')

    def save(self, name):
        self.add('</svg>')
        with open(os.path.join(OUT, name), 'w') as f:
            f.write('\n'.join(self.parts) + '\n')


def column(svg, x, y_base, w, h, cls):
    # 4px rounded data end, square at the baseline
    r = min(4, h / 2, w / 2)
    y = y_base - h
    svg.path(f'M{x:.1f},{y_base:.1f} V{y + r:.1f} Q{x:.1f},{y:.1f} {x + r:.1f},{y:.1f} H{x + w - r:.1f} '
             f'Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} V{y_base:.1f} Z', cls)


def widths(n):
    c = [0] * (3 * n - 2)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                c[i + j + k] += 1
    return c


LANE_FILL = ['f1', 'f2', 'f3', 'f4']
LANE_WASH = ['w1', 'w2', 'w3', 'w4']
LANE_STROKE = ['s1', 's2', 's3', 's4']
# label ink on a lane-colored fill: white on blue/orange, near-black on aqua/yellow
ON_LANE = ['style="fill:#ffffff"', 'style="fill:#ffffff"', 'style="fill:#0b0b0b"', 'style="fill:#0b0b0b"']

###############################################################################
# 1. speedup.svg: speedup over the original RAJA_CUDA kernel, by mesh size
###############################################################################

def fig_speedup():
    sizes = ['5³ mesh', '16³ mesh', '32³ mesh']
    raja = [0.484, 12.619, 100.506]
    series = [  # left to right, slot order 4,3,2,1 (adjacent pairs validated)
        ('original Base_CUDA unroll_64', 'f4', [0.413, 10.246, 82.009]),
        ('coop_256', 'f3', [0.254, 4.017, 31.015]),
        ('dagqueue_256', 'f2', [0.208, 3.124, 24.741]),
        ('daggraph_256', 'f1', [0.130, 2.911, 21.479]),
    ]
    W, H = 860, 400
    s = Svg(W, H, 'Speedup over the original RAJA_CUDA kernel by mesh size')
    s.text(24, 32, 'Speedup over the original RAJA_CUDA kernel', 'title')
    s.text(24, 52, 'unroll_64_min_blocks_per_sm_5 = 1×; higher is better. A100, 72 angles, 16 groups, min of 3 runs.', 'sub')
    # legend row
    lx = 24
    for name, cls, _ in series:
        s.rect(lx, 70, 12, 12, cls, 3)
        s.text(lx + 18, 81, name, 'lab t2')
        lx += 18 + 7.2 * len(name) + 28
    x0, x1, y0, y1 = 70, 836, 110, 340
    ymax = 6
    sy = lambda v: y1 - (y1 - y0) * v / ymax
    for t in range(0, ymax + 1):
        s.line(x0, sy(t), x1, sy(t), 'grid')
        s.text(x0 - 10, sy(t) + 4, f'{t}×', 'lab t3', 'end')
    bw, gap = 24, 2
    gw = len(series) * bw + (len(series) - 1) * gap
    band = (x1 - x0) / len(sizes)
    for c, size in enumerate(sizes):
        gx = x0 + band * c + (band - gw) / 2
        for k, (name, cls, ms) in enumerate(series):
            v = raja[c] / ms[c]
            bx = gx + k * (bw + gap)
            column(s, bx, y1, bw, y1 - sy(v), cls)
            if name == 'daggraph_256':
                s.text(bx + bw / 2, sy(v) - 6, f'{v:.1f}×', 'lab b', 'middle')
        s.text(x0 + band * c + band / 2, y1 + 22, size, 'lab t2', 'middle')
    s.line(x0, sy(1), x1, sy(1), 'ax')
    s.text(24, H - 16, 'Exact times for every variant are in the results table.', 'small t3')
    s.save('speedup.svg')

###############################################################################
# 2. progress.svg: 32^3 time per sweep across the optimization steps
###############################################################################

def fig_progress():
    steps = [('first DAG', 'version'), ('full-warp', 'shuffle mask'), ('reuse pivot', 'reciprocal'),
             ('queue: early', 'downstream lookup'), ('conflict-free', 'face_b padding'), ('branch-free', 'reciprocal')]
    series = [
        ('coop_256', 'f3', 's3', [34.7, 34.7, 32.9, 32.9, 31.4, 31.02]),
        ('dagqueue_256', 'f2', 's2', [30.2, 27.7, 27.0, 26.08, 25.37, 24.74]),
        ('daggraph_256', 'f1', 's1', [32.5, 24.8, 23.45, 23.45, 22.29, 21.48]),
    ]
    gi = [('dagqueuegi_256', 's2', 23.07), ('daggraphgi_256', 's1', 18.93)]
    W, H = 860, 440
    s = Svg(W, H, 'Time per sweep at 32 cubed across the optimization steps')
    s.text(24, 32, 'Time per sweep through the optimization steps (32³ mesh)', 'title')
    s.text(24, 52, 'ms per sweep, lower is better; the y axis starts at 15 ms. Original RAJA_CUDA: 100.5 ms, original Base_CUDA: 82.0 ms.', 'sub')
    lx = 24
    for name, fcls, _, _ in reversed(series):
        s.rect(lx, 70, 12, 12, fcls, 3)
        s.text(lx + 18, 81, name, 'lab t2')
        lx += 18 + 7.2 * len(name) + 28
    s.circle(lx + 6, 76, 5, 'hollow s1')
    s.text(lx + 18, 81, 'X kept group-innermost (sweep only)', 'lab t2')
    x0, x1, y0, y1 = 64, 760, 106, 360
    lo, hi = 15, 40
    sy = lambda v: y1 - (y1 - y0) * (v - lo) / (hi - lo)
    n = len(steps) + 1
    sx = lambda i: x0 + 30 + (x1 - x0 - 60) * i / (n - 1)
    for t in range(lo, hi + 1, 5):
        s.line(x0, sy(t), x1, sy(t), 'grid')
        s.text(x0 - 10, sy(t) + 4, f'{t}', 'lab t3', 'end')
    for i, (a, b) in enumerate(steps + [('X group-', 'innermost')]):
        s.text(sx(i), y1 + 20, a, 'small t2', 'middle')
        s.text(sx(i), y1 + 34, b, 'small t2', 'middle')
    for name, fcls, scls, vals in series:
        d = ' '.join(('M' if i == 0 else 'L') + f'{sx(i):.1f},{sy(v):.1f}' for i, v in enumerate(vals))
        s.path(d, f'ln {scls}')
        for i, v in enumerate(vals):
            s.circle(sx(i), sy(v), 4, f'{fcls} ring')
        s.text(sx(len(vals) - 1) + 10, sy(vals[-1]) + 4, f'{vals[-1]:.1f}', 'lab b')
    for name, scls, v in gi:
        s.circle(sx(n - 1), sy(v), 5, f'hollow {scls}')
        s.text(sx(n - 1) + 10, sy(v) + 4, f'{v:.1f}', 'lab b')
    s.text(24, H - 14, 'Steps are cumulative. The last column keeps X group-innermost and skips converting it back (see the X layout section).', 'small t3')
    s.save('progress.svg')

###############################################################################
# 3. original-schedule.svg: one block per (angle, group), 5^3 hyperplanes
###############################################################################

def fig_original():
    w5 = widths(5)
    W, H = 860, 400
    s = Svg(W, H, 'The original schedule: one block per angle and group walking its hyperplanes')
    s.text(24, 32, 'Original schedule: one thread block per (angle, group)', 'title')
    s.text(24, 52, 'Each block walks its angle\'s hyperplanes in order, one element per thread, with a block barrier between them.', 'sub')
    # left: the grid of blocks
    s.text(24, 90, 'Grid of blocks', 'lab b')
    gx, gy, c = 24, 102, 14
    for r in range(8):
        for q in range(12):
            cls = 'f1' if (r == 2 and q == 4) else 'box'
            s.rect(gx + q * (c + 3), gy + r * (c + 3), c, c, cls, 2)
    s.text(24, gy + 8 * (c + 3) + 18, '72 angles × 16 groups', 'lab t2')
    s.text(24, gy + 8 * (c + 3) + 34, '= 1,152 independent blocks', 'lab t2')
    s.text(24, gy + 8 * (c + 3) + 58, 'The unrolled kernel fits 6 blocks', 'small t3')
    s.text(24, gy + 8 * (c + 3) + 72, 'per SM: 1.78 waves on an A100,', 'small t3')
    s.text(24, gy + 8 * (c + 3) + 86, 'so the last wave is partial.', 'small t3')
    s.path(f'M{gx + 4 * (c + 3) + c},{gy + 2 * (c + 3) + c / 2} C 250,150 250,110 300,104', 'arrow')
    # right: one block's threads per hyperplane
    rx, ry = 312, 90
    s.text(rx, ry, 'One block (64 threads) on a 5³ mesh: threads busy per hyperplane', 'lab b')
    cw, ch, rowh = 6, 12, 19
    for h, n in enumerate(w5):
        y = ry + 14 + h * rowh
        s.text(rx + 26, y + 10, f'h={h}', 'small t3', 'end')
        for t in range(64):
            s.rect(rx + 34 + t * (cw + 1), y, cw, ch, 'f1' if t < n else 'box', 1)
        s.text(rx + 34 + 64 * (cw + 1) + 8, y + 10, f'{n} busy', 'small t2')
        if h < len(w5) - 1:
            s.line(rx + 34, y + ch + 3.5, rx + 34 + 64 * (cw + 1) - 1, y + ch + 3.5, 'shi')
    yb = ry + 14 + len(w5) * rowh + 6
    s.line(rx + 34, yb, rx + 54, yb, 'shi')
    s.text(rx + 60, yb + 4, '__syncthreads() between hyperplanes', 'small t2')
    s.text(rx, yb + 26, 'At most 19 of 64 threads ever have work; at 32³ hyperplanes reach 768 elements, so only the', 'small t3')
    s.text(rx, yb + 40, 'first and last few are narrow there. Each busy thread solves a whole 8×8 system on its own.', 'small t3')
    s.save('original-schedule.svg')

###############################################################################
# 4. warp-mapping.svg: lanes, groups, rows
###############################################################################

def fig_warp():
    W, H = 860, 500
    s = Svg(W, H, 'How a warp is divided among element solves')
    s.text(24, 32, 'A warp = 8 groups of 4 lanes; each group solves one 8×8 element system', 'title')
    s.text(24, 52, 'The two kernel families fill a warp\'s 8 groups differently. Lanes are colored by their index within the group.', 'sub')
    lw, lg = 20, 2
    gwid = 4 * lw + 3 * lg
    ggap = 11
    x0 = 24

    def lanes(y, labels, caption):
        s.text(x0, y - 10, caption, 'lab b')
        for gidx in range(8):
            gx = x0 + gidx * (gwid + ggap)
            for l in range(4):
                s.rect(gx + l * (lw + lg), y, lw, 22, LANE_FILL[l], 3)
                s.text(gx + l * (lw + lg) + lw / 2, y + 15, str(gidx * 4 + l), 'small', 'middle',
                       extra=ON_LANE[l])
            s.path(f'M{gx},{y + 30} v5 h{gwid} v-5', 'edge')
            s.text(gx + gwid / 2, y + 50, labels[gidx], 'lab t2', 'middle')

    lanes(100, [f'element e{k}' for k in range(8)],
          'coop_256: one block per (angle a, group g); the 8 groups take 8 different elements of the hyperplane')
    lanes(200, [f'group g+{k}' for k in range(8)],
          'dag kernels: one warp per (angle a, element e); the 8 groups are 8 energy groups of that element')
    s.text(x0, 278, 'dag: the 8 groups read the same A, B, M0 rows, F blocks and face table entries, so those loads are shared by the whole warp.', 'small t3')
    # zoom: one group's rows
    zy = 310
    s.text(x0, zy, 'Inside one group: lane l owns rows l and l + 4 of the system', 'lab b')
    cs = 16
    mx, my = x0 + 60, zy + 14
    for r in range(8):
        lane = r % 4
        s.text(mx - 8, my + r * (cs + 2) + 12, f'row {r}', 'small t3', 'end')
        for cc in range(8):
            s.rect(mx + cc * (cs + 2), my + r * (cs + 2), cs, cs, LANE_FILL[lane], 2)
        s.rect(mx + 8 * (cs + 2) + 10, my + r * (cs + 2), cs, cs, LANE_FILL[lane], 2)
    s.text(mx + 4 * (cs + 2), my + 8 * (cs + 2) + 14, '(A + σ·M0), 8 × 8', 'small t2', 'middle')
    s.text(mx + 8 * (cs + 2) + 18, my + 8 * (cs + 2) + 14, 'b', 'small t2', 'middle')
    tx = mx + 9 * (cs + 2) + 60
    lines = ['• 2 rows × 8 entries + 2 entries of b per lane, held in registers',
             '• row kk lives in lane kk % 4, slot kk / 4: compile-time constants in the',
             '   unrolled loops, so no indexing cost',
             '• the LU step broadcasts each pivot row with __shfl_sync inside the group',
             '• 4 lanes (rather than 8 or 1) balances registers per lane against how',
             '   many elements a warp solves at once']
    for k, t in enumerate(lines):
        s.text(tx, my + 14 + k * 20, t, 'lab t2')
    s.save('warp-mapping.svg')

###############################################################################
# 5. lu-step.svg: one elimination step
###############################################################################

def fig_lu():
    W, H = 860, 400
    s = Svg(W, H, 'One step of the cooperative LU factorization')
    s.text(24, 32, 'LU step kk = 2 inside one group (Doolittle, no pivoting)', 'title')
    s.text(24, 52, 'Rows are tinted by the lane that owns them. The system is diagonally dominant by construction, so no pivoting is needed.', 'sub')
    cs = 34
    mx, my = 80, 86
    kk = 2
    for r in range(8):
        lane = r % 4
        s.text(mx - 10, my + r * cs + 22, f'{r}', 'lab t3', 'end')
        for c in range(8):
            x, y = mx + c * cs, my + r * cs
            cls = LANE_WASH[lane]
            s.rect(x + 1, y + 1, cs - 2, cs - 2, cls, 3)
            label = ''
            if r == c and r < kk:
                label = '1/u'
            elif c < r and c < kk:
                label = 'l'
            elif r < kk and c > r:
                label = 'u'
            if label:
                s.text(x + cs / 2, y + 22, label, 'small t3', 'middle')
        s.text(mx + 8 * cs + 12, my + r * cs + 22, f'lane {lane}', 'small t2')
    # pivot row
    s.rect(mx + kk * cs + 1, my + kk * cs + 1, (8 - kk) * cs - 2, cs - 2, 'none', 3, 'style="fill:none;stroke:var(--fg);stroke-width:2"')
    s.text(mx + kk * cs + cs / 2, my + kk * cs + 22, 'u₂₂', 'small b', 'middle')
    # multipliers column and update block
    s.rect(mx + kk * cs + 1, my + (kk + 1) * cs + 1, cs - 2, (7 - kk) * cs - 2, 'none', 3, 'style="fill:none;stroke:var(--fg2);stroke-width:1.5;stroke-dasharray:4 3"')
    s.rect(mx + (kk + 1) * cs + 1, my + (kk + 1) * cs + 1, (7 - kk) * cs - 2, (7 - kk) * cs - 2, 'none', 3, 'style="fill:none;stroke:var(--fg2);stroke-width:1.5"')
    s.text(mx + kk * cs + cs / 2, my + 8 * cs + 18, 'l = a·r', 'small t2', 'middle')
    s.text(mx + 5.5 * cs, my + 8 * cs + 18, 'a -= l · (pivot row)', 'small t2', 'middle')
    tx = mx + 8 * cs + 80
    items = [('1', 'Broadcast. The owner of row 2 (lane 2) shares entries 2…7 of'),
             ('', 'its row with the other lanes: 6 __shfl_sync calls.'),
             ('2', 'Reciprocal. Every lane computes r = 1/u₂₂ with femsweepRcp:'),
             ('', 'a hardware estimate plus two Newton steps, no slow-path branch.'),
             ('3', 'Keep r. The owner overwrites u₂₂ with r, since only the back'),
             ('', 'substitution reads the diagonal again: it then multiplies by r'),
             ('', 'instead of dividing by u₂₂.'),
             ('4', 'Eliminate. Each lane scales column 2 of its rows below 2 by r'),
             ('', '(dashed) and subtracts multiples of the pivot row (solid box).'),
             ('', ''),
             ('', 'Per element: 8 steps, 36 + 15 shuffles for LU and substitutions,'),
             ('', '8 reciprocals, no divisions.')]
    y = my + 10
    for num, t in items:
        if num:
            s.circle(tx + 8, y - 4, 9, 'box')
            s.text(tx + 8, y, num, 'small b', 'middle')
        s.text(tx + 24, y, t, 'lab t2')
        y += 20
    s.save('lu-step.svg')

###############################################################################
# 6. face-gather.svg: face table entry, face terms, face_b, row sums
###############################################################################

def fig_face():
    W, H = 860, 500
    s = Svg(W, H, 'Gathering the upwind face terms into the right-hand side')
    s.text(24, 32, 'Step 1–2 of an element solve: upwind face terms into b', 'title')
    s.text(24, 52, 'Shown for a direction whose incoming faces are x-, y-, z-; the face-to-dof map below is illustrative.', 'sub')
    # face table entry
    s.text(24, 88, 'Face table entry for (element e, local face): 8 ints, built once at setup (femsweepFaceTable)', 'lab b')
    fields = [('Fdat offset', 1), ('target dofs', 1), ('unused', 2), ('upwind dof index dis₀…dis₃', 4)]
    x = 24
    for name, n in fields:
        w = 90 * n
        s.rect(x, 98, w - 4, 30, 'box', 4)
        s.text(x + (w - 4) / 2, 118, name, 'small t2', 'middle')
        x += w
    s.text(24, 146, 'Replaces the chain elem_to_faces -> F_g2l -> idx1/idx2, which needed 4 dependent loads per face. None of it depends on angle or group.', 'small t3')
    # lanes computing terms
    s.text(24, 180, 'Lane l computes face dof j = l on every face:', 'lab b')
    s.text(24, 202, 'F = Σᵢ Ffactor · F[j][i] · X[disᵢ]', 'mono')
    s.text(24, 222, 'if the face is incoming and interior,', 'lab t2')
    s.text(24, 240, 'and writes it to face_b[face][dof(j)].', 'lab t2')
    s.text(24, 266, '6 faces × 4 face dofs = 24 terms,', 'lab t2')
    s.text(24, 284, '6 per lane.', 'lab t2')
    s.text(24, 316, 'face_b lives in shared memory, one', 'small t3')
    s.text(24, 330, 'slice per group, stride 48 + 4 doubles', 'small t3')
    s.text(24, 344, '(the padding avoids bank conflicts).', 'small t3')
    # face_b grid
    faces = [('x-', [0, 2, 4, 6], True), ('x+', [1, 3, 5, 7], False), ('y-', [0, 1, 4, 5], True),
             ('y+', [2, 3, 6, 7], False), ('z-', [0, 1, 2, 3], True), ('z+', [4, 5, 6, 7], False)]
    gx, gy, cs = 360, 196, 34
    s.text(gx, gy - 14, 'face_b[face][element dof]', 'lab b')
    for d in range(8):
        s.text(gx + d * cs + cs / 2, gy + 6, f'dof {d}', 'small t3', 'middle')
    for fi, (fname, dofs, active) in enumerate(faces):
        y = gy + 14 + fi * cs
        s.text(gx - 10, y + 22, fname, 'lab t2', 'end')
        for d in range(8):
            on = active and d in dofs
            s.rect(gx + d * cs + 1, y + 1, cs - 2, cs - 2, 'w1' if on else 'box', 3)
            s.text(gx + d * cs + cs / 2, y + 21, 'F' if on else '0', 'small' if on else 'small t3', 'middle')
        s.text(gx + 8 * cs + 10, y + 21, 'incoming' if active else 'outgoing: stays 0', 'small t2' if active else 'small t3')
    # column sums
    yb = gy + 14 + 6 * cs + 10
    for d in range(8):
        cx = gx + d * cs + cs / 2
        s.line(cx, yb, cx, yb + 18, 'arrow')
        s.rect(gx + d * cs + 4, yb + 24, cs - 8, 24, LANE_FILL[d % 4], 4)
        s.text(cx, yb + 40, f'b{d}', 'small b', 'middle', extra=ON_LANE[d % 4])
    s.text(gx + 8 * cs + 10, yb + 40, 'lane d % 4 sums column d into b_d', 'small t2')
    s.save('face-gather.svg')

###############################################################################
# 7. levels.svg: combined hyperplane sizes vs GPU capacity (32^3)
###############################################################################

def fig_levels():
    w = widths(32)
    warps = [72 * n * 2 for n in w]   # (a, e, group octet) items, one warp each
    cap = 108 * 32                    # 4 blocks of 8 warps per SM
    W, H = 860, 380
    s = Svg(W, H, 'Work per combined hyperplane at 32 cubed versus what the GPU holds at once')
    s.text(24, 32, 'Work in each combined hyperplane (32³ mesh, all 72 angles × 16 groups)', 'title')
    s.text(24, 52, 'Warps of work per daglevels launch (log scale) against the 3,456 warps an A100 runs at once for this kernel.', 'sub')
    x0, x1, y0, y1 = 72, 836, 80, 320
    lo, hi = 2, 5.2   # log10 range
    sy = lambda v: y1 - (y1 - y0) * (math.log10(v) - lo) / (hi - lo)
    sx = lambda h: x0 + (x1 - x0) * h / (len(w) - 1)
    for t, lab in [(100, '100'), (1000, '1K'), (10000, '10K'), (100000, '100K')]:
        s.line(x0, sy(t), x1, sy(t), 'grid')
        s.text(x0 - 10, sy(t) + 4, lab, 'lab t3', 'end')
    area = f'M{sx(0):.1f},{y1} ' + ' '.join(f'L{sx(h):.1f},{sy(v):.1f}' for h, v in enumerate(warps)) + f' L{sx(len(w) - 1):.1f},{y1} Z'
    s.path(area, 'w1')
    s.path(' '.join(('M' if h == 0 else 'L') + f'{sx(h):.1f},{sy(v):.1f}' for h, v in enumerate(warps)), 'ln s1')
    s.line(x0, sy(cap), x1, sy(cap), 'shi')
    s.text(x1, sy(cap) - 6, 'GPU capacity: 108 SMs × 32 warps', 'small t2', 'end')
    under = [h for h, v in enumerate(warps) if v < cap]
    s.text(sx(46), sy(max(warps)) - 8, f'peak {max(warps):,} warps (h = 46)', 'small t2', 'middle')
    s.text(sx(3), sy(150) - 10, f'{len(under)} of {len(w)} hyperplanes', 'small t2')
    s.text(sx(3), sy(150) + 4, 'fit in less than one wave', 'small t2')
    for t in [0, 15, 31, 46, 62, 78, 93]:
        s.text(sx(t), y1 + 18, str(t), 'lab t3', 'middle')
    s.text((x0 + x1) / 2, y1 + 38, 'hyperplane h', 'lab t2', 'middle')
    s.save('levels.svg')

###############################################################################
# 8. schedules.svg: schematic timelines of the three schedules
###############################################################################

def fig_schedules():
    W, H = 860, 420
    s = Svg(W, H, 'Schematic timelines of the three schedules')
    s.text(24, 32, 'Three ways to schedule the same DAG (schematic, not to scale)', 'title')
    s.text(24, 52, 'Each bar is GPU work over time; color marks the kind of unit doing it.', 'sub')
    x0, x1 = 210, 836
    # a) per-(a,g) blocks
    y = 84
    s.text(24, y + 12, 'coop / original', 'lab b')
    s.text(24, y + 28, 'one block per (a, g)', 'small t2')
    for r in range(6):
        ln = (x1 - x0) * (0.62 if r < 4 else 0.0)
        yy = y + r * 12
        seg = (x1 - x0) * 0.62 / 9
        for k in range(9):
            s.rect(x0 + k * seg, yy, seg - 3, 9, 'f3', 2)
    for r in range(2):
        yy = y + (4 + r) * 12
    s.text(x0, y + 86, 'barrier per hyperplane inside each block; 1,152 blocks rarely fill whole waves, so a partial last wave:', 'small t3')
    for r in range(2):
        yy = y + 96 + r * 12
        seg = (x1 - x0) * 0.62 / 9
        for k in range(9):
            s.rect(x0 + (x1 - x0) * 0.62 + 6 + k * seg * 0.6, yy, seg * 0.6 - 2, 9, 'f3', 2)
    # b) levels
    y = 230
    s.text(24, y + 12, 'daglevels / daggraph', 'lab b')
    s.text(24, y + 28, 'one launch per hyperplane', 'small t2')
    prof = [1, 3, 6, 10, 15, 18, 19, 18, 15, 10, 6, 3, 1]
    tot = sum(max(p, 4) for p in prof) + 12 * 2
    sc = (x1 - x0) / tot
    xx = x0
    for p in prof:
        wid = max(p, 4) * sc
        hgt = 10 + 40 * p / 19
        s.rect(xx, y + 50 - hgt, wid, hgt, 'f1', 2)
        xx += wid + 2 * sc
    s.text(x0, y + 68, 'kernel boundary = barrier for every angle and group at once; gaps = launch overhead (graph: ~1.7 µs less each)', 'small t3')
    # c) queue
    y = 330
    s.text(24, y + 12, 'dagqueue', 'lab b')
    s.text(24, y + 28, 'one persistent launch', 'small t2')
    s.rect(x0, y, (x1 - x0) * 0.86, 30, 'f2', 3)
    s.text(x0 + 10, y + 20, 'elements start as soon as all their upwind neighbors are done; no global barriers', 'lab', extra='style="fill:#ffffff"')
    s.text(x0, y + 50, 'costs: a queue pop, up to 3 atomic count updates and a push per (angle, element)', 'small t3')
    s.save('schedules.svg')

###############################################################################
# 9. work-queue.svg
###############################################################################

def fig_queue():
    W, H = 860, 500
    s = Svg(W, H, 'The persistent work queue')
    s.text(24, 32, 'dagqueue: a write-once queue, dependency counts, and the release chain', 'title')
    s.text(24, 52, 'Every (angle, element) is pushed exactly once per sweep, so the queue is a plain array with no wraparound.', 'sub')
    # queue array
    qx, qy, cw = 24, 110, 46
    vals = ['(a,e)'] * 7 + ['-1'] * 8
    s.text(qx, qy - 26, 'queue[ ]  (one int per slot; -1 = not written yet)', 'lab b')
    for i, v in enumerate(vals):
        cls = 'w1' if i < 5 else ('w2' if i < 7 else 'box')
        s.rect(qx + i * cw, qy, cw - 4, 30, cls, 4)
        s.text(qx + i * cw + (cw - 4) / 2, qy + 20, v, 'small t2' if v == '-1' else 'small', 'middle')
        s.text(qx + i * cw + (cw - 4) / 2, qy + 46, str(i), 'small t3', 'middle')
    for slot, lab in [(7, 'pushed = 7'), (9, 'taken = 9')]:
        cx = qx + slot * cw + (cw - 4) / 2
        s.path(f'M{cx - 5:.1f},{qy - 10} L{cx + 5:.1f},{qy - 10} L{cx:.1f},{qy - 3} Z', 'ffg2')
        s.text(cx, qy - 14, lab, 'small b', 'middle')
    s.text(qx, qy + 70, 'slots 0–4: taken and being solved by warps · 5–6: pushed, will be taken next ·', 'small t3')
    s.text(qx, qy + 84, '7–8: taken by warps that spin (with __nanosleep) until a producer writes the slot', 'small t3')
    # warp loop
    ly = 236
    s.text(24, ly, 'Each warp loops:', 'lab b')
    steps = ['1  slot = atomicAdd(taken, 1); exit if slot >= n',
             '2  wait until queue[slot] >= 0 (acquire load), then reset it to -1 and count[item] to 0',
             '3  look up the downwind neighbors across outgoing faces (overlaps the solve)',
             '4  solve the element for all 16 groups (2 passes of 8 groups)',
             '5  __syncwarp, then for each downwind element d:',
             '     if atomicAdd(count[d], 1) + 1 == indeg[d]  (acq_rel)',
             '        queue[atomicAdd(pushed, 1)] = d  (release store)']
    for k, t in enumerate(steps):
        s.text(24, ly + 22 + k * 19, t, 'mono t2')
    # ordering chain
    cy = 420
    s.text(24, cy, 'Why the consumer sees every upwind X it reads:', 'lab b')
    chain = ['X writes', '__syncwarp', 'count += 1 (acq_rel)', 'push (release)', 'pop (acquire)', '__syncwarp', 'read upwind X']
    x = 24
    for k, c in enumerate(chain):
        w = 6.3 * len(c) + 16
        s.rect(x, cy + 12, w, 26, 'box', 4)
        s.text(x + w / 2, cy + 29, c, 'small', 'middle')
        if k < len(chain) - 1:
            s.line(x + w + 2, cy + 25, x + w + 13, cy + 25, 'arrow')
        x += w + 16
    s.text(24, cy + 58, 'No deadlock: a waiting warp holds no item, and every pushed item is held by a running warp, whose completion pushes more.', 'small t3')
    s.save('work-queue.svg')

###############################################################################
# 10. bank-conflicts.svg
###############################################################################

def fig_banks():
    W, H = 860, 400
    s = Svg(W, H, 'Shared memory banks touched by half a warp clearing face_b')
    s.text(24, 32, 'Shared memory banks hit by half a warp (4 groups × 4 lanes, one double each)', 'title')
    s.text(24, 52, 'A double spans 2 of the 32 four-byte banks. Group g\'s slice starts at g × stride doubles.', 'sub')
    bw = 21
    x0 = 112

    def panel(y, stride, caption):
        s.text(24, y - 10, caption, 'lab b')
        for b in range(32):
            if b % 4 == 0:
                s.text(x0 + b * bw + bw / 2, y + 4, str(b), 'small t3', 'middle')
        hits = [0] * 32
        for g in range(4):
            yy = y + 12 + g * 22
            s.text(x0 - 10, yy + 14, f'group {g}', 'small t2', 'end')
            start = (2 * g * stride) % 32
            for b in range(32):
                s.rect(x0 + b * bw + 1, yy, bw - 2, 18, 'box', 2)
            for k in range(8):
                b = (start + k) % 32
                hits[b] += 1
                s.rect(x0 + b * bw + 1, yy, bw - 2, 18, LANE_FILL[g], 2)
        yy = y + 12 + 4 * 22 + 6
        s.text(x0 - 10, yy + 13, 'requests', 'small t2', 'end')
        for b in range(32):
            if hits[b]:
                s.text(x0 + b * bw + bw / 2, yy + 13, str(hits[b]), 'small b' if hits[b] > 1 else 'small t2', 'middle')
        return max(hits)

    m1 = panel(92, 56, 'stride 56 doubles (right for 8 lanes per element): groups 0 & 2 and 1 & 3 collide')
    m2 = panel(252, 52, 'stride 52 = 48 + LPE doubles: the 4 groups tile all 32 banks')
    s.text(x0 + 32 * bw + 10, 92 + 12 + 4 * 22 + 19, f'{m1}-way', 'lab b')
    s.text(x0 + 32 * bw + 10, 252 + 12 + 4 * 22 + 19, f'{m2}-way', 'lab b')
    s.save('bank-conflicts.svg')

###############################################################################
# 11. x-layout.svg
###############################################################################

def fig_xlayout():
    W, H = 860, 390
    s = Svg(W, H, 'Upwind X loads under the two X layouts')
    s.text(24, 32, 'One warp gathering upwind dof x_d for its 8 groups', 'title')
    s.text(24, 52, 'dag kernels: the warp\'s 8 groups share (angle, element), so they read the same dof of 8 different groups.', 'sub')
    # canonical
    y = 92
    s.text(24, y, 'Xdat layout [a][g][e][dof]: groups are 256 KB apart (32³ mesh)', 'lab b')
    x0, x1 = 24, 836
    segw = (x1 - x0) / 8
    for g in range(8):
        s.rect(x0 + g * segw + 2, y + 14, segw - 4, 28, 'box', 4)
        s.text(x0 + g * segw + 10, y + 33, f'group {g}', 'small t3')
        px = x0 + g * segw + segw * 0.72
        s.rect(px, y + 14, 5, 28, 'f2', 1)
    s.text(24, y + 66, 'So every load instruction touches 8 cache lines (6 sectors on average with some faces inactive);', 'lab t2')
    s.text(24, y + 84, 'the X gather was 54% of all global memory sectors and the top stall location.', 'lab t2')
    # group inner
    y = 222
    s.text(24, y, 'Group-innermost [a][e][dof][g]: the 8 values are adjacent', 'lab b')
    cw = 30
    cells = [f'g{g}' for g in range(12, 16)] + [f'g{g}' for g in range(16)] + [f'g{g}' for g in range(4)]
    for k, lab in enumerate(cells):
        on = 4 <= k < 12   # the warp's group octet: g0..g7 of dof d
        s.rect(x0 + k * (cw + 2), y + 14, cw, 28, 'f1' if on else 'box', 3)
        s.text(x0 + k * (cw + 2) + cw / 2, y + 33, lab, 'small' if on else 'small t3', 'middle',
               extra='style="fill:#ffffff"' if on else '')
    for k0, k1, lab in [(0, 4, 'dof d-1'), (4, 20, 'dof d: its 16 groups'), (20, 24, 'dof d+1')]:
        s.path(f'M{x0 + k0 * (cw + 2)},{y + 50} v5 h{(k1 - k0) * (cw + 2) - 2} v-5', 'edge')
        s.text(x0 + (k0 + k1) / 2 * (cw + 2), y + 70, lab, 'small t3', 'middle')
    s.text(24, y + 96, 'So the warp\'s 8 groups (blue) read 64 contiguous bytes: 2 sectors in 1 cache line.', 'lab t2')
    s.text(24, y + 124, 'The sweep runs 12% faster at 32³, but converting back to Xdat\'s layout costs ~3.7 ms (4.8 GB moved), more than it saves.', 'lab t2')
    s.text(24, y + 142, 'It pays only if the application keeps X in this layout between sweeps.', 'lab t2')
    s.save('x-layout.svg')


for f in [fig_speedup, fig_progress, fig_original, fig_warp, fig_lu, fig_face, fig_levels, fig_schedules, fig_queue, fig_banks, fig_xlayout]:
    f()
print('ok')
