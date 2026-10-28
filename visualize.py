"""
visualize.py  --  koyu tema grafikler + yandan araç şeması / canlı grafik animasyonu
====================================================================================
plot_aero_curve   : C_L(h) eğrisi (ön/arka, tepe ve stall işaretli)
plot_single       : tek senaryo özet grafiği (doğrulama)
plot_comparison   : üç senaryo ride height / downforce zaman serileri
plot_summary      : banked downforce, tur zamanı, 300+ km/h downforce kaybı
animate           : üstte yandan araç şeması (gövde, yay-sönümleyici, ride height oku),
                    altta canlı ilerleyen grafik paneli
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, FancyArrowPatch, Rectangle
from matplotlib import cm, colors as mcolors

BG, PANEL, FG, GRID = "#0d1117", "#161b22", "#c9d1d9", "#30363d"
SCEN_COLORS = {"Pasif-Alçak": "#ff7b72", "Pasif-Yüksek": "#79c0ff", "Aktif": "#3fb950",
               "Aktif (PID)": "#d2a8ff", "Aktif (LQR)": "#3fb950"}
IDEAL_COLOR = "#8b949e"
ACCENT = "#f2cc60"


def color_of(name):
    return SCEN_COLORS.get(name, "#ffffff")


def apply_dark_style():
    plt.rcParams.update({
        "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
        "axes.edgecolor": GRID, "axes.labelcolor": FG, "axes.titlecolor": FG,
        "xtick.color": FG, "ytick.color": FG, "text.color": FG,
        "grid.color": GRID, "grid.alpha": 0.6, "axes.grid": True,
        "legend.facecolor": PANEL, "legend.edgecolor": GRID, "legend.labelcolor": FG,
        "font.size": 10, "axes.titlesize": 11, "lines.linewidth": 1.4,
        "font.family": "DejaVu Sans",
    })


apply_dark_style()


def _decim(n, target=3000):
    return max(1, n // target)


def _shade_corners(ax, t, mode):
    m = mode > 0.5
    edges = np.flatnonzero(np.diff(m.astype(int)))
    starts = list(edges[m[edges + 1]] + 1) if len(edges) else []
    ends = list(edges[~m[edges + 1]] + 1) if len(edges) else []
    if m[0]:
        starts = [0] + starts
    if m[-1]:
        ends = ends + [len(t) - 1]
    for a, b in zip(starts, ends):
        ax.axvspan(t[a], t[b], color="#ffffff", alpha=0.06, lw=0)


# ------------------------------------------------------------------ aero eğrisi
def plot_aero_curve(aero, path):
    h = np.linspace(0, 0.09, 800)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for curve, nm, c in ((aero.front, "ön eksen", "#79c0ff"), (aero.rear, "arka eksen", "#ffa657")):
        ax.plot(h * 1e3, curve.cl(h), color=c, label=f"C_L {nm}")
        ax.axvline(curve.h_peak * 1e3, color=c, ls=":", lw=1)
        ax.axvline(curve.h_stall * 1e3, color=c, ls="--", lw=1)
        ax.plot([curve.h_peak * 1e3], [curve.c_peak], "o", color=c)
        ax.annotate(f"tepe {curve.h_peak*1e3:.0f} mm", (curve.h_peak * 1e3, curve.c_peak),
                    xytext=(6, 6), textcoords="offset points", color=c, fontsize=9)
        ax.annotate(f"stall {curve.h_stall*1e3:.0f} mm", (curve.h_stall * 1e3, curve.cl(curve.h_stall)),
                    xytext=(-62, -14), textcoords="offset points", color=c, fontsize=9)
    ax.set_xlabel("ride height [mm]"); ax.set_ylabel("C_L (ClA) [m²]")
    ax.set_title("Floor stall'lu aero haritası: tepe + keskin düşüş")
    ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# --------------------------------------------------------------------- tek senaryo
def plot_single(res, aero, path):
    t, k = res["t"], _decim(len(res["t"]))
    sl = slice(None, None, k)
    c = color_of(res["name"])
    fig, axs = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True)
    ax = axs[0]
    _shade_corners(ax, t, res["mode"])
    ax.plot(t[sl], res["v"][sl] * 3.6, color=ACCENT, label="hız [km/h]")
    ax.set_ylabel("hız [km/h]")
    ax2 = ax.twinx(); ax2.grid(False)
    ax2.plot(t[sl], res["ay"][sl] / 9.81, color="#d2a8ff", lw=1, label="yanal ivme [g]")
    ax2.set_ylabel("yanal ivme [g]")
    ax.set_title(f"{res['name']} -- parkur girdisi (açık bantlar: viraj)")
    ax = axs[1]
    _shade_corners(ax, t, res["mode"])
    ax.plot(t[sl], res["hf"][sl] * 1e3, color="#79c0ff", label="h_f (ön)")
    ax.plot(t[sl], res["hr"][sl] * 1e3, color="#ffa657", label="h_r (arka)")
    for cv, cc in ((aero.front, "#79c0ff"), (aero.rear, "#ffa657")):
        ax.axhline(cv.h_peak * 1e3, color=cc, ls=":", lw=1)
        ax.axhline(cv.h_stall * 1e3, color=cc, ls="--", lw=1)
    ax.set_ylabel("ride height [mm]"); ax.legend(loc="upper right", ncol=2)
    ax.text(0.005, 0.04, "noktalı: aero tepe   kesikli: stall eşiği", transform=ax.transAxes, fontsize=8, color=IDEAL_COLOR)
    ax = axs[2]
    _shade_corners(ax, t, res["mode"])
    ax.plot(t[sl], res["F_ideal"][sl] / 1e3, color=IDEAL_COLOR, ls="--", label="ideal (tepe)")
    ax.plot(t[sl], res["F"][sl] / 1e3, color=c, label="gerçek")
    ax.set_ylabel("downforce [kN]"); ax.set_xlabel("zaman [s]"); ax.legend(loc="upper right")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# -------------------------------------------------------------------- karşılaştırma
def plot_comparison(results, aero, path):
    t = results[0]["t"]; k = _decim(len(t)); sl = slice(None, None, k)
    fig, axs = plt.subplots(5, 1, figsize=(13, 13), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1.5, 1.5, 1.5, 1.1]))
    ax = axs[0]
    _shade_corners(ax, t, results[0]["mode"])
    ax.plot(t[sl], results[0]["v"][sl] * 3.6, color=ACCENT)
    ax.axhline(300, color=IDEAL_COLOR, ls=":", lw=1)
    ax.set_ylabel("hız [km/h]"); ax.set_title("Ortak hız profili (açık bantlar: viraj) -- üç senaryo da aynı girdiyle koşuyor")
    for ax, key, cv, lab in ((axs[1], "hf", aero.front, "ön ride height h_f [mm]"),
                             (axs[2], "hr", aero.rear, "arka ride height h_r [mm]")):
        _shade_corners(ax, t, results[0]["mode"])
        ax.axhspan(cv.h_stall * 1e3, cv.h_peak * 1e3, color="#f85149", alpha=0.10, lw=0)
        ax.axhline(cv.h_peak * 1e3, color=IDEAL_COLOR, ls=":", lw=1)
        ax.axhline(cv.h_stall * 1e3, color="#f85149", ls="--", lw=1)
        for r in results:
            ax.plot(t[sl], r[key][sl] * 1e3, color=color_of(r["name"]), label=r["name"], lw=1.3)
        ax.set_ylabel(lab)
    axs[1].legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3, frameon=False)
    axs[1].set_title("kırmızı bant: aero tepe ile stall eşiği arası (downforce'un çöktüğü bölge)", fontsize=9,
                     color="#f85149", loc="left")
    ax = axs[3]; _shade_corners(ax, t, results[0]["mode"])
    ax.plot(t[sl], results[0]["F_ideal"][sl] / 1e3, color=IDEAL_COLOR, ls="--", label="ideal (her an tepe)")
    for r in results:
        ax.plot(t[sl], r["F"][sl] / 1e3, color=color_of(r["name"]), label=r["name"], lw=1.3)
    ax.set_ylabel("toplam downforce [kN]"); ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=4, frameon=False)
    ax = axs[4]; _shade_corners(ax, t, results[0]["mode"])
    for r in results:
        ax.plot(t[sl], r["eta"][sl] * 100, color=color_of(r["name"]), lw=1.2)
    ax.set_ylabel("verim η = F/F_ideal [%]"); ax.set_xlabel("zaman [s]"); ax.set_ylim(40, 105)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_summary(metrics, path, extra_note=""):
    names = list(metrics)
    cols = [color_of(n) for n in names]
    fig, axs = plt.subplots(1, 3, figsize=(14.5, 5.0))
    xt = [n.replace("-", "-\n") if "Pasif" in n else n for n in names]

    def bars(ax, vals, title, labels, ylabel, ylim):
        b = ax.bar(range(len(names)), vals, color=cols, width=0.62, zorder=3)
        ax.set_axisbelow(True)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(xt, fontsize=9)
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="x", visible=False)
        ax.set_ylim(*ylim)
        pad = (ylim[1] - ylim[0]) * 0.015
        for r, v, lab in zip(b, vals, labels):
            ax.text(r.get_x() + r.get_width() / 2, v + pad, lab, ha="center", va="bottom", fontsize=9, color=FG, zorder=4)

    bk = [metrics[n]["banked_MJ"] for n in names]
    bars(axs[0], bk, "Banked downforce  ∫F ds  (tur başına)",
         [f"{metrics[n]['banked_MJ']:.1f} MJ\n%{metrics[n]['banked_pct']:.0f} ideal" for n in names],
         "MJ  (= MN·m)", (0, max(bk) * 1.18))
    lt = [metrics[n]["lap_time"] for n in names]; best = min(lt)
    d = [x - best for x in lt]
    bars(axs[1], d, "Tur zamanı farkı (en hızlıya göre)",
         [f"{metrics[n]['lap_time']:.2f} s\n" + ("en hızlı" if x == best else f"+{x-best:.2f} s") for n, x in zip(names, lt)],
         "Δ tur zamanı [s]", (0, max(max(d) * 1.35, 0.1)))
    ls = [metrics[n]["loss300_pct"] for n in names]
    bars(axs[2], ls, "300+ km/h downforce kaybı",
         [f"%{v:.1f}" for v in ls], "kayıp [%]  (1 - F/F_ideal)", (0, max(ls) * 1.2 + 1))
    if extra_note:
        fig.text(0.5, 0.008, extra_note, ha="center", fontsize=8, color=IDEAL_COLOR)
    fig.tight_layout(rect=(0, 0.035, 1, 1)); fig.savefig(path, dpi=130); plt.close(fig)


# =========================================================================== animasyon
def _zigzag(x, y0, y1, n=9, amp=0.07):
    ys = np.linspace(y0, y1, 2 * n + 3)
    xs = np.full_like(ys, x)
    xs[2:-2:2] += amp; xs[3:-2:2] -= amp
    return xs, ys


class CarPanel:
    """Bir senaryo için yandan araç şeması.  Ride height ölçeği EX kat abartılıdır;
    gövde, zemin çizgisi (ön/arka ride height'tan geçen doğru) boyunca 'shear' edilir."""
    EX = 7.0               # ride height görsel büyütme (1 mm -> 7 mm çizim)
    HUB = 0.33             # tekerlek merkezi yüksekliği
    BODY = [(-2.4, 0.0), (3.1, 0.0), (3.1, 0.10), (2.1, 0.17), (1.3, 0.30), (0.75, 0.58), (0.25, 0.80),
            (-0.35, 0.80), (-0.45, 1.00), (-0.8, 1.00), (-0.9, 0.72), (-1.7, 0.56), (-2.4, 0.52)]
    FW = [(2.55, 0.0), (3.45, 0.0), (3.45, 0.05), (2.55, 0.05)]
    RW_PLANE = [(-2.62, 0.93), (-1.85, 0.93), (-1.85, 1.0), (-2.62, 1.0)]
    RW_END = [(-2.5, 0.50), (-2.38, 0.50), (-2.38, 1.02), (-2.5, 1.02)]

    def __init__(self, ax, name, params, aero):
        self.ax, self.name, self.p, self.aero = ax, name, params, aero
        self.xf, self.xr = params.a, -params.b
        ax.set_xlim(-3.3, 4.0); ax.set_ylim(-0.10, 1.72); ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_color(color_of(name)); sp.set_linewidth(1.6)
        ax.add_patch(Rectangle((-3.3, -0.10), 7.3, 0.10, color="#21262d", lw=0, zorder=0))
        ax.plot([-3.3, 4.0], [0, 0], color="#8b949e", lw=1.5, zorder=1)
        self.body = Polygon(self.BODY, closed=True, fc="#2ea043", ec="#e6edf3", lw=1.2, zorder=5)
        self.parts = [Polygon(pts, closed=True, fc="#e6edf3", ec="none", zorder=6) for pts in (self.FW, self.RW_PLANE, self.RW_END)]
        self.part_pts = [self.FW, self.RW_PLANE, self.RW_END]
        ax.add_patch(self.body)
        for pt in self.parts:
            ax.add_patch(pt)
        for xw in (self.xf, self.xr):      # tekerlekler gövdenin önünde, yarı saydam (yay görünsün)
            ax.add_patch(Circle((xw, self.HUB), self.HUB, fc="#0b0e13", ec="#6e7681", lw=2.0, alpha=0.62, zorder=7))
            ax.add_patch(Circle((xw, self.HUB), 0.13, fc="#6e7681", ec="none", zorder=7))
        self.spring = {k: ax.plot([], [], color="#f2cc60", lw=1.8, zorder=8)[0] for k in "fr"}
        self.damper = {k: ax.plot([], [], color="#a5d6ff", lw=3.2, solid_capstyle="butt", zorder=8)[0] for k in "fr"}
        self.rh = {k: FancyArrowPatch((0, 0), (0, 0), arrowstyle="<|-|>", mutation_scale=9, color="#ffffff", lw=1.3, zorder=9) for k in "fr"}
        self.df = {k: FancyArrowPatch((0, 0), (0, 0), arrowstyle="-|>", mutation_scale=14, color=ACCENT, lw=2.6, zorder=9) for k in "fr"}
        for a in list(self.rh.values()) + list(self.df.values()):
            ax.add_patch(a)
        self.txt_rh = {k: ax.text(0, 0, "", color="#ffffff", fontsize=8.5, ha="center", va="bottom", zorder=10,
                                  bbox=dict(boxstyle="round,pad=0.12", fc=PANEL, ec="none", alpha=0.9)) for k in "fr"}
        self.title = ax.text(-3.22, 1.66, name, color=color_of(name), fontsize=12, fontweight="bold", va="top", zorder=10)
        self.badge = ax.text(3.9, 1.66, "", color="#ff7b72", fontsize=11, fontweight="bold", ha="right", va="top", zorder=10)
        self.cmap = plt.get_cmap("RdYlGn"); self.norm = mcolors.Normalize(0.45, 1.0)

    def floor_y(self, x, hf, hr):
        e = self.EX
        return e * hr + (e * hf - e * hr) * (x - self.xr) / (self.xf - self.xr)

    def update(self, hf, hr, Ff, Fr, eta):
        e = self.EX
        shear = lambda pts: [(x, self.floor_y(x, hf, hr) + y) for x, y in pts]
        self.body.set_xy(shear(self.BODY))
        for pt, pts in zip(self.parts, self.part_pts):
            pt.set_xy(shear(pts))
        self.body.set_facecolor(self.cmap(self.norm(eta)))
        for k, x, h in (("f", self.xf, hf), ("r", self.xr, hr)):
            top = self.floor_y(x, hf, hr) + 0.62
            xs, ys = _zigzag(x - 0.13, self.HUB, top)
            self.spring[k].set_data(xs, ys)
            self.damper[k].set_data([x + 0.13, x + 0.13], [self.HUB, top])
            xa = x + (0.80 if k == "f" else -0.80)
            ya = self.floor_y(xa, hf, hr)
            self.rh[k].set_positions((xa, 0.0), (xa, ya))
            self.txt_rh[k].set_position((xa, ya + 0.06))
            self.txt_rh[k].set_text(f"h_{k} = {h*1e3:4.1f} mm")
            F = Ff if k == "f" else Fr
            L = 0.05 * F / 1e3
            y0 = top + 0.12
            self.df[k].set_positions((x, y0 + L + 0.05), (x, y0 + 0.05))
        cf, cr = self.aero.front, self.aero.rear
        if hf < cf.h_peak and cf.shape(hf) < 0.6 or hr < cr.h_peak and cr.shape(hr) < 0.6:
            self.badge.set_text("FLOOR STALL"); self.badge.set_color("#ff7b72")
        elif (hf > cf.h_peak and cf.shape(hf) < 0.85) or (hr > cr.h_peak and cr.shape(hr) < 0.85):
            self.badge.set_text("ZEMİN ETKİSİ ZAYIF (yüksek)"); self.badge.set_color("#79c0ff")
        else:
            self.badge.set_text("AERO TEPESİNDE"); self.badge.set_color("#3fb950")


class Readout:
    """Senaryo için sayısal gösterge paneli (araç şemasının sağında)."""
    def __init__(self, ax, name, aero):
        self.ax, self.name, self.aero = ax, name, aero
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        kw = dict(fontfamily="DejaVu Sans Mono", fontsize=10.5, va="center", color=FG)
        self.t1 = ax.text(0.03, 0.83, "", **kw)
        self.t2 = ax.text(0.03, 0.60, "", **kw)
        self.t3 = ax.text(0.03, 0.40, "", **kw)
        ax.add_patch(Rectangle((0.03, 0.10), 0.62, 0.12, fc="#21262d", ec="none"))
        self.bar = Rectangle((0.03, 0.10), 0.0, 0.12, fc="#3fb950", ec="none")
        ax.add_patch(self.bar)
        self.t4 = ax.text(0.67, 0.16, "", **kw)
        self.cmap = plt.get_cmap("RdYlGn"); self.norm = mcolors.Normalize(0.45, 1.0)

    def update(self, r, i):
        v = r["v"][i] * 3.6
        self.t1.set_text(f"v = {v:5.0f} km/h    a_y = {r['ay'][i]/9.81:4.1f} g    a_x = {r['ax'][i]/9.81:+4.1f} g")
        self.t2.set_text(f"h_f {r['hf'][i]*1e3:5.1f} mm (tepe {self.aero.front.h_peak*1e3:2.0f})   h_r {r['hr'][i]*1e3:5.1f} mm (tepe {self.aero.rear.h_peak*1e3:2.0f})")
        self.t3.set_text(f"downforce {r['F'][i]/1e3:5.1f} kN  (ideal {r['F_ideal'][i]/1e3:5.1f})   u = {r['uf'][i]/1e3:+4.1f}/{r['ur'][i]/1e3:+4.1f} kN")
        eta = r["eta"][i]
        self.bar.set_width(0.62 * max(min(eta, 1.0), 0.0)); self.bar.set_facecolor(self.cmap(self.norm(eta)))
        self.t4.set_text(f"verim η = {eta*100:3.0f} %")


def animate(results, aero, params, path="animasyon.mp4", fps=25, speedup=2.5, size=(1920, 1080)):
    from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
    t = results[0]["t"]
    n_frames = int(t[-1] / speedup * fps)
    idx = np.linspace(0, len(t) - 1, n_frames).astype(int)
    dpi = 100
    fig = plt.figure(figsize=(size[0] / dpi, size[1] / dpi), dpi=dpi)
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 0.95], hspace=0.10, left=0.05, right=0.985, top=0.975, bottom=0.065)
    n = len(results)
    top = outer[0].subgridspec(n, 2, wspace=0.03, hspace=0.12, width_ratios=[0.8, 1.0])
    panels = [CarPanel(fig.add_subplot(top[i, 0]), r["name"], params, aero) for i, r in enumerate(results)]
    reads = [Readout(fig.add_subplot(top[i, 1]), r["name"], aero) for i, r in enumerate(results)]
    bot = outer[1].subgridspec(3, 1, hspace=0.14, height_ratios=[1.25, 1.25, 0.8])
    axh, axf, axv = [fig.add_subplot(bot[i, 0]) for i in range(3)]
    k = _decim(len(t), 1800)
    ts = t[::k]
    cv = aero.front
    axh.axhspan(cv.h_stall * 1e3, cv.h_peak * 1e3, color="#f85149", alpha=0.12, lw=0)
    axh.axhline(cv.h_peak * 1e3, color=IDEAL_COLOR, ls=":", lw=1)
    axh.axhline(cv.h_stall * 1e3, color="#f85149", ls="--", lw=1)
    allh = np.concatenate([r["hf"] for r in results]) * 1e3
    axh.set_ylim(allh.min() - 2, allh.max() + 2)
    axh.set_ylabel("ön ride height [mm]", fontsize=9)
    axf.plot(ts, results[0]["F_ideal"][::k] / 1e3, color=IDEAL_COLOR, ls="--", lw=1, label="ideal")
    axf.set_ylabel("downforce [kN]", fontsize=9)
    axv.plot(ts, results[0]["v"][::k] * 3.6, color=ACCENT, lw=1.2)
    axv.set_ylabel("hız [km/h]", fontsize=9); axv.set_xlabel("zaman [s]")
    for a in (axh, axf, axv):
        a.set_xlim(0, t[-1]); a.tick_params(labelsize=8)
        _shade_corners(a, t, results[0]["mode"])
    axv.set_ylim(0, 360)
    axh.tick_params(labelbottom=False); axf.tick_params(labelbottom=False)
    lines_h, lines_f = [], []
    for r in results:
        c = color_of(r["name"])
        lines_h.append(axh.plot([], [], color=c, lw=1.6, label=r["name"])[0])
        lines_f.append(axf.plot([], [], color=c, lw=1.6)[0])
    axh.legend(loc="upper right", ncol=3, fontsize=8)
    cursors = [a.axvline(0, color="#ffffff", lw=0.9, alpha=0.8) for a in (axh, axf, axv)]
    fig.text(0.5, 0.012, f"{speedup:g}× hızlı oynatma  |  çizimde ride height {CarPanel.EX:g}× büyütülmüştür  |  açık bantlar: viraj  |  kırmızı bant: stall bölgesi",
             ha="center", fontsize=8.5, color=IDEAL_COLOR)
    hf_cols = [r["hf"][::k] * 1e3 for r in results]
    F_cols = [r["F"][::k] / 1e3 for r in results]

    def update(fi):
        i = idx[fi]
        j = i // k + 1
        for p, rd, r in zip(panels, reads, results):
            p.update(r["hf"][i], r["hr"][i], r["Ff"][i], r["Fr"][i], r["eta"][i])
            rd.update(r, i)
        for lh, lf, hc, fc in zip(lines_h, lines_f, hf_cols, F_cols):
            lh.set_data(ts[:j], hc[:j]); lf.set_data(ts[:j], fc[:j])
        for c in cursors:
            c.set_xdata([t[i], t[i]])
        return []

    anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
    try:
        writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=6000, extra_args=["-pix_fmt", "yuv420p"])
        anim.save(path, writer=writer, dpi=dpi)
    except Exception as e:                                        # ffmpeg yoksa GIF
        print("ffmpeg kullanılamadı, GIF'e düşülüyor:", e)
        path = path.rsplit(".", 1)[0] + ".gif"
        anim.save(path, writer=PillowWriter(fps=min(fps, 12)), dpi=50)
    plt.close(fig)
    return path
