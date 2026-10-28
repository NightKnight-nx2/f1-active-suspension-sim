"""
scenario.py  --  Test parkuru, hız profili, yol bozuklukları, tur zamanı çözücüsü
=================================================================================

Parkur: düzlük + viraj segmentleri (kapalı tur).  Eğrilik κ(s) segmentlerden kurulup
yumuşatılır.

Hız profili (referans): nokta-kütle, sürtünme-çemberi, motor gücü sınırlı ileri/geri
geçiş çözücüsü.  Referans profil İDEAL aero (η=1) ile hesaplanır ve ÜÇ senaryoya da
birebir aynı hız girdisi olarak verilir.

Tur zamanı: simülasyonda her senaryonun gerçekleşen downforce verimi
η(s) = F_gerçek / F_ideal bulunur; çözücü bu η(s) ile (daha az downforce -> daha az
viraj tutuşu, ama daha az indüklenmiş sürükleme) yeniden çalıştırılır.  Böylece
"aynı hız profiliyle koşturma" (adil downforce kıyası) ile "senaryoya özgü tur zamanı"
birlikte elde edilir.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

G = 9.81


# ------------------------------------------------------------------ parkur
@dataclass
class Segment:
    kind: str          # "straight" | "corner"
    length: float      # [m]
    radius: float = 0  # corner için [m]


def default_track():
    return [
        Segment("straight", 1200),
        Segment("corner", 140, 45),      # T1 : yavaş
        Segment("straight", 300),
        Segment("corner", 260, 110),     # T2 : hızlı süpürme (downforce'a duyarlı)
        Segment("straight", 1100),       # arka düzlük
        Segment("corner", 110, 35),      # T3 : saç tokası
        Segment("straight", 450),
        Segment("corner", 200, 85),      # T4 : orta
        Segment("straight", 400),
    ]


class Track:
    def __init__(self, segments, ds=2.0, smooth=30.0):
        self.segments = segments
        self.S = float(sum(s.length for s in segments))
        self.N = int(round(self.S / ds))
        self.ds = self.S / self.N
        self.s = np.arange(self.N) * self.ds
        k = np.zeros(self.N)
        pos = 0.0
        for sg in segments:
            if sg.kind == "corner":
                m = (self.s >= pos) & (self.s < pos + sg.length)
                k[m] = 1.0 / sg.radius
            pos += sg.length
        # periyodik hareketli ortalama (clothoid benzeri geçiş)
        w = max(int(smooth / self.ds), 1)
        ker = np.ones(w) / w
        kk = np.concatenate([k[-w:], k, k[:w]])
        self.kappa = np.convolve(kk, ker, mode="same")[w:-w]

    def kappa_at(self, s):
        return np.interp(np.mod(s, self.S), self.s, self.kappa, period=self.S)


# ------------------------------------------------------------ yol bozukluğu
@dataclass
class Road:
    """Gauss tümsek/çukurlar:  w(s) = Σ A_j exp(-(s-s_j)²/(2σ_j²))  (periyodik)."""
    bumps: list            # (s_j [m], A_j [m], sigma_j [m])
    S: float
    gain: float = 1.0

    def eval(self, s):
        s = np.mod(np.asarray(s, float), self.S)
        w = np.zeros_like(s); d1 = np.zeros_like(s); d2 = np.zeros_like(s)
        for sj, A, sg in self.bumps:
            for shift in (-self.S, 0.0, self.S):
                x = s - sj + shift
                g = self.gain * A * np.exp(-x * x / (2 * sg * sg))
                w += g
                d1 += g * (-x / sg ** 2)
                d2 += g * (x * x / sg ** 4 - 1.0 / sg ** 2)
        return w, d1, d2


def default_road(track: Track, gain=1.0):
    bumps = [(600, 3.5e-3, 1.3), (1000, -3.0e-3, 1.5), (1335, 6.0e-3, 1.0),
             (2300, 4.0e-3, 1.4), (2650, -3.5e-3, 1.5), (3600, 3.0e-3, 1.2)]
    return Road(bumps, track.S, gain)


# --------------------------------------------------- hız profili çözücüsü
@dataclass
class DriveParams:
    m: float = 800.0
    power: float = 700e3     # tekerlekte etkin güç [W]
    mu: float = 1.9          # lastik sürtünme katsayısı
    v_max: float = 97.0      # [m/s]  (~349 km/h sınırlayıcı)


def solve_speed_profile(track: Track, aero, dp: DriveParams, eta=None):
    """Kapalı turda v(s) [m/s].  eta: pist noktalarında downforce verimi (None -> 1)."""
    N, ds, kap = track.N, track.ds, track.kappa
    eta = np.ones(N) if eta is None else np.asarray(eta)
    cla = aero.cla_peak * eta
    rho, m, mu = aero.rho, dp.m, dp.mu
    # viraj sınırı
    den = kap - mu * rho * cla / (2 * m)
    with np.errstate(divide="ignore", invalid="ignore"):
        vc = np.where(den > 1e-9, np.sqrt(mu * G / np.maximum(den, 1e-9)), dp.v_max)
    v = np.minimum(vc, dp.v_max)

    def forces(i, vi):
        q = 0.5 * rho * vi * vi
        grip = mu * (m * G + q * cla[i])
        lat = m * vi * vi * kap[i]
        Fav = math.sqrt(max(grip * grip - lat * lat, 0.0))
        drag = q * float(aero.drag_cda(cla[i]))
        return Fav, drag

    i0 = int(np.argmin(v))
    for k in range(N):                         # ileri geçiş (hızlanma)
        i, j = (i0 + k) % N, (i0 + k + 1) % N
        Fav, drag = forces(i, v[i])
        Fx = min(dp.power / max(v[i], 1.0), Fav)
        a = (Fx - drag) / m
        v[j] = min(v[j], math.sqrt(max(v[i] ** 2 + 2 * a * ds, 0.0)))
    for k in range(N):                         # geri geçiş (frenleme)
        i, j = (i0 - k) % N, (i0 - k - 1) % N
        Fav, drag = forces(i, v[i])
        d = (Fav + drag) / m
        v[j] = min(v[j], math.sqrt(v[i] ** 2 + 2 * d * ds))
    return v


def lap_time(track: Track, v):
    vn = np.roll(v, -1)
    return float(np.sum(2 * track.ds / (v + vn)))


# ----------------------------------------------------------------- senaryo
class Signal:
    __slots__ = ("s", "v", "ax", "ay", "mode", "wdd_f", "wdd_r")

    def __init__(self, row):
        self.s, self.v, self.ax, self.ay, self.mode, self.wdd_f, self.wdd_r = row


class Scenario:
    """Zaman tablosu: simülasyonun hız / ivme / yol girdileri (tüm senaryolar için ortak)."""
    COLS = ("s", "v", "ax", "ay", "mode", "wdd_f", "wdd_r")

    def __init__(self, track: Track, v_of_s, wheelbase, road: Road | None = None,
                 dt=1e-3, accel_smooth=0.12):
        self.track, self.dt = track, dt
        # s -> t
        vn = np.roll(v_of_s, -1)
        dts = 2 * track.ds / (v_of_s + vn)
        t_nodes = np.concatenate([[0.0], np.cumsum(dts)])
        s_nodes = np.concatenate([track.s, [track.S]])
        v_nodes = np.concatenate([v_of_s, [v_of_s[0]]])
        self.t_end = float(t_nodes[-1])
        t = np.arange(0.0, self.t_end + dt, dt)
        v = np.interp(t, t_nodes, v_nodes)
        # ivmeyi yumuşat (fren/gaz geçişleri kırık) -> ax
        sig_n = max(int(accel_smooth / dt), 1)
        kx = np.arange(-4 * sig_n, 4 * sig_n + 1)
        ker = np.exp(-0.5 * (kx / sig_n) ** 2); ker /= ker.sum()
        vp = np.pad(v, 4 * sig_n, mode="edge")
        v = np.convolve(vp, ker, mode="same")[4 * sig_n:-4 * sig_n]
        ax = np.gradient(v, dt)
        s = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * dt)])
        s *= track.S / s[-1] if abs(s[-1] - track.S) / track.S < 0.01 else 1.0
        kap = track.kappa_at(s)
        ay = v * v * kap
        mode = np.clip(kap / 0.008, 0, 1)
        mode = mode * mode * (3 - 2 * mode)           # smoothstep
        if road is None:
            wdd_f = wdd_r = np.zeros_like(t)
        else:
            _, d1f, d2f = road.eval(s)
            _, d1r, d2r = road.eval(s - wheelbase)    # arka teker aynı bozukluğu L/v sonra görür
            wdd_f = d2f * v * v + d1f * ax
            wdd_r = d2r * v * v + d1r * ax
        self.t = t
        self.table = np.column_stack([s, v, ax, ay, mode, wdd_f, wdd_r])
        self.n = len(t)

    def at(self, t):
        x = t / self.dt
        i = int(x)
        if i >= self.n - 1:
            return Signal(self.table[-1])
        f = x - i
        r0 = self.table[i]
        return Signal(r0 + f * (self.table[i + 1] - r0))

    def at_array(self, t):
        return {c: np.interp(t, self.t, self.table[:, k]) for k, c in enumerate(self.COLS)}


class ConstantScenario:
    """Sabit hız / sabit ivme, yol bozukluğu yok -- doğrulama testleri için."""
    def __init__(self, v, ax=0.0, t_end=3.0, mode=0.0, ay=0.0):
        self.v, self.ax, self.t_end, self.mode, self.ay = v, ax, t_end, mode, ay

    def at(self, t):
        return Signal([self.v * t, self.v, self.ax, self.ay, self.mode, 0.0, 0.0])

    def at_array(self, t):
        t = np.asarray(t, float)
        one = np.ones_like(t)
        return dict(s=self.v * t, v=self.v * one, ax=self.ax * one, ay=self.ay * one,
                    mode=self.mode * one, wdd_f=0 * one, wdd_r=0 * one)
