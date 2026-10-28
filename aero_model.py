"""
aero_model.py  --  Ride height'a bağlı aerodinamik downforce modeli
====================================================================

C_L(h) eğrisi "tepe + keskin düşüş" biçimindedir (ground-effect / floor stall):

    h büyük   : zemin etkisi zayıflar, C_L yavaşça C_base (kanatçıklar) değerine iner
    h = h_peak: C_L maksimum (C_peak)
    h < h_peak: zemin ile yol arası daralır, akış tıkanır -> h_stall civarında
                C_L aniden çöker ("floor stall" / porpoising'in kaynağı)

Parametrik form (iki farklı genişlik ve şekil üssü olan, asimetrik
"stretched-exponential" -- iki üstel yarı-eğrinin tepe noktasında birleşimi):

    d = h - h_peak
    g(h) = exp( -ln2 * ( d  / w_high )^q_high )      d >= 0   (yumuşak, yüksek taraf)
    g(h) = exp( -ln2 * (-d  / w_stall)^p_low  )      d <  0   (sert, düşük taraf)
    C_L(h) = C_base + (C_peak - C_base) * g(h)

    w_high  = h_half_high        -> tepe üstünde g = 0.5 olduğu yükseklik farkı
    w_stall = h_peak - h_stall   -> stall eşiği: g(h_stall) = 0.5 (C_L zemin kazancının yarısını kaybeder)
    p_low   >> 1 ise düşüş keskin (varsayılan 5), q_high > 1 ise tepe C1-sürekli

Tepe noktası (h_peak) ve stall eşiği (h_stall) serbestçe ayarlanabilir.
Ön ve arka eksen için ayrı eğri kullanılır (arka tepe genelde daha yüksek: "rake").
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np

LN2 = math.log(2.0)


@dataclass(frozen=True)
class CLCurve:
    """Tek eksen için C_L(h).  Tüm yükseklikler METRE."""
    c_peak: float          # tepe C_L (A_ref ile çarpılmış "ClA" olarak düşünülebilir)
    c_base: float          # zemin etkisi kaybolunca kalan C_L (kanatçıklar vb.)
    h_peak: float          # maksimum downforce ride height'ı [m]
    h_stall: float         # floor stall eşiği [m]  (h_stall < h_peak)
    h_half_high: float     # tepe üstünde zemin kazancının yarıya indiği mesafe [m]
    p_low: float = 5.0     # stall tarafı keskinlik üssü
    q_high: float = 1.4    # yüksek taraf şekil üssü

    def __post_init__(self):
        if not self.h_stall < self.h_peak:
            raise ValueError("h_stall < h_peak olmalı")

    # g(h) in [0, 1] ---------------------------------------------------------
    def shape(self, h):
        h = np.asarray(h, dtype=float)
        d = h - self.h_peak
        w_s = self.h_peak - self.h_stall
        low = np.exp(-LN2 * (np.maximum(-d, 0.0) / w_s) ** self.p_low)
        high = np.exp(-LN2 * (np.maximum(d, 0.0) / self.h_half_high) ** self.q_high)
        return np.where(d < 0.0, low, high)

    def cl(self, h):
        return self.c_base + (self.c_peak - self.c_base) * self.shape(h)

    def cl_scalar(self, h: float) -> float:
        """Hızlı skaler sürüm (ODE sağ tarafı için)."""
        d = h - self.h_peak
        if d < 0.0:
            g = math.exp(-LN2 * ((-d) / (self.h_peak - self.h_stall)) ** self.p_low)
        else:
            g = math.exp(-LN2 * (d / self.h_half_high) ** self.q_high)
        return self.c_base + (self.c_peak - self.c_base) * g

    def dcl_dh(self, h, eps=1e-5):
        return (self.cl(h + eps) - self.cl(h - eps)) / (2 * eps)


@dataclass(frozen=True)
class AeroModel:
    """Ön + arka eksen aerodinamiği.  F = 0.5 rho v^2 * C_L(h)  (aşağı doğru, N)."""
    front: CLCurve
    rear: CLCurve
    rho: float = 1.225
    cd0_a: float = 1.00      # parazit sürükleme CdA [m^2]
    k_wing: float = 0.24     # kanat downforce'u başına indüklenmiş sürükleme (L/D ≈ 4)
    k_floor: float = 0.03    # zemin (ground-effect) downforce'u başına sürükleme (L/D ≈ 30+)

    # --- kuvvetler ----------------------------------------------------------
    def forces(self, hf: float, hr: float, v: float):
        q = 0.5 * self.rho * v * v
        return q * self.front.cl_scalar(hf), q * self.rear.cl_scalar(hr)

    def ideal_force(self, v):
        """Her iki eksen de tam tepe noktasındaymış gibi downforce (üst sınır)."""
        q = 0.5 * self.rho * np.asarray(v) ** 2
        return q * (self.front.c_peak + self.rear.c_peak)

    def total_cla(self, hf, hr):
        return self.front.cl(hf) + self.rear.cl(hr)

    @property
    def cla_peak(self):
        return self.front.c_peak + self.rear.c_peak

    @property
    def cla_wing(self):
        """Ride height'tan bağımsız kalan (kanat) downforce katsayısı."""
        return self.front.c_base + self.rear.c_base

    def drag_cda(self, cla):
        """CdA = cd0 + k_wing*ClA_kanat + k_floor*ClA_zemin.  Zemin etkisi 'verimli'
        downforce'tur; floor stall'da kaybolan downforce sürüklemeyi çok az azaltır."""
        floor = np.maximum(cla - self.cla_wing, 0.0)
        return self.cd0_a + self.k_wing * self.cla_wing + self.k_floor * floor

    def tweak(self, **kw):
        """Örn. aero.tweak(front=replace(aero.front, h_stall=0.02))"""
        return replace(self, **kw)


def default_aero() -> AeroModel:
    """F1-benzeri varsayılanlar: ClA_peak ≈ 3.8 m², aero balans ≈ %41 ön."""
    front = CLCurve(c_peak=1.55, c_base=0.70, h_peak=0.028, h_stall=0.023, h_half_high=0.022)
    rear = CLCurve(c_peak=2.25, c_base=1.05, h_peak=0.046, h_stall=0.040, h_half_high=0.026)
    return AeroModel(front, rear)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    aero = default_aero()
    h = np.linspace(0, 0.09, 600)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(h * 1e3, aero.front.cl(h), label="ön")
    ax.plot(h * 1e3, aero.rear.cl(h), label="arka")
    ax.set_xlabel("ride height [mm]"); ax.set_ylabel("C_L (ClA, m²)"); ax.legend(); ax.grid(alpha=.3)
    fig.savefig("aero_curve.png", dpi=120)
    print("aero_curve.png yazıldı; ClA_peak =", aero.cla_peak)
