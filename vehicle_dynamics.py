"""
vehicle_dynamics.py  --  2-DOF (heave + pitch) F1 aracı, sympy türetimi + solve_ivp
=====================================================================================

Model
-----
Tek rijit gövde (kütle m, pitch eylemsizliği Iy), CG'den ön aksa a, arka aksa b.
Lastikler rijit kabul edilir; gövde ile yol arasında ön/arka yay-sönümleyici vardır.
(Yani 2 serbestlik derecesi: z = heave, θ = pitch;  ride height'lar bunların
 doğrusal dönüşümüdür.)

    y_f = z + a θ ,  y_r = z - b θ       (gövde noktalarının mutlak yüksekliği, θ burun-yukarı +)
    h_i = y_i - w_i                       (ride height = gövde noktası - yol profili)

Lagrange denklemleri y_f, y_r üzerinden sympy ile türetilir:

    T = ½ m ż² + ½ Iy θ̇²
    V = ½ k_f (h_f - h0_f)² + ½ k_r (h_r - h0_r)² + m g z      (h0 = yayın serbest yüksekliği)
    R = ½ c_f ḣ_f² + ½ c_r ḣ_r²                                (Rayleigh sönüm)
    Q_i = -F_aero,i - ΔF_yük-aktarımı,i + u_i                   (genelleştirilmiş kuvvetler)

Çözüm durum vektörü  (ride height'lar durum değişkeni):

    x = [h_f, h_r, ḣ_f, ḣ_r,  I_f, I_r,  ua_f, ua_r]
         ride height ve hızları | kontrolcü integratörleri | aktüatör kuvvetleri (1. derece gecikme)

ḧ_i = ÿ_i - ẅ_i  (ẅ: yol profilinden gelen bozucu).
Boylamsal ivme (fren/gaz) gövdeye pitch momenti olarak girer: ΔF_f = -m a_x h_cg / L = -ΔF_r.
Yanal ivme heave/pitch'e doğrudan girmez (roll modellenmiyor); yalnızca tur zamanı
modelinde (scenario.py) tutuşu belirler.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

from aero_model import AeroModel

G = 9.81


# ---------------------------------------------------------------- parametreler
@dataclass
class VehicleParams:
    m: float = 800.0          # kütle (sürücü+yakıt dahil) [kg]
    Iy: float = 1100.0        # pitch eylemsizliği [kg m²]
    a: float = 1.98           # CG - ön aks [m]
    b: float = 1.62           # CG - arka aks [m]
    h_cg: float = 0.28        # CG yüksekliği [m]
    k_f: float = 850e3        # ön efektif yay rijitliği [N/m]
    k_r: float = 950e3        # arka
    c_f: float = 9.0e3        # ön sönüm [N s/m]
    c_r: float = 10.0e3       # arka
    tau_act: float = 0.015    # aktüatör zaman sabiti [s]
    u_max: float = 15e3       # aktüatör kuvvet sınırı (eksen başına) [N]

    @property
    def L(self):
        return self.a + self.b


@dataclass
class Setup:
    """Statik (v=0) ride height ayarı [m]; yay ön-yükü bundan hesaplanır."""
    h_f: float
    h_r: float
    name: str = ""


# ------------------------------------------------------------ sympy türetimi
_SYM_CACHE = {}


def derive_equations():
    """Euler-Lagrange denklemlerini sembolik türetir.  Sonuç önbelleğe alınır."""
    if "eq" in _SYM_CACHE:
        return _SYM_CACHE["eq"]
    t = sp.symbols("t")
    m, I, a, b, g = sp.symbols("m I_y a b g", positive=True)
    kf, kr, cf, cr = sp.symbols("k_f k_r c_f c_r", positive=True)
    h0f, h0r = sp.symbols("h0_f h0_r")
    Faf, Far, Tf, Tr, uf, ur = sp.symbols("F_af F_ar T_f T_r u_f u_r")

    yf, yr, wf, wr = [sp.Function(n)(t) for n in ("y_f", "y_r", "w_f", "w_r")]
    L = a + b
    z = (b * yf + a * yr) / L
    th = (yf - yr) / L
    hf, hr = yf - wf, yr - wr

    T = sp.Rational(1, 2) * m * z.diff(t) ** 2 + sp.Rational(1, 2) * I * th.diff(t) ** 2
    V = (sp.Rational(1, 2) * kf * (hf - h0f) ** 2 + sp.Rational(1, 2) * kr * (hr - h0r) ** 2
         + m * g * z)
    R = sp.Rational(1, 2) * cf * hf.diff(t) ** 2 + sp.Rational(1, 2) * cr * hr.diff(t) ** 2
    Lag = T - V
    Qf = -Faf - Tf + uf
    Qr = -Far - Tr + ur

    eqs = []
    for q, Q in ((yf, Qf), (yr, Qr)):
        qd = q.diff(t)
        eqs.append(sp.diff(Lag.diff(qd), t) - Lag.diff(q) + R.diff(qd) - Q)

    # türevleri düz sembollere çevir
    ydf, ydr, yddf, yddr = sp.symbols("yd_f yd_r ydd_f ydd_r")
    wfd, wrd = sp.symbols("wd_f wd_r")
    hF, hR, hdF, hdR, wF, wR = sp.symbols("hF hR hdF hdR wF wR")
    rep = {yf.diff(t, 2): yddf, yr.diff(t, 2): yddr, yf.diff(t): ydf, yr.diff(t): ydr,
           wf.diff(t): wfd, wr.diff(t): wrd}
    eqs = [e.subs(rep) for e in eqs]
    # y = h + w  ->  ride height değişkenleriyle ifade et
    rep2 = {yf: hF + wF, yr: hR + wR, wf: wF, wr: wR}
    eqs = [e.subs(rep2) for e in eqs]
    eqs = [e.subs({ydf: hdF + wfd, ydr: hdR + wrd}) for e in eqs]

    sol = sp.solve(eqs, [yddf, yddr], dict=True)[0]
    ydd_f, ydd_r = sp.simplify(sol[yddf]), sp.simplify(sol[yddr])

    # Kütle matrisi (y koordinatlarında sabit)
    A, _ = sp.linear_eq_to_matrix(eqs, [yddf, yddr])
    M = sp.simplify(A)

    out = dict(
        ydd=(ydd_f, ydd_r), M=M, eqs=eqs,
        args=(hF, hR, hdF, hdR, Faf, Far, Tf, Tr, uf, ur),
        params=(m, I, a, b, g, kf, kr, cf, cr, h0f, h0r),
    )
    _SYM_CACHE["eq"] = out
    return out


def print_equations():
    d = derive_equations()
    sp.init_printing(use_unicode=True)
    print("Kütle matrisi M (y_f, y_r koordinatlarında):")
    sp.pprint(d["M"])
    print("\nHareket denklemleri  M·ÿ = Q - ∂V/∂y - ∂R/∂ẏ :")
    for e in d["eqs"]:
        sp.pprint(sp.Eq(e, 0))
    print("\nÇözülmüş ivmeler:")
    for n, e in zip(("ÿ_f", "ÿ_r"), d["ydd"]):
        print(n, "=", e)


# ------------------------------------------------------------------- Araç
class HeavePitchCar:
    def __init__(self, p: VehicleParams, aero: AeroModel, setup: Setup):
        self.p, self.aero, self.setup = p, aero, setup
        L = p.L
        # yay serbest yüksekliği: v=0'da statik ağırlığı setup yüksekliğinde taşısın
        Wf, Wr = p.m * G * p.b / L, p.m * G * p.a / L
        self.h0f = setup.h_f + Wf / p.k_f   # V=½k(h-h0)² : yük altında h < h0
        self.h0r = setup.h_r + Wr / p.k_r

        d = derive_equations()
        subs = dict(zip(d["params"], (p.m, p.Iy, p.a, p.b, G, p.k_f, p.k_r, p.c_f, p.c_r,
                                      self.h0f, self.h0r)))
        ydd = [e.subs(subs) for e in d["ydd"]]
        self._ydd = sp.lambdify(d["args"], ydd, modules="math")
        self.M = np.array(d["M"].subs(subs).tolist(), dtype=float)
        self.K = np.diag([p.k_f, p.k_r])
        self.C = np.diag([p.c_f, p.c_r])

    # --- yardımcılar --------------------------------------------------------
    def load_transfer(self, ax):
        """Boylamsal ivmeden eksen yük aktarımı (aşağı +):  fren (ax<0) -> ön artar."""
        T = -self.p.m * ax * self.p.h_cg / self.p.L
        return T, -T

    def accel_y(self, hf, hr, hdf, hdr, Ff, Fr, Tf, Tr, uf=0.0, ur=0.0):
        return self._ydd(hf, hr, hdf, hdr, Ff, Fr, Tf, Tr, uf, ur)

    def net_force(self, hf, hr, hdf, hdr, v, ax):
        """Aktüatör hariç eksen net kuvveti (yukarı +) = M ÿ|u=0 ."""
        Ff, Fr = self.aero.forces(hf, hr, v)
        Tf, Tr = self.load_transfer(ax)
        ydd = np.array(self.accel_y(hf, hr, hdf, hdr, Ff, Fr, Tf, Tr))
        return self.M @ ydd

    def equilibrium(self, v, ax=0.0, guess=None, u=(0.0, 0.0)):
        """Verilen hız ve ivmede statik ride height dengesi (yakınsayan ilk kök)."""
        g0 = np.array(guess if guess is not None else [self.setup.h_f, self.setup.h_r])

        def f(h):
            return self.net_force(h[0], h[1], 0.0, 0.0, v, ax) + np.array(u)
        sol, info, ier, msg = fsolve(f, g0, full_output=True, xtol=1e-12)
        return sol, ier == 1

    # --- ODE ---------------------------------------------------------------
    def make_rhs(self, scenario, controller):
        p, aero = self.p, self.aero
        fb = self.accel_y
        load_t = self.load_transfer
        umax, tau = p.u_max, p.tau_act
        active = controller.active

        def rhs(t, x):
            hf, hr, hdf, hdr, If, Ir, uaf, uar = x
            sig = scenario.at(t)
            v, ax = sig.v, sig.ax
            Ff, Fr = aero.forces(hf, hr, v)
            Tf, Tr = load_t(ax)
            ydf, ydr = fb(hf, hr, hdf, hdr, Ff, Fr, Tf, Tr, uaf, uar)
            hddf = ydf - sig.wdd_f
            hddr = ydr - sig.wdd_r
            if active:
                rf, rr = controller.reference(v, sig.mode)
                ef, er = rf - hf, rr - hr
                cf_, cr_ = controller.command(ef, er, -hdf, -hdr, If, Ir, uaf, uar)
                # doyma + integratör anti-windup (koşullu integrasyon)
                ucf, ucr = max(-umax, min(umax, cf_)), max(-umax, min(umax, cr_))
                dIf = 0.0 if (cf_ != ucf and cf_ * ef > 0) else ef
                dIr = 0.0 if (cr_ != ucr and cr_ * er > 0) else er
            else:
                ucf = ucr = dIf = dIr = 0.0
            return [hdf, hdr, hddf, hddr, dIf, dIr, (ucf - uaf) / tau, (ucr - uar) / tau]

        return rhs

    # --- simülasyon ----------------------------------------------------------
    def initial_state(self, scenario, controller):
        sig = scenario.at(0.0)
        if controller.active:
            rf, rr = controller.reference(sig.v, sig.mode)
            h = np.array([rf, rr])
            Fn = self.net_force(h[0], h[1], 0.0, 0.0, sig.v, sig.ax)
            ua = -Fn                       # M·ÿ = Fn + u = 0
            I0 = controller.init_integrator(ua)
            return np.array([h[0], h[1], 0, 0, I0[0], I0[1], ua[0], ua[1]])
        h, ok = self.equilibrium(sig.v, sig.ax)
        return np.array([h[0], h[1], 0, 0, 0, 0, 0, 0])

    def simulate(self, scenario, controller, x0=None, dt_out=0.002, rtol=1e-6, atol=1e-8, max_step=0.004):
        x0 = self.initial_state(scenario, controller) if x0 is None else np.asarray(x0, float)
        t_eval = np.arange(0.0, scenario.t_end, dt_out)
        rhs = self.make_rhs(scenario, controller)
        sol = solve_ivp(rhs, (0.0, scenario.t_end), x0, t_eval=t_eval, method="RK45",
                        rtol=rtol, atol=atol, max_step=max_step)
        if not sol.success:
            raise RuntimeError(sol.message)
        return self.postprocess(sol.t, sol.y, scenario, controller)

    def postprocess(self, t, y, scenario, controller):
        hf, hr = y[0], y[1]
        sig = scenario.at_array(t)
        aero = self.aero
        Ff = 0.5 * aero.rho * sig["v"] ** 2 * aero.front.cl(hf)
        Fr = 0.5 * aero.rho * sig["v"] ** 2 * aero.rear.cl(hr)
        F = Ff + Fr
        F_ideal = aero.ideal_force(sig["v"])
        return dict(
            name=controller.name, t=t, s=sig["s"], v=sig["v"], ax=sig["ax"], ay=sig["ay"],
            mode=sig["mode"], hf=hf, hr=hr, hdf=y[2], hdr=y[3],
            Ff=Ff, Fr=Fr, F=F, F_ideal=F_ideal, eta=np.divide(F, F_ideal, out=np.ones_like(F), where=F_ideal > 0),
            cla=aero.total_cla(hf, hr), uf=y[6], ur=y[7],
            href_f=np.full_like(t, np.nan), href_r=np.full_like(t, np.nan),
        )


def linear_modes(car: HeavePitchCar):
    """Sönümsüz doğal frekanslar [Hz] ve mod şekilleri (doğrulama için)."""
    w2, vec = np.linalg.eig(np.linalg.solve(car.M, car.K))
    order = np.argsort(w2)
    return np.sqrt(w2[order].real) / (2 * np.pi), vec[:, order]


if __name__ == "__main__":
    print_equations()
