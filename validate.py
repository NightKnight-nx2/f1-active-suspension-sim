"""
validate.py  --  Pasif araç modelinin doğrulaması (kontrolcü yok)
==================================================================
1. sympy denklemi  vs  bağımsız elle yazılmış ivme formülü (rastgele durumlar)
2. v=0 serbest titreşim  vs  matris üstel çözümü expm(A t)  (tam doğrusal -> hata ~ çözücü toleransı)
3. Sönümsüz araçta enerji korunumu
4. Sabit hızda dengeye oturma  vs  bağımsız skaler kuvvet dengesi (brentq)
5. Fren yük aktarımı (statik pitch kayması)  vs  analitik
6. Tam tur (Pasif-Alçak): fiziksel makullük özeti
"""
from __future__ import annotations
import dataclasses
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from aero_model import default_aero
from vehicle_dynamics import VehicleParams, Setup, HeavePitchCar, linear_modes, G
from controllers import PassiveController
from scenario import ConstantScenario, Track, default_track, default_road, DriveParams, \
    solve_speed_profile, Scenario, lap_time

OK = "\033[92mGEÇTİ\033[0m"
BAD = "\033[91mKALDI\033[0m"


def check(name, cond, detail=""):
    print(f"  [{OK if cond else BAD}] {name}  {detail}")
    return bool(cond)


def run_validation(setup_low=Setup(0.030, 0.047, "Pasif-Alçak"), make_plot=True, outdir="outputs"):
    p, aero = VehicleParams(), default_aero()
    car = HeavePitchCar(p, aero, setup_low)
    ok = True
    rng = np.random.default_rng(1)
    pas = PassiveController()
    L = p.L

    print("\n[1] sympy ivmeleri vs bağımsız formül")
    err = 0.0
    for _ in range(50):
        hf, hr = rng.uniform(0.02, 0.06, 2); hdf, hdr = rng.uniform(-0.3, 0.3, 2)
        Ff, Fr, Tf, Tr, uf, ur = rng.uniform(-5e3, 8e3, 6)
        y_sym = np.array(car.accel_y(hf, hr, hdf, hdr, Ff, Fr, Tf, Tr, uf, ur))
        # elle: M ÿ = [uf - Ff - Tf - kf(hf-h0f) - cf hdf - m g b/L ,  ...a/L]
        rhs = np.array([uf - Ff - Tf - p.k_f * (hf - car.h0f) - p.c_f * hdf - p.m * G * p.b / L,
                        ur - Fr - Tr - p.k_r * (hr - car.h0r) - p.c_r * hdr - p.m * G * p.a / L])
        Mman = np.array([[p.m * p.b ** 2 + p.Iy, p.m * p.a * p.b - p.Iy],
                         [p.m * p.a * p.b - p.Iy, p.m * p.a ** 2 + p.Iy]]) / L ** 2
        err = max(err, np.abs(y_sym - np.linalg.solve(Mman, rhs)).max())
    ok &= check("ÿ eşleşmesi", err < 1e-8, f"(maks fark {err:.2e} m/s²)")

    print("\n[2] v=0 serbest titreşim vs expm(A t)")
    Mi = np.linalg.inv(car.M)
    A = np.block([[np.zeros((2, 2)), np.eye(2)], [-Mi @ car.K, -Mi @ car.C]])
    x0 = np.array([car.setup.h_f + 3e-3, car.setup.h_r - 1e-3, 0, 0])
    sc0 = ConstantScenario(v=0.0, t_end=1.5)
    res = car.simulate(sc0, pas, x0=np.r_[x0, np.zeros(4)], rtol=1e-9, atol=1e-12)
    ref = np.array([expm(A * t) @ (x0 - np.r_[car.setup.h_f, car.setup.h_r, 0, 0]) for t in res["t"]])
    ref[:, 0] += car.setup.h_f; ref[:, 1] += car.setup.h_r
    e2 = max(np.abs(res["hf"] - ref[:, 0]).max(), np.abs(res["hr"] - ref[:, 1]).max())
    ok &= check("expm ile uyum", e2 < 1e-7, f"(maks sapma {e2*1e3:.2e} mm)")
    f_n, _ = linear_modes(car)
    ev = np.linalg.eigvals(A); ev = ev[ev.imag > 0]
    zeta = -ev.real / np.abs(ev)
    print(f"      doğal frekanslar: {f_n[0]:.2f} Hz (heave-ağırlıklı), {f_n[1]:.2f} Hz (pitch-ağırlıklı)")
    print(f"      sönüm oranları  : ζ = {np.sort(zeta)[0]:.3f}, {np.sort(zeta)[1]:.3f}")

    print("\n[3] enerji korunumu (c=0)")
    p0 = dataclasses.replace(p, c_f=0.0, c_r=0.0)
    car0 = HeavePitchCar(p0, aero, setup_low)
    r0 = car0.simulate(sc0, pas, x0=np.r_[x0, np.zeros(4)], rtol=1e-10, atol=1e-13)
    def energy(r):
        yd = np.vstack([r["hdf"], r["hdr"]])
        T = 0.5 * np.einsum("it,ij,jt->t", yd, car0.M, yd)
        V = 0.5 * p.k_f * (r["hf"] - car0.h0f) ** 2 + 0.5 * p.k_r * (r["hr"] - car0.h0r) ** 2 \
            + p.m * G * (p.b * r["hf"] + p.a * r["hr"]) / L
        return T + V
    E = energy(r0); drift = (E.max() - E.min()) / (0.5 * p.k_f * (3e-3) ** 2)
    ok &= check("enerji sapması", drift < 1e-6, f"(göreli {drift:.1e})")

    print("\n[4] sabit hızda denge vs bağımsız kuvvet dengesi (brentq)")
    for v in (30.0, 60.0, 85.0):
        sol, conv = car.equilibrium(v)
        Wf, Wr = p.m * G * p.b / L, p.m * G * p.a / L
        q = 0.5 * aero.rho * v * v
        hf_b = brentq(lambda h: p.k_f * (car.h0f - h) - Wf - q * aero.front.cl_scalar(h), 0.005, 0.09)
        hr_b = brentq(lambda h: p.k_r * (car.h0r - h) - Wr - q * aero.rear.cl_scalar(h), 0.005, 0.12)
        # zaman integrali bu dengeye oturuyor mu? (başlangıç: statik ayar, 4 s)
        sc = ConstantScenario(v=v, t_end=4.0)
        r = car.simulate(sc, pas, x0=np.r_[setup_low.h_f, setup_low.h_r, np.zeros(6)])
        e = max(abs(r["hf"][-1] - hf_b), abs(r["hr"][-1] - hr_b))
        branch = "stall bölgesi" if r["eta"][-1] < 0.85 else "tepe/yüksek taraf"
        ok &= check(f"v={v*3.6:5.0f} km/h", abs(sol[0] - hf_b) < 1e-7 and e < 1e-4,
                    f"h_f={hf_b*1e3:5.1f} mm h_r={hr_b*1e3:5.1f} mm  η={r['eta'][-1]:.3f} ({branch}; sim-denge farkı {e*1e3:.1e} mm)")

    print("\n[5] fren yük aktarımı")
    ax = -20.0
    sc = ConstantScenario(v=0.0, ax=ax, t_end=2.0)
    r = car.simulate(sc, pas, x0=np.r_[setup_low.h_f, setup_low.h_r, np.zeros(6)])
    T = -p.m * ax * p.h_cg / L
    dh_f_an, dh_r_an = -T / p.k_f, +T / p.k_r
    e5 = max(abs((r["hf"][-1] - setup_low.h_f) - dh_f_an), abs((r["hr"][-1] - setup_low.h_r) - dh_r_an))
    ok &= check("ön çöker / arka kalkar", e5 < 1e-6 and dh_f_an < 0 < dh_r_an,
                f"(Δh_f={dh_f_an*1e3:+.2f} mm, Δh_r={dh_r_an*1e3:+.2f} mm; hata {e5:.1e} m)")

    print("\n[6] Pasif-Alçak tam tur (yol bozukluklu)")
    tr = Track(default_track()); v = solve_speed_profile(tr, aero, DriveParams())
    sc = Scenario(tr, v, p.L, default_road(tr))
    res = car.simulate(sc, PassiveController("Pasif-Alçak"))
    fast = res["v"] > 300 / 3.6
    slow = res["v"] < 150 / 3.6
    print(f"      tur süresi (ideal profil) {sc.t_end:.2f} s,  v_maks {res['v'].max()*3.6:.0f} km/h")
    print(f"      ride height ön [mm]: min {res['hf'].min()*1e3:.1f} / maks {res['hf'].max()*1e3:.1f};"
          f" arka: min {res['hr'].min()*1e3:.1f} / maks {res['hr'].max()*1e3:.1f}")
    print(f"      η (F/F_ideal): 300+ km/h ort. {res['eta'][fast].mean():.3f},  <150 km/h ort. {res['eta'][slow].mean():.3f}")
    ok &= check("ride height fiziksel aralıkta (0 < h < 90 mm)", 0 < res["hf"].min() and res["hf"].max() < 0.09)
    ok &= check("düşük hızda tepe, yüksek hızda stall", res["eta"][slow].mean() > 0.9 > res["eta"][fast].mean())

    if make_plot:
        import os
        import visualize as viz
        os.makedirs(outdir, exist_ok=True)
        viz.plot_single(res, aero, f"{outdir}/dogrulama_pasif_alcak.png")
        print(f"      grafik: {outdir}/dogrulama_pasif_alcak.png")
    print("\nGENEL SONUÇ:", OK if ok else BAD)
    return ok, res


if __name__ == "__main__":
    run_validation()
