"""
main.py  --  F1 pasif / aktif süspansiyon karşılaştırma aracı
==============================================================
    python main.py curve                 # aero C_L(h) eğrisi
    python main.py validate              # tek senaryo (Pasif-Alçak) + doğrulama testleri
    python main.py compare [--controller lqr|pid] [--no-road] [--anim]
    python main.py animate [--controller lqr|pid] [--speedup 2]
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

from aero_model import default_aero
from vehicle_dynamics import VehicleParams, Setup, HeavePitchCar
from controllers import PassiveController, PIDController, LQRController, ReferencePolicy
from scenario import (Track, default_track, default_road, DriveParams, Scenario,
                      solve_speed_profile, lap_time)
import visualize as viz

OUT = "outputs"
SETUPS = {                                   # statik ride height ayarları [m]
    "Pasif-Alçak": Setup(0.030, 0.047, "Pasif-Alçak"),
    "Pasif-Yüksek": Setup(0.048, 0.065, "Pasif-Yüksek"),
    "Aktif": Setup(0.036, 0.054, "Aktif"),   # aktüatör kuvveti ±~9 kN aralığında
}
V_FAST = 300 / 3.6


def build_world(road_gain=1.0):
    p, aero, dp = VehicleParams(), default_aero(), DriveParams()
    tr = Track(default_track())
    v_ref = solve_speed_profile(tr, aero, dp)            # İDEAL aero ile ortak referans profil
    road = default_road(tr, road_gain) if road_gain > 0 else None
    sc = Scenario(tr, v_ref, p.L, road)
    return p, aero, dp, tr, v_ref, sc


def run_scenarios(controller_kind="lqr", road_gain=1.0, verbose=True):
    p, aero, dp, tr, v_ref, sc = build_world(road_gain)
    pol = ReferencePolicy(aero.front.h_peak, aero.rear.h_peak)
    results = []
    for name in ("Pasif-Alçak", "Pasif-Yüksek", "Aktif"):
        car = HeavePitchCar(p, aero, SETUPS[name])
        if name == "Aktif":
            ctrl = LQRController(car, pol, name="Aktif") if controller_kind == "lqr" \
                else PIDController(pol, name="Aktif")
        else:
            ctrl = PassiveController(name)
        if verbose:
            print(f"  simülasyon: {name} ...", flush=True)
        results.append(car.simulate(sc, ctrl))
    return p, aero, dp, tr, v_ref, sc, results


def eta_of_s(res, tr, smooth_m=30.0):
    """Gerçekleşen downforce verimi η'nın pist konumuna göre (ortalama) haritası."""
    s = np.mod(res["s"], tr.S)
    bins = np.floor(s / tr.ds).astype(int) % tr.N
    num = np.bincount(bins, weights=res["eta"], minlength=tr.N)
    cnt = np.bincount(bins, minlength=tr.N)
    eta = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
    idx = np.arange(tr.N); good = ~np.isnan(eta)
    eta = np.interp(idx, idx[good], eta[good], period=tr.N)
    w = max(int(smooth_m / tr.ds), 1)
    ker = np.ones(w) / w
    ee = np.concatenate([eta[-w:], eta, eta[:w]])
    return np.clip(np.convolve(ee, ker, mode="same")[w:-w], 0.3, 1.0)


def compute_metrics(results, tr, aero, dp, v_ref):
    ideal_lap = lap_time(tr, v_ref)
    out = {}
    for r in results:
        s, F, Fi = r["s"], r["F"], r["F_ideal"]
        banked = np.trapezoid(F, s)            # ∫F ds  [J = N·m]
        banked_ideal = np.trapezoid(Fi, s)
        fast = r["v"] >= V_FAST
        loss_pct = 100 * (1 - F[fast].sum() / Fi[fast].sum()) if fast.any() else float("nan")
        eta_s = eta_of_s(r, tr)
        v_s = solve_speed_profile(tr, aero, dp, eta=eta_s)
        out[r["name"]] = dict(
            banked_MJ=banked / 1e6, banked_pct=100 * banked / banked_ideal,
            lap_time=lap_time(tr, v_s), lap_time_ideal=ideal_lap,
            loss300_pct=loss_pct, t_fast=float(fast.mean() * (r["t"][-1])),
            eta_mean=float(r["eta"].mean()),
            hf_min_mm=float(r["hf"].min() * 1e3), hf_max_mm=float(r["hf"].max() * 1e3),
            u_max_kN=float(max(np.abs(r["uf"]).max(), np.abs(r["ur"]).max()) / 1e3),
            v_top_kmh=float(v_s.max() * 3.6),
        )
    return out


def print_table(m):
    names = list(m)
    best = min(m[n]["lap_time"] for n in names)
    print("\n" + "=" * 86)
    print(f"{'':26s}" + "".join(f"{n:>20s}" for n in names))
    print("-" * 86)
    rows = [
        ("Banked downforce [MJ]", lambda d: f"{d['banked_MJ']:.2f}"),
        ("  ideale oranı [%]", lambda d: f"{d['banked_pct']:.1f}"),
        ("Tur zamanı [s]", lambda d: f"{d['lap_time']:.3f}"),
        ("  en hızlıya fark [s]", lambda d: f"+{d['lap_time']-best:.3f}"),
        ("300+ km/h downforce kaybı [%]", lambda d: f"{d['loss300_pct']:.1f}"),
        ("300+ km/h'de geçen süre [s]", lambda d: f"{d['t_fast']:.1f}"),
        ("Ön ride height aralığı [mm]", lambda d: f"{d['hf_min_mm']:.1f}-{d['hf_max_mm']:.1f}"),
        ("Aktüatör tepe kuvveti [kN]", lambda d: f"{d['u_max_kN']:.1f}"),
    ]
    for lab, f in rows:
        print(f"{lab:26s}" + "".join(f"{f(m[n]):>20s}" for n in names))
    print("=" * 86)
    print(f"Referans (ideal aero) tur zamanı: {m[names[0]]['lap_time_ideal']:.3f} s")


def cmd_compare(args):
    os.makedirs(OUT, exist_ok=True)
    print(f"Üç senaryo çalıştırılıyor (kontrolcü: {args.controller.upper()}, yol bozukluğu: {'kapalı' if args.no_road else 'açık'})")
    p, aero, dp, tr, v_ref, sc, results = run_scenarios(args.controller, 0.0 if args.no_road else 1.0)
    metrics = compute_metrics(results, tr, aero, dp, v_ref)
    print_table(metrics)
    viz.plot_aero_curve(aero, f"{OUT}/aero_egri.png")
    viz.plot_comparison(results, aero, f"{OUT}/karsilastirma_zaman_serileri.png")
    note = ("Banked downforce: ortak referans hız profilinde ∫F ds.  Tur zamanı: her senaryonun gerçekleşen downforce verimi η(s) "
            "ile nokta-kütle çözücüde yeniden hesaplanır.")
    viz.plot_summary(metrics, f"{OUT}/karsilastirma_ozet.png", note)
    with open(f"{OUT}/metrikler.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    np.savez_compressed(f"{OUT}/sonuclar.npz", **{f"{r['name']}__{k}": v for r in results for k, v in r.items()
                                                  if isinstance(v, np.ndarray)})
    print(f"\nÇıktılar -> {OUT}/ (aero_egri.png, karsilastirma_zaman_serileri.png, karsilastirma_ozet.png, metrikler.json, sonuclar.npz)")
    if args.anim:
        path = viz.animate(results, aero, p, f"{OUT}/animasyon.mp4", speedup=args.speedup)
        print("animasyon:", path)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("cmd", choices=["curve", "validate", "compare", "animate"])
    ap.add_argument("--controller", choices=["lqr", "pid"], default="lqr")
    ap.add_argument("--no-road", action="store_true", help="yol bozukluklarını kapat")
    ap.add_argument("--anim", action="store_true", help="compare sonrası animasyon da üret")
    ap.add_argument("--speedup", type=float, default=2.5)
    args = ap.parse_args()
    if args.cmd == "curve":
        os.makedirs(OUT, exist_ok=True)
        viz.plot_aero_curve(default_aero(), f"{OUT}/aero_egri.png"); print(f"{OUT}/aero_egri.png")
    elif args.cmd == "validate":
        import validate
        validate.run_validation(SETUPS["Pasif-Alçak"], outdir=OUT)
    elif args.cmd == "compare":
        cmd_compare(args)
    elif args.cmd == "animate":
        os.makedirs(OUT, exist_ok=True)
        p, aero, dp, tr, v_ref, sc, results = run_scenarios(args.controller)
        print("animasyon:", viz.animate(results, aero, p, f"{OUT}/animasyon.mp4", speedup=args.speedup))


if __name__ == "__main__":
    main()
