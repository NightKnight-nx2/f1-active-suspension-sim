"""
controllers.py  --  Pasif / PID / LQR ride-height kontrolcüleri
================================================================

Aktüatör: yayla paralel çalışan eksen-başı kuvvet kaynağı u_i (N), 1. derece
gecikmeli (tau_act) ve ±u_max ile sınırlı (vehicle_dynamics.py içinde).

Kontrol hatası:  e_i = h_ref,i - h_i      (i = ön, arka)
Doğrusal durum:  z = [e (2), ė (2), ∫e (2), u_act (2)]     ->  u_cmd = -K z

Referans politikası (ReferencePolicy):
    h_ref = h_peak (aero-optimal) + güvenlik payı
    pay, düzlükte hızın karesiyle artar (yüksek hızda dinamik salınım stall'a
    sokmasın), viraj modunda (mode→1) küçülür (düşük hız, bozucu az -> tepeye yaslan).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_continuous_are


# --------------------------------------------------------------- referans
@dataclass
class ReferencePolicy:
    h_peak_f: float
    h_peak_r: float
    margin0: float = 0.0005        # taban güvenlik payı [m]
    margin_v: float = 0.0015       # (v/v_ref)² ile artan pay [m]
    v_ref: float = 90.0            # [m/s]
    corner_scale: float = 0.4      # viraj modunda payın çarpanı

    def __call__(self, v: float, mode: float):
        m = (self.margin0 + self.margin_v * (v / self.v_ref) ** 2)
        m *= 1.0 - (1.0 - self.corner_scale) * mode
        return self.h_peak_f + m, self.h_peak_r + m


# --------------------------------------------------------- temel sınıf
class Controller:
    name = "base"
    active = False
    Kmat: np.ndarray | None = None   # 2x8 ;  u_cmd = -Kmat @ z

    def reference(self, v, mode):
        raise NotImplementedError

    def command(self, ef, er, edf, edr, If, Ir, uaf, uar):
        z = np.array([ef, er, edf, edr, If, Ir, uaf, uar])
        u = -self.Kmat @ z
        return float(u[0]), float(u[1])

    def init_integrator(self, ua):
        """e=0, ė=0 ve u_cmd = ua olacak şekilde integratör başlangıcı."""
        Ki, Ku = self.Kmat[:, 4:6], self.Kmat[:, 6:8]
        return np.linalg.solve(-Ki, (np.eye(2) + Ku) @ np.asarray(ua))


class PassiveController(Controller):
    name = "Pasif"
    active = False

    def __init__(self, name="Pasif"):
        self.name = name

    def reference(self, v, mode):
        return None, None

    def command(self, *a):
        return 0.0, 0.0


class PIDController(Controller):
    """Eksen başına bağımsız PID:  u = kp e + ki ∫e + kd ė   (ė = -ḣ)."""
    active = True

    def __init__(self, policy: ReferencePolicy, kp=2.0e6, ki=3.0e7, kd=5.0e4, name="Aktif (PID)"):
        self.policy, self.name = policy, name
        self.kp, self.ki, self.kd = kp, ki, kd
        K = np.zeros((2, 8))
        for i in range(2):
            K[i, i], K[i, 2 + i], K[i, 4 + i] = -kp, -kd, -ki
        self.Kmat = K

    def reference(self, v, mode):
        return self.policy(v, mode)

    def command(self, ef, er, edf, edr, If, Ir, uaf, uar):
        kp, ki, kd = self.kp, self.ki, self.kd
        return (kp * ef + ki * If + kd * edf, kp * er + ki * Ir + kd * edr)

    def init_integrator(self, ua):
        return np.asarray(ua) / self.ki


class LQRController(Controller):
    """İntegral etkili LQR (ride-height hatası, hızı, ∫e ve aktüatör kuvveti durumda)."""
    active = True

    def __init__(self, car, policy: ReferencePolicy, q_e=4e6, q_ed=300.0, q_i=3e8, q_u=0.0,
                 r=1.0 / 4e3 ** 2, name="Aktif (LQR)"):
        self.policy, self.name = policy, name
        A, B = linear_error_model(car)
        Q = np.diag([q_e] * 2 + [q_ed] * 2 + [q_i] * 2 + [q_u] * 2)
        R = np.eye(2) * r
        P = solve_continuous_are(A, B, Q, R)
        self.Kmat = np.linalg.solve(R, B.T @ P)

    def reference(self, v, mode):
        return self.policy(v, mode)


# ------------------------------------------------- doğrusal model / analiz
def linear_error_model(car):
    """z = [e, ė, ∫e, u_act] için  ż = A z + B u_cmd  (aero rijitliği ihmal)."""
    Mi = np.linalg.inv(car.M)
    tau = car.p.tau_act
    I2, Z = np.eye(2), np.zeros((2, 2))
    A = np.block([
        [Z, I2, Z, Z],
        [-Mi @ car.K, -Mi @ car.C, Z, -Mi],
        [I2, Z, Z, Z],
        [Z, Z, Z, -I2 / tau],
    ])
    B = np.vstack([Z, Z, Z, I2 / tau])
    return A, B


def closed_loop_poles(car, ctrl):
    A, B = linear_error_model(car)
    return np.linalg.eigvals(A - B @ ctrl.Kmat)
