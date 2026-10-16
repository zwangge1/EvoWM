"""三段腿落震的 Exudyn 多体模型（工程验证用，真实鸟类尺度）。

拓扑沿用 FFY 的高保真模型（FFY 仓库 src/stage6_surrogate/hf_exudyn.py 与 src/stage10_v2/physics_v2.py，只读参考）：
地面 ←机身面内平动← 机身 ←髋（转动副 + 扭簧阻尼）← 股骨 ←膝← 胫跗骨 ←踝← 跗跖骨 → 足端球 ↔ 地面罚接触。
仿真一条腿，m 为这条腿分担的载荷（体重的一半）。

与 FFY v2.5 一致的部分：机身在 x-z 平面内自由平动、转动锁死（水平力由足端摩擦承担）；髋与踝、膝同式阻尼 c = τ·k；
接触阻尼 cc = 2ζ√(kc·m)，ζ 由地面刚度导出；摩擦系数由地面刚度导出；初始离地间隙 = 0.3 r_foot；
腿行程 = 触地后髋足相对压缩的最大值（FFY D-8）；塌陷与深陷判据；落震指标（FFY 的 _metrics）。

为真实鸟类尺度改动的部分（FFY 的常数是 kg 级无人机的）：
- 地面刚度、关节参数、速度都按 FFY 参考档（12 kg，参考腿长 0.457 m）的无量纲值换算到鸟的尺度：
  kc* = kc·L/(m g)，κ = k/(m g L)，τ* = τ/√(L/g)；
- 足端半径、杆件线密度与截面、机身等效半径都由原腿 O 定一次，所有腿相同（不随被比较的腿变）；
- 关节扭簧刚度与阻尼用绝对值，所有腿相同，即"阻尼材料相同"。
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import asdict, dataclass

import numpy as np

G = 9.81

# FFY 参考档：m_ref = 12 kg（bioprior.M_REF_KG），L_ref = L1_med(12 kg)·(1 + r2_med + r3_med) = 118.7 mm × 3.85
FFY_M, FFY_L = 12.0, 0.4570
# FFY 的三种地面（evalcfg.CONDS 的 k_c），换成无量纲刚度；ζ_c 用 FFY 的 zeta_of_kc 在原始 k_c 上算
_FFY_KC = {"concrete": 1.0e6, "turf": 1.0e5, "wetsand": 5.0e4}


def zeta_of_kc(kc: float) -> float:
    """FFY factory_v2.zeta_of_kc：越软越耗散。"""
    return float(np.clip(10 ** (1.771 - 0.4875 * np.log10(kc)), 0.05, 0.35))


def mu_of_kc(kc: float) -> float:
    """FFY physics_v2.mu_of_kc：足端摩擦系数随地面刚度变化（FFY 的工程估计，不是实测）。"""
    return float(np.clip(2.916 * float(kc) ** (-0.1461), 0.30, 0.65))


TERRAIN = {k: dict(kc_star=kc * FFY_L / (FFY_M * G), zeta_c=zeta_of_kc(kc), mu=mu_of_kc(kc))
           for k, kc in _FFY_KC.items()}

# FFY v2.5 的参数盒（bioprior：KAP_RANGE_V21、ZETA_RANGE_V21、THA_RANGE、THK_RANGE、Q1_RANGE_V25），线性均匀；
# τ 换成无量纲 τ* = τ / √(L_ref/g)，L_ref 取 FFY 参考腿长
_TAU_S = np.sqrt(FFY_L / G)
BOX = dict(kappa_a=(0.75, 8.0), kappa_k=(1.5, 8.0), kappa_h=(3.0, 32.0),
           tau_star=(0.005 / _TAU_S, 0.10 / _TAU_S),
           theta_ankle=(113.0, 160.0), theta_knee=(118.0, 157.0), q1_0=(27.6, 54.8))

COLLAPSE = 0.6   # 腿行程超过腿长的 60% 记为塌陷（FFY）


@dataclass(frozen=True)
class Leg:
    """三段长度，单位 m：tmt 跗跖骨，tbt 胫跗骨，fem 股骨。"""
    tmt: float
    tbt: float
    fem: float

    @property
    def total(self) -> float:
        return self.tmt + self.tbt + self.fem

    def scaled(self, total: float) -> "Leg":
        """比例不变，总长缩放到 total。"""
        s = total / self.total
        return Leg(self.tmt * s, self.tbt * s, self.fem * s)


@dataclass(frozen=True)
class Params:
    """所有腿共用的一组参数：关节扭簧（N·m/rad）、阻尼（N·m·s/rad）与触地姿态（度）。"""
    k_ankle: float
    k_knee: float
    k_hip: float
    c_ankle: float
    c_knee: float
    c_hip: float
    theta_ankle: float
    theta_knee: float
    q1_0: float


@dataclass(frozen=True)
class Scenario:
    """一个落震工况。由 bird_scenario 从原腿 O 构造，比较中的所有腿共用。"""
    m: float              # 一条腿分担的载荷（kg）
    v0: float             # 竖直触地速度（m/s）
    terrain: str
    l_ref: float          # 原腿 O 的总长（m），所有无量纲量的长度尺度
    r_foot: float         # 足端半径（m）
    lin_density: float    # 杆件线密度（kg/m）
    rod_width: float      # 杆件截面边长（m），只影响转动惯量
    T: float              # 仿真时长（s）
    h: float              # 步长（s）

    @property
    def kc(self) -> float:
        return TERRAIN[self.terrain]["kc_star"] * self.m * G / self.l_ref


def bird_scenario(body_mass_kg: float, leg_o: Leg, v0: float, terrain: str = "turf",
                  leg_mass_frac: float = 0.10, h: float | None = None) -> Scenario:
    """真实鸟类尺度的工况：载荷 = 体重的一半；足端半径 = 0.2 × O 的跗跖骨（FFY v2.2 的口径，只取一次）；
    一条腿的质量 = leg_mass_frac × 载荷（按段长分配）；时长与步长按 √(L/g) 从 FFY 的 1.0 s、2e-4 s 缩放。"""
    m = 0.5 * body_mass_kg
    L = leg_o.total
    ts = np.sqrt(L / G) / _TAU_S
    return Scenario(m=m, v0=v0, terrain=terrain, l_ref=L, r_foot=0.2 * leg_o.tmt,
                    lin_density=leg_mass_frac * m / L, rod_width=0.06 * L,
                    T=round(0.6 * ts, 4), h=h if h is not None else 2e-4 * ts)


def froude_v0(v0_ffy: float, l_ref: float) -> float:
    """FFY 参考档的触地速度按 Froude 数换到长度 l_ref：v = v_ffy·√(L/L_ffy)。"""
    return float(v0_ffy * np.sqrt(l_ref / FFY_L))


def param_box(n: int, seed: int, sc: Scenario, box: dict | None = None) -> list[Params]:
    """在 FFY v2.5 参数盒里拉丁超立方取 n 组参数，所有腿共用同一批（配对比较）。"""
    box = box or BOX
    rng = np.random.default_rng(seed)
    keys = list(box)
    u = np.empty((n, len(keys)))
    for j in range(len(keys)):
        u[:, j] = rng.permutation((np.arange(n) + rng.random(n)) / n)
    unit = sc.m * G * sc.l_ref
    tau_scale = np.sqrt(sc.l_ref / G)
    out = []
    for row in u:
        v = {k: box[k][0] + (box[k][1] - box[k][0]) * row[i] for i, k in enumerate(keys)}
        ka, kk, kh = v["kappa_a"] * unit, v["kappa_k"] * unit, v["kappa_h"] * unit
        tau = v["tau_star"] * tau_scale
        out.append(Params(ka, kk, kh, tau * ka, tau * kk, tau * kh,
                          v["theta_ankle"], v["theta_knee"], v["q1_0"]))
    return out


NAN = dict(peak_a=np.nan, a_res=np.nan, stroke=np.nan, leg_stroke=np.nan, sink=np.nan, eta=np.nan, cfe=np.nan,
           peak_jerk=np.nan, E_abs=np.nan, F_peak=np.nan, rebound=np.nan, n_bounce=np.nan,
           t_settle=np.nan, slip=np.nan, x_drift=np.nan, peak_grf_bw=np.nan, e_gain=np.nan)


@contextlib.contextmanager
def _silence():
    """屏蔽 Exudyn C++ 内核直接写到文件描述符的横幅（FFY 同款）；EXU_VERBOSE=1 时不屏蔽。"""
    if os.environ.get("EXU_VERBOSE"):
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = (os.dup(1), os.dup(2))
    try:
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(devnull)
        os.close(saved[0])
        os.close(saved[1])


def _rot_y(a: float) -> np.ndarray:
    """局部 x 轴指向 x-z 平面内与水平夹角 a 的方向（z 向上）。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


def _metrics(t, z, az, m, v0):
    """FFY 的落震指标（hf_exudyn._metrics），口径不变。"""
    h = t[1] - t[0]
    F = m * (az + G)
    thr = 0.02 * m * G
    idx = np.where(F > thr)[0]
    if len(idx) == 0:
        return {}, None, None
    i0 = int(idx[0])
    imin = int(np.argmin(z))
    if imin <= i0:
        imin = min(i0 + 1, len(z) - 1)
    zc, Fc = z[i0:imin + 1], F[i0:imin + 1]
    s_max = float(z[i0] - z[imin])
    Fmax = float(np.max(Fc))
    E = float(np.trapezoid(Fc, x=-zc))
    eta = E / (Fmax * s_max) if Fmax > 0 and s_max > 1e-9 else np.nan
    cfe = float(np.mean(Fc) / Fmax) if Fmax > 0 else np.nan
    w = max(3, int(0.005 / h))
    az_s = np.convolve(az, np.ones(w) / w, mode="same")
    jerk = float(np.max(np.abs(np.diff(az_s) / h)))
    reb = float(max(0.0, np.max(z[imin:]) - z[i0]))
    off = F[i0:] < thr
    wmin = max(1, int(0.005 / h))
    n_bounce, run = 0, 0
    for o in off:
        run = run + 1 if o else 0
        if run == wmin:
            n_bounce += 1
    v = np.gradient(z, h)
    hot = np.where(np.abs(v) > 0.05 * abs(v0))[0]
    t_settle = float(t[hot[-1]] - t[i0]) if len(hot) else 0.0
    out = dict(eta=eta, cfe=cfe, peak_jerk=jerk, E_abs=E, F_peak=Fmax, rebound=reb,
               n_bounce=n_bounce, t_settle=t_settle)
    return out, i0, imin


def drop(leg: Leg, p: Params, sc: Scenario, trace: bool = False) -> dict:
    """一次落震；返回指标字典（失败时为 NaN 并带 fail 字段）。trace=True 时附时程。"""
    import exudyn as exu
    from exudyn.utilities import InertiaCuboid, InertiaSphere, SensorBody

    l1, l2, l3 = leg.tmt, leg.tbt, leg.fem
    m, v0 = sc.m, sc.v0
    ter = TERRAIN[sc.terrain]
    kc = sc.kc
    cc = 2.0 * ter["zeta_c"] * np.sqrt(kc * m)
    r_foot = sc.r_foot
    a1 = np.radians(p.q1_0)
    a2 = a1 + (np.pi - np.radians(p.theta_ankle))
    a3 = a2 - (np.pi - np.radians(p.theta_knee))
    d = lambda a: np.array([np.cos(a), 0.0, np.sin(a)])  # noqa: E731
    F = np.array([0.0, 0.0, 1.3 * r_foot])          # 初始离地间隙 0.3 r_foot（FFY v2）
    A = F + l1 * d(a1)
    K = A + l2 * d(a2)
    H = K + l3 * d(a3)
    w = sc.rod_width

    SC = exu.SystemContainer()
    mbs = SC.AddSystem()
    ground = mbs.CreateGround(referencePosition=[0, 0, 0])

    def rod(name, P0, P1, ang, L):
        mass = sc.lin_density * L
        inertia = InertiaCuboid(density=mass / (L * w * w), sideLengths=[L, w, w])
        return mbs.CreateRigidBody(name=name, referencePosition=list(0.5 * (P0 + P1)),
                                   referenceRotationMatrix=_rot_y(ang), initialVelocity=[0, 0, -v0],
                                   inertia=inertia, gravity=[0, 0, -G])

    tarso = rod("tarso", F, A, a1, l1)
    tibio = rod("tibio", A, K, a2, l2)
    femur = rod("femur", K, H, a3, l3)
    body = mbs.CreateRigidBody(name="payload", referencePosition=list(H), initialVelocity=[0, 0, -v0],
                               inertia=InertiaSphere(mass=m, radius=0.3 * sc.l_ref), gravity=[0, 0, -G])
    # 机身面内自由平动（放开 x、z，锁死 y 与全部转动），FFY v2.5 planar=True
    mbs.CreateGenericJoint(bodyNumbers=[ground, body], position=list(H), constrainedAxes=[0, 1, 0, 1, 1, 1])
    for b0, b1, P, k, c in ((body, femur, H, p.k_hip, p.c_hip),
                            (femur, tibio, K, p.k_knee, p.c_knee),
                            (tibio, tarso, A, p.k_ankle, p.c_ankle)):
        mbs.CreateRevoluteJoint(bodyNumbers=[b0, b1], position=list(P), axis=[0, 1, 0])
        mbs.CreateTorsionalSpringDamper(bodyNumbers=[b0, b1], position=list(P), axis=[0, 1, 0],
                                        stiffness=k, damping=c)
    big = 50.0 * sc.l_ref
    quad = [[-big, -big, 0.0], [big, -big, 0.0], [big, big, 0.0], [-big, big, 0.0]]
    mbs.CreateSphereQuadContact(bodyNumbers=[tarso, ground], localPosition0=[-0.5 * l1, 0.0, 0.0],
                                radiusSphere=r_foot, quadPoints=quad, contactStiffness=kc,
                                contactDamping=cc, dynamicFriction=ter["mu"],
                                frictionProportionalZone=1e-3 * np.sqrt(sc.l_ref / FFY_L))
    s_acc = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True,
                                     outputVariableType=exu.OutputVariableType.Acceleration))
    s_pos = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True,
                                     outputVariableType=exu.OutputVariableType.Position))
    s_foot = mbs.AddSensor(SensorBody(bodyNumber=tarso, localPosition=[-0.5 * l1, 0.0, 0.0], storeInternal=True,
                                      outputVariableType=exu.OutputVariableType.Position))
    s_rot = mbs.AddSensor(SensorBody(bodyNumber=tarso, storeInternal=True,
                                     outputVariableType=exu.OutputVariableType.Rotation))
    mbs.Assemble()
    ss = exu.SimulationSettings()
    ss.timeIntegration.endTime = sc.T
    ss.timeIntegration.numberOfSteps = int(round(sc.T / sc.h))
    ss.timeIntegration.generalizedAlpha.spectralRadius = 0.7
    # 触地瞬间接触单元会建议极小的步长，自适应步长一路降到下限后求解失败（Exudyn 1.13.6 实测，
    # FFY 的设置下同样如此）；关掉这个建议后，结果与步长无关（见 scripts/exploratory/drop_smoke.py）。
    ss.timeIntegration.discontinuous.useRecommendedStepSize = False
    ss.timeIntegration.verboseMode = 0
    ss.solutionSettings.writeSolutionToFile = False
    ss.solutionSettings.sensorsWritePeriod = sc.h
    try:
        with _silence():
            mbs.SolveDynamic(ss)
    except Exception:
        return dict(NAN, fail="solver")
    acc, pos, foot, rot = (mbs.GetSensorStoredData(s) for s in (s_acc, s_pos, s_foot, s_rot))
    n = min(len(acc), len(pos), len(foot), len(rot))
    t, ax, az, z, xb = acc[:n, 0], acc[:n, 1], acc[:n, 3], pos[:n, 3], pos[:n, 1]
    xf, zf, phi = foot[:n, 1], foot[:n, 3], rot[:n, 2]
    if not np.all(np.isfinite(az)):
        return dict(NAN, fail="nonfinite")
    # 腿行程（FFY D-8）：足球心开始侵入地面以后，髋足相对高度的最大压缩
    rel = z - zf
    ic = np.flatnonzero(zf <= r_foot * (1.0 + 1e-9))
    i0c = int(ic[0]) if len(ic) else None
    leg_stroke = float(max(0.0, np.max(rel[i0c] - rel[i0c:]))) if i0c is not None else 0.0
    sink = float(max(0.0, r_foot - np.min(zf)))
    if leg_stroke > COLLAPSE * leg.total:
        return dict(NAN, fail="collapse")
    if sink > 0.9 * r_foot:
        return dict(NAN, fail="deep_sink")
    out = dict(NAN)
    out.update(peak_a=float(np.max(np.abs(az))), a_res=float(np.max(np.hypot(ax, az))),
               stroke=float(z[0] - np.min(z)), leg_stroke=leg_stroke, sink=sink)
    met, _, _ = _metrics(t, z, az, m, v0)
    out.update(met)
    if i0c is not None:
        # 打滑（FFY _lateral 的口径）：竖直冲击窗口（触地 → 机身最低点）内，足端净滑移的最大值，
        # 净滑移 = 足球心水平位移 − 滚动 r·Δφ
        imin = i0c + int(np.argmin(z[i0c:]))
        sl = (xf - xf[i0c]) - r_foot * (phi - phi[i0c])
        out["slip"] = float(np.max(np.abs(sl[i0c:imin + 1])))
        out["x_drift"] = float(xb[-1] - xb[0])
    out["peak_grf_bw"] = float(np.max(m * (az + G)) / (m * G))
    # 数值健康体检（FFY）：末态机身总能不应高于初态
    vz, vx = np.gradient(z, t), np.gradient(xb, t)
    ke0 = max(0.5 * m * (vx[0] ** 2 + vz[0] ** 2), 1e-12)
    out["e_gain"] = float((0.5 * m * (vx[-1] ** 2 + vz[-1] ** 2) + m * G * z[-1]
                           - 0.5 * m * (vx[0] ** 2 + vz[0] ** 2) - m * G * z[0]) / ke0)
    out["fail"] = ""
    if trace:
        out["trace"] = dict(t=t, z=z, az=az, ax=ax, x_foot=xf, z_foot=zf, x_body=xb, phi=phi)
    return out


@dataclass(frozen=True)
class Criteria:
    """可行性判据（FFY v2.5 的四闸口径；数值在预注册里定）。"""
    g_cap: float = 10.0          # 峰值加速度上限（g）
    s_max_frac: float = 0.15     # 腿行程上限（占腿长的比例）
    rebound_cap: float = 0.05    # 回弹高度 / (v0²/2g) 的上限
    slip_frac: float = 0.5       # 冲击段足端相对机身走开 / 跗跖骨长 的上限

    def ok(self, r: dict, leg: Leg, sc: Scenario) -> bool:
        if r.get("fail") or not np.isfinite(r["peak_a"]):
            return False
        return bool(r["a_res"] <= self.g_cap * G and r["leg_stroke"] <= self.s_max_frac * leg.total
                    and r["rebound"] <= self.rebound_cap * sc.v0 ** 2 / (2 * G)
                    and r["slip"] <= self.slip_frac * leg.tmt)


def _one(args):
    leg, p, sc = args
    return drop(leg, p, sc)


def evaluate(legs: dict[str, Leg], params: list[Params], scenarios: list[Scenario], workers: int = 1) -> list[dict]:
    """每条腿 × 每组参数 × 每个工况跑一次；返回逐次结果（带 leg、j、工况字段）。
    以脚本运行且 workers > 1 时，调用方必须在 `if __name__ == "__main__":` 之下调用（spawn 子进程会重新导入主模块）。"""
    jobs, meta = [], []
    for si, sc in enumerate(scenarios):
        for name, leg in legs.items():
            for j, p in enumerate(params):
                jobs.append((leg, p, sc))
                meta.append(dict(leg=name, j=j, scenario=si, **{f"sc_{k}": v for k, v in asdict(sc).items()}))
    if workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(_one, jobs, chunksize=4)
    else:
        res = [_one(a) for a in jobs]
    return [dict(mt, **r) for mt, r in zip(meta, res)]
