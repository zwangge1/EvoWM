"""三段腿奔跑一步（或双腿跳跃一步）的 Exudyn 模型（工程验证 v2，真实鸟类尺度）。

在 drop.py（v1 落震）的基础上改，三段刚性杆、膝踝扭簧阻尼、足端球与地面罚接触、地面模型都沿用 v1：
- 触地时机身有前向速度 u，行进方向为 −x；整条腿绕足端转到着地角 α（髋足连线与地面的夹角，髋在足端的后上方），
  所有腿用同一个 α。膝在髋足连线前方、踝在后方，与鸟腿一致。
- 摆腿回收：触地时整条腿绕髋刚性转动，使足端的水平速度为零（足端对地速度匹配）；否则带质量的腿以 u 横扫着地，
  支撑期一开始就打滑，这是模型的假象。
- 髋关节自由转动（弹簧质量类模型的做法）；hip_spring=True 时改回 v1 的髋扭簧阻尼（敏感性）。
- 载荷 m = load × 体重：奔跑一步为一条腿支撑（load = 1），负重奔跑 1.5、2，双腿跳跃各担一半（load = 0.5）。
- 关节扭簧、阻尼、杆件质量与地面刚度的绝对值与 v1 相同（单位：体重的一半 × g × 原腿总长），不随载荷变：
  同一条腿、同一种地面。
- 踩空：多下落 Δh 折算成触地竖直速度 √(v_平² + 2gΔh)，下落时腿的姿态不变（被动腿，各段同加速度下落）。
- 仿真到离地（足端离开地面并保持一小段时间）或到 T 结束；离地后的抛体最高点不仿真。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from evowm.engineering.drop import FFY_L, G, TERRAIN, Leg, _rot_y, _silence

# 参数盒。刚度单位同 v1（体重的一半 × g × 原腿总长）；阻尼 τ* 同 v1（FFY）。
# 膝、踝扭簧按鹌鹑奔跑标定（Andrada 等 2013：峰值地面反力约 2 倍体重、腿压缩约 25%）：κ 约 1.6 时模型给出
# 峰值 2.2 倍体重、压缩 19%，0.8 以下塌陷；取 1.2 到 4。
# 触地姿态取鸟类奔跑实测（Stoessel 与 Fischer 2012 的鹌鹑、䳍、寒鸦段角）：膝内角约 96° 到 111°，踝内角约 105° 到 112°，
# 髋足连线与地面约 57° 到 67°。髋扭簧 kappa_h 照常抽样，只在 hip_spring 时使用，主分析与敏感性用同一批参数。
_TAU_S = np.sqrt(FFY_L / G)
RUN_BOX = dict(kappa_a=(1.2, 4.0), kappa_k=(1.2, 4.0), kappa_h=(3.0, 32.0),
               tau_star=(0.005 / _TAU_S, 0.10 / _TAU_S),
               theta_ankle=(95.0, 125.0), theta_knee=(90.0, 120.0), alpha=(55.0, 70.0))

COLLAPSE = 0.6        # 腿（髋足距离）压缩超过腿长的 60% 记为塌陷（同 v1）
HIP_MIN = 0.25        # 支撑期髋的高度低于 0.25 倍腿长也记为塌陷（腿被压倒、机身落地）
OFF_WIN = 0.02        # 离地判定：足端连续离地的时长（单位 √(L/g)）


@dataclass(frozen=True)
class RunParams:
    """所有腿共用的一组参数：关节扭簧（N·m/rad）、阻尼（N·m·s/rad）、触地时的踝角与膝角（度，内角）、着地角（度）。"""
    k_ankle: float
    k_knee: float
    k_hip: float
    c_ankle: float
    c_knee: float
    c_hip: float
    theta_ankle: float
    theta_knee: float
    alpha: float


@dataclass(frozen=True)
class RunScenario:
    """一个工况。由 ground_scenario 从原腿 O 构造，比较中的所有腿共用。"""
    m_body: float         # 鸟的体重（kg）
    load: float           # 这条腿承担的载荷 / 体重
    u: float              # 前向速度（m/s）
    v_td: float           # 触地竖直速度（m/s，向下为正）
    dh: float             # 踩空的下落高度（腿长的倍数，只作记录）
    terrain: str
    l_ref: float          # 原腿 O 的总长（m）
    r_foot: float         # 足端半径（m）
    lin_density: float    # 杆件线密度（kg/m）
    rod_width: float      # 杆件截面边长（m）
    T: float              # 仿真时长（s）
    h: float              # 步长（s）
    hip_spring: bool = False

    @property
    def m(self) -> float:
        return self.load * self.m_body

    @property
    def m_unit(self) -> float:
        """关节与地面参数的质量单位，同 v1：体重的一半。"""
        return 0.5 * self.m_body

    @property
    def kc(self) -> float:
        return TERRAIN[self.terrain]["kc_star"] * self.m_unit * G / self.l_ref


def ground_scenario(body_mass_kg: float, leg_o: Leg, speed: float, dh: float = 0.0, terrain: str = "turf",
                    load: float = 1.0, v_level: float = 0.2, hip_spring: bool = False,
                    h: float | None = None, T: float = 3.0) -> RunScenario:
    """speed、v_level 以 √(gL) 为单位，dh 以腿长 L 为单位，T 以 √(L/g) 为单位（L 为 O 的总长）。
    足端半径、杆件线密度与截面、步长的取法同 v1 的 bird_scenario（杆件质量 = 体重一半的 10%，按段长分配）。"""
    L = leg_o.total
    vg = np.sqrt(G * L)
    m_unit = 0.5 * body_mass_kg
    ts = np.sqrt(L / G) / _TAU_S
    v0 = v_level * vg
    return RunScenario(m_body=body_mass_kg, load=load, u=speed * vg, v_td=float(np.sqrt(v0 ** 2 + 2 * G * dh * L)),
                       dh=dh, terrain=terrain, l_ref=L, r_foot=0.2 * leg_o.tmt, lin_density=0.10 * m_unit / L,
                       rod_width=0.06 * L, T=round(T * np.sqrt(L / G), 4), h=h if h is not None else 2e-4 * ts,
                       hip_spring=hip_spring)


def run_box(n: int, seed: int, sc: RunScenario, box: dict | None = None) -> list[RunParams]:
    """在 RUN_BOX 里拉丁超立方取 n 组参数（线性均匀），所有腿、所有工况共用同一批（配对比较）。
    刚度单位同 v1：体重的一半 × g × 原腿总长；阻尼 c = τ·k，τ = τ*·√(L/g)。"""
    box = box or RUN_BOX
    rng = np.random.default_rng(seed)
    keys = list(box)
    u = np.empty((n, len(keys)))
    for j in range(len(keys)):
        u[:, j] = rng.permutation((np.arange(n) + rng.random(n)) / n)
    unit = sc.m_unit * G * sc.l_ref
    tau_scale = np.sqrt(sc.l_ref / G)
    out = []
    for row in u:
        v = {k: box[k][0] + (box[k][1] - box[k][0]) * row[i] for i, k in enumerate(keys)}
        ka, kk, kh = v["kappa_a"] * unit, v["kappa_k"] * unit, v["kappa_h"] * unit
        tau = v["tau_star"] * tau_scale
        out.append(RunParams(ka, kk, kh, tau * ka, tau * kk, tau * kh, v["theta_ankle"], v["theta_knee"], v["alpha"]))
    return out


def touchdown_pose(leg: Leg, p: RunParams):
    """触地姿态：返回各段与水平的夹角 (a1, a2, a3) 与足端、踝、膝、髋的位置（足端在原点）。
    先按 v1 的约定由踝角、膝角搭出折线腿，再整体绕足端转动，使髋足连线与地面成 α，髋在足端后方（+x）。"""
    a1 = 0.0
    a2 = a1 + (np.pi - np.radians(p.theta_ankle))
    a3 = a2 - (np.pi - np.radians(p.theta_knee))
    d = lambda a: np.array([np.cos(a), 0.0, np.sin(a)])  # noqa: E731
    H0 = leg.tmt * d(a1) + leg.tbt * d(a2) + leg.fem * d(a3)
    rot = np.radians(p.alpha) - np.arctan2(H0[2], H0[0])
    a1, a2, a3 = a1 + rot, a2 + rot, a3 + rot
    F = np.zeros(3)
    A = F + leg.tmt * d(a1)
    K = A + leg.tbt * d(a2)
    H = K + leg.fem * d(a3)
    return (a1, a2, a3), (F, A, K, H)


NAN = dict(peak_f=np.nan, peak_fz=np.nan, compression=np.nan, sink=np.nan, hip_min=np.nan,
           slip=np.nan, slip_stance=np.nan,
           t_stance=np.nan, vx_ratio=np.nan, vz_to=np.nan, e_loss=np.nan, apex=np.nan, e_gain=np.nan)


def _signed(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """x-z 平面内由 u 转到 v 的有向角（逆时针为正，z 向上、x 向右看）。u、v 形如 (n, 2)：列为 x、z。"""
    return np.arctan2(u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0], u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1])


def step(leg: Leg, p: RunParams, sc: RunScenario, trace: bool = False) -> dict:
    """奔跑（或跳跃）的一步；返回指标字典（失败时带 fail 字段：pose、solver、nonfinite、collapse、hyperext、
    deep_sink、no_takeoff、no_contact）。力以鸟的体重 m_body·g 为单位（每条腿）。"""
    import exudyn as exu
    from exudyn.utilities import InertiaCuboid, InertiaSphere, SensorBody

    l1, l2, l3 = leg.tmt, leg.tbt, leg.fem
    m = sc.m
    ter = TERRAIN[sc.terrain]
    kc = sc.kc
    cc = 2.0 * ter["zeta_c"] * np.sqrt(kc * sc.m_unit)
    r_foot = sc.r_foot
    (a1, a2, a3), (F, A, K, H) = touchdown_pose(leg, p)
    # 姿态检查：踝、膝、髋都要在足端之上（否则杆件插进地面）
    if min(A[2], K[2], H[2]) <= 0.0:
        return dict(NAN, fail="pose")
    lift = np.array([0.0, 0.0, 1.3 * r_foot])          # 初始离地间隙 0.3 r_foot（同 v1）
    F, A, K, H = F + lift, A + lift, K + lift, H + lift
    w = sc.rod_width
    v_body = np.array([-sc.u, 0.0, -sc.v_td])
    # 摆腿回收：腿绕髋以 ω_y 刚性转动，ω × (F − H) 抵消足端的水平速度
    om = -sc.u / (H[2] - F[2])
    omega = np.array([0.0, om, 0.0])
    vel = lambda P: list(v_body + np.cross(omega, P - H))  # noqa: E731

    SC = exu.SystemContainer()
    mbs = SC.AddSystem()
    ground = mbs.CreateGround(referencePosition=[0, 0, 0])

    def rod(name, P0, P1, ang, L):
        mass = sc.lin_density * L
        inertia = InertiaCuboid(density=mass / (L * w * w), sideLengths=[L, w, w])
        return mbs.CreateRigidBody(name=name, referencePosition=list(0.5 * (P0 + P1)),
                                   referenceRotationMatrix=_rot_y(ang), initialVelocity=vel(0.5 * (P0 + P1)),
                                   initialAngularVelocity=list(omega), inertia=inertia, gravity=[0, 0, -G])

    tarso = rod("tarso", F, A, a1, l1)
    tibio = rod("tibio", A, K, a2, l2)
    femur = rod("femur", K, H, a3, l3)
    body = mbs.CreateRigidBody(name="payload", referencePosition=list(H), initialVelocity=list(v_body),
                               inertia=InertiaSphere(mass=m, radius=0.3 * sc.l_ref), gravity=[0, 0, -G])
    # 机身面内自由平动、转动锁死（同 v1）
    mbs.CreateGenericJoint(bodyNumbers=[ground, body], position=list(H), constrainedAxes=[0, 1, 0, 1, 1, 1])
    mbs.CreateRevoluteJoint(bodyNumbers=[body, femur], position=list(H), axis=[0, 1, 0])
    if sc.hip_spring:
        mbs.CreateTorsionalSpringDamper(bodyNumbers=[body, femur], position=list(H), axis=[0, 1, 0],
                                        stiffness=p.k_hip, damping=p.c_hip)
    for b0, b1, P, k, c in ((femur, tibio, K, p.k_knee, p.c_knee), (tibio, tarso, A, p.k_ankle, p.c_ankle)):
        mbs.CreateRevoluteJoint(bodyNumbers=[b0, b1], position=list(P), axis=[0, 1, 0])
        mbs.CreateTorsionalSpringDamper(bodyNumbers=[b0, b1], position=list(P), axis=[0, 1, 0],
                                        stiffness=k, damping=c)
    big = 50.0 * sc.l_ref
    quad = [[-big, -big, 0.0], [big, -big, 0.0], [big, big, 0.0], [-big, big, 0.0]]
    mbs.CreateSphereQuadContact(bodyNumbers=[tarso, ground], localPosition0=[-0.5 * l1, 0.0, 0.0],
                                radiusSphere=r_foot, quadPoints=quad, contactStiffness=kc,
                                contactDamping=cc, dynamicFriction=ter["mu"],
                                frictionProportionalZone=1e-3 * np.sqrt(sc.l_ref / FFY_L))
    OVT = exu.OutputVariableType
    sens = dict(
        pos=SensorBody(bodyNumber=body, storeInternal=True, outputVariableType=OVT.Position),
        vel=SensorBody(bodyNumber=body, storeInternal=True, outputVariableType=OVT.Velocity),
        acc=SensorBody(bodyNumber=body, storeInternal=True, outputVariableType=OVT.Acceleration),
        foot=SensorBody(bodyNumber=tarso, localPosition=[-0.5 * l1, 0.0, 0.0], storeInternal=True,
                        outputVariableType=OVT.Position),
        ank=SensorBody(bodyNumber=tarso, localPosition=[0.5 * l1, 0.0, 0.0], storeInternal=True,
                       outputVariableType=OVT.Position),
        kne=SensorBody(bodyNumber=tibio, localPosition=[0.5 * l2, 0.0, 0.0], storeInternal=True,
                       outputVariableType=OVT.Position),
        rot=SensorBody(bodyNumber=tarso, storeInternal=True, outputVariableType=OVT.Rotation),
    )
    sid = {k: mbs.AddSensor(s) for k, s in sens.items()}
    mbs.Assemble()
    ss = exu.SimulationSettings()
    ss.timeIntegration.endTime = sc.T
    ss.timeIntegration.numberOfSteps = int(round(sc.T / sc.h))
    ss.timeIntegration.generalizedAlpha.spectralRadius = 0.7
    ss.timeIntegration.discontinuous.useRecommendedStepSize = False   # 同 v1
    ss.timeIntegration.verboseMode = 0
    ss.solutionSettings.writeSolutionToFile = False
    ss.solutionSettings.sensorsWritePeriod = sc.h
    try:
        with _silence():
            mbs.SolveDynamic(ss)
    except Exception:
        return dict(NAN, fail="solver")
    D = {k: mbs.GetSensorStoredData(i) for k, i in sid.items()}
    n = min(len(v) for v in D.values())
    t = D["pos"][:n, 0]
    xb, zb = D["pos"][:n, 1], D["pos"][:n, 3]
    vx, vz = D["vel"][:n, 1], D["vel"][:n, 3]
    ax, az = D["acc"][:n, 1], D["acc"][:n, 3]
    xf, zf = D["foot"][:n, 1], D["foot"][:n, 3]
    xa, za = D["ank"][:n, 1], D["ank"][:n, 3]
    xk, zk = D["kne"][:n, 1], D["kne"][:n, 3]
    phi = D["rot"][:n, 2]
    if not (np.all(np.isfinite(az)) and np.all(np.isfinite(zb))):
        return dict(NAN, fail="nonfinite")

    on = zf <= r_foot * (1.0 + 1e-9)            # 足球心侵入地面即为接触
    ic = np.flatnonzero(on)
    if len(ic) == 0:
        return dict(NAN, fail="no_contact")
    i0 = int(ic[0])
    # 离地：触地以后，足端连续 OFF_WIN·√(L/g) 不接触，且这一段开始时机身已在上升；取这一段的起点。
    # 触地瞬间足端的短暂弹起（机身仍在下降）不算离地；足端离地而机身一直下降，记为没有离地。
    wn = max(2, int(round(OFF_WIN * np.sqrt(sc.l_ref / G) / sc.h)))
    off = ~on
    off[:i0] = False
    run, ito = 0, None
    for i in range(i0, n):
        run = run + 1 if off[i] else 0
        if run >= wn and vz[i - run + 1] >= 0.0:
            ito = i - run + 1
            break

    seg = slice(i0, (ito if ito is not None else n - 1) + 1)
    lleg = np.hypot(xb - xf, zb - zf)
    compression = float(max(0.0, lleg[i0] - np.min(lleg[seg])))
    sink = float(max(0.0, r_foot - np.min(zf[seg])))
    hip_min = float(np.min(zb[seg]))
    # 关节反折：踝处由跗跖转到胫跗的有向角、膝处由胫跗转到股骨的有向角，与触地时异号即越过伸直位
    u1 = np.stack([xa - xf, za - zf], axis=1)
    u2 = np.stack([xk - xa, zk - za], axis=1)
    u3 = np.stack([xb - xk, zb - zk], axis=1)
    turn_a, turn_k = _signed(u1, u2), _signed(u2, u3)
    hyper = bool(np.any(turn_a[seg] * turn_a[0] <= 0) or np.any(turn_k[seg] * turn_k[0] <= 0))
    Fx, Fz = m * ax, m * (az + G)               # 机身受到的合力（经髋传来），近似地面反力
    bw = sc.m_body * G
    out = dict(NAN)
    out.update(compression=compression, sink=sink, peak_f=float(np.max(np.hypot(Fx, Fz)[seg]) / bw),
               peak_fz=float(np.max(Fz[seg]) / bw), hip_min=hip_min / leg.total)
    # 打滑（同 v1）：竖直冲击窗口（触地 → 机身最低点）内足端净滑移的最大值，净滑移 = 水平位移 − 滚动 r·Δφ；
    # 另记整个支撑期的净滑移（描述）
    sl = (xf - xf[i0]) - r_foot * (phi - phi[i0])
    imin = i0 + int(np.argmin(zb[seg]))
    out["slip"] = float(np.max(np.abs(sl[i0:imin + 1])))
    out["slip_stance"] = float(np.max(np.abs(sl[seg])))
    e = 0.5 * m * (vx ** 2 + vz ** 2) + m * G * zb
    ke0 = 0.5 * m * (vx[i0] ** 2 + vz[i0] ** 2)
    if compression > COLLAPSE * leg.total or hip_min < HIP_MIN * leg.total:
        out["fail"] = "collapse"
    elif hyper:
        out["fail"] = "hyperext"
    elif sink > 0.9 * r_foot:
        out["fail"] = "deep_sink"
    elif ito is None:
        out["fail"] = "no_takeoff"
    else:
        out.update(t_stance=float(t[ito] - t[i0]), vx_ratio=float(-vx[ito] / sc.u) if sc.u > 0 else np.nan,
                   vz_to=float(vz[ito]), e_loss=float((e[i0] - e[ito]) / ke0),
                   apex=float((zb[ito] + max(vz[ito], 0.0) ** 2 / (2 * G) - zb[i0]) / leg.total))
        out["fail"] = ""
    # 数值体检：从开始到结束，机身总能不应增加（腿与弹簧的能量另算，只作粗查）
    out["e_gain"] = float((e[seg][-1] - e[0]) / max(0.5 * m * (vx[0] ** 2 + vz[0] ** 2), 1e-12))
    if trace:
        out["trace"] = dict(t=t, xb=xb, zb=zb, vx=vx, vz=vz, Fx=Fx / bw, Fz=Fz / bw, xf=xf, zf=zf,
                            lleg=lleg, turn_a=turn_a, turn_k=turn_k, i0=i0, ito=ito)
    return out


@dataclass(frozen=True)
class RunCriteria:
    """一步的判据（数值在预注册里定）。"""
    f_cap: float | None = None       # 峰值合力上限（鸟的体重的倍数）；None 表示不用绝对上限
    f_rel: float | None = None       # 相对上限：不超过同组参数、同工况平地（Δh = 0）的峰值 + f_rel 倍体重
    vx_min: float = 0.5              # 离地时前向速度 / 触地时的下限
    slip_frac: float = 0.5           # 冲击窗口内足端净滑移 / 跗跖骨长的上限

    def ok(self, r: dict, leg: Leg, level_peak: float | None = None) -> bool:
        if r.get("fail"):
            return False
        if not (r["vz_to"] > 0 and r["vx_ratio"] >= self.vx_min and r["slip"] <= self.slip_frac * leg.tmt):
            return False
        if self.f_cap is not None and r["peak_f"] > self.f_cap:
            return False
        if self.f_rel is not None:
            if level_peak is None or not np.isfinite(level_peak) or r["peak_f"] > level_peak + self.f_rel:
                return False
        return True


def _one(args):
    leg, p, sc = args
    return step(leg, p, sc)


def evaluate_run(legs: dict[str, Leg], params: list[RunParams], scenarios: list[RunScenario],
                 workers: int = 1) -> list[dict]:
    """每条腿 × 每组参数 × 每个工况跑一步；返回逐次结果（带 leg、j、scenario 字段与工况字段）。
    以脚本运行且 workers > 1 时，调用方必须在 `if __name__ == "__main__":` 之下调用。"""
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
