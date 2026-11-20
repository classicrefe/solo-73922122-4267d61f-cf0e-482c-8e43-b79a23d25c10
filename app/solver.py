"""权重校正求解器（纯函数，不依赖 Web/数据库）。

问题（凸二次规划 QP）::

    最小化   sum_i (w_i - d_i)^2 / d_i
    约束     A w = p           （每个目标人群一行，元素为 0/1，允许线性冗余）
             0.5 d_i <= w_i <= 2 d_i

可行性由线性规划 (LP) 独立判定，QP 的数值失败绝不被当作“无解”。
无可行解时返回一个最小不可行目标子集（IIS）：删去其中任一项目标后，
其余目标在个人权重界限下变为可行。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linprog, minimize

TOL = 1.0e-10          # 内部数值容差
RESIDUAL_TOL = 1.0e-6  # 对外承诺的目标合计误差上限
BOUND_EPS = 1.0e-10


class SolverError(RuntimeError):
    """数值求解失败（区别于“不可行”）。"""


@dataclass
class QPResult:
    status: str                       # "solved" | "infeasible" | "failed"
    weights: list[float] | None = None
    objective: float | None = None
    residual_abs_sum: float | None = None
    residual_max_abs: float | None = None
    infeasible_targets: list[int] | None = None  # 目标行索引（IIS）
    reason: str | None = None
    iterations: int = 0


@dataclass
class _LPFeasibility:
    feasible: bool
    slack_min: float          # 最小松弛量；>0 可行，<0 不可行
    status_code: int
    message: str


# ---------------------------------------------------------------------------
# 数据整理
# ---------------------------------------------------------------------------

def build_matrix(samples: list[dict], targets: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """根据目标维度构造 0/1 约束矩阵 A 与人口向量 p。

    目标未覆盖任何样本时该行全 0；可行性由 LP 统一判定（例如 p>0 即不可行）。
    """
    n = len(samples)
    m = len(targets)
    a = np.zeros((m, n), dtype=float)
    p = np.zeros(m, dtype=float)
    for j, t in enumerate(targets):
        dim = t["dimension"]
        want = t["code"]
        for i, s in enumerate(samples):
            if s[dim] == want:
                a[j, i] = 1.0
        p[j] = float(t["population"])
    return a, p


# ---------------------------------------------------------------------------
# 可行性（LP：最小化总松弛，松弛必须非负；最小总松弛为 0 即可行）
# ---------------------------------------------------------------------------

def _bounds(d: np.ndarray) -> np.ndarray:
    return np.column_stack((0.5 * d, 2.0 * d))


def lp_feasible(
    a: np.ndarray,
    p: np.ndarray,
    d: np.ndarray,
    rows: list[int] | None = None,
) -> _LPFeasibility:
    """判定给定目标行子集在个人权重界限下是否可行。

    变量 x = (w_1..w_n, s_1..s_m)，约束  A' w - s = p，s >= 0，min sum(s)。
    """
    if rows is None:
        rows = list(range(a.shape[0]))
    a2 = a[rows, :] if rows else np.zeros((0, a.shape[1]))
    p2 = p[rows] if rows else np.zeros(0)
    n = a.shape[1]
    k = len(rows)

    if k == 0:
        return _LPFeasibility(True, 1.0, 0, "no targets")

    c = np.concatenate((np.zeros(n), np.ones(k)))
    # 等式 A' w - s = p
    aeq = np.hstack((a2, -np.eye(k)))
    beq = p2
    bounds = list(_bounds(d)) + [(0.0, None)] * k

    best: _LPFeasibility | None = None
    for method in ("highs",):
        try:
            res = linprog(
                c,
                A_eq=aeq,
                b_eq=beq,
                bounds=bounds,
                method=method,
                options={"presolve": True},
            )
        except Exception as exc:  # pragma: no cover - 防御性
            best = _LPFeasibility(False, np.nan, -99, f"lp exception: {exc}")
            continue
        if res.status == 0:
            slack = res.x[n:]
            s_min = float(np.min(slack))
            s_sum = float(res.fun)
            # 与人口量级匹配的零判据
            scale = max(1.0, float(np.max(np.abs(p2))))
            feasible = s_sum <= 1.0e-7 * scale and s_min <= 1.0e-7 * scale + 1.0e-9
            best = _LPFeasibility(feasible, s_min, res.status, res.message)
        elif res.status == 2:
            best = _LPFeasibility(False, -1.0, res.status, res.message)
        else:
            best = _LPFeasibility(False, np.nan, res.status, res.message)

    assert best is not None
    if np.isnan(best.slack_min):
        raise SolverError(f"可行性 LP 数值失败: {best.message}")
    return best


def find_iis(a: np.ndarray, p: np.ndarray, d: np.ndarray) -> list[int]:
    """返回最小不可行目标行索引集合（删去其中任一项即变可行）。

    采用标准贪心删除法：逐行尝试删除，删除后仍不可行则永久删除，
    最终剩下的每行都是不可行性的必要条件。
    """
    full = lp_feasible(a, p, d)
    if np.isnan(full.slack_min):
        raise SolverError("可行性 LP 数值失败，无法判定冲突目标")
    if full.feasible:
        return []

    kept = list(range(a.shape[0]))
    for idx in list(kept):
        trial = [r for r in kept if r != idx]
        probe = lp_feasible(a, p, d, trial)
        if np.isnan(probe.slack_min):
            raise SolverError("可行性 LP 数值失败，无法完成冲突定位")
        if not probe.feasible:
            kept = trial  # idx 对冲突并非必要，删除
    # 防御性核验：kept 不可行，且删去任意一行后可行
    assert not lp_feasible(a, p, d, kept).feasible
    for idx in kept:
        assert lp_feasible(a, p, d, [r for r in kept if r != idx]).feasible
    return kept


# ---------------------------------------------------------------------------
# QP 求解
# ---------------------------------------------------------------------------

def _objective(w: np.ndarray, d: np.ndarray) -> float:
    return float(np.sum((w - d) ** 2 / d))


def _grad(w: np.ndarray, d: np.ndarray) -> np.ndarray:
    return 2.0 * (w - d) / d


def _residuals(w: np.ndarray, a: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    r = a @ w - p
    return float(np.sum(np.abs(r))), float(np.max(np.abs(r)))


def _slsqp(a: np.ndarray, p: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = d.size
    constraints = {"type": "eq", "fun": lambda w: a @ w - p, "jac": lambda w: a}
    w0 = np.minimum(np.maximum(d.copy(), 0.5 * d), 2.0 * d)
    res = minimize(
        lambda w: _objective(w, d),
        w0,
        jac=lambda w: _grad(w, d),
        method="SLSQP",
        bounds=[(float(0.5 * d[i]), float(2.0 * d[i])) for i in range(n)],
        constraints=[constraints],
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    if not res.success:
        raise SolverError(f"SLSQP 未收敛: {res.message}")
    return np.asarray(res.x, dtype=float)


def _active_set_refine(
    w: np.ndarray, a: np.ndarray, p: np.ndarray, d: np.ndarray
) -> np.ndarray:
    """活动集 KKT 精修：在活动约束（等式 + 活动边界）上解 KKT 线性系统。

    目标函数 Hessian H = diag(2/d) 正定；反复加入被违反的边界、
    移除乘子符号错误的活动边界，直到 KKT 条件满足。
    """
    n = d.size
    lower, upper = 0.5 * d, 2.0 * d
    h = np.diag(2.0 / d)
    active_lo = np.zeros(n, dtype=bool)
    active_hi = np.zeros(n, dtype=bool)
    w = w.copy()

    for _ in range(200):
        # 1) 边界违反者直接纳入活动集
        active_lo |= w <= lower + 1e-9
        active_hi |= w >= upper - 1e-9
        active_lo &= ~active_hi

        b_idx_lo = np.where(active_lo)[0]
        b_idx_hi = np.where(active_hi)[0]
        nb = b_idx_lo.size + b_idx_hi.size
        # KKT: [H  C^T] [w]   [-2]   （梯度常数项为 -2，因 grad = 2w/d - 2）
        #      [C  0 ] [λ] = [b ]
        blocks = [a]
        rhs_b = [p]
        if b_idx_lo.size:
            c = np.zeros((b_idx_lo.size, n))
            c[np.arange(b_idx_lo.size), b_idx_lo] = 1.0
            blocks.append(c)
            rhs_b.append(lower[b_idx_lo])
        if b_idx_hi.size:
            c = np.zeros((b_idx_hi.size, n))
            c[np.arange(b_idx_hi.size), b_idx_hi] = 1.0
            blocks.append(c)
            rhs_b.append(upper[b_idx_hi])
        cmat = np.vstack(blocks)
        k = cmat.shape[0]
        kkt = np.block([
            [h, cmat.T],
            [cmat, np.zeros((k, k))],
        ])
        rhs = np.concatenate((np.full(n, -2.0), np.concatenate(rhs_b)))
        try:
            sol, *_ = np.linalg.lstsq(kkt, rhs, rcond=None)
        except np.linalg.LinAlgError as exc:
            raise SolverError(f"KKT 系统奇异: {exc}") from exc
        w_new, lam = sol[:n], sol[n:]
        lam_eq = lam[: a.shape[0]]
        lam_bnd = lam[a.shape[0]:]

        # 2) 检查新解是否落在界限内
        viol_lo = w_new < lower - 1e-8
        viol_hi = w_new > upper + 1e-8
        if not (viol_lo.any() or viol_hi.any()):
            w = np.clip(w_new, lower, upper)
            # 3) 活动边界乘子符号检查：
            #    下界乘子 <= 0、上界乘子 >= 0 才是最优活动集
            lam_lo = lam_bnd[: b_idx_lo.size]
            lam_hi = lam_bnd[b_idx_lo.size:]
            drop_lo = lam_lo > 1e-7
            drop_hi = lam_hi < -1e-7
            if drop_lo.any():
                active_lo[b_idx_lo[drop_lo]] = False
                continue
            if drop_hi.any():
                active_hi[b_idx_hi[drop_hi]] = False
                continue
            return w
        # 有违反：把违反边界加入活动集后重解
        w = np.clip(w_new, lower, upper)
        active_lo |= viol_lo
        active_hi |= viol_hi
        active_lo &= ~active_hi
    raise SolverError("活动集精修迭代未收敛")


def _trust_constr(a: np.ndarray, p: np.ndarray, d: np.ndarray) -> np.ndarray:
    from scipy.optimize import LinearConstraint, Bounds

    n = d.size
    res = minimize(
        lambda w: _objective(w, d),
        d.copy(),
        jac=lambda w: _grad(w, d),
        method="trust-constr",
        bounds=Bounds(0.5 * d, 2.0 * d),
        constraints=[LinearConstraint(a, p, p)],
        options={"maxiter": 1000, "gtol": 1e-12, "xtol": 1e-12,
                 "barrier_tol": 1e-12, "verbose": 0},
    )
    if not res.success:
        raise SolverError(f"trust-constr 未收敛: {res.message}")
    return np.asarray(res.x, dtype=float)


def kkt_check(a: np.ndarray, p: np.ndarray, d: np.ndarray,
              w: np.ndarray, active_tol: float = 1e-7
              ) -> tuple[float, bool]:
    """凸 QP 的 KKT 对偶可行性核验。

    平稳条件  g = Aᵀλ + ν − η，其中
      下界活动约束 w_i=0.5d_i 的乘子 ν_i ≥ 0（列取 +e_i），
      上界活动约束 w_i=2d_i   的乘子 η_i ≥ 0（列取 −e_i）。
    用最小二乘求全部乘子，返回梯度拟合残差与乘子符号是否合规。
    凸问题原始可行 + 对偶可行即全局最优。
    """
    n = d.size
    g = _grad(w, d)
    lo_idx = np.where(w <= 0.5 * d + active_tol)[0]
    hi_idx = np.where(w >= 2.0 * d - active_tol)[0]
    cols = [a.T]
    for i in lo_idx:
        e = np.zeros((n, 1))
        e[i] = 1.0
        cols.append(e)
    for i in hi_idx:
        e = np.zeros((n, 1))
        e[i] = -1.0
        cols.append(e)
    mmat = np.hstack(cols)
    coef, *_ = np.linalg.lstsq(mmat, g, rcond=None)
    resid = float(np.max(np.abs(mmat @ coef - g)))
    multipliers = coef[a.shape[0]:]
    sign_ok = bool(multipliers.size == 0 or np.min(multipliers) >= -1e-6)
    return resid, sign_ok


def solve_qp(samples: list[dict], targets: list[dict]) -> QPResult:
    """完整求解流程：先 LP 判可行性，再 QP 求最优权重。"""
    d = np.array([float(s["d"]) for s in samples], dtype=float)
    a, p = build_matrix(samples, targets)

    # 无目标：唯一可行即 d 本身（目标函数最小值 0）
    if a.shape[0] == 0:
        w = d.copy()
        return QPResult(
            status="solved",
            weights=[float(x) for x in w],
            objective=0.0,
            residual_abs_sum=0.0,
            residual_max_abs=0.0,
            iterations=0,
        )

    # 1) 可行性独立判定（数值失败抛 SolverError，不混同于不可行）
    feas = lp_feasible(a, p, d)
    if not feas.feasible:
        iis = find_iis(a, p, d)
        return QPResult(status="infeasible", infeasible_targets=iis)

    # 2) QP 数值求解，多策略兜底
    attempts: list[str] = []
    w: np.ndarray | None = None
    for name, fn in (("SLSQP", _slsqp), ("trust-constr", _trust_constr)):
        try:
            w_try = fn(a, p, d)
            w_try = np.clip(w_try, 0.5 * d, 2.0 * d)
            l1, mx = _residuals(w_try, a, p)
            if mx <= RESIDUAL_TOL:
                w = w_try
                attempts.append(f"{name}:ok")
                break
            attempts.append(f"{name}:residual={mx:.2e}")
            # 残差不达标 -> 活动集精修
            try:
                w_ref = _active_set_refine(w_try, a, p, d)
                l1, mx = _residuals(w_ref, a, p)
                if mx <= RESIDUAL_TOL:
                    w = w_ref
                    attempts.append(f"{name}+activeset:ok")
                    break
                attempts.append(f"{name}+activeset:residual={mx:.2e}")
            except SolverError as exc:
                attempts.append(f"{name}+activeset:fail({exc})")
        except SolverError as exc:
            attempts.append(f"{name}:fail({exc})")
        except Exception as exc:  # pragma: no cover - 防御性
            attempts.append(f"{name}:error({type(exc).__name__}:{exc})")

    if w is None:
        return QPResult(
            status="failed",
            reason="；".join(attempts),
        )

    l1, mx = _residuals(w, a, p)
    return QPResult(
        status="solved",
        weights=[float(x) for x in w],
        objective=_objective(w, d),
        residual_abs_sum=l1,
        residual_max_abs=mx,
        iterations=len(attempts),
        reason="；".join(attempts),
    )
