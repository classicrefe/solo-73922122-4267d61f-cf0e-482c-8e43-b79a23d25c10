"""对账求解：两份录入清单 + 复核加权赞成合计 → 可确认分歧 / 需回访分歧。

舍入规则（与需求严格一致）：
- 每人权重先四舍五入（ROUND_HALF_UP）到 6 位小数，之后全部换算为整数
  微单位（1 微单位 = 1e-6）精确运算，不再经过浮点；
- 每组赞成权重加总后四舍五入到 2 位小数。合计 S 等于复核值 T 当且仅当
      T·10⁶ − 5000 ≤ S（微单位） ≤ T·10⁶ + 4999
  （恰好 x.xx5 时按四舍五入归入高的一侧，故下界含、上界不含）。

分歧建模：两份清单在分歧处必为一是一否，“从两份清单中取一个值”等价于
一个 0/1 变量 x_j（1 = 赞成）。每项复核给出一条整数区间约束，用 HiGHS
MILP（scipy.optimize.milp）判定可行性；系数全为整数，float64 求和精确，
MILP 返回的解还会用 Python 整数算术复核。数值求解失败返回
"indeterminate"，绝不混同于“汇总不相容”。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
from scipy.optimize import LinearConstraint, Bounds, milp

MAX_DISCREPANCIES = 30      # 分歧上限（超出即 422）
MAX_REVIEWS = 12            # 复核项上限
MAX_APPROVE_TOTAL = 1_000_000  # 复核赞成合计的卫生上限（理论最大值仅 200×200）
MILP_TIME_LIMIT = 30.0      # 单次 MILP 求解的防御性时限（秒）

_HALF_CENT_MICRO = 5000     # 0.005 的微单位数
_MICRO = 1_000_000


class ReconcileError(RuntimeError):
    """对账内部错误（防御性，正常流程不应触发）。"""


@dataclass
class ReconcileOutcome:
    status: str                       # "ok" | "incompatible" | "indeterminate"
    reason: str | None = None
    discrepancy_count: int = 0
    confirmed: list[tuple[str, bool]] = field(default_factory=list)  # (编号, 答复)
    needs_callback: list[str] = field(default_factory=list)
    # status=incompatible 时给出最小冲突复核子集（删去任一项即恢复相容）
    incompatible_reviews: list[dict] | None = None


# ---------------------------------------------------------------------------
# 舍入与换算
# ---------------------------------------------------------------------------

def weight_to_micro(w: float) -> int:
    """权重四舍五入到 6 位小数，返回整数微单位。"""
    q = Decimal(str(w)).quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    return int(q * _MICRO)


def total_to_micro(value: float) -> int:
    """复核赞成合计（至多两位小数，调用前已校验）换算为整数微单位。"""
    return int(Decimal(str(value)) * 100) * 10_000


# ---------------------------------------------------------------------------
# MILP 可行性判定（整数精确复核）
# ---------------------------------------------------------------------------

def _milp_feasible(
    rows: list[list[int]],
    lbs: list[int],
    ubs: list[int],
    k: int,
    fixed: dict[int, int] | None = None,
) -> tuple[str, list[int] | None]:
    """判定 Σ_j row[j]·x_j ∈ [lb, ub]（逐行）且 x ∈ {0,1}^k 是否可行。

    返回 ("feasible", x) / ("infeasible", None) / ("error", None)。
    "error" 表示数值求解失败，绝不混同于不可行。
    """
    fixed = fixed or {}
    if k == 0:
        ok = all(lo <= 0 <= hi for lo, hi in zip(lbs, ubs))
        return ("feasible" if ok else "infeasible"), ([] if ok else None)
    if not rows:
        return "feasible", [fixed.get(j, 0) for j in range(k)]

    a = np.array(rows, dtype=float)
    lb = np.array(lbs, dtype=float)
    ub = np.array(ubs, dtype=float)
    vlb = np.zeros(k)
    vub = np.ones(k)
    for j, v in fixed.items():
        vlb[j] = vub[j] = float(v)

    try:
        res = milp(
            np.zeros(k),
            constraints=LinearConstraint(a, lb, ub),
            bounds=Bounds(vlb, vub),
            integrality=np.ones(k),
            options={"time_limit": MILP_TIME_LIMIT},
        )
    except Exception:  # pragma: no cover - 防御性
        return "error", None
    if res.status == 2:
        return "infeasible", None
    if res.status != 0 or res.x is None:
        return "error", None

    # 整数精确复核：变量须为 0/1，且每行区间约束严格成立
    x = []
    for v in res.x:
        r = int(round(float(v)))
        if abs(float(v) - r) > 1e-6:
            return "error", None
        x.append(r)
    for row, lo, hi in zip(rows, lbs, ubs):
        s = sum(c * xv for c, xv in zip(row, x))
        if not (lo <= s <= hi):
            return "error", None
    return "feasible", x


def _minimal_incompatible_subset(
    rows: list[list[int]],
    lbs: list[int],
    ubs: list[int],
    k: int,
) -> list[int] | None:
    """最小不可行复核行子集（贪心删除法）：删去其中任一行即恢复可行。

    数值失败时返回 None（此时整体不可行的结论仍然成立，只是无法定位子集）。
    """
    kept = list(range(len(rows)))
    for idx in list(kept):
        trial = [r for r in kept if r != idx]
        status, _ = _milp_feasible(
            [rows[t] for t in trial],
            [lbs[t] for t in trial],
            [ubs[t] for t in trial],
            k,
        )
        if status == "error":
            return None
        if status == "infeasible":
            kept = trial  # idx 对冲突并非必要，删除
    return kept


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def reconcile(
    samples: list[dict],
    weights: list[float],
    answers_a: dict[str, bool],
    answers_b: dict[str, bool],
    reviews: list[dict],
) -> ReconcileOutcome:
    """对账主流程。

    参数
        samples:   作业快照样本（含 person_id/region/age_group），顺序即快照顺序
        weights:   与 samples 对齐的作业权重（未舍入的原始浮点值）
        answers_a/answers_b: 两份清单的逐人答复（person_id -> bool），
                             已校验与快照人员完全一致
        reviews:   复核项 [{dimension, code, approve_total}]，至多 12 项，
                   approve_total 至多两位小数
    """
    n = len(samples)
    micro = [weight_to_micro(w) for w in weights]
    disc = [i for i, s in enumerate(samples)
            if answers_a[s["person_id"]] != answers_b[s["person_id"]]]
    if len(disc) > MAX_DISCREPANCIES:  # 端点已拦截，此处防御
        raise ReconcileError(f"分歧 {len(disc)} 处，超过上限 {MAX_DISCREPANCIES}")
    disc_pos = {i: p for p, i in enumerate(disc)}
    k = len(disc)

    # 每项复核 -> 一条整数区间约束（一致答复计入常数 base，分歧计入变量系数）
    rows: list[list[int]] = []
    lbs: list[int] = []
    ubs: list[int] = []
    for rv in reviews:
        dim, code = rv["dimension"], rv["code"]
        target_micro = total_to_micro(rv["approve_total"])
        base = 0
        row = [0] * k
        for i, s in enumerate(samples):
            if s[dim] != code:
                continue
            if i in disc_pos:
                row[disc_pos[i]] = micro[i]
            elif answers_a[s["person_id"]]:  # 一致答复：两清单相同，取其一即可
                base += micro[i]
        rows.append(row)
        lbs.append(target_micro - _HALF_CENT_MICRO - base)
        ubs.append(target_micro + _HALF_CENT_MICRO - 1 - base)

    status, x = _milp_feasible(rows, lbs, ubs, k)
    if status == "error":
        return ReconcileOutcome(
            status="indeterminate",
            reason="数值求解失败，无法可靠判定分歧取值（未判定为汇总不相容），"
                   "请重试或人工复核",
            discrepancy_count=k,
        )
    if status == "infeasible":
        iis = _minimal_incompatible_subset(rows, lbs, ubs, k)
        if iis is None:
            return ReconcileOutcome(
                status="incompatible",
                reason="复核汇总不相容：不存在同时满足全部复核合计的取值方案"
                       "（冲突子集定位时数值失败，未能给出最小冲突集）",
                discrepancy_count=k,
                incompatible_reviews=None,
            )
        subset = [reviews[t] for t in iis]
        return ReconcileOutcome(
            status="incompatible",
            reason="复核汇总不相容：不存在同时满足全部复核合计的取值方案；"
                   "incompatible_reviews 给出最小冲突复核子集"
                   "（删去其中任一项即恢复相容），请核对复核汇总后重试",
            discrepancy_count=k,
            incompatible_reviews=subset,
        )

    assert x is not None
    # 逐分歧判定：已知 x 是一个可行选法，只需检验每个分歧能否取相反值
    confirmed: list[tuple[str, bool]] = []
    callback: list[str] = []
    for pos, i in enumerate(disc):
        pid = samples[i]["person_id"]
        other = 1 - x[pos]
        st, _ = _milp_feasible(rows, lbs, ubs, k, fixed={pos: other})
        if st == "error":
            return ReconcileOutcome(
                status="indeterminate",
                reason="数值求解失败，无法可靠判定全部分歧取值"
                       "（未判定为汇总不相容），请重试或人工复核",
                discrepancy_count=k,
            )
        if st == "infeasible":
            confirmed.append((pid, bool(x[pos])))  # 所有可行选法取值相同
        else:
            callback.append(pid)

    return ReconcileOutcome(
        status="ok",
        discrepancy_count=k,
        confirmed=confirmed,
        needs_callback=callback,
    )
