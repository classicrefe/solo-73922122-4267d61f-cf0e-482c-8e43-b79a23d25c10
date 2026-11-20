"""求解器单元测试：可行性、冗余、冲突 IIS、界限、容差、最优性。"""

import numpy as np
import pytest

from app import solver


def sample(n=10, seed=0, dmax=10):
    rng = np.random.default_rng(seed)
    regions = ["E", "S", "W", "N"]
    ages = ["y", "m", "s"]
    out = []
    for i in range(n):
        out.append({
            "person_id": f"P{i}",
            "region": regions[rng.integers(0, len(regions))],
            "age_group": ages[rng.integers(0, len(ages))],
            "d": int(rng.integers(1, dmax + 1)),
        })
    return out


def targets_from(samples, scale=1.0):
    """以 d 的分组合计（缩放后取整风格）构造可行目标。"""
    tg = []
    for dim in ("region", "age_group"):
        codes = sorted({s[dim] for s in samples})
        for c in codes:
            pop = sum(s["d"] for s in samples if s[dim] == c) * scale
            tg.append({"dimension": dim, "code": c,
                       "population": int(round(pop))})
    return tg


def check_solved(samples, targets):
    res = solver.solve_qp(samples, targets)
    assert res.status == "solved", res.reason
    d = np.array([float(s["d"]) for s in samples])
    w = np.array(res.weights)
    # 个人界限
    assert np.all(w >= 0.5 * d - 1e-9)
    assert np.all(w <= 2.0 * d + 1e-9)
    # 合计误差
    a, p = solver.build_matrix(samples, targets)
    r = a @ w - p
    assert np.max(np.abs(r)) <= 1e-6
    assert np.sum(np.abs(r)) <= 1e-6
    # 最优性：凸 QP 的 KKT 对偶可行性核验（原始可行 + 对偶可行 => 全局最优）
    kkt_resid, sign_ok = solver.kkt_check(a, p, d, w)
    assert sign_ok, "活动边界乘子符号错误"
    assert kkt_resid <= 1e-6, f"KKT 平稳残差过大: {kkt_resid}"
    return res


def test_basic_feasible_random_cases():
    for seed in range(8):
        s = sample(n=20, seed=seed)
        t = targets_from(s)
        check_solved(s, t)


def test_max_size_200_samples_12_targets():
    s = sample(n=200, seed=42)
    t = targets_from(s)[:12]
    res = check_solved(s, t)
    assert len(res.weights) == 200


def test_redundant_targets_solve_normally():
    # 行线性冗余：ALL 行（覆盖全员）= 各 region 行之和，约束矩阵行线性相关。
    s = sample(n=15, seed=3)
    for x in s:
        x["age_group"] = "ALL"
    total = sum(x["d"] for x in s)
    t = [{"dimension": "age_group", "code": "ALL", "population": total}]
    for c in sorted({x["region"] for x in s}):
        t.append({
            "dimension": "region", "code": c,
            "population": sum(x["d"] for x in s if x["region"] == c),
        })
    a, _ = solver.build_matrix(s, t)
    assert np.linalg.matrix_rank(a) < a.shape[0]  # 确有冗余行
    res = check_solved(s, t)
    assert res.status == "solved"


def test_zero_population_target():
    s = sample(n=6, seed=5)
    t = targets_from(s)
    t[0]["population"] = 0
    res = solver.solve_qp(s, t)
    # 该组每个人下界之和 > 0，故不可行
    assert res.status == "infeasible"
    # IIS 最小性：删去其中任一项目标后可行
    a, p = solver.build_matrix(s, t)
    d = np.array([float(x["d"]) for x in s])
    iis = res.infeasible_targets
    assert iis == [0]
    assert solver.lp_feasible(a, p, d, []).feasible


def test_conflict_pairwise_iis_minimality():
    s = [
        {"person_id": "X", "region": "R1", "age_group": "a1", "d": 1},
        {"person_id": "Y", "region": "R1", "age_group": "a2", "d": 1},
        {"person_id": "Z", "region": "R2", "age_group": "a1", "d": 1},
    ]
    t = [
        {"dimension": "region", "code": "R1", "population": 4},
        {"dimension": "age_group", "code": "a1", "population": 1},
    ]
    res = solver.solve_qp(s, t)
    assert res.status == "infeasible"
    assert set(res.infeasible_targets) == {0, 1}
    a, p = solver.build_matrix(s, t)
    d = np.ones(3)
    # 整体不可行
    assert not solver.lp_feasible(a, p, d).feasible
    # 删去 IIS 中任一项 => 可行
    for idx in res.infeasible_targets:
        rest = [r for r in range(len(t)) if r != idx]
        assert solver.lp_feasible(a, p, d, rest).feasible


def test_solution_at_bounds():
    # 目标人口压到下界总和：权重全部贴 0.5d（d 取偶数保证人口为整数）
    s = sample(n=6, seed=7)
    for x in s:
        x["d"] = 2
    t = []
    for c in sorted({x["region"] for x in s}):
        pop = sum(0.5 * x["d"] for x in s if x["region"] == c)
        t.append({"dimension": "region", "code": c, "population": int(pop)})
    res = solver.solve_qp(s, t)
    assert res.status == "solved"
    for x, w in zip(s, res.weights):
        assert w == pytest.approx(0.5 * x["d"], abs=1e-8)


def test_objective_zero_when_targets_match_d():
    s = sample(n=8, seed=11)
    t = targets_from(s, scale=1.0)
    res = solver.solve_qp(s, t)
    assert res.status == "solved"
    assert res.objective == pytest.approx(0.0, abs=1e-16)
    assert np.allclose(res.weights, [x["d"] for x in s], atol=1e-9)


def test_no_targets_returns_d():
    s = sample(n=4, seed=1)
    res = solver.solve_qp(s, [])
    assert res.status == "solved"
    assert res.weights == [float(x["d"]) for x in s]
    assert res.objective == 0.0


def test_uncovered_target_infeasible():
    s = sample(n=4, seed=2)
    t = [{"dimension": "region", "code": "MARS", "population": 3}]
    res = solver.solve_qp(s, t)
    assert res.status == "infeasible"
    assert res.infeasible_targets == [0]
    # 人口为 0 的未覆盖目标可行（0=0）
    t2 = [{"dimension": "region", "code": "MARS", "population": 0}]
    res2 = solver.solve_qp(s, t2)
    assert res2.status == "solved"


def test_solver_failure_is_not_infeasible(monkeypatch):
    s = sample(n=4, seed=2)
    t = targets_from(s)

    def boom(*a, **k):
        raise solver.SolverError("simulated LP crash")

    monkeypatch.setattr(solver, "lp_feasible", boom)
    with pytest.raises(solver.SolverError):
        solver.solve_qp(s, t)


def test_d_limits_respected():
    s = [
        {"person_id": f"P{i}", "region": "E", "age_group": "m", "d": 100}
        for i in range(3)
    ]
    t = [{"dimension": "region", "code": "E", "population": 450}]
    res = solver.solve_qp(s, t)
    assert res.status == "solved"
    assert all(abs(w - 150) < 1e-7 for w in res.weights)
    assert res.residual_max_abs <= 1e-6
