"""对账功能测试：舍入规则、分歧判定、不相容诊断、作业失效、严格校验。"""

import importlib
import time
from types import SimpleNamespace

import pytest

from app import reconcile
from app.reconcile import total_to_micro, weight_to_micro


# ---------------------------------------------------------------- 工具
def samples4():
    return [
        {"person_id": "P1", "region": "R", "age_group": "a1", "d": 1},
        {"person_id": "P2", "region": "R", "age_group": "a1", "d": 1},
        {"person_id": "P3", "region": "R", "age_group": "a2", "d": 1},
        {"person_id": "P4", "region": "R", "age_group": "a2", "d": 1},
    ]


def answers(true_map):
    """list_a 全部取真值；list_b 在 P2、P4 翻转（2 处分歧）。"""
    a = dict(true_map)
    b = dict(true_map)
    for pid in ("P2", "P4"):
        b[pid] = not b[pid]
    return a, b


TRUE4 = {"P1": True, "P2": False, "P3": True, "P4": False}
W4 = [1.0, 2.0, 4.0, 2.0]  # P1..P4


# ---------------------------------------------------------------- 舍入
def test_weight_rounding_half_up_to_6dp():
    assert weight_to_micro(2.5) == 2_500_000
    assert weight_to_micro(1.0) == 1_000_000
    assert weight_to_micro(0.1234564) == 123_456
    assert weight_to_micro(0.1234565) == 123_457   # 第 7 位为 5，四舍五入进位
    assert weight_to_micro(1.0000005) == 1_000_001
    assert weight_to_micro(0.3333333333333333) == 333_333


def test_total_to_micro():
    assert total_to_micro(2) == 2_000_000
    assert total_to_micro(2.01) == 2_010_000
    assert total_to_micro(0.05) == 50_000


def test_group_sum_rounding_boundary_half_up():
    """加总后恰为 x.xx5 时按四舍五入归入高的一侧。"""
    s = [{"person_id": "P1", "region": "R", "age_group": "a", "d": 1},
         {"person_id": "P2", "region": "R", "age_group": "a", "d": 1}]
    ans = ({"P1": True, "P2": True}, {"P1": True, "P2": True})
    w = [1.004, 1.001]  # 合计 2.005 -> 2.01
    ok = reconcile.reconcile(s, w, *ans,
                             [{"dimension": "region", "code": "R",
                               "approve_total": 2.01}])
    assert ok.status == "ok"
    bad = reconcile.reconcile(s, w, *ans,
                              [{"dimension": "region", "code": "R",
                                "approve_total": 2.00}])
    assert bad.status == "incompatible"
    bad2 = reconcile.reconcile(s, w, *ans,
                               [{"dimension": "region", "code": "R",
                                 "approve_total": 2.02}])
    assert bad2.status == "incompatible"


# ---------------------------------------------------------------- 判定逻辑
def test_all_discrepancies_callback_when_ambiguous():
    a, b = answers(TRUE4)
    out = reconcile.reconcile(
        samples4(), W4, a, b,
        [{"dimension": "region", "code": "R", "approve_total": 7.0}])
    assert out.status == "ok"
    assert out.discrepancy_count == 2
    assert out.confirmed == []
    assert out.needs_callback == ["P2", "P4"]


def test_discrepancies_confirmed_when_forced():
    a, b = answers(TRUE4)
    out = reconcile.reconcile(
        samples4(), W4, a, b,
        [{"dimension": "region", "code": "R", "approve_total": 7.0},
         {"dimension": "age_group", "code": "a1", "approve_total": 3.0}])
    assert out.status == "ok"
    # a1 组迫使 P2=赞成，地区组随之迫使 P4=反对
    assert out.confirmed == [("P2", True), ("P4", False)]
    assert out.needs_callback == []


def test_mixed_confirmed_and_callback():
    # 5 人同一地区：P2 被锁定，P4/P5 等重可互换 -> 一个确认 + 两个回访
    s = [{"person_id": f"P{i}", "region": "R", "age_group": "a", "d": 1}
         for i in range(1, 6)]
    w = [1.0, 2.0, 2.0, 3.0, 3.0]
    true = {"P1": True, "P2": False, "P3": True, "P4": False, "P5": True}
    a = dict(true)
    b = dict(true)
    for pid in ("P2", "P4", "P5"):
        b[pid] = not b[pid]
    # base = P1+P3 = 3；目标 8 -> 分歧变量须凑 5 = P2 + (P4 或 P5)
    out = reconcile.reconcile(
        s, w, a, b,
        [{"dimension": "region", "code": "R", "approve_total": 8.0}])
    assert out.status == "ok"
    assert out.discrepancy_count == 3
    assert out.confirmed == [("P2", True)]
    assert out.needs_callback == ["P4", "P5"]


def test_incompatible_with_minimal_subset():
    a, b = answers(TRUE4)
    out = reconcile.reconcile(
        samples4(), W4, a, b,
        [{"dimension": "region", "code": "R", "approve_total": 7.0},
         {"dimension": "age_group", "code": "a1", "approve_total": 1.5}])
    assert out.status == "incompatible"
    assert out.confirmed == [] and out.needs_callback == []
    # 最小冲突子集只含 a1 复核（地区复核单独可行）
    assert out.incompatible_reviews == [
        {"dimension": "age_group", "code": "a1", "approve_total": 1.5}]


def test_incompatible_constant_row_via_milp():
    """某复核组不含分歧人但合计对不上 -> 整体不相容（零行约束）。"""
    a, b = answers(TRUE4)
    out = reconcile.reconcile(
        samples4(), W4, a, b,
        [{"dimension": "age_group", "code": "a2", "approve_total": 9.0}])
    assert out.status == "incompatible"
    assert out.incompatible_reviews == [
        {"dimension": "age_group", "code": "a2", "approve_total": 9.0}]


def test_zero_reviews_all_callback():
    a, b = answers(TRUE4)
    out = reconcile.reconcile(samples4(), W4, a, b, [])
    assert out.status == "ok"
    assert out.confirmed == []
    assert out.needs_callback == ["P2", "P4"]


def test_zero_discrepancies():
    s = samples4()
    same = (dict(TRUE4), dict(TRUE4))
    ok = reconcile.reconcile(
        s, W4, *same,
        [{"dimension": "region", "code": "R", "approve_total": 5.0}])
    assert ok.status == "ok"
    assert ok.discrepancy_count == 0
    assert ok.confirmed == [] and ok.needs_callback == []
    bad = reconcile.reconcile(
        s, W4, *same,
        [{"dimension": "region", "code": "R", "approve_total": 6.0}])
    assert bad.status == "incompatible"


def test_unknown_group_code():
    a, b = answers(TRUE4)
    ok = reconcile.reconcile(
        samples4(), W4, a, b,
        [{"dimension": "region", "code": "NOWHERE", "approve_total": 0.0}])
    assert ok.status == "ok"
    bad = reconcile.reconcile(
        samples4(), W4, a, b,
        [{"dimension": "region", "code": "NOWHERE", "approve_total": 1.0}])
    assert bad.status == "incompatible"


def test_person_outside_reviewed_groups_always_callback():
    s = samples4()
    for x in s:
        x["region"] = "OTHER"  # 地区复核覆盖不到任何人
    a, b = answers(TRUE4)
    out = reconcile.reconcile(
        s, W4, a, b,
        [{"dimension": "age_group", "code": "a1", "approve_total": 3.0}])
    assert out.status == "ok"
    # P2 被 a1 复核锁定，P4 不在任何复核组 -> 回访
    assert out.confirmed == [("P2", True)]
    assert out.needs_callback == ["P4"]


def test_indeterminate_on_solver_failure(monkeypatch):
    def boom(*args, **kwargs):
        return SimpleNamespace(status=4, x=None)  # HiGHS 异常状态

    monkeypatch.setattr(reconcile, "milp", boom)
    a, b = answers(TRUE4)
    out = reconcile.reconcile(
        samples4(), W4, a, b,
        [{"dimension": "region", "code": "R", "approve_total": 7.0}])
    assert out.status == "indeterminate"
    assert out.reason
    assert out.confirmed == [] and out.needs_callback == []


def test_max_discrepancies_30_ok():
    s = [{"person_id": f"P{i}", "region": "R", "age_group": "a", "d": 1}
         for i in range(30)]
    w = [1.0] * 30
    a = {f"P{i}": False for i in range(30)}
    b = {f"P{i}": True for i in range(30)}
    out = reconcile.reconcile(s, w, a, b, [])
    assert out.status == "ok"
    assert out.discrepancy_count == 30
    assert len(out.needs_callback) == 30


def test_over_30_discrepancies_rejected():
    s = [{"person_id": f"P{i}", "region": "R", "age_group": "a", "d": 1}
         for i in range(31)]
    w = [1.0] * 31
    a = {f"P{i}": False for i in range(31)}
    b = {f"P{i}": True for i in range(31)}
    with pytest.raises(reconcile.ReconcileError):
        reconcile.reconcile(s, w, a, b, [])


def test_randomized_cross_check_against_brute_force():
    """MILP 判定结果与 2^k 暴力枚举完全一致（固定种子，可复现）。"""
    import itertools
    import random

    rng = random.Random(20261120)
    regions = ["R1", "R2"]
    ages = ["a1", "a2", "a3"]
    for _ in range(30):
        n = 6
        s = [{"person_id": f"P{i}",
              "region": rng.choice(regions),
              "age_group": rng.choice(ages),
              "d": 1} for i in range(n)]
        w = [round(rng.uniform(0.5, 4.0), 6) for _ in range(n)]
        true = {f"P{i}": rng.random() < 0.5 for i in range(n)}
        flips = rng.sample(range(n), rng.randint(0, 4))
        a = dict(true)
        b = dict(true)
        for i in flips:
            b[f"P{i}"] = not b[f"P{i}"]
        # 复核项：随机选若干组，合计由真值推导（保证相容）
        reviews = []
        groups = [(d, c) for d in ("region", "age_group")
                  for c in sorted({x[d] for x in s})]
        rng.shuffle(groups)
        for dim, code in groups[: rng.randint(0, 5)]:
            micro = sum(reconcile.weight_to_micro(w[i])
                        for i, x in enumerate(s)
                        if x[dim] == code and true[x["person_id"]])
            reviews.append({"dimension": dim, "code": code,
                            "approve_total": ((micro + 5000) // 10000) / 100})

        out = reconcile.reconcile(s, w, a, b, reviews)
        assert out.status == "ok"

        # 暴力枚举所有分歧选法，独立复核区间约束
        disc = [i for i in range(n) if a[f"P{i}"] != b[f"P{i}"]]
        micro = [reconcile.weight_to_micro(x) for x in w]
        feasible_values = {i: set() for i in disc}
        found = 0
        for combo in itertools.product((0, 1), repeat=len(disc)):
            assign = dict(true)
            for pos, i in enumerate(disc):
                assign[f"P{i}"] = bool(combo[pos])
            ok = True
            for rv in reviews:
                tot = sum(micro[i] for i, x in enumerate(s)
                          if x[rv["dimension"]] == rv["code"]
                          and assign[x["person_id"]])
                t = reconcile.total_to_micro(rv["approve_total"])
                if not (t - 5000 <= tot <= t + 4999):
                    ok = False
                    break
            if ok:
                found += 1
                for pos, i in enumerate(disc):
                    feasible_values[i].add(bool(combo[pos]))
        assert found >= 1  # 真值选法必然可行
        exp_confirmed = {f"P{i}": next(iter(v)) for i, v in
                         feasible_values.items() if len(v) == 1}
        exp_callback = {f"P{i}" for i, v in feasible_values.items()
                        if len(v) == 2}
        assert dict(out.confirmed) == exp_confirmed
        assert set(out.needs_callback) == exp_callback


# ---------------------------------------------------------------- API 端到端
@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("WEIGHTS_DB", str(tmp_path / "test.db"))
    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as c:
        yield c, main
    main.app.state.jobs.shutdown()


SAMPLES = [
    {"person_id": "P1", "region": "R", "age_group": "a1", "d": 1},
    {"person_id": "P2", "region": "R", "age_group": "a1", "d": 1},
    {"person_id": "P3", "region": "R", "age_group": "a2", "d": 1},
    {"person_id": "P4", "region": "R", "age_group": "a2", "d": 1},
]
TARGETS = [{"dimension": "region", "code": "R", "population": 4}]
MOCK_W = [1.0, 2.0, 4.0, 2.0]


def poll(c, job_id, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = c.get(f"/jobs/{job_id}")
        if r.status_code == 404:
            return r.json()
        body = r.json()
        if body["status"] not in ("queued", "running"):
            return body
        time.sleep(0.05)
    raise TimeoutError


def run_mocked_job(c, monkeypatch, weights=MOCK_W, samples=SAMPLES):
    """提交样本/目标并用受控权重完成一次成功作业，返回 job_id。"""
    import app.solver as solver_mod

    def fake_solve(samples_arg, targets_arg):
        return solver_mod.QPResult(
            status="solved", weights=list(weights),
            objective=0.0, residual_abs_sum=0.0, residual_max_abs=0.0)

    monkeypatch.setattr(solver_mod, "solve_qp", fake_solve)
    assert c.put("/samples", json={"samples": samples}).status_code == 200
    assert c.put("/targets", json={"targets": TARGETS}).status_code == 200
    job_id = c.post("/calculate").json()["job_id"]
    assert poll(c, job_id)["status"] == "succeeded"
    return job_id


def lists(true_map, flips=("P2", "P4")):
    la = [{"person_id": p, "answer": v} for p, v in true_map.items()]
    lb = [{"person_id": p, "answer": (not v if p in flips else v)}
          for p, v in true_map.items()]
    return la, lb


def test_api_reconcile_ok_then_confirmed(client, monkeypatch):
    c, _ = client
    job_id = run_mocked_job(c, monkeypatch)
    la, lb = lists(TRUE4)

    # 仅地区复核：P2/P4 二选一，都需回访
    r = c.post("/reconcile", json={
        "job_id": job_id, "list_a": la, "list_b": lb,
        "reviews": [{"dimension": "region", "code": "R",
                     "approve_total": 7.0}]})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["discrepancy_count"] == 2
    assert body["review_count"] == 1
    assert body["confirmed"] == []
    assert body["needs_callback"] == ["P2", "P4"]
    assert body["reason"] is None
    assert body["job_id"] == job_id

    # 追加年龄段复核：两处分歧都被锁定，全部确认
    r = c.post("/reconcile", json={
        "job_id": job_id, "list_a": la, "list_b": lb,
        "reviews": [
            {"dimension": "region", "code": "R", "approve_total": 7.0},
            {"dimension": "age_group", "code": "a1", "approve_total": 3.0}]})
    body = r.json()
    assert body["status"] == "ok"
    assert body["confirmed"] == [
        {"person_id": "P2", "answer": True},
        {"person_id": "P4", "answer": False}]
    assert body["needs_callback"] == []


def test_api_reconcile_incompatible(client, monkeypatch):
    c, _ = client
    job_id = run_mocked_job(c, monkeypatch)
    la, lb = lists(TRUE4)
    r = c.post("/reconcile", json={
        "job_id": job_id, "list_a": la, "list_b": lb,
        "reviews": [{"dimension": "region", "code": "R",
                     "approve_total": 99.0}]})
    body = r.json()
    assert body["status"] == "incompatible"
    assert body["reason"]
    assert body["confirmed"] == [] and body["needs_callback"] == []
    assert body["incompatible_reviews"] == [
        {"dimension": "region", "code": "R", "approve_total": 99.0}]


def test_api_reconcile_rounding_via_weights(client, monkeypatch):
    """权重先舍入到 6 位小数，加总再舍入到 2 位（x.xx5 进高位）。"""
    c, _ = client
    job_id = run_mocked_job(c, monkeypatch, weights=[1.004, 1.001, 1.0, 1.0])
    same = [{"person_id": p["person_id"], "answer": True} for p in SAMPLES]
    r = c.post("/reconcile", json={
        "job_id": job_id, "list_a": same, "list_b": same,
        "reviews": [{"dimension": "region", "code": "R",
                     "approve_total": 4.01}]})
    assert r.json()["status"] == "ok"  # 1.004+1.001+1+1 = 4.005 -> 4.01
    r = c.post("/reconcile", json={
        "job_id": job_id, "list_a": same, "list_b": same,
        "reviews": [{"dimension": "region", "code": "R",
                     "approve_total": 4.00}]})
    assert r.json()["status"] == "incompatible"


def test_api_job_invalid_cases(client, monkeypatch):
    c, _ = client
    la, lb = lists(TRUE4)
    payload = {"job_id": "missing", "list_a": la, "list_b": lb,
               "reviews": []}
    # 未知作业
    body = c.post("/reconcile", json=payload).json()
    assert body["status"] == "job_invalid"
    assert body["reason"]
    assert body["confirmed"] == [] and body["needs_callback"] == []

    # 成功作业 -> 修改输入 -> 旧作业失效
    job_id = run_mocked_job(c, monkeypatch)
    c.put("/samples", json={"samples": SAMPLES})
    body = c.post("/reconcile", json={**payload, "job_id": job_id}).json()
    assert body["status"] == "job_invalid"

    # 未成功作业（数值失败）不可对账
    import app.solver as solver_mod

    def boom(samples_arg, targets_arg):
        return solver_mod.QPResult(status="failed", reason="simulated")

    monkeypatch.setattr(solver_mod, "solve_qp", boom)
    assert c.put("/samples", json={"samples": SAMPLES}).status_code == 200
    assert c.put("/targets", json={"targets": TARGETS}).status_code == 200
    job_id = c.post("/calculate").json()["job_id"]
    assert poll(c, job_id)["status"] == "failed"
    body = c.post("/reconcile", json={**payload, "job_id": job_id}).json()
    assert body["status"] == "job_invalid"
    assert "failed" in body["reason"]


def test_api_validation_errors(client, monkeypatch):
    c, _ = client
    job_id = run_mocked_job(c, monkeypatch)
    la, lb = lists(TRUE4)
    base = {"job_id": job_id, "list_a": la, "list_b": lb, "reviews": []}

    # 清单缺人 / 多人 / 快照外人员
    r = c.post("/reconcile", json={**base, "list_b": la[:-1]})
    assert r.status_code == 422
    r = c.post("/reconcile", json={**base, "list_b": la + [la[0]]})
    assert r.status_code == 422
    r = c.post("/reconcile", json={
        **base, "list_b": [{"person_id": "QX", "answer": True}] + la[1:]})
    assert r.status_code == 422
    # 答复必须是布尔
    r = c.post("/reconcile", json={
        **base, "list_b": [{"person_id": p["person_id"], "answer": 1}
                           for p in SAMPLES]})
    assert r.status_code == 422
    # 复核项超 12 / 重复 / 三位小数 / 负数 / 布尔
    r = c.post("/reconcile", json={**base, "reviews": [
        {"dimension": "region", "code": f"R{i}", "approve_total": 1}
        for i in range(13)]})
    assert r.status_code == 422
    r = c.post("/reconcile", json={**base, "reviews": [
        {"dimension": "region", "code": "R", "approve_total": 1},
        {"dimension": "region", "code": "R", "approve_total": 2}]})
    assert r.status_code == 422
    r = c.post("/reconcile", json={**base, "reviews": [
        {"dimension": "region", "code": "R", "approve_total": 1.234}]})
    assert r.status_code == 422
    r = c.post("/reconcile", json={**base, "reviews": [
        {"dimension": "region", "code": "R", "approve_total": -1}]})
    assert r.status_code == 422
    r = c.post("/reconcile", json={**base, "reviews": [
        {"dimension": "region", "code": "R", "approve_total": True}]})
    assert r.status_code == 422
    # 额外字段禁止
    r = c.post("/reconcile", json={**base, "extra": 1})
    assert r.status_code == 422


def test_api_discrepancy_limit(client, monkeypatch):
    c, _ = client
    samples31 = [{"person_id": f"P{i}", "region": "R", "age_group": "a",
                  "d": 1} for i in range(31)]
    job_id = run_mocked_job(c, monkeypatch, weights=[1.0] * 31,
                            samples=samples31)
    la = [{"person_id": f"P{i}", "answer": False} for i in range(31)]
    lb = [{"person_id": f"P{i}", "answer": True} for i in range(31)]
    r = c.post("/reconcile", json={
        "job_id": job_id, "list_a": la, "list_b": lb, "reviews": []})
    assert r.status_code == 422
    assert "31" in r.json()["detail"]
    # 30 处分歧可通过
    la30, lb30 = la[:-1], lb[:-1]
    r = c.post("/reconcile", json={
        "job_id": job_id, "list_a": la30, "list_b": lb30, "reviews": []})
    assert r.status_code == 422  # 清单未覆盖快照全员
    samples30 = samples31[:-1]
    job_id = run_mocked_job(c, monkeypatch, weights=[1.0] * 30,
                            samples=samples30)
    r = c.post("/reconcile", json={
        "job_id": job_id, "list_a": la30, "list_b": lb30, "reviews": []})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["discrepancy_count"] == 30
    assert len(body["needs_callback"]) == 30


FEAS_SAMPLES = [
    {"person_id": "P1", "region": "East",  "age_group": "youth",  "d": 1},
    {"person_id": "P2", "region": "East",  "age_group": "mid",    "d": 1},
    {"person_id": "P3", "region": "East",  "age_group": "senior", "d": 2},
    {"person_id": "P4", "region": "South", "age_group": "youth",  "d": 1},
    {"person_id": "P5", "region": "South", "age_group": "mid",    "d": 3},
    {"person_id": "P6", "region": "West",  "age_group": "youth",  "d": 1},
    {"person_id": "P7", "region": "West",  "age_group": "senior", "d": 2},
    {"person_id": "P8", "region": "North", "age_group": "mid",    "d": 1},
]
FEAS_TARGETS = [
    {"dimension": "region", "code": "East", "population": 4},
    {"dimension": "region", "code": "South", "population": 2},
    {"dimension": "region", "code": "West", "population": 3},
    {"dimension": "region", "code": "North", "population": 1},
    {"dimension": "age_group", "code": "youth", "population": 4},
    {"dimension": "age_group", "code": "mid", "population": 4},
    {"dimension": "age_group", "code": "senior", "population": 2},
]


def test_api_end_to_end_with_real_solver(client):
    """真实 QP 权重 + 由真实答复推导的复核合计：结论必须与真值一致。"""
    c, _ = client
    assert c.put("/samples", json={"samples": FEAS_SAMPLES}).status_code == 200
    assert c.put("/targets", json={"targets": FEAS_TARGETS}).status_code == 200
    job_id = c.post("/calculate").json()["job_id"]
    job = poll(c, job_id)
    assert job["status"] == "succeeded"
    weights = {w["person_id"]: w["w"] for w in job["weights"]}

    true = {s["person_id"]: (i % 2 == 0)
            for i, s in enumerate(FEAS_SAMPLES)}
    flips = ("P2", "P5", "P7")
    la = [{"person_id": p, "answer": v} for p, v in true.items()]
    lb = [{"person_id": p, "answer": (not v if p in flips else v)}
          for p, v in true.items()]

    def group_total(dim, code):
        micro = sum(reconcile.weight_to_micro(weights[s["person_id"]])
                    for s in FEAS_SAMPLES
                    if s[dim] == code and true[s["person_id"]])
        return ((micro + 5000) // 10000) / 100  # 四舍五入到两位

    reviews = []
    for dim in ("region", "age_group"):
        for code in sorted({s[dim] for s in FEAS_SAMPLES}):
            reviews.append({"dimension": dim, "code": code,
                            "approve_total": group_total(dim, code)})
    assert len(reviews) == 7

    r = c.post("/reconcile", json={
        "job_id": job_id, "list_a": la, "list_b": lb, "reviews": reviews})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["discrepancy_count"] == 3
    handled = {x["person_id"] for x in body["confirmed"]} | \
        set(body["needs_callback"])
    assert handled == set(flips)
    # 可确认答复必与真值一致（真值选法本身可行）
    for item in body["confirmed"]:
        assert item["answer"] == true[item["person_id"]]


def test_docs_include_reconcile(client):
    c, _ = client
    assert c.get("/docs").status_code == 200
    spec = c.get("/openapi.json").json()
    assert "/reconcile" in spec["paths"]
