#!/usr/bin/env python3
"""端到端演示（仅用标准库 urllib）。

用法:
    python3 examples/demo.py [BASE_URL]

依次演示：
  1. 可行场景（目标行含线性冗余：四个地区之和 = 三个年龄段之和），
     轮询到成功，校验 0.5d<=w<=2d 与合计误差 <=1e-6；
  2. 冲突场景，返回最小冲突目标子集（删去任一项即可行）；
  3. 输入修改后旧作业 404、/result 返回 409，证明旧结果永久失效。
"""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
HERE = Path(__file__).resolve().parent


def call(method: str, path: str, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE + path, data=data, method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


def wait_job(job_id: str, timeout: float = 30.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        status, body = call("GET", f"/jobs/{job_id}")
        if status == 404:
            return status, body
        if body["status"] not in ("queued", "running"):
            return status, body
        time.sleep(0.2)
    raise TimeoutError(job_id)


def main() -> None:
    print("=" * 70)
    print("场景 1：可行输入（含线性冗余目标）")
    print("=" * 70)
    samples = json.loads((HERE / "feasible_samples.json").read_text())
    targets = json.loads((HERE / "feasible_targets.json").read_text())
    print("PUT /samples ->", call("PUT", "/samples", samples)[0])
    print("PUT /targets ->", call("PUT", "/targets", targets)[0])
    _, ack = call("POST", "/calculate")
    print("POST /calculate ->", ack)
    status, job = wait_job(ack["job_id"])
    print("终态:", job["status"], "| 目标值:", job["objective"])
    assert job["status"] == "succeeded"
    for item in job["weights"]:
        assert item["lower"] <= item["w"] <= item["upper"] + 1e-12, item
    print("逐人权重 (person_id, d -> w):")
    for item in job["weights"]:
        print(f"  {item['person_id']}: d={item['d']} -> w={item['w']:.10f} "
              f"[{item['lower']}, {item['upper']}]")
    print("各目标合计与误差:")
    for t in job["target_totals"]:
        print(f"  {t['dimension']}/{t['code']}: 合计={t['weighted_total']:.10f} "
              f"目标={t['population']} 误差={t['abs_error']:.2e}")
        assert t["abs_error"] <= 1e-6, t
    assert job["residual_max_abs"] <= 1e-6
    print(f"残差: L1={job['residual_abs_sum']:.2e} "
          f"max={job['residual_max_abs']:.2e} (<= 1e-6)")

    status, result = call("GET", "/result")
    print("GET /result ->", status, "(仅提供与当前输入一致的完整结果)")
    assert status == 200

    print()
    print("=" * 70)
    print("场景 2：冲突输入（R1=4 与 a1=1 在界限下无法同时满足）")
    print("=" * 70)
    cs = json.loads((HERE / "conflict_samples.json").read_text())
    ct = json.loads((HERE / "conflict_targets.json").read_text())
    call("PUT", "/samples", cs)
    call("PUT", "/targets", ct)
    _, ack = call("POST", "/calculate")
    _, job = wait_job(ack["job_id"])
    print("终态:", job["status"])
    assert job["status"] == "infeasible"
    print("最小冲突目标子集（删去任一项即不再冲突）:")
    for t in job["infeasible_targets"]:
        print(" ", t)
    assert {(t["dimension"], t["code"]) for t in job["infeasible_targets"]} == {
        ("region", "R1"), ("age_group", "a1")}
    status, _ = call("GET", "/result")
    print("冲突时 GET /result ->", status, "(不开放半份结果)")
    assert status == 409

    print()
    print("=" * 70)
    print("场景 3：修改输入使旧作业与旧结果永久失效")
    print("=" * 70)
    old_job = ack["job_id"]
    call("PUT", "/samples", cs)  # 即使内容相同，任何已提交修改都使旧结果失效
    status, body = call("GET", f"/jobs/{old_job}")
    print("旧作业查询 ->", status, body.get("detail", ""))
    assert status == 404
    status, _ = call("GET", "/result")
    print("GET /result ->", status, "(须重新计算)")
    assert status == 409

    print("\n全部演示断言通过 ✔")


if __name__ == "__main__":
    main()
