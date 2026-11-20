"""后台计算调度。

- 计算在固定线程池中执行，HTTP 请求只负责入队，不阻塞样本/目标更正；
- 作业启动前与求解过程中检查 epoch：一旦输入被修改（cancel 事件置位
  或库内 epoch 变化），在途作业立即放弃且不写任何结果；
- 只有作业仍属于当前 epoch 时才写入终态结果（完整权重或冲突集）。
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from . import solver
from .db import Database


class JobManager:
    def __init__(self, db: Database, max_workers: int = 4):
        self.db = db
        self.pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="weight-worker"
        )

    def submit(self) -> dict:
        snap = self.db.snapshot()
        job = self.db.create_job(snap)
        self.pool.submit(self._run, job["job_id"])
        return job

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    def _cancelled(self, job_id: str, job_epoch: int) -> bool:
        if self.db.current_cancel_event().is_set():
            return True
        if self.db.get_epoch() != job_epoch:
            return True
        if self.db.get_job(job_id) is None:  # 已被新 epoch 清除
            return True
        return False

    def _run(self, job_id: str) -> None:
        job = self.db.get_job(job_id)
        if job is None:
            return
        job_epoch = job["epoch"]
        cancel = self.db.current_cancel_event()
        if self._cancelled(job_id, job_epoch):
            return

        self.db.mark_running(job_id)
        if cancel.is_set() or self._cancelled(job_id, job_epoch):
            return

        snap = job["input_snapshot"]
        import json
        snapshot = json.loads(snap)
        samples, targets = snapshot["samples"], snapshot["targets"]

        try:
            # LP 可行性检查点
            if cancel.is_set() or self._cancelled(job_id, job_epoch):
                return
            res = solver.solve_qp(samples, targets)
            if cancel.is_set() or self._cancelled(job_id, job_epoch):
                return
        except solver.SolverError as exc:
            # 数值失败 ≠ 无解：单独标记 failed，保留输入，绝不返回冲突集
            self._finish_if_current(job_id, job_epoch, "failed", None, str(exc))
            return
        except Exception as exc:  # pragma: no cover - 防御性
            self._finish_if_current(
                job_id, job_epoch, "failed", None,
                f"{type(exc).__name__}: {exc}",
            )
            return

        if res.status == "infeasible":
            iis_rows = [targets[i] for i in res.infeasible_targets]
            self._finish_if_current(
                job_id, job_epoch, "infeasible",
                {"infeasible_targets": iis_rows}, None,
            )
            return

        if res.status == "failed":
            self._finish_if_current(
                job_id, job_epoch, "failed", None,
                f"数值求解失败（未判定为无解）：{res.reason}",
            )
            return

        # 成功：组装完整结果与输入依据
        weights, totals = self._assemble(samples, targets, res.weights)
        payload = {
            "weights": weights,
            "objective": res.objective,
            "residual_abs_sum": res.residual_abs_sum,
            "residual_max_abs": res.residual_max_abs,
            "target_totals": totals,
        }
        self._finish_if_current(job_id, job_epoch, "succeeded", payload, None)

    def _finish_if_current(self, job_id, job_epoch, status, result, error):
        if self.db.get_epoch() != job_epoch:
            return  # 输入已变更，结果永久作废，不写入
        if self.db.get_job(job_id) is None:
            return
        self.db.finish_job(job_id, status, result, error)

    @staticmethod
    def _assemble(samples, targets, w):
        weights = []
        for s, wi in zip(samples, w):
            weights.append({
                "person_id": s["person_id"],
                "d": s["d"],
                "w": wi,
                "lower": 0.5 * s["d"],
                "upper": 2.0 * s["d"],
            })
        a, p = solver.build_matrix(samples, targets)
        import numpy as np
        wa = np.array(w, dtype=float)
        totals = []
        for j, t in enumerate(targets):
            total = float(a[j] @ wa)
            totals.append({
                "dimension": t["dimension"],
                "code": t["code"],
                "population": t["population"],
                "weighted_total": total,
                "abs_error": abs(total - t["population"]),
            })
        return weights, totals
