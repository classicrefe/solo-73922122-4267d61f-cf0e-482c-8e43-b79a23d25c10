"""FastAPI 应用：样本/目标更正、后台权重计算、结果取用、录入对账。"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from . import reconcile as reconcile_mod
from .db import Database
from .jobs import JobManager
from .schemas import (
    ConfirmedAnswer,
    JobResult,
    ReconcilePayload,
    ReconcileResponse,
    SamplesPayload,
    TargetsPayload,
    TargetTotalsItem,
    WeightItem,
)

DB_PATH = os.environ.get("WEIGHTS_DB", "/data/weights.db")


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    app.state.db = Database(DB_PATH)
    app.state.jobs = JobManager(app.state.db)
    yield
    app.state.jobs.shutdown()
    app.state.db.close()


app = FastAPI(
    title="回访问卷样本权重校正 API",
    version="1.1.0",
    description=(
        "根据地区/年龄段人口总量，对至多 200 人的基础权重 d 做二次规划校正，"
        "满足 0.5d ≤ w ≤ 2d 且目标人群权重合计等于人口总量，"
        "最小化 Σ(w−d)²/d。任何样本/目标修改都会令旧结果与在途计算永久失效。\n\n"
        "对账：基于一次成功且仍有效的权重作业，比对两份逐人是/否录入清单与"
        "按地区/年龄段复核的加权赞成合计，判定哪些分歧可直接确认、哪些仍需回访。"
    ),
    lifespan=lifespan,
)


def db() -> Database:
    return app.state.db


def jobs() -> JobManager:
    return app.state.jobs


# ---------------------------------------------------------------- 输入更正
@app.put("/samples", status_code=200, tags=["输入数据"],
         summary="整体替换样本（至多 200 人；使所有旧结果与在途计算失效）")
def put_samples(payload: SamplesPayload):
    samples = [s.model_dump() for s in payload.samples]
    epoch = db().replace_samples(samples)
    return {"epoch": epoch, "count": len(samples),
            "message": "样本已更新，旧结果与在途计算已永久失效，须重新计算"}


@app.get("/samples", tags=["输入数据"], summary="查看当前样本")
def get_samples():
    snap = db().snapshot()
    return {"epoch": snap["epoch"], "samples": snap["samples"]}


@app.put("/targets", status_code=200, tags=["输入数据"],
         summary="整体替换人口目标（至多 12 项；使所有旧结果与在途计算失效）")
def put_targets(payload: TargetsPayload):
    targets = [t.model_dump() for t in payload.targets]
    epoch = db().replace_targets(targets)
    return {"epoch": epoch, "count": len(targets),
            "message": "目标已更新，旧结果与在途计算已永久失效，须重新计算"}


@app.get("/targets", tags=["输入数据"], summary="查看当前人口目标")
def get_targets():
    snap = db().snapshot()
    return {"epoch": snap["epoch"], "targets": snap["targets"]}


# ---------------------------------------------------------------- 计算/结果
@app.post("/calculate", status_code=202, tags=["计算"],
          summary="提交后台权重计算（不阻塞，返回 job_id）")
def calculate():
    snap = db().snapshot()
    if not snap["samples"]:
        raise HTTPException(422, "当前没有样本，无法计算：请先 PUT /samples")
    if not snap["targets"]:
        raise HTTPException(422, "当前没有目标，无法计算：请先 PUT /targets")
    job = jobs().submit()
    return JSONResponse(
        status_code=202,
        content={"job_id": job["job_id"], "epoch": job["epoch"],
                 "status": job["status"],
                 "poll": f"/jobs/{job['job_id']}"},
    )


def _serialize_job(row: dict) -> dict:
    result = json.loads(row["result"]) if row["result"] else None
    weights = totals = None
    objective = l1 = lmax = None
    iis = None
    if row["status"] == "succeeded" and result:
        weights = [WeightItem(**x).model_dump() for x in result["weights"]]
        totals = [TargetTotalsItem(**x).model_dump() for x in result["target_totals"]]
        objective = result["objective"]
        l1 = result["residual_abs_sum"]
        lmax = result["residual_max_abs"]
    elif row["status"] == "infeasible" and result:
        iis = result["infeasible_targets"]

    out = JobResult(
        job_id=row["job_id"],
        epoch=row["epoch"],
        status=row["status"],
        submitted_at=row["submitted_at"],
        finished_at=row["finished_at"],
        input_snapshot=json.loads(row["input_snapshot"]),
        error=row["error"],
        weights=weights,
        objective=objective,
        residual_abs_sum=l1,
        residual_max_abs=lmax,
        target_totals=totals,
        infeasible_targets=iis,
    )
    return out.model_dump(exclude_none=False)


@app.get("/jobs", tags=["计算"], summary="列出最近作业")
def list_jobs():
    return {"epoch": db().get_epoch(), "jobs": db().list_jobs()}


@app.get("/jobs/{job_id}", tags=["计算"],
         summary="查询作业状态与完整结果（仅成功时返回权重，失败/中断无半份结果）")
def get_job(job_id: str):
    row = db().get_job(job_id)
    if row is None:
        raise HTTPException(404, "作业不存在或已因输入变更而永久失效")
    return _serialize_job(row)


@app.get("/result", tags=["计算"],
         summary="取用与当前输入一致的完整权重；无可用完整结果时返回 409")
def get_result():
    row = db().latest_succeeded_job()
    if row is None:
        raise HTTPException(
            409,
            "当前没有与现有输入一致的完整权重结果（尚未计算、计算中、冲突、"
            "数值失败或进程重启均不开放半份结果）",
        )
    data = _serialize_job(dict(row))
    # 一致性核对：作业快照必须与当前输入逐字段一致
    current = db().snapshot()
    snap = data["input_snapshot"]
    if snap.get("samples") != current["samples"] or \
            snap.get("targets") != current["targets"] or \
            snap.get("epoch") != current["epoch"]:
        # 理论上不会发生（epoch 变更即清除作业），双保险
        raise HTTPException(409, "结果与当前输入不一致，已拒绝取用，请重新计算")
    return data


@app.get("/health", tags=["系统"], summary="健康检查")
def health():
    return {"status": "ok", "epoch": db().get_epoch()}


# ---------------------------------------------------------------- 对账
def _reconcile_refused(payload: ReconcilePayload, reason: str) -> ReconcileResponse:
    """作业失效时的统一应答：说明原因，不给任何确认项。"""
    return ReconcileResponse(
        job_id=payload.job_id,
        epoch=db().get_epoch(),
        status="job_invalid",
        reason=reason,
        review_count=len(payload.reviews),
    )


@app.post(
    "/reconcile",
    tags=["对账"],
    summary="对账：用复核汇总判定两份录入清单的分歧是否仍需回访",
    response_description="对账结论；非 ok 时说明原因且不给确认项",
)
def reconcile_lists(payload: ReconcilePayload) -> ReconcileResponse:
    """接收成功且仍有效的权重作业号、两份逐人是/否清单和至多 12 项按
    地区/年龄段复核的加权赞成人数。

    规则：每人权重先四舍五入到 6 位小数，各组赞成权重加总后再四舍五入到
    2 位小数；保留两份清单一致的答复，每处分歧从两份清单中取一个值，所有
    复核合计须同时吻合。某处分歧在所有可行选法中取值都相同时，返回编号和
    可确认答复（confirmed）；其余分歧列出编号供回访（needs_callback）。
    汇总不相容（incompatible）、无法可靠判定（indeterminate）或作业失效
    （job_invalid）时说明原因，不给确认项。

    演示流程：先 PUT /samples 与 /targets，再 POST /calculate 并轮询
    /jobs/{job_id} 至 succeeded，然后把该 job_id 填入本接口请求体试调。
    """
    row = db().get_job(payload.job_id)
    if row is None:
        return _reconcile_refused(
            payload,
            "作业不存在或已因输入变更而永久失效（任何样本/目标修改都会删除"
            "历史作业），请重新 POST /calculate 后再对账",
        )
    if row["status"] != "succeeded":
        return _reconcile_refused(
            payload,
            f"作业当前状态为 {row['status']}，只有成功（succeeded）且仍有效"
            "的作业可用于对账",
        )
    if row["epoch"] != db().get_epoch():
        # 理论上不会发生（epoch 变更即删除作业），双保险
        return _reconcile_refused(
            payload, "作业已不属于当前输入版本，结果已失效，请重新计算"
        )

    snap = json.loads(row["input_snapshot"])
    samples = snap["samples"]
    result = json.loads(row["result"])
    w_by_id = {w["person_id"]: w["w"] for w in result["weights"]}
    weights = [w_by_id[s["person_id"]] for s in samples]

    # 两份清单必须各含快照中每人一条答复（不多不少、不重复——重复已由
    # 请求模型拦截）
    snapshot_ids = [s["person_id"] for s in samples]
    id_set = set(snapshot_ids)
    answers: dict[str, dict[str, bool]] = {}
    for name, lst in (("list_a", payload.list_a), ("list_b", payload.list_b)):
        got = [a.person_id for a in lst]
        if len(got) != len(snapshot_ids) or set(got) != id_set:
            raise HTTPException(
                422,
                f"{name} 必须恰好覆盖作业快照中的 {len(snapshot_ids)} 人："
                "每人一条答复，且不含快照外人员",
            )
        answers[name] = {a.person_id: a.answer for a in lst}

    disc_count = sum(
        1 for pid in snapshot_ids
        if answers["list_a"][pid] != answers["list_b"][pid]
    )
    if disc_count > reconcile_mod.MAX_DISCREPANCIES:
        raise HTTPException(
            422,
            f"两份清单分歧 {disc_count} 处，超过上限 "
            f"{reconcile_mod.MAX_DISCREPANCIES} 处",
        )

    reviews = [r.model_dump() for r in payload.reviews]
    outcome = reconcile_mod.reconcile(
        samples, weights, answers["list_a"], answers["list_b"], reviews
    )

    # 计算期间输入被修改的防御性复核：作业一旦失效，结论一并作废
    if db().get_epoch() != row["epoch"] or db().get_job(payload.job_id) is None:
        return _reconcile_refused(
            payload, "对账期间输入发生变更，作业已失效，结论作废，请重新计算后再对账"
        )

    return ReconcileResponse(
        job_id=payload.job_id,
        epoch=row["epoch"],
        status=outcome.status,
        reason=outcome.reason,
        review_count=len(payload.reviews),
        discrepancy_count=outcome.discrepancy_count,
        confirmed=[ConfirmedAnswer(person_id=p, answer=a)
                   for p, a in outcome.confirmed],
        needs_callback=outcome.needs_callback,
        incompatible_reviews=outcome.incompatible_reviews,
    )
