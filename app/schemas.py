"""请求/响应模型与严格输入校验。"""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

MAX_SAMPLES = 200
MAX_TARGETS = 12
MAX_D = 100
MAX_CODE_LEN = 64
MAX_DISCREPANCIES = 30
MAX_REVIEWS = 12
MAX_APPROVE_TOTAL = 1_000_000


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _strict_int(name: str, minimum: int, maximum: int | None = None):
    # mode="before"：在 Pydantic 类型强转之前拦截，
    # 否则 True 会先被强转为 1，布尔冒充整数将无法识别。
    def _v(value):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 必须是整数")
        if value < minimum:
            raise ValueError(f"{name} 必须 >= {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"{name} 必须 <= {maximum}")
        return value
    return field_validator(name, mode="before")(_v)


def _strict_code(name: str):
    def _v(value):
        if isinstance(value, bool) or not isinstance(value, str):
            raise ValueError(f"{name} 必须是非空字符串")
        value = value.strip()
        if not value:
            raise ValueError(f"{name} 不能为空字符串")
        if len(value) > MAX_CODE_LEN:
            raise ValueError(f"{name} 长度须 <= {MAX_CODE_LEN}")
        return value
    return field_validator(name, mode="before")(_v)


class SampleIn(StrictModel):
    person_id: str = Field(description="样本人员编号，非空，单次提交内唯一")
    region: str = Field(description="地区代码")
    age_group: str = Field(description="年龄段代码")
    d: int = Field(description="正整数基础权重，1 <= d <= 100")

    _v_id = _strict_code("person_id")
    _v_region = _strict_code("region")
    _v_age = _strict_code("age_group")
    _v_d = _strict_int("d", 1, MAX_D)


class SamplesPayload(StrictModel):
    samples: list[SampleIn] = Field(description=f"至多 {MAX_SAMPLES} 人")

    @field_validator("samples")
    @classmethod
    def _check(cls, v: list[SampleIn]):
        if len(v) > MAX_SAMPLES:
            raise ValueError(f"样本数至多 {MAX_SAMPLES} 人")
        ids = [s.person_id for s in v]
        if len(set(ids)) != len(ids):
            raise ValueError("person_id 在本次提交中存在重复")
        return v


class TargetIn(StrictModel):
    dimension: Literal["region", "age_group"] = Field(
        description="目标维度：region 或 age_group"
    )
    code: str = Field(description="维度取值代码（对应样本的 region/age_group）")
    population: int = Field(description="非负整数人口总量")

    _v_code = _strict_code("code")
    _v_pop = _strict_int("population", 0)


class TargetsPayload(StrictModel):
    targets: list[TargetIn] = Field(description=f"至多 {MAX_TARGETS} 项")

    @field_validator("targets")
    @classmethod
    def _check(cls, v: list[TargetIn]):
        if len(v) > MAX_TARGETS:
            raise ValueError(f"目标至多 {MAX_TARGETS} 项")
        keys = [(t.dimension, t.code) for t in v]
        if len(set(keys)) != len(keys):
            raise ValueError("(dimension, code) 在本次提交中存在重复")
        return v


class WeightItem(BaseModel):
    person_id: str
    d: int
    w: float
    lower: float
    upper: float


class TargetTotalsItem(BaseModel):
    dimension: str
    code: str
    population: int
    weighted_total: float
    abs_error: float


class JobResult(BaseModel):
    job_id: str
    epoch: int
    status: str
    submitted_at: str
    finished_at: str | None = None
    input_snapshot: dict
    error: str | None = None
    # status=succeeded 时完整给出，其余状态为 None，绝不开放半份结果
    weights: list[WeightItem] | None = None
    objective: float | None = None
    residual_abs_sum: float | None = None
    residual_max_abs: float | None = None
    target_totals: list[TargetTotalsItem] | None = None
    # status=infeasible 时给出最小冲突目标子集
    infeasible_targets: list[dict] | None = None


# ---------------------------------------------------------------- 对账
class AnswerItem(StrictModel):
    person_id: str = Field(description="样本人员编号（须为作业快照中的人）")
    answer: StrictBool = Field(description="该人答复：true=是/赞成，false=否")

    _v_id = _strict_code("person_id")


class ReviewItem(StrictModel):
    dimension: Literal["region", "age_group"] = Field(
        description="复核维度：region 或 age_group"
    )
    code: str = Field(description="维度取值代码（对应样本的 region/age_group）")
    approve_total: float = Field(
        description="该组复核后的加权赞成人数（非负，至多两位小数）"
    )

    _v_code = _strict_code("code")

    @field_validator("approve_total", mode="before")
    @classmethod
    def _v_total(cls, v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError("approve_total 必须是数字")
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError("approve_total 必须是有限数")
        if v < 0:
            raise ValueError("approve_total 必须 >= 0")
        if v > MAX_APPROVE_TOTAL:
            raise ValueError(f"approve_total 必须 <= {MAX_APPROVE_TOTAL}")
        # 至多两位小数（复核合计本身即按两位小数填报）
        if Decimal(str(v)).as_tuple().exponent < -2:
            raise ValueError("approve_total 至多两位小数")
        return v


class ReconcilePayload(StrictModel):
    job_id: str = Field(
        description="成功且仍有效的权重作业号（POST /calculate 返回、"
                    "状态 succeeded 且未因输入修改而失效）"
    )
    list_a: list[AnswerItem] = Field(
        description="清单 A：作业快照中每人一条是/否答复"
    )
    list_b: list[AnswerItem] = Field(
        description="清单 B：同一批人员的另一份录入，与清单 A 分歧至多 30 处"
    )
    reviews: list[ReviewItem] = Field(
        description=f"按地区/年龄段复核的加权赞成人数，至多 {MAX_REVIEWS} 项"
    )

    _v_job = _strict_code("job_id")

    @field_validator("list_a", "list_b")
    @classmethod
    def _check_list(cls, v: list[AnswerItem]):
        if len(v) > MAX_SAMPLES:
            raise ValueError(f"每份清单至多 {MAX_SAMPLES} 条")
        ids = [a.person_id for a in v]
        if len(set(ids)) != len(ids):
            raise ValueError("同一清单内 person_id 不得重复")
        return v

    @field_validator("reviews")
    @classmethod
    def _check_reviews(cls, v: list[ReviewItem]):
        if len(v) > MAX_REVIEWS:
            raise ValueError(f"复核项至多 {MAX_REVIEWS} 项")
        keys = [(r.dimension, r.code) for r in v]
        if len(set(keys)) != len(keys):
            raise ValueError("(dimension, code) 在本次提交中存在重复")
        return v

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [{
                "job_id": "粘贴 POST /calculate 返回且已 succeeded 的 job_id",
                "list_a": [
                    {"person_id": "P1", "answer": True},
                    {"person_id": "P2", "answer": False},
                ],
                "list_b": [
                    {"person_id": "P1", "answer": True},
                    {"person_id": "P2", "answer": True},
                ],
                "reviews": [
                    {"dimension": "region", "code": "East",
                     "approve_total": 2.5},
                    {"dimension": "age_group", "code": "youth",
                     "approve_total": 1.75},
                ],
            }]
        },
    )


class ConfirmedAnswer(BaseModel):
    person_id: str
    answer: bool = Field(description="所有可行选法中一致的可确认答复")


class ReconcileResponse(BaseModel):
    job_id: str
    epoch: int | None = None
    status: str = Field(
        description="ok=对账完成；incompatible=复核汇总不相容；"
                    "indeterminate=无法可靠判定；job_invalid=作业失效"
    )
    reason: str | None = Field(
        default=None, description="非 ok 时的原因说明（此时不给任何确认项）"
    )
    review_count: int
    discrepancy_count: int | None = Field(
        default=None, description="两份清单的分歧处数（job_invalid 时为 null）"
    )
    confirmed: list[ConfirmedAnswer] = Field(
        default_factory=list,
        description="所有可行选法中取值都相同的分歧：编号 + 可确认答复",
    )
    needs_callback: list[str] = Field(
        default_factory=list, description="仍需回访的分歧编号"
    )
    incompatible_reviews: list[dict] | None = Field(
        default=None,
        description="status=incompatible 时的最小冲突复核子集"
                    "（删去任一项即恢复相容）",
    )
