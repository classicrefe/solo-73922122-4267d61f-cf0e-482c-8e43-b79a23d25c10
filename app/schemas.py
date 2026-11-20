"""请求/响应模型与严格输入校验。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_SAMPLES = 200
MAX_TARGETS = 12
MAX_D = 100
MAX_CODE_LEN = 64


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
