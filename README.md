# 回访问卷样本权重校正 API

调查数据工程师用于校正回访问卷样本偏差的服务：为每位受访者求一个校正权重
`w`，使各目标人群（按地区或年龄段）的权重合计等于给定人口总量，同时保持
个人权重界限 `0.5d ≤ w ≤ 2d`，并最小化与基础权重 `d` 的加权二次距离
`Σ (w−d)² / d`。

## 数学模型

给定 n 名样本（n ≤ 200），基础权重 `d_i ∈ {1,…,100}`，以及 m 项目标
（m ≤ 12，维度为 `region` 或 `age_group`）：

```
min   Σ_i (w_i − d_i)² / d_i
s.t.  A w = p                # 每行一个目标人群，A 为 0/1 成员矩阵（允许线性冗余）
      0.5 d_i ≤ w_i ≤ 2 d_i
```

- **可行性与求解分离**：先用 SciPy `linprog`（HiGHS）解松弛 LP 独立判定可行性；
  可行后再用 SLSQP 解凸 QP，残差不达标时以活动集 KKT 精修、`trust-constr` 兜底。
  **数值求解失败返回 `failed`，绝不被当作无解。**
- **无解诊断**：不可行时返回一个最小不可行目标子集（IIS，贪心删除法），
  删去其中任意一项目标后其余目标即在个人界限下恢复可行；个人权重界限始终保留。
- **精度**：每个目标人群的合计误差 `|Σw − population| ≤ 1e-6`（响应中逐项目回报）。
- **冗余目标**：约束矩阵行线性相关（如“全员合计 = 各地区之和”）时正常求解。

## 失效与一致性语义

- 计算在后台线程池执行，HTTP 请求仅入队，**不阻塞样本/目标更正**。
- 任何一次已提交的样本或目标修改（即使内容与原来完全相同、即使之后又改回去）
  都会使全局版本号 `epoch` 加一：所有历史作业被删除、在途计算通过取消事件
  永久放弃且不写结果。旧结果不会因“改回原值”而恢复，必须重新计算。
- 只有作业成功结束时才原子写入**完整**权重结果；`queued/running/failed/
  infeasible/interrupted` 状态均不返回任何半份权重。
- 进程重启后，未完成作业标记为 `interrupted`；`GET /result` 仅提供与当前
  输入逐字段一致的完整结果，否则返回 `409`。

## 录入对账（POST /reconcile）

回访中心对同一题收到两份录入清单时，用复核过的汇总判断哪些分歧仍需回访。
请求接收：

- `job_id`：**成功且仍有效**的权重作业号（`succeeded` 且未因输入修改失效）；
- `list_a` / `list_b`：两份逐人是/否清单，各含该作业快照中**每人一条**答复，
  两份清单分歧至多 **30** 处；
- `reviews`：至多 **12** 项按 `region` 或 `age_group` 复核的加权赞成人数
  （非负、至多两位小数）。

舍入规则：每人权重先**四舍五入到 6 位小数**，各组赞成权重加总后再
**四舍五入到 2 位小数**（恰好 x.xx5 时入高的一侧）。保留两份清单一致的
答复，每处分歧从两份清单中取一个值，所有复核合计须**同时**吻合。

响应 `status`：

- `ok`：`confirmed` 给出在所有可行选法中取值都相同的分歧（编号 + 可确认
  答复），`needs_callback` 列出其余仍需回访的分歧编号；
- `incompatible`：复核汇总不相容，`incompatible_reviews` 给出最小冲突复核
  子集（删去任一项即恢复相容）；
- `indeterminate`：数值求解失败，无法可靠判定（不混同于不相容）；
- `job_invalid`：作业不存在、未成功或已失效。

后三种情况均说明原因（`reason`）且**不给任何确认项**。清单与快照人员不
一致、分歧超 30 处、复核超 12 项等输入问题返回 `422`。

## 快速启动（Docker Compose）

```bash
docker compose up --build
# API 与交互式文档：
#   http://localhost:8000/docs      （Swagger UI，可直接试调）
#   http://localhost:8000/redoc
#   http://localhost:8000/openapi.json
```

SQLite 数据库保存在命名卷 `weights-data`（容器内 `/data/weights.db`）。

## 本地无 Docker 运行（可选）

```bash
python3 -m pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 接口一览

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| PUT | `/samples` | 整体替换样本（≤200 人；使旧结果与在途计算失效） |
| GET | `/samples` | 查看当前样本与 epoch |
| PUT | `/targets` | 整体替换人口目标（≤12 项；同上失效） |
| GET | `/targets` | 查看当前目标 |
| POST | `/calculate` | 提交后台计算，返回 `job_id`（202） |
| GET | `/jobs/{job_id}` | 查询作业；成功时含完整权重与输入快照 |
| GET | `/jobs` | 最近作业列表 |
| GET | `/result` | 取用与当前输入一致的完整权重，否则 409 |
| POST | `/reconcile` | 对账：两份录入清单 + 复核汇总 → 可确认/需回访分歧 |
| GET | `/health` | 健康检查 |

输入为严格整数：`d` 必须是 1–100 的整数（布尔值、小数、越界均 422），
`population` 为非负整数；`person_id` 与 `(dimension, code)` 单次提交内不得重复。

## 示例与演示

`examples/` 提供可行与冲突两套数据：

```bash
# 先启动服务，然后：
python3 examples/demo.py                      # 默认 http://127.0.0.1:8000
python3 examples/demo.py http://host:8000
```

手动调用：

```bash
curl -X PUT localhost:8000/samples -H 'Content-Type: application/json' \
  --data @examples/feasible_samples.json
curl -X PUT localhost:8000/targets -H 'Content-Type: application/json' \
  --data @examples/feasible_targets.json
curl -X POST localhost:8000/calculate
curl localhost:8000/jobs/<job_id>
curl localhost:8000/result
```

冲突示例（`conflict_*.json`）：地区 R1 要求合计 4，但 R1 两人权重上限之和
仅为 4，与年龄段 a1 要求合计 1 无法同时满足，返回两项目标组成的最小冲突集。

演示脚本最后会接着做一次**对账**：用场景 1 的成功作业，对两份存在分歧的
录入清单和按地区/年龄段复核的加权赞成合计调用 `POST /reconcile`，展示
可确认分歧与需回访分歧。也可以在 `/docs` 页面中对 `/reconcile` 点击
“Try it out”，把 `/calculate` 返回的 `job_id` 填入请求体直接试调。

## 测试

```bash
pip install pytest
pytest -q
```

覆盖：随机可行案例的 KKT 最优性、200×12 满规模、冗余等式、零/未覆盖目标、
IIS 最小性、边界解、严格校验、失效语义、在途修改取消、重启中断、
数值失败与无解的区分；对账的舍入边界（6 位/2 位四舍五入）、分歧确认与
回访判定、汇总不相容最小冲突子集、作业失效、数值失败与输入校验。
