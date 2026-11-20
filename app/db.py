"""SQLite 存储层。

版本化规则（核心）：
- 每次成功提交样本或目标修改（PUT/DELETE）都会令全局 epoch +1；
- epoch 变更会删除全部旧作业记录，并置位取消事件，使在途计算永久失效
  （即使把数据改回原值也不会恢复旧结果，必须重新计算）；
- 结果只在作业成功完成时随作业原子写入，不存在“半份结果”；
- 进程重启后所有未完成作业标记为 interrupted，不开放任何部分结果。
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    epoch INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    person_id TEXT PRIMARY KEY,
    region TEXT NOT NULL,
    age_group TEXT NOT NULL,
    d INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS targets (
    dimension TEXT NOT NULL,
    code TEXT NOT NULL,
    population INTEGER NOT NULL,
    PRIMARY KEY (dimension, code)
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    epoch INTEGER NOT NULL,
    status TEXT NOT NULL,            -- queued|running|succeeded|infeasible|failed|interrupted
    submitted_at TEXT NOT NULL,
    finished_at TEXT,
    input_snapshot TEXT NOT NULL,    -- 提交时刻完整输入快照（JSON）
    result TEXT,                     -- 仅终态写入（完整权重+依据，或冲突集，或错误）
    error TEXT
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        self._write_lock = threading.RLock()
        self._cancel = threading.Event()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        cur = self._conn.execute("SELECT epoch FROM meta WHERE id=1")
        row = cur.fetchone()
        if row is None:
            self._conn.execute("INSERT INTO meta(id, epoch) VALUES (1, 1)")
            self._conn.commit()
            self.epoch = 1
        else:
            self.epoch = int(row["epoch"])
        # 重启：未完成作业一律中断，不允许读到半成品
        self._conn.execute(
            "UPDATE jobs SET status='interrupted', finished_at=?, "
            "error='process restarted' WHERE status IN ('queued','running')",
            (_now(),),
        )
        self._conn.commit()

    def close(self) -> None:
        with self._write_lock:
            self._conn.close()

    # ------------------------------------------------------------------ 读取
    def get_epoch(self) -> int:
        with self._write_lock:
            return self.epoch

    def list_samples(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT person_id, region, age_group, d FROM samples ORDER BY rowid"
        ).fetchall()
        return [dict(r) for r in rows]

    def list_targets(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT dimension, code, population FROM targets ORDER BY rowid"
        ).fetchall()
        return [dict(r) for r in rows]

    def snapshot(self) -> dict:
        with self._write_lock:
            return {
                "epoch": self.epoch,
                "samples": self.list_samples(),
                "targets": self.list_targets(),
            }

    def get_job(self, job_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def latest_succeeded_job(self) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE status='succeeded' "
            "ORDER BY finished_at DESC LIMIT 1"
        ).fetchone()
        return dict(row) if row else None

    def list_jobs(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT job_id, epoch, status, submitted_at, finished_at, error "
            "FROM jobs ORDER BY submitted_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ 写入
    def _bump_epoch_locked(self) -> int:
        """调用方必须持有 _write_lock。"""
        self.epoch += 1
        self._conn.execute("UPDATE meta SET epoch=? WHERE id=1", (self.epoch,))
        # 旧作业（含在途）永久失效
        self._conn.execute("DELETE FROM jobs")
        self._cancel.set()
        # 唤醒等待者后立即重建事件，供下一轮作业使用
        self._cancel = threading.Event()
        return self.epoch

    def replace_samples(self, samples: list[dict]) -> int:
        with self._write_lock:
            try:
                self._conn.execute("DELETE FROM samples")
                self._conn.executemany(
                    "INSERT INTO samples(person_id, region, age_group, d) "
                    "VALUES (:person_id, :region, :age_group, :d)",
                    samples,
                )
                epoch = self._bump_epoch_locked()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return epoch

    def replace_targets(self, targets: list[dict]) -> int:
        with self._write_lock:
            try:
                self._conn.execute("DELETE FROM targets")
                self._conn.executemany(
                    "INSERT INTO targets(dimension, code, population) "
                    "VALUES (:dimension, :code, :population)",
                    targets,
                )
                epoch = self._bump_epoch_locked()
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise
            return epoch

    # ------------------------------------------------------------------ 作业
    def current_cancel_event(self) -> threading.Event:
        with self._write_lock:
            return self._cancel

    def create_job(self, snapshot: dict) -> dict:
        with self._write_lock:
            job_id = uuid.uuid4().hex
            now = _now()
            epoch = self.epoch
            self._conn.execute(
                "INSERT INTO jobs(job_id, epoch, status, submitted_at, input_snapshot) "
                "VALUES (?,?,?,?,?)",
                (job_id, epoch, "queued", now, json.dumps(snapshot, ensure_ascii=False)),
            )
            self._conn.commit()
            return self.get_job(job_id)  # type: ignore[return-value]

    def mark_running(self, job_id: str) -> None:
        with self._write_lock:
            self._conn.execute(
                "UPDATE jobs SET status='running' WHERE job_id=? AND status='queued'",
                (job_id,),
            )
            self._conn.commit()

    def finish_job(self, job_id: str, status: str, result: dict | None,
                   error: str | None) -> None:
        with self._write_lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, result=?, error=? "
                "WHERE job_id=?",
                (status, _now(),
                 json.dumps(result, ensure_ascii=False) if result is not None else None,
                 error, job_id),
            )
            self._conn.commit()
