import threading
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ProgressState:
    task_id: str
    created_at: float
    status: str  # queued|running|done|error
    progress: float  # 0..1
    message: str
    result: Optional[dict] = None
    canceled: bool = False


_lock = threading.Lock()
_tasks: Dict[str, ProgressState] = {}


def new_task() -> ProgressState:
    tid = uuid.uuid4().hex
    st = ProgressState(
        task_id=tid,
        created_at=time.time(),
        status="queued",
        progress=0.0,
        message="queued",
        result=None,
    )
    with _lock:
        _tasks[tid] = st
    return st


def get_task(task_id: str) -> Optional[ProgressState]:
    with _lock:
        return _tasks.get(task_id)


def update(task_id: str, **kwargs) -> None:
    with _lock:
        st = _tasks.get(task_id)
        if not st:
            return
        for k, v in kwargs.items():
            setattr(st, k, v)


def finish(task_id: str, result: dict) -> None:
    update(task_id, status="done", progress=1.0, message="done", result=result)


def fail(task_id: str, error: str) -> None:
    update(task_id, status="error", message=error)


def cancel(task_id: str) -> None:
    # Mark as canceled; worker should stop as soon as it notices.
    update(task_id, canceled=True, status="error", message="已终止")


def cleanup(max_age_seconds: float = 3600.0) -> int:
    """Remove old tasks to prevent unbounded memory growth."""
    now = time.time()
    removed = 0
    with _lock:
        old = [k for k, v in _tasks.items() if now - v.created_at > max_age_seconds]
        for k in old:
            _tasks.pop(k, None)
            removed += 1
    return removed
