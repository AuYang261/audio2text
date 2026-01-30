import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from faster_whisper import WhisperModel
from tqdm import tqdm

from auth import (
    SESSION_COOKIE_NAME,
    get_logged_in_user,
    make_session_value,
    verify_credentials,
)

from progress import cleanup as progress_cleanup
from progress import cancel as progress_cancel
from progress import fail as progress_fail
from progress import finish as progress_finish
from progress import get_task as progress_get_task
from progress import new_task as progress_new_task
from progress import update as progress_update

APP_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))


def _env(key: str, default: str) -> str:
    v = os.getenv(key)
    return v if v not in (None, "") else default


MODEL_SIZE = _env("WHISPER_MODEL_SIZE", "large-v3")
DEVICE = _env("WHISPER_DEVICE", "cuda")
COMPUTE_TYPE = _env("WHISPER_COMPUTE_TYPE", "float16")
LAST_TASK_COOKIE = _env("APP_LAST_TASK_COOKIE", "whisper_last_task")

app = FastAPI(title="Faster-Whisper Web Demo")

_model: Optional[WhisperModel] = None


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def _fmt_time(sec: float) -> str:
    total = int(sec)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_file(
    audio_path: str,
    beam_size: int = 5,
    task_id: Optional[str] = None,
) -> dict:

    model = get_model()

    segments, info = model.transcribe(audio_path, beam_size=beam_size)
    print(
        "Detected language '%s' with probability %f, duration %.2f sec, after VAD %.2f sec"
        % (
            info.language,
            info.language_probability,
            info.duration,
            info.duration_after_vad,
        )
    )
    plain_parts = []
    verbose_lines = []

    total_dur = float(info.duration_after_vad or info.duration or 0.0)
    last_progress = -1.0

    if task_id:
        # 尽快把探测到的语言等 meta 写入任务，便于前端在转写开始阶段就显示。
        progress_update(
            task_id,
            status="running",
            progress=0.0,
            message="running",
            result={
                "language": info.language,
                "language_probability": info.language_probability,
                "duration": info.duration,
                "duration_after_vad": info.duration_after_vad,
            },
        )

    for s in segments:
        if task_id:
            st = progress_get_task(task_id)
            if st is not None and getattr(st, "canceled", False):
                raise RuntimeError("已终止")

        plain_parts.append(s.text)
        verbose_lines.append(f"[{_fmt_time(s.start)} -> {_fmt_time(s.end)}] {s.text}")

        if task_id and total_dur > 0:
            p = max(0.0, min(1.0, float(s.end) / total_dur))
            # Reduce update frequency (avoid too many writes).
            if p - last_progress >= 0.01 or p == 1.0:
                last_progress = p
                progress_update(
                    task_id,
                    progress=p,
                    message=f"{p * 100:.0f}%",
                )

    return {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
        "duration_after_vad": info.duration_after_vad,
        "text": " ".join(plain_parts).strip(),
        "segments": verbose_lines,
    }


@app.get("/api/task/{task_id}")
async def api_task_status(request: Request, task_id: str):
    if not get_logged_in_user(request):
        return JSONResponse(status_code=401, content={"error": "未登录"})

    st = progress_get_task(task_id)
    if st is None:
        return JSONResponse(status_code=404, content={"error": "任务不存在或已过期"})

    payload = {
        "task_id": st.task_id,
        "status": st.status,
        "progress": st.progress,
        "message": st.message,
    }
    # 允许在 running 阶段就返回 result（例如 language/language_probability 等提前可用信息）。
    if st.result is not None:
        payload["result"] = st.result
    return payload


@app.post("/api/task/{task_id}/cancel")
async def api_task_cancel(request: Request, task_id: str):
    if not get_logged_in_user(request):
        return JSONResponse(status_code=401, content={"error": "未登录"})

    st = progress_get_task(task_id)
    if st is None:
        return JSONResponse(status_code=404, content={"error": "任务不存在或已过期"})

    progress_cancel(task_id)
    return {"ok": True}


@app.post("/api/transcribe_async")
async def api_transcribe_async(
    request: Request,
    file: UploadFile = File(...),
    beam_size: int = Form(5),
):
    if not get_logged_in_user(request):
        return JSONResponse(status_code=401, content={"error": "未登录"})

    progress_cleanup()

    suffix = Path(file.filename or "audio").suffix
    st = progress_new_task()
    progress_update(st.task_id, message="uploading")

    # We keep the uploaded file in a temp dir that survives until the background thread finishes.
    td = tempfile.TemporaryDirectory(prefix="whisper_upload_")
    out_path = Path(td.name) / f"upload{suffix}"
    with out_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    def _worker():
        try:
            progress_update(
                st.task_id, status="running", progress=0.0, message="running"
            )
            result = transcribe_file(
                str(out_path), beam_size=int(beam_size), task_id=st.task_id
            )
            progress_finish(st.task_id, result)
        except Exception as e:
            progress_fail(st.task_id, str(e))
        finally:
            try:
                td.cleanup()
            except Exception:
                pass

    threading.Thread(target=_worker, daemon=True).start()

    resp = JSONResponse(content={"task_id": st.task_id})
    # Save last task id so page refresh can restore results.
    resp.set_cookie(
        key=LAST_TASK_COOKIE,
        value=st.task_id,
        httponly=True,
        samesite="lax",
    )
    return resp


@app.get("/api/me")
async def api_me(request: Request):
    if not get_logged_in_user(request):
        return JSONResponse(status_code=401, content={"error": "未登录"})

    last_task_id = request.cookies.get(LAST_TASK_COOKIE, "")
    return {"last_task_id": last_task_id}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    user = get_logged_in_user(request)
    if not user:
        root_path = request.scope.get("root_path", "")
        return RedirectResponse(url=f"{root_path}/login", status_code=302)

    last_task_id = request.cookies.get(LAST_TASK_COOKIE, "")
    return TEMPLATES.TemplateResponse(
        "index.html",
        {
            "request": request,
            "default_model": MODEL_SIZE,
            "default_device": DEVICE,
            "default_compute": COMPUTE_TYPE,
            "last_task_id": last_task_id,
        },
    )


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    # Already logged in? Go straight to the main page.
    if get_logged_in_user(request):
        root_path = request.scope.get("root_path", "")
        return RedirectResponse(url=f"{root_path}/", status_code=302)
    return TEMPLATES.TemplateResponse("login.html", {"request": request})


@app.post("/api/login")
async def api_login(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "无效的请求体"})

    username = str(payload.get("username", "")).strip()
    password = str(payload.get("password", ""))
    if not username or not password:
        return JSONResponse(status_code=400, content={"error": "请输入用户名和密码"})

    if not verify_credentials(username, password):
        return JSONResponse(status_code=401, content={"error": "用户名或密码错误"})

    resp = JSONResponse(content={"ok": True})
    # HttpOnly cookie so JS can't read it.
    resp.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=make_session_value(username),
        httponly=True,
        samesite="lax",
    )
    return resp


@app.post("/api/logout")
async def api_logout():
    resp = JSONResponse(content={"ok": True})
    resp.delete_cookie(key=SESSION_COOKIE_NAME)
    return resp


@app.post("/api/transcribe")
async def api_transcribe(
    request: Request,
    file: UploadFile = File(...),
    beam_size: int = Form(5),
):
    # Protect API endpoint as well (not only the page).
    if not get_logged_in_user(request):
        return JSONResponse(status_code=401, content={"error": "未登录"})

    suffix = Path(file.filename or "audio").suffix
    with tempfile.TemporaryDirectory(prefix="whisper_upload_") as td:
        out_path = Path(td) / f"upload{suffix}"
        with out_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            result = transcribe_file(str(out_path), beam_size=int(beam_size))
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return result
