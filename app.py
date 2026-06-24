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

# 空闲多久后自动释放模型（秒）。0 表示禁用。
MODEL_IDLE_UNLOAD_SECONDS = int(_env("WHISPER_MODEL_IDLE_UNLOAD_SECONDS", "0"))

app = FastAPI(title="Faster-Whisper Web Demo")


class _StripPrefixMiddleware:
    """Allow the app to be accessed with or without the /audio2text URL prefix.

    Supported cases:
    - direct access:              GET /                  → path stays /, root_path = ""
    - prefixed access:            GET /audio2text        → path becomes /, root_path = /audio2text
    - proxy strips prefix first:  GET /                  → path stays /, root_path = /audio2text
                                   (detected from forwarded headers)
    """

    _PREFIX = "/audio2text"
    _FORWARDED_PREFIX_HEADER = b"x-forwarded-prefix"
    _FORWARDED_PROTO_HEADER = b"x-forwarded-proto"
    _FORWARDED_HOST_HEADER = b"x-forwarded-host"
    _FORWARDED_FOR_HEADER = b"x-forwarded-for"

    def __init__(self, inner):
        self._inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            p: str = scope.get("path", "/")
            scope = dict(scope)

            headers = {k.lower(): v for k, v in scope.get("headers", [])}
            forwarded_prefix = headers.get(self._FORWARDED_PREFIX_HEADER, b"").decode("latin-1").rstrip("/")
            is_forwarded = any(
                headers.get(h)
                for h in (
                    self._FORWARDED_PROTO_HEADER,
                    self._FORWARDED_HOST_HEADER,
                    self._FORWARDED_FOR_HEADER,
                )
            )

            if p == self._PREFIX or p.startswith(self._PREFIX + "/"):
                new_path = p[len(self._PREFIX) :] or "/"
                scope["path"] = new_path
                scope["raw_path"] = new_path.encode("latin-1")
                scope["root_path"] = self._PREFIX
            elif not scope.get("root_path") and (
                forwarded_prefix == self._PREFIX or (not forwarded_prefix and is_forwarded)
            ):
                scope["root_path"] = self._PREFIX

        await self._inner(scope, receive, send)


_model: Optional[WhisperModel] = None
_model_lock = threading.Lock()
_model_last_used_at = 0.0


def _touch_model_last_used() -> None:
    global _model_last_used_at
    _model_last_used_at = time.time()


def _try_unload_model_if_idle(now: float) -> bool:
    """空闲超过阈值时卸载模型，释放显存；返回是否发生卸载。"""
    global _model
    if MODEL_IDLE_UNLOAD_SECONDS <= 0:
        return False

    with _model_lock:
        if _model is None:
            return False

        idle = now - float(_model_last_used_at or 0.0)
        if idle < MODEL_IDLE_UNLOAD_SECONDS:
            return False

        # Drop reference; memory will be released when GC runs.
        _model = None

    # Best-effort CUDA cache cleanup.
    try:
        if str(DEVICE).lower().startswith("cuda"):
            import torch  # type: ignore

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Some builds may not have ipc_collect; keep it best-effort.
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
    except Exception:
        pass

    return True


def _start_model_reaper_thread() -> None:
    if MODEL_IDLE_UNLOAD_SECONDS <= 0:
        return

    def _reaper():
        interval = max(15, min(60, MODEL_IDLE_UNLOAD_SECONDS // 3))
        while True:
            time.sleep(interval)
            _try_unload_model_if_idle(time.time())

    threading.Thread(target=_reaper, daemon=True, name="whisper-model-reaper").start()


def get_model() -> WhisperModel:
    global _model
    with _model_lock:
        if _model is None:
            _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
        _touch_model_last_used()
        return _model


# Start background model reaper (idle unload).
_start_model_reaper_thread()


def _fmt_time(sec: float) -> str:
    total = int(sec)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def transcribe_file(
    audio_path: str,
    beam_size: int = 5,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    task_id: Optional[str] = None,
) -> dict:

    model = get_model()
    _touch_model_last_used()

    segments, info = model.transcribe(
        audio_path,
        beam_size=beam_size,
        language=language or None,
        initial_prompt=initial_prompt or None,
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
        _touch_model_last_used()
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
async def api_transcribe_async(request: Request):
    """Phase 1 – create a transcription task (no file).  Returns immediately."""
    if not get_logged_in_user(request):
        return JSONResponse(status_code=401, content={"error": "未登录"})

    progress_cleanup()

    # Parse optional parameters from JSON body --------------------------------
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    beam_size = int(payload.get("beam_size", 5))
    language = str(payload.get("language", "")).strip()
    initial_prompt = str(payload.get("initial_prompt", "")).strip()
    filename = str(payload.get("filename", "")).strip()

    upload_dir = tempfile.mkdtemp(prefix="whisper_upload_")
    st = progress_new_task()
    st.params = {
        "beam_size": beam_size,
        "language": language,
        "initial_prompt": initial_prompt,
        "filename": filename,
        "upload_dir": upload_dir,
        "chunks_received": [],
    }
    progress_update(st.task_id, status="awaiting_upload", message="等待文件上传…")

    resp = JSONResponse(content={"task_id": st.task_id})
    resp.set_cookie(
        key=LAST_TASK_COOKIE,
        value=st.task_id,
        httponly=True,
        samesite="lax",
    )
    return resp


@app.post("/api/upload_chunk/{task_id}")
async def api_upload_chunk(
    request: Request,
    task_id: str,
    chunk: int = Form(0),
    file: UploadFile = File(...),
):
    """Phase 2 – upload one chunk of the audio file."""
    if not get_logged_in_user(request):
        return JSONResponse(status_code=401, content={"error": "未登录"})

    st = progress_get_task(task_id)
    if st is None:
        return JSONResponse(status_code=404, content={"error": "任务不存在或已过期"})
    if getattr(st, "canceled", False):
        return JSONResponse(status_code=400, content={"error": "任务已取消"})

    upload_dir = (st.params or {}).get("upload_dir")
    if not upload_dir:
        return JSONResponse(status_code=400, content={"error": "任务未初始化上传目录"})

    chunk_path = os.path.join(upload_dir, f"chunk_{chunk:06d}")
    with open(chunk_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if st.params is not None:
        st.params.setdefault("chunks_received", []).append(chunk)

    progress_update(
        task_id,
        status="awaiting_upload",
        message=f"已接收 chunk {chunk}",
    )
    return {"ok": True, "chunk": chunk}


@app.post("/api/upload_done/{task_id}")
async def api_upload_done(request: Request, task_id: str):
    """Phase 3 – all chunks uploaded; assemble file and start transcription."""
    if not get_logged_in_user(request):
        return JSONResponse(status_code=401, content={"error": "未登录"})

    st = progress_get_task(task_id)
    if st is None:
        return JSONResponse(status_code=404, content={"error": "任务不存在或已过期"})

    params = st.params or {}
    upload_dir = params.get("upload_dir")
    if not upload_dir:
        return JSONResponse(status_code=400, content={"error": "上传目录不存在"})

    beam_size = int(params.get("beam_size", 5))
    language = params.get("language") or None
    initial_prompt = params.get("initial_prompt") or None
    original_filename = params.get("filename", "audio")

    suffix = Path(original_filename).suffix
    out_path = os.path.join(upload_dir, f"upload{suffix}")

    # Assemble chunks in order --------------------------------------------------
    chunks_received = sorted(params.get("chunks_received", []))
    if not chunks_received:
        return JSONResponse(status_code=400, content={"error": "未收到任何文件块"})

    with open(out_path, "wb") as outf:
        for i in chunks_received:
            chunk_path = os.path.join(upload_dir, f"chunk_{i:06d}")
            with open(chunk_path, "rb") as inf:
                shutil.copyfileobj(inf, outf)
            # Free disk space early
            try:
                os.unlink(chunk_path)
            except Exception:
                pass

    def _worker() -> None:
        try:
            progress_update(
                task_id, status="running", progress=0.0, message="running"
            )
            result = transcribe_file(
                out_path,
                beam_size=beam_size,
                language=language,
                initial_prompt=initial_prompt,
                task_id=task_id,
            )
            progress_finish(task_id, result)
        except Exception as e:
            progress_fail(task_id, str(e))
        finally:
            try:
                shutil.rmtree(upload_dir, ignore_errors=True)
            except Exception:
                pass

    threading.Thread(target=_worker, daemon=True).start()
    return {"ok": True}


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
    language: str = Form(""),
    initial_prompt: str = Form(""),
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
            result = transcribe_file(
                str(out_path),
                beam_size=int(beam_size),
                language=language,
                initial_prompt=initial_prompt,
            )
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    return result


# Wrap after all routes so the FastAPI instance is fully configured first.
app = _StripPrefixMiddleware(app)  # noqa: F811
