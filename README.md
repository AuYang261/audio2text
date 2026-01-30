# Faster-Whisper Web Demo

这是一个基于 `faster-whisper` 的本地网页：上传音频/视频文件，然后在浏览器里拿到转写文本（含时间戳分段）。

## 功能

- 网页端上传录音文件（`wav/mp3/m4a/flac/mp4` 等）
- 服务端调用 `faster_whisper.WhisperModel.transcribe()`
- 返回：
  - 纯文本（拼接）
  - 带时间戳的分段列表
  - 检测语言、时长信息

## 安装

建议使用虚拟环境。

- 安装依赖：见 `requirements.txt`

> 说明：`faster-whisper` 依赖音频解码工具（常见是 FFmpeg）。如果你的系统没有 FFmpeg，可能会出现无法解码 m4a/mp3 的报错。

## 运行

- 启动 Web 服务后，浏览器打开 `http://127.0.0.1:8000/`

服务端可用环境变量：

- `WHISPER_MODEL_SIZE`（默认 `large-v3`）
- `WHISPER_DEVICE`（默认 `cuda`，没 GPU 可设成 `cpu`）
- `WHISPER_COMPUTE_TYPE`（默认 `float16`，CPU 场景可用 `int8`）

鉴权相关环境变量：

- `APP_USERNAME`（默认 `admin`）
- `APP_PASSWORD`（默认 `admin`）
- `APP_SESSION_SECRET`（默认 `change-me-please`，建议务必修改）

## 接口

- `POST /api/transcribe`
  - form-data:
    - `file`: 上传文件
    - `beam_size`: 可选，默认 5

响应 JSON：

- `text`: 拼接后的文本
- `segments`: 带时间戳的行列表
- `language`, `language_probability`, `duration`, `duration_after_vad`

## 文件结构

- `app.py`: FastAPI 后端
- `templates/index.html`: 简单前端页面
- `requirements.txt`: Python 依赖
