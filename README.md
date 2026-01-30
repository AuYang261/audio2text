# Faster-Whisper Web Demo

*本项目基本由 VSCode Copilot with GPT5.2 完成（Vibe coding 大法好）*

> 声明：本项目的语音转写能力由 **Faster-Whisper**提供。

[![GitHub](https://img.shields.io/badge/GitHub-SYSTRAN%2Ffaster--whisper-181717?logo=github&logoColor=white)](https://github.com/SYSTRAN/faster-whisper)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Systran/faster--whisper--large--v3-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/Systran/faster-whisper-large-v3)

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

## 运行

### 1) 安装依赖

- Python 依赖：见 `requirements.txt`
- 系统依赖：建议安装 FFmpeg（用于解码 m4a/mp3 等）

### 2) 配置鉴权（必须）

当前 `auth.py` **强制要求**设置以下环境变量，否则服务会启动失败：

- `APP_USERNAME`
- `APP_PASSWORD`
- `APP_SESSION_SECRET`：建议用随机长字符串（用于 Cookie 签名）

示例（bash）：

```bash
export APP_USERNAME="admin"
export APP_PASSWORD="admin"
export APP_SESSION_SECRET="please-change-to-a-long-random-string"
```

### 3) 启动服务

项目提供了脚本：

- `run_web.sh`：启动 uvicorn（+ 可选 frpc）

> 当前默认使用 `--root-path /audio2text`，方便后续挂到站点子路径。

启动后访问（本机）：

- `http://127.0.0.1:8000/audio2text/`

服务端可用环境变量：

- `WHISPER_MODEL_SIZE`（默认 `large-v3`）
- `WHISPER_DEVICE`（默认 `cuda`，没 GPU 可设成 `cpu`）
- `WHISPER_COMPUTE_TYPE`（默认 `float16`，CPU 场景可用 `int8`）

## 使用说明（网页）

1. 打开页面并登录
2. 选择音频/视频文件，设置 `beam_size`
3. 点击“上传并转写”
4. 页面会轮询任务进度，并显示：
   - 识别语言/概率/时长
   - 转写文本
   - 分段时间戳结果
5. 转写过程中可点击“终止当前转写”
6. 刷新页面会尝试恢复上一次任务结果（服务未重启 + 任务未过期时）

## 部署：公网云服务器（宝塔）+ 内网算力机（audio2text）的 frp 转发

适用场景：

- 云服务器（公网、性能弱）跑宝塔/Nginx，只负责 HTTPS/路由
- 内网服务器（性能强）跑本项目 uvicorn
- 通过 frp 把云服务器的某个本地端口转发到内网的 uvicorn

### 架构

- 云服务器 A：运行 `frps`（对外/对内网机提供转发表）
- 内网服务器 B：运行 `frpc`，把本机 `127.0.0.1:8000` 映射到云服务器 A 的某个端口（例如 `18000`）
- 宝塔 Nginx：把 `/audio2text/` 反代到 `http://127.0.0.1:18000/`

### A（云服务器）准备 frps

- 放行安全组：`7000`（frps 控制端口）、`18000`（转发端口，方案示例）
- `frps.ini` 示例：

```ini
[common]
bind_port = 7000
```

启动 frps 后，等待内网机连接。

### B（内网服务器）配置 frpc

本项目目录中的 `frpc_web.toml` 示例：

```toml
serverAddr = "<云服务器A公网IP>"
serverPort = 7000

[[proxies]]
name = "audio2text_18000"
type = "tcp"
localIP = "127.0.0.1"
localPort = 8000
remotePort = 18000
```

内网机上启动：

```bash
# 1) 先启动 uvicorn（建议回环 + root-path）
uvicorn app:app --host 127.0.0.1 --port 8000 --root-path /audio2text

# 2) 再启动 frpc
frpc -c ./frpc_web.toml
```

### A（云服务器）宝塔 Nginx 配置

把 `/audio2text/` 反代到本机 18000（由 frps 接入并转发到内网机）：

```nginx
location ^~ /audio2text/ {
    proxy_pass http://127.0.0.1:18000/;

    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;

    client_max_body_size 1024m;
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;
    proxy_read_timeout 600s;
}
```

### 常见排错

- 宝塔报 502 且日志含 `connect() failed (111: Connection refused)`：
  - 说明 Nginx 连接不上 upstream（例如 `127.0.0.1:18000` 没有监听）
  - 先在云服务器 A 上 `ss -ltnp | grep 18000`，确认 frps 是否在监听
  - 再看 frps/frpc 日志是否已建立 proxy

- 访问 `/audio2text/` 跳转到 `/login`（不带前缀）导致 404：
  - 多半是后端重定向写死 `/login` 或未设置 `--root-path /audio2text`
  - 本项目已修复重定向，并建议始终使用 `--root-path /audio2text`

- 上传时报 `413 Content Too Large`：
  - 这是 Nginx/宝塔限制了上传体积（不是 FastAPI 报错）
  - 在对应 `location` 或 `server` 中增大限制，例如：

```nginx
client_max_body_size 1024m;
```

## 文件结构

- `app.py`: FastAPI 后端
- `templates/index.html`: 简单前端页面
- `templates/login.html`: 登录页
- `auth.py`: 简单 Cookie Session（HMAC 签名）
- `progress.py`: 内存任务状态（进度/取消/结果）
- `run_web.sh`: 启动脚本
- `frpc_web.toml`: FRP 客户端示例配置（内网服务器使用）
- `requirements.txt`: Python 依赖

## 接口

### 页面

- `GET /`：主页面（未登录会重定向到 `/login`，支持子路径部署）
- `GET /login`：登录页

### 登录

- `POST /api/login`：JSON `{username, password}`
- `POST /api/logout`

### 异步转写（网页使用）

- `POST /api/transcribe_async`
  - form-data:
    - `file`
    - `beam_size`（默认 5）
  - 返回：`{ task_id }`

- `GET /api/task/{task_id}`
  - 返回：`{ status, progress, message, result? }`
  - 说明：后端会尽早把 `result.language` 等信息写入任务，因此 **running 阶段也可能返回 result**。

- `POST /api/task/{task_id}/cancel`：终止任务（协作式取消）

### 同步转写（保留接口）

- `POST /api/transcribe`
  - form-data:
    - `file`: 上传文件
    - `beam_size`: 可选，默认 5

响应 JSON：

- `text`: 拼接后的文本
- `segments`: 带时间戳的行列表
- `language`, `language_probability`, `duration`, `duration_after_vad`
