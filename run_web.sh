export APP_USERNAME=
export APP_PASSWORD=
export APP_SESSION_SECRET=""
nohup uvicorn app:app --host 127.0.0.1 --port 8000 --root-path /audio2text > output.log 2>&1 &
echo $! > uvicorn.pid
nohup ./frpc -c ./frpc_web.toml > frpc_web.log 2>&1 &
echo $! > frpc_web.pid
echo "Server is running on http://127.0.0.1:8000 or check output.log for details."
cat uvicorn.pid
echo "Username and password can be set via environment variables APP_USERNAME and APP_PASSWORD. Defaults are 'admin'/'admin'."
echo "Username = ${APP_USERNAME:-admin}"
echo "Password = ${APP_PASSWORD:-admin}"