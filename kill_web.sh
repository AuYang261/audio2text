kill -9 $(cat uvicorn.pid)
rm uvicorn.pid
echo "Server stopped."
kill -9 $(cat frpc_web.pid)
rm frpc_web.pid
echo "frpc stopped."