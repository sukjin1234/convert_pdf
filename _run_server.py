"""서버 런처 — 환경변수 PORT로 포트 지정 가능"""
import os
import uvicorn

port = int(os.getenv("PORT", "8000"))
uvicorn.run(
    "app:app",
    host="0.0.0.0",
    port=port,
    reload=False,
    log_level="info",
    timeout_keep_alive=600,
)
