# main.py
# MindVerse Router Proxy (Auto-detect FastAPI/Gradio)
# ENV:
#   ROUTER_URL   (required)
#   ROUTER_TYPE  ("fastapi" | "gradio" | "" -> auto)
#   HF_TOKEN     (optional, ถ้า Router/Space เป็น private)
#   ALLOW_ORIGINS (optional, comma-separated; default="*")
#   PORT         (optional, default=10000)
#
# Run: uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}

import os
import json
import time
from typing import Any, Dict, Optional, Set

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path)

import httpx
from fastapi import FastAPI, HTTPException, Body, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===================== ENV =====================
ROUTER_URL   = os.getenv("ROUTER_URL", "").rstrip("/")
ROUTER_TYPE  = os.getenv("ROUTER_TYPE", "").strip().lower()  # "fastapi" | "gradio" | ""(auto)
HF_TOKEN     = os.getenv("HF_TOKEN", "").strip()
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").strip()
PORT         = int(os.getenv("PORT", "10000"))

if not ROUTER_URL:
    raise RuntimeError("ENV ROUTER_URL is required")

# ===================== FastAPI =====================
app = FastAPI(title="MindVerse Router Proxy", version="2.0.2")

allow_origins = (
    [o.strip() for o in ALLOW_ORIGINS.split(",") if o.strip()]
    if ALLOW_ORIGINS else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Schemas =====================
class ChatPayload(BaseModel):
    input: str = Field(..., description="ข้อความจากผู้ใช้")
    meta: Optional[Dict[str, Any]] = Field(default=None)
    session: Optional[str] = Field(default="default")

class RagPayload(BaseModel):
    query: str = Field(..., description="ข้อความ/คำถาม")
    top_k: int = Field(default=4, ge=1, le=10)
    session: Optional[str] = Field(default="rag")

# ===================== Router Detection =====================
_detect_cache: Dict[str, Any] = {
    "type": None,          # "fastapi" | "gradio" | "unknown"
    "seen_paths": set(),   # เฉพาะ fastapi: รายการ paths
    "last_probe_ok": False,
    "ts": None,
    "error": None
}

def _auth_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h

async def detect_router() -> Dict[str, Any]:
    # หากบังคับชนิดจาก ENV ให้เชื่อก่อน
    if ROUTER_TYPE in ("fastapi", "gradio"):
        _detect_cache.update({
            "type": ROUTER_TYPE,
            "seen_paths": set(),
            "last_probe_ok": True,
            "ts": time.time(),
            "error": None
        })
        return _detect_cache

    headers = _auth_headers()
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
        # 1) FastAPI: /openapi.json
        try:
            r = await client.get(f"{ROUTER_URL}/openapi.json", headers=headers)
            if r.status_code == 200:
                try:
                    j = r.json()
                    if isinstance(j, dict) and "paths" in j:
                        paths = set(j["paths"].keys())
                        _detect_cache.update({
                            "type": "fastapi",
                            "seen_paths": paths,
                            "last_probe_ok": True,
                            "ts": time.time(),
                            "error": None
                        })
                        return _detect_cache
                except Exception as e:
                    _detect_cache["error"] = f"openapi parse error: {e}"
        except Exception as e:
            _detect_cache["error"] = f"openapi.json probe: {e}"

        # 2) Gradio: /config หรือ /info
        for hint in ("/config", "/info"):
            try:
                r = await client.get(f"{ROUTER_URL}{hint}", headers=headers)
                if r.status_code == 200:
                    _detect_cache.update({
                        "type": "gradio",
                        "seen_paths": set(),
                        "last_probe_ok": True,
                        "ts": time.time(),
                        "error": None
                    })
                    return _detect_cache
            except Exception as e:
                _detect_cache["error"] = f"gradio probe: {e}"

    _detect_cache.update({
        "type": "unknown",
        "seen_paths": set(),
        "last_probe_ok": False,
        "ts": time.time()
    })
    return _detect_cache

# ===================== Helpers =====================
async def _post_json(url: str, payload: Dict[str, Any]) -> httpx.Response:
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        return await client.post(url, headers=_auth_headers(), json=payload)

def _extract_text_from_router(resp_json: Any) -> str:
    # รองรับหลายรูปแบบ response
    if isinstance(resp_json, dict):
        for key in ("reply", "text", "message", "output", "result"):
            if key in resp_json and isinstance(resp_json[key], (str, int, float)):
                return str(resp_json[key])
        if "data" in resp_json:  # Gradio
            data = resp_json["data"]
            if isinstance(data, list) and data:
                return data[0] if isinstance(data[0], str) else json.dumps(data[0], ensure_ascii=False)
            return json.dumps(data, ensure_ascii=False)
        if "prediction" in resp_json:  # บาง Space
            return str(resp_json["prediction"])
        return json.dumps(resp_json, ensure_ascii=False)
    return str(resp_json)

async def router_call_chat(user_text: str) -> str:
    info = await detect_router()
    rtype: str = info["type"] or "unknown"
    paths: Set[str] = info["seen_paths"] or set()

    if rtype == "fastapi":
        if "/chat" in paths:
            r = await _post_json(f"{ROUTER_URL}/chat", {"input": user_text, "meta": {"source": "mv-proxy"}})
            if r.status_code < 400:
                return _extract_text_from_router(r.json())
        if "/rag-chat" in paths:
            r = await _post_json(f"{ROUTER_URL}/rag-chat", {"query": user_text, "top_k": 4, "session": "auto"})
            if r.status_code < 400:
                return _extract_text_from_router(r.json())

    if rtype == "gradio":
        try:
            r = await _post_json(f"{ROUTER_URL}/api/predict", {"data": [user_text]})
            if r.status_code < 400:
                return _extract_text_from_router(r.json())
        except Exception:
            pass
        for ep, body in (
            ("/chat", {"input": user_text}),
            ("/rag-chat", {"query": user_text, "top_k": 4}),
        ):
            try:
                r = await _post_json(f"{ROUTER_URL}{ep}", body)
                if r.status_code < 400:
                    return _extract_text_from_router(r.json())
            except Exception:
                continue

    # Unknown: ไล่ทีละแบบ
    for ep, body in (
        ("/chat", {"input": user_text}),
        ("/rag-chat", {"query": user_text, "top_k": 4}),
        ("/api/predict", {"data": [user_text]}),
    ):
        try:
            r = await _post_json(f"{ROUTER_URL}{ep}", body)
            if r.status_code < 400:
                return _extract_text_from_router(r.json())
        except Exception:
            continue

    raise HTTPException(status_code=502, detail="Router did not accept any known chat endpoints.")

async def router_call_rag(query_text: str, top_k: int = 4) -> str:
    info = await detect_router()
    rtype: str = info["type"] or "unknown"
    paths: Set[str] = info["seen_paths"] or set()

    if rtype == "fastapi":
        if "/rag-chat" in paths:
            r = await _post_json(f"{ROUTER_URL}/rag-chat", {"query": query_text, "top_k": top_k, "session": "rag"})
            if r.status_code < 400:
                return _extract_text_from_router(r.json())
        if "/chat" in paths:
            r = await _post_json(f"{ROUTER_URL}/chat", {"input": query_text, "meta": {"mode": "rag-fallback"}})
            if r.status_code < 400:
                return _extract_text_from_router(r.json())

    if rtype == "gradio":
        try:
            r = await _post_json(f"{ROUTER_URL}/api/predict", {"data": [query_text]})
            if r.status_code < 400:
                return _extract_text_from_router(r.json())
        except Exception:
            pass
        for ep, body in (
            ("/rag-chat", {"query": query_text, "top_k": top_k}),
            ("/chat", {"input": query_text}),
        ):
            try:
                r = await _post_json(f"{ROUTER_URL}{ep}", body)
                if r.status_code < 400:
                    return _extract_text_from_router(r.json())
            except Exception:
                continue

    for ep, body in (
        ("/rag-chat", {"query": query_text, "top_k": top_k}),
        ("/chat", {"input": query_text}),
        ("/api/predict", {"data": [query_text]}),
    ):
        try:
            r = await _post_json(f"{ROUTER_URL}{ep}", body)
            if r.status_code < 400:
                return _extract_text_from_router(r.json())
        except Exception:
            continue

    raise HTTPException(status_code=502, detail="Router did not accept any known RAG endpoints.")

# ===================== Endpoints =====================
@app.get("/")
async def root():
    return {"ok": True, "service": "MindVerse Router Proxy", "router_url": ROUTER_URL}

@app.get("/health")
async def health():
    info = await detect_router()
    return {
        "ok": True,
        "router_url": ROUTER_URL,
        "router_type_env": ROUTER_TYPE or "(auto)",
        "router_type_detected": info["type"],
        "seen_paths": sorted(list(info["seen_paths"])) if info["seen_paths"] else [],
        "last_probe_ok": info["last_probe_ok"],
        "ts": time.time(),
    }

@app.post("/chat")
async def chat(payload: ChatPayload = Body(...)):
    user = (payload.input or "").strip()
    if not user:
        raise HTTPException(status_code=400, detail="input is empty")
    text = await router_call_chat(user)
    return {"reply": text, "session": payload.session}

@app.post("/rag-chat")
async def rag_chat(payload: RagPayload = Body(...)):
    q = (payload.query or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is empty")
    text = await router_call_rag(q, top_k=payload.top_k)
    return {"reply": text, "session": payload.session}

# Proxy generic -> Router (แนบ query string ครบ)
@app.api_route("/router-proxy/{path:path}", methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS","HEAD"])
async def router_proxy(path: str, request: Request):
    target = f"{ROUTER_URL}/{path}".rstrip("/")
    if request.url.query:
        target = f"{target}?{request.url.query}"

    method = request.method.upper()
    out_headers = dict(request.headers)
    # ลบ header ที่ httpx จะจัดการเอง
    for k in ("host", "content-length", "transfer-encoding"):
        out_headers.pop(k, None)

    # บังคับ Content-Type หากต้นทางไม่ได้ส่งมา
    if "content-type" not in {k.lower(): v for k, v in out_headers.items()}:
        out_headers["Content-Type"] = "application/json"

    # ใช้ HF_TOKEN ถ้ามี (ทับของเดิม เพื่อความแน่นอน)
    if HF_TOKEN:
        out_headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        body = await request.body()
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            r = await client.request(method, target, headers=_auth_headers() | out_headers, content=body or None)
            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/octet-stream"),
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"proxy error: {e}")
        
