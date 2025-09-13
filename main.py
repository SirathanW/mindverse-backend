# main.py
# MindVerse Router Proxy (Auto-detect FastAPI/Gradio) - Improved for HF Space
# ENV:
#   ROUTER_URL    (required)  e.g. https://org-space.hf.space
#   ROUTER_TYPE   ("fastapi" | "gradio" | "" -> auto)
#   HF_TOKEN      (optional, ถ้า Router/Space เป็น private)
#   ALLOW_ORIGINS (optional, comma-separated; default="*")
#   PORT          (optional, default=10000)
#
# Run: uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}

import os
import json
import time
from typing import Any, Dict, Optional, Set, List

from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path)

import httpx
from fastapi import FastAPI, HTTPException, Body, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===================== ENV =====================
ROUTER_URL    = os.getenv("ROUTER_URL", "").rstrip("/")
ROUTER_TYPE   = os.getenv("ROUTER_TYPE", "").strip().lower()  # "fastapi" | "gradio" | ""(auto)
HF_TOKEN      = os.getenv("HF_TOKEN", "").strip()
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").strip()
PORT          = int(os.getenv("PORT", "10000"))

if not ROUTER_URL:
    raise RuntimeError("ENV ROUTER_URL is required")

# ===================== FastAPI =====================
app = FastAPI(title="MindVerse Router Proxy", version="2.1.0")

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

# ===================== Detection Cache =====================
_detect_cache: Dict[str, Any] = {
    "type": None,            # "fastapi" | "gradio" | "unknown"
    "seen_paths": set(),     # สำหรับ fastapi: รายการ paths
    "gradio_fn_indices": [], # รายการ fn_index ที่พบจาก /config|/info
    "last_probe_ok": False,
    "ts": None,
    "error": None
}

# ===================== Helpers: headers / http =====================
def _auth_headers() -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    if HF_TOKEN:
        h["Authorization"] = f"Bearer {HF_TOKEN}"
    return h

async def _get_json(url: str, timeout: float = 10.0) -> Optional[dict]:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        r = await client.get(url, headers=_auth_headers())
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return None
        return None

async def _post_json(url: str, payload: Dict[str, Any], timeout: float = 60.0) -> httpx.Response:
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
        return await client.post(url, headers=_auth_headers(), json=payload)

# ===================== Gradio utils =====================
def _parse_gradio_fn_indices(cfg: dict) -> List[int]:
    """อ่าน fn_index จาก /config หรือ /info ของ Gradio (รองรับหลายรูปแบบเวอร์ชัน)"""
    fn_list: List[int] = []
    if not isinstance(cfg, dict):
        return fn_list

    # รูปแบบเก่า: dependencies / dependencies_cache
    deps = cfg.get("dependencies") or cfg.get("dependencies_cache")
    if isinstance(deps, list):
        for d in deps:
            try:
                idx = d.get("fn_index")
                if isinstance(idx, int):
                    fn_list.append(idx)
            except Exception:
                continue

    # รูปแบบใหม่ของ Gradio v4+: routes -> path = "/api/predict/{fn_index}"
    routes = cfg.get("routes")
    if isinstance(routes, list):
        for rt in routes:
            path = rt.get("path") if isinstance(rt, dict) else None
            if isinstance(path, str) and path.startswith("/api/predict/"):
                try:
                    idx = int(path.rsplit("/", 1)[-1])
                    fn_list.append(idx)
                except Exception:
                    pass

    # unique + keep order
    seen = set()
    uniq = []
    for x in fn_list:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq

def _extract_text_from_router(resp_json: Any) -> str:
    """พยายามดึงข้อความหลักจาก response หลากหลายรูปแบบ"""
    if isinstance(resp_json, dict):
        # เคสทั่วไป: ฝั่ง FastAPI/Proxy มักคืนคีย์เหล่านี้
        for key in ("reply", "text", "message", "output", "result", "generated_text"):
            v = resp_json.get(key, None)
            if isinstance(v, (str, int, float)):
                return str(v)

        # เคส Gradio: {"data": [...]}
        if "data" in resp_json:
            data = resp_json["data"]
            if isinstance(data, list) and data:
                first = data[0]
                if isinstance(first, (str, int, float)):
                    return str(first)
                if isinstance(first, dict):
                    for k in ("text", "label", "value", "message"):
                        if k in first and isinstance(first[k], (str, int, float)):
                            return str(first[k])
                try:
                    return json.dumps(first, ensure_ascii=False)
                except Exception:
                    return str(first)
            try:
                return json.dumps(data, ensure_ascii=False)
            except Exception:
                return str(data)

        # บาง Space: {"prediction": "..."}
        if "prediction" in resp_json:
            return str(resp_json["prediction"])

        # สุดท้าย: แปลงทั้ง dict
        try:
            return json.dumps(resp_json, ensure_ascii=False)
        except Exception:
            return str(resp_json)
    return str(resp_json)

async def _gradio_predict_try(fn_index: Optional[int], user_text: str) -> Optional[str]:
    """ลองเรียก /api/predict/{fn_index} (หรือ /api/predict ถ้าระบุ fn_index ไม่ได้)"""
    try:
        if fn_index is None:
            url = f"{ROUTER_URL}/api/predict"
        else:
            url = f"{ROUTER_URL}/api/predict/{fn_index}"
        r = await _post_json(url, {"data": [user_text]})
        if r.status_code < 400:
            return _extract_text_from_router(r.json())
    except Exception:
        pass
    return None

# ===================== Router Detection =====================
async def detect_router() -> Dict[str, Any]:
    # บังคับชนิดจาก ENV
    if ROUTER_TYPE in ("fastapi", "gradio"):
        _detect_cache.update({
            "type": ROUTER_TYPE,
            "last_probe_ok": True,
            "ts": time.time(),
            "error": None
        })
        return _detect_cache

    info: Dict[str, Any] = {
        "type": None,
        "seen_paths": set(),
        "gradio_fn_indices": [],
        "last_probe_ok": False,
        "ts": time.time(),
        "error": None
    }

    # 1) FastAPI: /openapi.json
    try:
        j = await _get_json(f"{ROUTER_URL}/openapi.json", timeout=8.0)
        if isinstance(j, dict) and "paths" in j:
            info.update({
                "type": "fastapi",
                "seen_paths": set(j["paths"].keys()),
                "last_probe_ok": True
            })
            _detect_cache.update(info)
            return _detect_cache
    except Exception as e:
        info["error"] = f"openapi.json probe error: {e}"

    # 2) Gradio: /config หรือ /info
    for hint in ("/config", "/info"):
        try:
            cfg = await _get_json(f"{ROUTER_URL}{hint}", timeout=8.0)
            if isinstance(cfg, dict):
                fns = _parse_gradio_fn_indices(cfg)
                info.update({
                    "type": "gradio",
                    "gradio_fn_indices": fns,
                    "last_probe_ok": True
                })
                _detect_cache.update(info)
                return _detect_cache
        except Exception as e:
            info["error"] = f"gradio probe error: {e}"

    # 3) ไม่ทราบชนิด
    info.update({"type": "unknown"})
    _detect_cache.update(info)
    return _detect_cache

# ===================== High-level Calls =====================
async def router_call_chat(user_text: str) -> str:
    info = await detect_router()
    rtype: str = info.get("type") or "unknown"
    paths: Set[str] = info.get("seen_paths") or set()
    fn_list: List[int] = info.get("gradio_fn_indices") or []

    # ------ FastAPI ------
    if rtype == "fastapi":
        if "/chat" in paths:
            r = await _post_json(f"{ROUTER_URL}/chat", {"input": user_text, "meta": {"source": "mv-proxy"}})
            if r.status_code < 400:
                return _extract_text_from_router(r.json())
        if "/rag-chat" in paths:
            r = await _post_json(f"{ROUTER_URL}/rag-chat", {"query": user_text, "top_k": 4, "session": "auto"})
            if r.status_code < 400:
                return _extract_text_from_router(r.json())

    # ------ Gradio ------
    if rtype == "gradio":
        # 1) ลองตาม fn_index ที่พบจาก /config|/info
        for fn in fn_list:
            out = await _gradio_predict_try(fn, user_text)
            if out is not None:
                return out
        # 2) ลอง /api/predict (ไม่มี fn_index)
        out = await _gradio_predict_try(None, user_text)
        if out is not None:
            return out
        # 3) เผื่อ Space มี custom path
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

    # ------ Unknown → ไล่ยิงแบบเดิม ------
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
    rtype: str = info.get("type") or "unknown"
    paths: Set[str] = info.get("seen_paths") or set()
    fn_list: List[int] = info.get("gradio_fn_indices") or []

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
        for fn in fn_list:
            out = await _gradio_predict_try(fn, query_text)
            if out is not None:
                return out
        out = await _gradio_predict_try(None, query_text)
        if out is not None:
            return out
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
        "gradio_fn_indices": info.get("gradio_fn_indices", []),
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

# Generic proxy → Router (แนบ query string ครบ)
@app.api_route("/router-proxy/{path:path}", methods=["GET","POST","PUT","PATCH","DELETE","OPTIONS","HEAD"])
async def router_proxy(path: str, request: Request):
    target = f"{ROUTER_URL}/{path}".rstrip("/")
    if request.url.query:
        target = f"{target}?{request.url.query}"

    method = request.method.upper()

    # นำ headers จาก client (ลบที่ httpx จะจัดการเอง)
    out_headers = {k: v for k, v in request.headers.items()
                   if k.lower() not in ("host", "content-length", "transfer-encoding")}

    # บังคับ Content-Type เฉพาะถ้าลูกค้าไม่ได้ส่งมา
    if "content-type" not in {k.lower(): v for k, v in out_headers.items()}:
        out_headers["Content-Type"] = "application/json"

    # ถ้ามี HF_TOKEN → ใส่ Authorization (service-to-service)
    if HF_TOKEN:
        out_headers["Authorization"] = f"Bearer {HF_TOKEN}"

    try:
        body = await request.body()
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
            r = await client.request(method, target, headers=out_headers, content=body or None)
            return Response(
                content=r.content,
                status_code=r.status_code,
                media_type=r.headers.get("content-type", "application/octet-stream"),
            )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"proxy error: {e}")
