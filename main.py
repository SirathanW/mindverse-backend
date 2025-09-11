# main.py
# FastAPI proxy for MindVerse
# - RAG ด้วย HF Embeddings (HF_EMBED_TOKEN / HF_EMBED_MODEL)
# - Chat ด้วย Claude (ANTRHOPIC_VERSION / ANTHROPIC_API_KEY / ANTHROPIC_MODEL)
# - CORS พร้อมสำหรับ Flutter/Web
# รัน: uvicorn main:app --reload --port %PORT%  (ค่าเริ่มต้น 8080)

import os
from dotenv import load_dotenv, find_dotenv

# โหลด .env (รองรับทั้งกรณีรันจากรากโปรเจกต์หรือ IDE)
dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)

import math
import re
import time
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Body, Request, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ===================== ENV =====================
ANTRHOPIC_VERSION = "2023-06-01"  # ใช้คงเดิม
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514").strip()

HF_EMBED_TOKEN = os.getenv("HF_EMBED_TOKEN", "").strip()
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").strip()

PORT = int(os.getenv("PORT", "10000"))

if not HF_EMBED_TOKEN:
    print("[WARN] HF_EMBED_TOKEN is empty — /index และ /rag-chat จะ error จนกว่าจะตั้งค่า.")
if not ANTHROPIC_API_KEY:
    print("[WARN] ANTHROPIC_API_KEY is empty — /chat และ /rag-chat จะเรียก Claude ไม่ได้.")

# ===================== FastAPI =====================
app = FastAPI(title="MindVerse Proxy", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://601411bb.mv-backend.pages.dev"],  # ปรับให้ตรง frontend ของคุณ
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Schemas (เดิม) =====================
class IndexItem(BaseModel):
    id: str = Field(..., description="รหัสเอกสารไม่ซ้ำ")
    text: str = Field(..., description="เนื้อหา plain text สำหรับทำ embedding")

class IndexPayload(BaseModel):
    items: List[IndexItem]

class ChatPayload(BaseModel):
    input: str = Field(..., description="คำถาม/ข้อความจากผู้ใช้")
    meta: Optional[Dict[str, Any]] = Field(default=None, description="เมตาข้อมูลเสริม (เช่น name/lang)")
    session: Optional[str] = Field(default="default", description="session id สำหรับจำบริบท")

class RagPayload(BaseModel):
    query: str = Field(..., description="คำถาม")
    top_k: int = Field(default=4, ge=1, le=10)
    session: Optional[str] = Field(default="rag", description="session id (แชร์ memory กับ /chat ได้)")

# ===================== Schemas (ใหม่สำหรับ RAG เพิ่มเติม) =====================
class UrlIndexPayload(BaseModel):
    urls: List[str] = Field(..., description="ลิงก์ข้อความ/เว็บเพจที่จะดึงมา index")
    id_prefix: Optional[str] = Field(default="url", description="คำนำหน้าสำหรับ gen id อัตโนมัติ")

class SearchPayload(BaseModel):
    query: str = Field(..., description="ข้อความค้นหา")
    top_k: int = Field(default=5, ge=1, le=20)

# ===================== Store (In-Memory) =====================
VECTOR_STORE: List[Dict[str, Any]] = []  # [{id, text, vec: List[float]}]
MEMORIES: Dict[str, List[Dict[str, str]]] = {}  # ต่อ session

def push_memory(session: str, role: str, content: str, max_len: int = 10):
    arr = MEMORIES.setdefault(session, [])
    arr.append({"role": role, "content": content})
    if len(arr) > max_len:
        del arr[0: len(arr)-max_len]

# ===================== Utils =====================
def cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        x = float(a[i]); y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def _strip_html(raw: str) -> str:
    # ตัด tag ง่าย ๆ ให้เหลือข้อความ
    txt = re.sub(r"(?is)<script.*?>.*?</script>", " ", raw)
    txt = re.sub(r"(?is)<style.*?>.*?</style>", " ", txt)
    txt = re.sub(r"(?s)<[^>]+>", " ", txt)
    txt = re.sub(r"[ \t\r\f\v]+", " ", txt)
    return re.sub(r"\n+", "\n", txt).strip()

async def hf_embed(texts: List[str]) -> List[List[float]]:
    if not HF_EMBED_TOKEN:
        raise HTTPException(status_code=400, detail="HF_EMBED_TOKEN is missing on server")

    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBED_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_EMBED_TOKEN}",
        "Content-Type": "application/json"
    }

    outs: List[List[float]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for t in texts:
            data = {"inputs": t, "truncate": True}
            r = await client.post(url, headers=headers, json=data)
            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"HuggingFace error: {r.text}")
            v = r.json()
            # [seq_len, dim] → average pooling
            if isinstance(v, list) and v and isinstance(v[0], list):
                if isinstance(v[0][0], (int, float)):
                    dim = len(v[0]); seq_len = len(v)
                    pooled = [0.0] * dim
                    for row in v:
                        for j in range(dim):
                            pooled[j] += float(row[j])
                    pooled = [x / float(seq_len) for x in pooled]
                    outs.append(pooled)
                else:
                    raise HTTPException(status_code=500, detail="Unexpected embedding format")
            elif isinstance(v, list) and v and isinstance(v[0], (int, float)):
                outs.append([float(x) for x in v])
            else:
                raise HTTPException(status_code=500, detail="Invalid embedding response format")
    return outs

async def anthropic_chat(messages: List[Dict[str, str]], system: Optional[str] = None) -> str:
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY is missing on server")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTRHOPIC_VERSION,
        "content-type": "application/json",
    }

    conv = []
    for m in messages:
        role = m.get("role", "user"); content = m.get("content", "")
        if role == "system":
            continue
        conv.append({"role": role, "content": content})

    payload = {
        "model": ANTHROPIC_MODEL,
        "max_tokens": 512,
        "messages": conv
    }
    if system:
        payload["system"] = system

    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        r = await client.post(url, headers=headers, json=payload)
        if r.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Anthropic API error: {r.text}")
        data = r.json()
        blocks = data.get("content", [])
        if not blocks:
            return "(no content)"
        text_parts = []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                text_parts.append(b.get("text", ""))
        return "\n".join([t for t in text_parts if t])

# ===================== Endpoints เดิม =====================
@app.get("/health")
async def health():
    return {
        "ok": True,
        "anthropic_model": ANTHROPIC_MODEL,
        "anthropic_key": bool(ANTHROPIC_API_KEY),
        "hf_embed_model": HF_EMBED_MODEL,
        "hf_embed_key": bool(HF_EMBED_TOKEN),
        "vector_store_count": len(VECTOR_STORE),
        "mem_sessions": len(MEMORIES),
        "ts": time.time(),
    }

# ---------- FIX 405: อนุญาตทั้ง GET และ POST ----------
@app.api_route("/index", methods=["GET", "POST"])
async def index_docs(request: Request, payload: Optional[IndexPayload] = Body(None)):
    if request.method == "GET":
        return {
            "usage": "POST JSON to /index",
            "example": {"items": [{"id": "doc1", "text": "เนื้อหา..."}]}
        }
    if not payload or not payload.items:
        raise HTTPException(status_code=400, detail="No items")
    texts = [it.text for it in payload.items]
    vecs = await hf_embed(texts)
    for it, v in zip(payload.items, vecs):
        # ลบของเก่าที่ id ซ้ำ
        for i, old in enumerate(VECTOR_STORE):
            if old["id"] == it.id:
                del VECTOR_STORE[i]
                break
        VECTOR_STORE.append({"id": it.id, "text": it.text, "vec": v})
    return {"ok": True, "count": len(payload.items), "store_size": len(VECTOR_STORE)}

@app.post("/clear-index")
async def clear_index():
    VECTOR_STORE.clear()
    return {"ok": True, "store_size": 0}

# ---------- FIX 405: /chat ----------
@app.api_route("/chat", methods=["GET", "POST"])
async def chat(request: Request, payload: Optional[ChatPayload] = Body(None)):
    if request.method == "GET":
        return {
            "usage": "POST JSON to /chat",
            "example": {"input": "สวัสดี", "session": "default", "meta": {"lang": "th"}}
        }
    if not payload or not payload.input.strip():
        raise HTTPException(status_code=400, detail="input is empty")

    user = payload.input.strip()
    session = payload.session or "default"
    lang = (payload.meta or {}).get("lang", "th")
    # name = (payload.meta or {}).get("name", "MindVerse User")  # เก็บไว้ใช้ภายหลังได้

    history = MEMORIES.get(session, [])
    system_prompt = (
        f"คุณคือ MindVerse AI Assistant ตอบเป็นภาษา { 'ไทย' if lang.lower().startswith('th') else lang } "
        "อย่างสุภาพ กระชับ และให้คุณค่าจริง ไม่แต่งเรื่อง ถ้าไม่ทราบให้บอกตรงๆ"
    )

    messages: List[Dict[str, str]] = []
    for h in history[-8:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user})

    reply = await anthropic_chat(messages, system=system_prompt)
    push_memory(session, "user", user)
    push_memory(session, "assistant", reply)

    return {"reply": reply, "session": session}

# ---------- FIX 405: /rag-chat (เดิม) ----------
@app.api_route("/rag-chat", methods=["GET", "POST"])
async def rag_chat(request: Request, payload: Optional[RagPayload] = Body(None)):
    if request.method == "GET":
        return {
            "usage": "POST JSON to /rag-chat",
            "example": {"query": "ถามจากคลัง", "top_k": 4, "session": "rag"}
        }
    if not payload or not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is empty")
    if not VECTOR_STORE:
        raise HTTPException(status_code=400, detail="No documents in index. Call /index first.")

    q = payload.query.strip()
    qvec = (await hf_embed([q]))[0]

    scored = []
    for doc in VECTOR_STORE:
        score = cosine(qvec, doc["vec"])
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    picks = [d for _, d in scored[:payload.top_k]]

    context_blocks = []
    for i, d in enumerate(picks, 1):
        context_blocks.append(f"[{i}] ({d['id']}): {d['text']}")
    context_text = "\n\n".join(context_blocks)

    system_prompt = (
        "คุณคือผู้ช่วย MindVerse ตอบอย่างกระชับ ชัดเจน อ้างอิงความรู้จาก 'Context' ด้านล่างเป็นหลัก "
        "ห้ามเดา ถ้าในบริบทไม่มีคำตอบ ให้บอกว่าไม่พบข้อมูล\n\n"
        f"Context:\n{context_text}\n\n"
        "แนวทางตอบ: สรุปประเด็นสำคัญสั้นๆ และอธิบายเพิ่มเฉพาะที่จำเป็น"
    )

    messages = [{"role": "user", "content": q}]
    answer = await anthropic_chat(messages, system=system_prompt)

    return {
        "reply": answer,
        "references": [{"id": d["id"], "preview": d["text"][:160]} for d in picks]
    }

# ===================== RAG: Endpoints เพิ่มเติม (ใหม่) =====================

@app.post("/search")
async def search_docs(payload: SearchPayload):
    """
    ค้นหาเอกสารที่ใกล้เคียงที่สุดจาก VECTOR_STORE (ไม่เรียก Claude)
    ใช้เพื่อ debug/ทดสอบ RAG ก่อนคุยจริง
    """
    if not payload.query.strip():
        raise HTTPException(status_code=400, detail="query is empty")
    if not VECTOR_STORE:
        raise HTTPException(status_code=400, detail="No documents in index. Call /index first.")

    qvec = (await hf_embed([payload.query.strip()]))[0]

    scored = []
    for doc in VECTOR_STORE:
        s = cosine(qvec, doc["vec"])
        scored.append((s, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:payload.top_k]

    return {
        "count": len(top),
        "results": [
            {"id": d["id"], "score": float(s), "preview": d["text"][:200]}
            for (s, d) in top
        ]
    }

@app.get("/list-index")
async def list_index():
    """
    ดูเอกสารทั้งหมดที่ index แล้ว (เพื่อความสะดวก)
    """
    return {
        "size": len(VECTOR_STORE),
        "items": [
            {"id": d["id"], "preview": d["text"][:200]}
            for d in VECTOR_STORE
        ]
    }

@app.delete("/delete/{doc_id}")
async def delete_doc(doc_id: str = Path(..., description="รหัสเอกสาร")):
    for i, d in enumerate(VECTOR_STORE):
        if d["id"] == doc_id:
            del VECTOR_STORE[i]
            return {"ok": True, "removed": doc_id, "store_size": len(VECTOR_STORE)}
    raise HTTPException(status_code=404, detail="doc not found")

@app.post("/index-urls")
async def index_from_urls(payload: UrlIndexPayload):
    """
    ดึงข้อความจาก URL แล้วทำ embedding เก็บเข้าดัชนี
    รองรับไฟล์ text ธรรมดา/markdown และเว็บเพจ (จะ strip HTML ให้)
    """
    if not payload.urls:
        raise HTTPException(status_code=400, detail="No urls")

    fetched_texts: List[str] = []
    ids: List[str] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for i, u in enumerate(payload.urls, start=1):
            try:
                r = await client.get(u)
                if r.status_code >= 400:
                    raise HTTPException(status_code=502, detail=f"fetch error {u}: {r.text}")
                raw = r.text or ""
                # ถ้าเป็นเว็บเพจ ให้ strip HTML
                if "text/html" in r.headers.get("content-type", "").lower():
                    text = _strip_html(raw)
                else:
                    text = raw
                text = text.strip()
                if not text:
                    continue
                fetched_texts.append(text)
                ids.append(f"{payload.id_prefix}-{i}")
            except Exception as e:
                # ข้าม URL ที่พัง แล้วไปต่อ URL ถัดไป
                print(f"[index-urls] skip {u}: {e}")

    if not fetched_texts:
        raise HTTPException(status_code=400, detail="No text fetched from urls")

    vecs = await hf_embed(fetched_texts)
    for _id, t, v in zip(ids, fetched_texts, vecs):
        # ถ้ามี id ซ้ำ ให้ลบก่อน
        for i, old in enumerate(VECTOR_STORE):
            if old["id"] == _id:
                del VECTOR_STORE[i]
                break
        VECTOR_STORE.append({"id": _id, "text": t, "vec": v})

    return {"ok": True, "added": len(ids), "store_size": len(VECTOR_STORE), "ids": ids}
