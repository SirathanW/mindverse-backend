# main.py
# FastAPI proxy for MindVerse
# - RAG ด้วย HF Embeddings (HF_EMBED_TOKEN / HF_EMBED_MODEL)
# - Chat ด้วย Claude (ANTHROPIC_API_KEY / ANTHROPIC_MODEL)
# - CORS พร้อมสำหรับ Flutter/Web
# รัน: uvicorn main:app --reload --port %PORT%  (ค่าเริ่มต้น 8080)

import os
from dotenv import load_dotenv, find_dotenv

# โหลด .env (รองรับทั้งกรณีรันจากรากโปรเจกต์หรือ IDE)
dotenv_path = find_dotenv(usecwd=True)
load_dotenv(dotenv_path)

import math
import json
import time
from typing import List, Dict, Any, Optional

import httpx
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ===================== ENV =====================
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()
ANTHROPIC_MODEL   = os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022").strip()

HF_EMBED_TOKEN = os.getenv("HF_EMBED_TOKEN", "").strip()
HF_EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-base").strip()

PORT = int(os.getenv("PORT", "8080"))

if not HF_EMBED_TOKEN:
    print("[WARN] HF_EMBED_TOKEN is empty — /index และ /rag-chat จะ error จนกว่าจะตั้งค่า.")
if not ANTHROPIC_API_KEY:
    print("[WARN] ANTHROPIC_API_KEY is empty — /chat และ /rag-chat จะเรียก Claude ไม่ได้.")

# ===================== FastAPI =====================
app = FastAPI(title="MindVerse Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # lock ให้แคบลงภายหลัง (เช่น http://localhost:3000, app scheme)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Schemas =====================
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

# ===================== Store (In-Memory) =====================
# เอกสารที่ทำ embedding แล้ว: [{"id": str, "text": str, "vec": List[float]}]
VECTOR_STORE: List[Dict[str, Any]] = []

# ความจำบทสนทนาต่อ session (เก็บล่าสุด ~10 ข้อความ)
MEMORIES: Dict[str, List[Dict[str, str]]] = {}

def push_memory(session: str, role: str, content: str, max_len: int = 10):
    arr = MEMORIES.setdefault(session, [])
    arr.append({"role": role, "content": content})
    if len(arr) > max_len:
        del arr[0: len(arr)-max_len]

# ===================== Utils =====================
def cosine(a: List[float], b: List[float]) -> float:
    # เลี่ยง numpy เพื่อความง่าย
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(len(a)):
        x = float(a[i])
        y = float(b[i])
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

async def hf_embed(texts: List[str]) -> List[List[float]]:
    """
    เรียก HF Inference API เพื่อทำ embedding ด้วย model ที่กำหนดใน HF_EMBED_MODEL
    ใช้ endpoint feature-extraction
    """
    if not HF_EMBED_TOKEN:
        raise HTTPException(status_code=400, detail="HF_EMBED_TOKEN is missing on server")

    # Inference API: https://api-inference.huggingface.co/pipeline/feature-extraction/{model}
    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBED_MODEL}"
    headers = {
        "Authorization": f"Bearer {HF_EMBED_TOKEN}",
        "Content-Type": "application/json"
    }

    # ใส่เป็น batch ทีละรายการ (บางรุ่นรองรับ batch; ทำทีละชิ้นเพื่อความชัวร์)
    outs: List[List[float]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:
        for t in texts:
            data = {"inputs": t, "truncate": True}
            r = await client.post(url, headers=headers, json=data)
            if r.status_code >= 400:
                raise HTTPException(status_code=502, detail=f"HuggingFace error: {r.text}")
            v = r.json()
            # บางรุ่นคืนหลายมิติ (เช่น sequence x dim) → เอาค่าเฉลี่ยตามแกน sequence
            if isinstance(v, list) and v and isinstance(v[0], list):
                # ถ้าเป็น [seq_len, dim] → average pooling
                if isinstance(v[0][0], (int, float)):
                    dim = len(v[0])
                    seq_len = len(v)
                    # average
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
    """
    เรียก Anthropic Messages API
    messages: [{"role": "user"/"assistant"/"system"(ไม่ใช้ใน body), "content": "..."}]
    """
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=400, detail="ANTHROPIC_API_KEY is missing on server")

    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    # แปลงเป็นรูปแบบของ Anthropic:
    # - ไม่รองรับ role=system ใน list -> ใช้ "system": "<text>"
    conv = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "system":
            # ข้าม เพราะจะใส่ใน system parameter ด้านล่าง
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
        # โครงสร้างโต้ตอบ: {"content":[{"type":"text","text":"..."}], ...}
        blocks = data.get("content", [])
        if not blocks:
            return "(no content)"
        text_parts = []
        for b in blocks:
            if isinstance(b, dict) and b.get("type") == "text":
                text_parts.append(b.get("text", ""))
        return "\n".join([t for t in text_parts if t])

# ===================== Endpoints =====================
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

@app.post("/index")
async def index_docs(payload: IndexPayload):
    if not payload.items:
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

@app.post("/chat")
async def chat(payload: ChatPayload):
    """
    แชตทั่วไป (ไม่มี RAG) + จำบริบทแบบสั้นตาม session
    Flutter ส่ง: { "input": "สวัสดี", "session": "abc", "meta": {"lang":"th"} }
    """
    user = payload.input.strip()
    if not user:
        raise HTTPException(status_code=400, detail="input is empty")

    session = payload.session or "default"
    lang = (payload.meta or {}).get("lang", "th")
    name = (payload.meta or {}).get("name", "MindVerse User")

    # ความจำบทสนทนา
    history = MEMORIES.get(session, [])

    # system prompt (ไทยตามต้องการ)
    system_prompt = (
        f"คุณคือ MindVerse AI Assistant ตอบเป็นภาษา { 'ไทย' if lang.lower().startswith('th') else lang } "
        "อย่างสุภาพ กระชับ และให้คุณค่าจริง ไม่แต่งเรื่อง ถ้าไม่ทราบให้บอกตรงๆ"
    )

    messages: List[Dict[str, str]] = []
    # นำ history (user/assistant) ใส่
    for h in history[-8:]:
        messages.append({"role": h["role"], "content": h["content"]})
    messages.append({"role": "user", "content": user})

    reply = await anthropic_chat(messages, system=system_prompt)

    # อัปเดตความจำ
    push_memory(session, "user", user)
    push_memory(session, "assistant", reply)

    return {"reply": reply, "session": session}

@app.post("/rag-chat")
async def rag_chat(payload: RagPayload):
    """
    แชต + อ้างอิงคลังความรู้ (RAG)
    - สร้าง embedding ของคำถาม
    - หาความคล้ายจาก VECTOR_STORE
    - ส่ง context เข้าหา Claude
    """
    q = payload.query.strip()
    if not q:
        raise HTTPException(status_code=400, detail="query is empty")
    if not VECTOR_STORE:
        raise HTTPException(status_code=400, detail="No documents in index. Call /index first.")

    # 1) ฝังคิวรี
    qvec = (await hf_embed([q]))[0]

    # 2) คำนวณ cosine แล้วเลือก top_k
    scored = []
    for doc in VECTOR_STORE:
        score = cosine(qvec, doc["vec"])
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    picks = [d for _, d in scored[:payload.top_k]]

    # 3) รวม context
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
