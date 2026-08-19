import os
import json
import uuid
import sqlite3
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_pipeline import build_chain, strip_think

app = FastAPI(title="PDFQUERY API")

# Let the React app (different port) talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "pdfquery.db"
CHAINS = {}  # conv_id -> (chain, vectorstore), kept in memory per server run


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        doc_name TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY,
        conv_id TEXT,
        role TEXT,
        content TEXT,
        sources TEXT
    )""")
    conn.commit()
    conn.close()


init_db()


class ChatRequest(BaseModel):
    conversation_id: str
    question: str


@app.post("/conversations")
def create_conversation():
    conv_id = uuid.uuid4().hex
    conn = get_db()
    conn.execute("INSERT INTO conversations (id, title) VALUES (?, ?)",
                 (conv_id, "New chat"))
    conn.commit()
    conn.close()
    return {"id": conv_id}


@app.get("/conversations")
def list_conversations():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, doc_name FROM conversations ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


@app.get("/conversations/{conv_id}/messages")
def get_messages(conv_id: str):
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, sources FROM messages WHERE conv_id = ? ORDER BY rowid",
        (conv_id,),
    ).fetchall()
    conn.close()
    return [
        {"role": r["role"], "content": r["content"],
         "sources": json.loads(r["sources"] or "[]")}
        for r in rows
    ]


@app.post("/conversations/{conv_id}/upload")
async def upload_pdf(conv_id: str, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.pdf")
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    try:
        chain, vectorstore = build_chain(tmp_path)
    except Exception as e:
        raise HTTPException(400, f"Could not process PDF: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if conv_id in CHAINS:
        try:
            CHAINS[conv_id][1].delete_collection()
        except Exception:
            pass
    CHAINS[conv_id] = (chain, vectorstore)

    conn = get_db()
    conn.execute("UPDATE conversations SET doc_name = ? WHERE id = ?",
                 (file.filename, conv_id))
    conn.commit()
    conn.close()
    return {"status": "indexed", "doc_name": file.filename}


@app.post("/chat")
def chat(req: ChatRequest):
    if req.conversation_id not in CHAINS:
        raise HTTPException(400, "Upload a PDF in this conversation first.")

    chain, _ = CHAINS[req.conversation_id]
    result = chain.invoke(req.question)
    answer = strip_think(result["answer"])
    sources = [d.page_content for d in result["context"]]

    conn = get_db()
    conn.execute(
        "INSERT INTO messages (id, conv_id, role, content, sources) VALUES (?, ?, ?, ?, ?)",
        (uuid.uuid4().hex, req.conversation_id, "user", req.question, ""))
    conn.execute(
        "INSERT INTO messages (id, conv_id, role, content, sources) VALUES (?, ?, ?, ?, ?)",
        (uuid.uuid4().hex, req.conversation_id, "assistant", answer, json.dumps(sources)))
    conn.execute(
        "UPDATE conversations SET title = ? WHERE id = ? AND title = 'New chat'",
        (req.question[:40], req.conversation_id))
    conn.commit()
    conn.close()
    return {"answer": answer, "sources": sources}


@app.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: str):
    if conv_id in CHAINS:
        try:
            CHAINS[conv_id][1].delete_collection()
        except Exception:
            pass
        CHAINS.pop(conv_id)
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted"}