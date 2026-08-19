import { useEffect, useRef, useState } from "react";
import "./App.css";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

export default function App() {
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [doc, setDoc] = useState(null); // {name, status: "uploading"|"ready"}
  const [notice, setNotice] = useState("");
  const fileRef = useRef(null);
  const bottomRef = useRef(null);

  useEffect(() => { loadConversations(); }, []);
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, thinking]);

  async function loadConversations() {
    const res = await fetch(`${API}/conversations`);
    setConversations(await res.json());
  }

  async function loadMessages(id) {
    const res = await fetch(`${API}/conversations/${id}/messages`);
    setMessages(await res.json());
  }

  async function newChat() {
    const res = await fetch(`${API}/conversations`, { method: "POST" });
    const { id } = await res.json();
    setActiveId(id);
    setMessages([]);
    setDoc(null);
    setNotice("");
    loadConversations();
  }

  async function selectConversation(id) {
    setActiveId(id);
    setDoc(null);
    setNotice("");
    loadMessages(id);
  }

  async function deleteConversation(id, e) {
    e.stopPropagation();
    await fetch(`${API}/conversations/${id}`, { method: "DELETE" });
    if (activeId === id) { setActiveId(null); setMessages([]); setDoc(null); }
    loadConversations();
  }

  async function ensureConversation() {
    if (activeId) return activeId;
    const res = await fetch(`${API}/conversations`, { method: "POST" });
    const { id } = await res.json();
    setActiveId(id);
    return id;
  }

  async function handleUpload(e) {
    const file = e.target.files[0];
    if (!file) return;
    e.target.value = "";
    setUploading(true);
    setNotice("");
    setDoc({ name: file.name, status: "uploading" });   // dim chip appears
    const id = await ensureConversation();
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${API}/conversations/${id}/upload`, {
      method: "POST",
      body: form,
    });
    setUploading(false);
    if (!res.ok) {
      const err = await res.json();
      setNotice(err.detail || "Upload failed");
      setDoc(null);
      return;
    }
    setDoc({ name: file.name, status: "ready" });       // chip turns bright
    loadConversations();
  }

  async function send() {
    const question = input.trim();
    if (!question || thinking) return;
    setNotice("");
    const id = await ensureConversation();
    setInput("");
    setMessages((m) => [...m, { role: "user", content: question }]);
    setThinking(true);
    try {
      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation_id: id, question }),
      });
      const data = await res.json();
      if (!res.ok) {
        setNotice(data.detail || "Something went wrong");
      } else {
        setMessages((m) => [
          ...m,
          { role: "assistant", content: data.answer, sources: data.sources },
        ]);
      }
    } catch {
      setNotice("Is the FastAPI server running? (uvicorn server:app --port 8000)");
    }
    setThinking(false);
    loadConversations();
  }

  const activeConv = conversations.find((c) => c.id === activeId);
  const chip = doc || (activeConv?.doc_name
    ? { name: activeConv.doc_name, status: "ready" }
    : null);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="sidebar-brand">PDFQUERY</div>
        <button className="new-chat" onClick={newChat}>+ New chat</button>
        <div className="conv-list">
          {conversations.map((c) => (
            <div
              key={c.id}
              className={`conv-item ${c.id === activeId ? "active" : ""}`}
              onClick={() => selectConversation(c.id)}
            >
              <div className="conv-text">
                <span className="conv-title">{c.title}</span>
                {c.doc_name && <span className="conv-doc">{c.doc_name}</span>}
              </div>
              <button className="conv-del" onClick={(e) => deleteConversation(c.id, e)}>
                ×
              </button>
            </div>
          ))}
        </div>
      </aside>

      <main className="main">
        {messages.length === 0 ? (
          <div className="home">
            <h1 className="greeting">What's on your mind today?</h1>
            <p className="sub-greeting">
              {chip
                ? `Ready — ask anything about ${chip.name}`
                : "Click the + button to upload a PDF, then ask anything about it."}
            </p>
          </div>
        ) : (
          <div className="chat">
            {messages.map((m, i) => (
              <div key={i} className={`msg ${m.role}`}>
                <div>
                  <div className="bubble">{m.content}</div>
                  {m.sources && m.sources.length > 0 && (
                    <details className="sources">
                      <summary>Sources</summary>
                      {m.sources.map((s, j) => (
                        <p key={j}>[{j + 1}] {s.slice(0, 300)}...</p>
                      ))}
                    </details>
                  )}
                </div>
              </div>
            ))}
            {thinking && (
              <div className="msg assistant">
                <div className="bubble thinking">Thinking…</div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>
        )}

        {/* 📄 The ChatGPT-style file chip */}
        {chip && (
          <div className={`file-chip ${chip.status}`}>
            <span className="file-icon">📄</span>
            <span className="file-name">{chip.name}</span>
            <span className="file-status">
              {chip.status === "uploading" ? "Indexing…" : "Ready"}
            </span>
          </div>
        )}

        {notice && <div className="notice">{notice}</div>}

        <div className="composer">
          <input type="file" accept=".pdf" hidden ref={fileRef}
                 onChange={handleUpload} />
          <button className="plus" onClick={() => fileRef.current.click()}
                  disabled={uploading}>
            {uploading ? "…" : "+"}
          </button>
          <input
            className="chat-input"
            placeholder="Ask anything"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
          />
          <button className="send" onClick={send}
                  disabled={!input.trim() || thinking}>
            ↑
          </button>
        </div>
      </main>
    </div>
  );
}