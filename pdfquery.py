import os
import gc
import json
import uuid
import sqlite3

import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import build_chain, strip_think

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
DB_PATH = os.path.join(BASE_DIR, "pdfquery.db")
os.makedirs(PDF_DIR, exist_ok=True)

CHAINS = {}  
st.set_page_config(
    page_title="PDFQUERY",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>

    #MainMenu {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    .stApp {
        background: #0d0f14;
    }

    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 7rem;
        max-width: 950px;
    }

    .brand {
        position: fixed;
        top: 1.15rem;
        left: 1.5rem;

        font-size: 1.05rem;
        font-weight: 750;
        letter-spacing: 0.5px;

        color: #f2f2f2;

        z-index: 999999;
    }

    .home-container {
        width: 100%;
        text-align: center;

        padding-top: 27vh;
    }

    .greeting {
        font-size: 2.15rem;
        font-weight: 650;

        color: #f4f4f5;

        margin-bottom: 0.7rem;

        letter-spacing: -0.7px;
    }

    .sub-greeting {
        color: #8d9098;
        font-size: 1rem;

        margin-bottom: 2rem;
    }

    [data-testid="stChatMessage"] {
        background: transparent !important;
    }

    [data-testid="stChatMessageContent"] {
        color: #e8e8ea;
    }

    div[data-testid="stPopover"] > button {

        width: 48px !important;
        height: 48px !important;

        min-width: 48px !important;

        padding: 0 !important;

        border-radius: 14px !important;

        border: 1px solid #343842 !important;

        background: #151821 !important;

        color: #f2f2f2 !important;

        font-size: 1.25rem !important;

        display: flex !important;
        align-items: center !important;
        justify-content: center !important;

        box-shadow: none !important;
    }

    div[data-testid="stPopover"] > button:hover {
        background: #1c202a !important;
        border-color: #4a4f5c !important;
    }

    [data-testid="stChatInput"] {

        padding-bottom: 1rem !important;
    }

    [data-testid="stChatInput"] > div {

        border-radius: 17px !important;

        background: #232630 !important;

        border: 1px solid #2d303a !important;

        box-shadow: none !important;

        min-height: 58px !important;
    }

    [data-testid="stChatInput"] textarea {

        color: #eeeeef !important;

        font-size: 1rem !important;

        padding-left: 1rem !important;
        padding-right: 3.5rem !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #8b8e97 !important;
    }

    [data-testid="stChatInput"] > div:focus-within {

        border-color: #444853 !important;

        box-shadow: 0 0 0 1px #444853 !important;
    }

    .composer-row {
        display: flex;

        align-items: center;

        gap: 10px;

        width: 100%;
    }

    [data-testid="stExpander"] {

        border: 1px solid #30333c !important;

        border-radius: 10px !important;

        background: transparent !important;
    }

    section[data-testid="stFileUploaderDropzone"] {

        background: #171a22 !important;

        border: 1px dashed #3a3e49 !important;

        border-radius: 12px !important;
    }

    .stButton > button {

        border-radius: 10px;

        border: 1px solid #353945;

        background: #171a22;

        color: #eeeeef;
    }

    .stButton > button:hover {
        border-color: #505562;
        background: #1d2029;
    }

    /* --- NEW: sidebar history styling --- */
    [data-testid="stSidebar"] {
        background: #151821;
    }

    .stSidebar .stButton > button {
        justify-content: flex-start;
        text-align: left;
    }

    .conv-doc {
        color: #8d9098;
        font-size: .72rem;
        margin: -0.4rem 0 0.5rem 0.2rem;
    }

    /* --- NEW: file chip (like the React app) --- */
    .file-chip {
        display: flex;
        align-items: center;
        gap: .6rem;
        padding: .55rem .8rem;
        border-radius: 12px;
        border: 1px solid #3b82f6;
        background: #1a2233;
        margin: 0 auto .6rem;
        width: fit-content;
        max-width: 100%;
    }

    .file-name {
        font-size: .85rem;
        color: #e8e8ea;
    }

    .file-status {
        font-size: .72rem;
        color: #60a5fa;
    }

    @media (max-width: 700px) {

        .block-container {
            padding-left: 0.8rem;
            padding-right: 0.8rem;
        }

        .brand {
            left: 1rem;
        }

        .home-container {
            padding-top: 25vh;
        }

        .greeting {
            font-size: 1.75rem;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY, title TEXT, doc_name TEXT,
        created_at TEXT DEFAULT (datetime('now')))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS messages (
        id TEXT PRIMARY KEY, conv_id TEXT, role TEXT, content TEXT, sources TEXT)""")
    conn.commit()
    conn.close()

init_db()

def list_conversations():
    conn = get_db()
    rows = conn.execute(
        "SELECT id, title, doc_name FROM conversations ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_conversation(conv_id):
    conn = get_db()
    row = conn.execute(
        "SELECT id, title, doc_name FROM conversations WHERE id = ?", (conv_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None

def get_messages(conv_id):
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

def create_conversation():
    conv_id = uuid.uuid4().hex
    conn = get_db()
    conn.execute("INSERT INTO conversations (id, title) VALUES (?, ?)",
                 (conv_id, "New chat"))
    conn.commit()
    conn.close()
    return conv_id

def delete_conversation(conv_id):
    if conv_id in CHAINS:
        try:
            CHAINS[conv_id][1].delete_collection()
        except Exception:
            pass
        CHAINS.pop(conv_id)
    try:
        os.remove(os.path.join(PDF_DIR, f"{conv_id}.pdf"))
    except OSError:
        pass
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
    conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
    conn.commit()
    conn.close()

def add_message(conv_id, role, content, sources=None):
    conn = get_db()
    conn.execute(
        "INSERT INTO messages (id, conv_id, role, content, sources) VALUES (?, ?, ?, ?, ?)",
        (uuid.uuid4().hex, conv_id, role, content, json.dumps(sources or [])))
    conn.commit()
    conn.close()

def set_title(conv_id, title):
    conn = get_db()
    conn.execute("UPDATE conversations SET title = ? WHERE id = ? AND title = 'New chat'",
                 (title, conv_id))
    conn.commit()
    conn.close()

def set_doc_name(conv_id, name):
    conn = get_db()
    conn.execute("UPDATE conversations SET doc_name = ? WHERE id = ?", (name, conv_id))
    conn.commit()
    conn.close()

def clear_messages(conv_id):
    conn = get_db()
    conn.execute("DELETE FROM messages WHERE conv_id = ?", (conv_id,))
    conn.commit()
    conn.close()

def get_chain(conv_id):
    """Returns (chain, vectorstore), rebuilding from the saved PDF if needed."""
    if conv_id in CHAINS:
        return CHAINS[conv_id]
    pdf_path = os.path.join(PDF_DIR, f"{conv_id}.pdf")
    if not os.path.exists(pdf_path):
        return None
    CHAINS[conv_id] = build_chain(pdf_path)
    return CHAINS[conv_id]

st.session_state.setdefault("active_id", None)
active_id = st.session_state.active_id
active = get_conversation(active_id) if active_id else None
messages = get_messages(active_id) if active_id else []

with st.sidebar:
    if st.button("+ New chat", use_container_width=True):
        st.session_state.active_id = create_conversation()
        st.rerun()
    st.write("")
    for conv in list_conversations():
        c1, c2 = st.columns([7, 1])
        with c1:
            if st.button(conv["title"], key=f"open_{conv['id']}",
                         use_container_width=True):
                st.session_state.active_id = conv["id"]
                st.rerun()
        with c2:
            if st.button("×", key=f"del_{conv['id']}"):
                if st.session_state.active_id == conv["id"]:
                    st.session_state.active_id = None
                delete_conversation(conv["id"])
                st.rerun()
        if conv["doc_name"]:
            st.markdown(f'<div class="conv-doc">{conv["doc_name"]}</div>',
                        unsafe_allow_html=True)

st.markdown(
    '<div class="brand">PDFQUERY</div>',
    unsafe_allow_html=True
)

if not messages:

    st.markdown(
        """
        <div class="home-container">
            <div class="greeting">
                What's on your mind today?
            </div>
        """,
        unsafe_allow_html=True
    )

    if active and active["doc_name"]:

        st.markdown(
            f"""
            <div class="sub-greeting">
                Ready — ask anything about
                <b>{active["doc_name"]}</b>
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="sub-greeting">
                Click the ＋ button to upload a PDF,
                then ask anything about it.
            </div>
            </div>
            """,
            unsafe_allow_html=True
        )

for msg in messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg.get("sources"):

            with st.expander("Sources"):

                for i, source in enumerate(
                    msg["sources"], 1
                ):

                    st.caption(
                        f"[{i}] {source[:300]}..."
                    )

if active and active["doc_name"]:
    st.markdown(
        f'<div class="file-chip"><span>📄</span>'
        f'<span class="file-name">{active["doc_name"]}</span>'
        f'<span class="file-status">Ready</span></div>',
        unsafe_allow_html=True,
    )

plus_col, chat_col = st.columns(
    [0.65, 12],
    vertical_alignment="center"
)

with plus_col:

    with st.popover("＋"):

        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            label_visibility="collapsed"
        )

        st.write("")

        if active_id and st.button(
            "Clear conversation",
            use_container_width=True
        ):

            clear_messages(active_id)

            st.rerun()

with chat_col:

    question = st.chat_input(
        "Ask anything"
    )

if uploaded_file is not None:

    conv_id = active_id or create_conversation()
    st.session_state.active_id = conv_id
    current = get_conversation(conv_id)

    if current["doc_name"] != uploaded_file.name:

        if conv_id in CHAINS:

            try:
                CHAINS[conv_id][1].delete_collection()
            except Exception:
                pass

            CHAINS.pop(conv_id)

            gc.collect()

        pdf_path = os.path.join(PDF_DIR, f"{conv_id}.pdf")

        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Reading your PDF..."):

            try:
                get_chain(conv_id)

            except Exception as e:

                st.error(
                    f"Could not process the PDF: {str(e)}"
                )

                st.stop()

        set_doc_name(conv_id, uploaded_file.name)

        st.rerun()

if question:

    conv_id = active_id or create_conversation()
    st.session_state.active_id = conv_id

    if not os.path.exists(os.path.join(PDF_DIR, f"{conv_id}.pdf")):

        st.warning(
            "Please upload a PDF first using the ＋ button."
        )

    else:

        add_message(conv_id, "user", question)

        with st.spinner("Thinking..."):

            chain, _ = get_chain(conv_id)  # rebuilds index for old chats automatically

            result = chain.invoke(question)

            docs = result["context"]
            answer = strip_think(result["answer"])

        add_message(conv_id, "assistant", answer,
                    [d.page_content for d in docs])

        set_title(conv_id, question[:40])

        st.rerun()