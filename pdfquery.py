import os
import gc
import json
import uuid
import sqlite3
import textwrap

import streamlit as st
from dotenv import load_dotenv
from rag_pipeline import build_chain, strip_think

load_dotenv()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_DIR = os.path.join(BASE_DIR, "data", "pdfs")
DB_PATH = os.path.join(BASE_DIR, "pdfquery.db")
os.makedirs(PDF_DIR, exist_ok=True)

st.set_page_config(
    page_title="PDFQUERY",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>

#MainMenu,
header,
footer {
    visibility: hidden;
}

.stApp {
    background: #0d0f14;
    color: #e8e8ea;
}

.block-container {
    max-width: 1050px;
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}
   section[data-testid="stMain"] .block-container,
section.main .block-container,
div[data-testid="stMainBlockContainer"],
.block-container.stMainBlockContainer {
    min-height: calc(100vh - 3rem);
    display: flex;
    flex-direction: column;
}

.brand {
    position: fixed;
    top: 1rem;
    left: 1.35rem;

    color: #f4f4f5;
    font-size: 1.05rem;
    font-weight: 700;
    letter-spacing: .4px;

    z-index: 999999;
}

[data-testid="stSidebar"] {
    background: #151821;
    border-right: 1px solid #242731;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.2rem;
}
/* New chat */

[data-testid="stSidebar"] .st-key-new_chat_btn button {

    height: 42px;

    border-radius: 10px;

    background: #1d2029;

    border: 1px solid #343842;

    color: #f1f1f3;

    font-weight: 600;
}

[data-testid="stSidebar"] .st-key-new_chat_btn button:hover {

    background: #242832;

    border-color: #4a4f5c;
}
/* Sidebar section title */

.sidebar-section {
    margin-top: 1.5rem;
    margin-bottom: .55rem;

    color: #737782;

    font-size: .68rem;

    font-weight: 700;

    letter-spacing: 1px;

    text-transform: uppercase;
}
/* Conversation buttons */

[data-testid="stSidebar"] .stButton > button {

    justify-content: flex-start;

    text-align: left;

    background: transparent;

    border: none;

    color: #c6c8d0;

    box-shadow: none;

    border-radius: 8px;

    padding: .55rem .65rem;

    font-size: .88rem;
}

[data-testid="stSidebar"] .stButton > button:hover {

    background: #1d2029;

    color: #f4f4f5;
}
/* Delete buttons */

[data-testid="stSidebar"] .st-key-del_ button {

    color: #686c76;

    justify-content: center;
}

[data-testid="stSidebar"] .st-key-del_ button:hover {

    color: #f87171;

    background: transparent;
}

.empty-state {

    min-height: calc(100vh - 7.5rem);

    display: flex;

    flex-direction: column;

    justify-content: center;

    align-items: center;

    text-align: center;

    padding: 2rem;
}

.empty-title {

    font-size: 2rem;

    font-weight: 650;

    color: #f4f4f5;

    letter-spacing: -.7px;

    margin-bottom: .65rem;
}

.empty-description {

    max-width: 520px;

    color: #858994;

    font-size: .95rem;

    line-height: 1.6;
}
.document-card {

    display: flex;

    align-items: center;

    gap: 13px;

    padding: 13px 16px;

    margin: 1rem 0 2rem;

    background: #151821;

    border: 1px solid #292d37;

    border-radius: 12px;
}

.document-icon {

    width: 38px;

    height: 38px;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 9px;

    background: #20242e;

    font-size: 1.05rem;
}

.document-info {

    min-width: 0;

    flex: 1;
}

.document-name {

    color: #eeeeef;

    font-size: .9rem;

    font-weight: 600;

    white-space: nowrap;

    overflow: hidden;

    text-overflow: ellipsis;
}

.document-meta {

    margin-top: 3px;

    color: #777b86;

    font-size: .72rem;
}

.document-status {

    color: #6ee7a0;

    font-size: .72rem;

    white-space: nowrap;
}

[data-testid="stChatMessage"] {

    background: transparent !important;

    padding: 1rem 0 !important;

    border: none !important;
}


[data-testid="stChatMessageContent"] {

    color: #dedfe4;

    font-size: .95rem;

    line-height: 1.7;
}
/* Hide default avatars */

[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
[data-testid="stChatMessageAvatarCustom"] {

    display: none !important;
}

[data-testid="stChatMessage"]:has(
    [data-testid="stChatMessageContent"]
) {

    max-width: 100%;
}
.message-label {

    color: #8b8f9a;

    font-size: .72rem;

    font-weight: 600;

    letter-spacing: .3px;

    margin-bottom: .35rem;
}

.user-label {

    color: #a8abb5;
}

.assistant-label {

    color: #7aa2ff;
}


.source-header {

    color: #858994;

    font-size: .72rem;

    font-weight: 600;

    margin-top: 1rem;

    margin-bottom: .5rem;
}

.source-card {

    display: flex;

    align-items: center;

    gap: 9px;

    padding: 9px 11px;

    margin-bottom: 6px;

    border-radius: 8px;

    background: #151821;

    border: 1px solid #292d37;

    color: #aeb1ba;

    font-size: .75rem;
}

.source-icon {

    font-size: .8rem;
}

.st-key-file_chip {
    position: sticky;
    bottom: 6.6rem;
    margin-left: auto;
    margin-right: auto;
    width: min(420px, 100%);
    z-index: 999997;
}

.file-chip {
    display: flex;
    align-items: center;
    gap: 10px;
    width: fit-content;
    max-width: 100%;
    margin: 0 auto;
    padding: 9px 13px;
    border-radius: 11px;
    background: #171a22;
    border: 1px solid #30343f;
    box-shadow: 0 8px 24px rgba(0,0,0,.25);
}

.file-chip-icon {
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 7px;
    background: #242833;
    font-size: .85rem;
    flex: 0 0 28px;
}

.file-chip-info {
    min-width: 0;
    text-align: left;
}

.file-chip-name {
    color: #e8e8ea;
    font-size: .78rem;
    font-weight: 600;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 270px;
}

.file-chip-status {
    color: #6ee7a0;
    font-size: .66rem;
    margin-top: 2px;
}

.st-key-composer_bar {

    position: sticky;

    bottom: 1.15rem;

    margin: auto auto 0;

    width: min(1000px, 100%);

    z-index: 999998;
}


.st-key-composer_bar
[data-testid="stHorizontalBlock"] {

    display: flex !important;

    align-items: center !important;

    gap: 10px !important;
}

.st-key-composer_bar
[data-testid="stHorizontalBlock"] > div:first-child {

    flex: 0 0 52px !important;

    width: 52px !important;

    min-width: 52px !important;

    max-width: 52px !important;

    padding: 0 !important;
}

.st-key-composer_bar
[data-testid="stHorizontalBlock"] > div:nth-child(2) {

    flex: 1 1 auto !important;

    min-width: 0 !important;

    padding: 0 !important;
}

.st-key-composer_bar
[data-testid="stFileUploaderDropzone"] {

    visibility: hidden;

    width: 52px !important;

    height: 52px !important;

    min-height: 52px !important;

    padding: 0 !important;

    margin: 0 !important;

    border: none !important;

    background: transparent !important;

    overflow: visible !important;
}
.st-key-composer_bar
[data-testid="stFileUploaderDropzone"] button {

    visibility: visible !important;

    position: relative !important;

    width: 52px !important;

    height: 52px !important;

    min-width: 52px !important;

    padding: 0 !important;

    border-radius: 14px !important;

    border: 1px solid #343842 !important;

    background: #151821 !important;

    color: transparent !important;

    font-size: 0 !important;

    box-shadow: none !important;
}

.st-key-composer_bar
[data-testid="stFileUploaderDropzone"] button::after {

    content: "+";

    position: absolute;

    inset: 0;

    display: flex;

    align-items: center;

    justify-content: center;

    color: #eeeeef;

    font-size: 1.45rem;

    font-weight: 400;
}

.st-key-composer_bar
[data-testid="stFileUploaderDropzone"] button:hover {

    background: #1c202a !important;

    border-color: #4a4f5c !important;
}

.st-key-composer_bar
[data-testid="stFileUploaderFile"] {

    display: none !important;
}

.st-key-composer_bar
[data-testid="stChatInput"] {

    width: 100% !important;

    margin: 0 !important;

    padding: 0 !important;
}


.st-key-composer_bar
[data-testid="stChatInput"] > div {

    width: 100% !important;

    height: 52px !important;

    min-height: 52px !important;

    border-radius: 16px !important;

    background: #232630 !important;

    border: 1px solid #2d303a !important;

    box-shadow: none !important;

    padding: 0 !important;

    position: relative !important;
}

.st-key-composer_bar
[data-testid="stChatInput"] textarea {

    color: #eeeeef !important;

    font-size: .95rem !important;

    height: 52px !important;

    min-height: 52px !important;

    max-height: 52px !important;

    line-height: 52px !important;

    padding: 0 62px 0 18px !important;

    box-sizing: border-box !important;
}

.st-key-composer_bar
[data-testid="stChatInput"] textarea::placeholder {

    color: #858994 !important;
}

.st-key-composer_bar
[data-testid="stChatInput"] > div:focus-within {

    border-color: #414550 !important;

    box-shadow: 0 0 0 1px #414550 !important;
}

.st-key-composer_bar
[data-testid="stChatInput"] button {

    position: absolute !important;

    top: 50% !important;

    right: 8px !important;

    transform: translateY(-50%) !important;

    width: 40px !important;

    height: 40px !important;

    border-radius: 50% !important;

    background: #343743 !important;

    border: none !important;
}

@media (max-width: 700px) {

        .st-key-file_chip {
        bottom: 5.6rem;
        width: 100%;
    }

    .file-chip-name {
        max-width: 220px;
    }

    .block-container {

        padding-left: 1rem;

        padding-right: 1rem;
    }

    .st-key-composer_bar {

        width: 100%;

        bottom: .7rem;
    }

    .empty-title {

        font-size: 1.65rem;
    }

    .document-card {

        margin-top: .7rem;

        margin-bottom: 1.2rem;
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
    chains = st.session_state.chains
    if conv_id in chains:
        try:
            chains[conv_id][1].delete_collection()
        except Exception:
            pass
        chains.pop(conv_id)
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

    chains = st.session_state.chains
    if conv_id in chains:
        return chains[conv_id]
    pdf_path = os.path.join(PDF_DIR, f"{conv_id}.pdf")
    if not os.path.exists(pdf_path):
        return None
    chains[conv_id] = build_chain(pdf_path)
    return chains[conv_id]


st.session_state.setdefault("active_id", None)
st.session_state.setdefault("chains", {})
active_id = st.session_state.active_id
active = get_conversation(active_id) if active_id else None
messages = get_messages(active_id) if active_id else []

with st.sidebar:

    st.markdown(
        '<div style="font-size:.72rem;color:#777b86;'
        'margin-bottom:10px;">YOUR DOCUMENT ASSISTANT</div>',
        unsafe_allow_html=True
    )
    if st.button(
        "+  New chat",
        use_container_width=True,
        key="new_chat_btn"
    ):
        st.session_state.active_id = create_conversation()
        st.rerun()
    st.markdown(
        '<div class="sidebar-section">Chats</div>',
        unsafe_allow_html=True
    )

    conversations = list_conversations()

    if not conversations:

        st.markdown(
            '<div style="color:#696d77;font-size:.8rem;'
            'padding:.5rem;">No conversations yet.</div>',
            unsafe_allow_html=True
        )

    for conv in conversations:

        c1, c2 = st.columns([8, 1], gap="small")

        with c1:

            title = conv["title"] or "New conversation"

            if st.button(
                title,
                key=f"open_{conv['id']}",
                use_container_width=True
            ):

                st.session_state.active_id = conv["id"]

                st.rerun()

        with c2:

            if st.button(
                "×",
                key=f"del_{conv['id']}"
            ):

                if st.session_state.active_id == conv["id"]:

                    st.session_state.active_id = None

                delete_conversation(conv["id"])

                st.rerun()

    if active_id and messages:

        st.markdown("---")

        if st.button(
            "Clear conversation",
            use_container_width=True,
            key="clear_conv_btn"
        ):

            clear_messages(active_id)

            st.rerun()
st.markdown(
    '<div class="brand">PDFQUERY</div>',
    unsafe_allow_html=True
)

if not messages:

    st.markdown(
    '<div class="brand">PDFQUERY</div>',
    unsafe_allow_html=True
)

if not messages and not active:

    st.markdown(
        textwrap.dedent(
            """
            <div class="empty-state">
                <div class="empty-title">Ask your PDF anything.</div>
                <div class="empty-description">
                    Upload a PDF and have a conversation with its contents.
                    Ask questions, summarize sections, find specific information,
                    or explore the document.
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True
    )

elif not messages and active:

    st.markdown(
        textwrap.dedent(
            """
            <div class="empty-state">
                <div class="empty-title">What would you like to know?</div>
                <div class="empty-description">
                    Ask a question about this document.
                    I’ll find the relevant information and explain it.
                </div>
            </div>
            """
        ),
        unsafe_allow_html=True
    )

for msg in messages:

    role = msg["role"]

    if role == "user":

        with st.chat_message(
            "user",
            avatar=None
        ):

            st.markdown(
                '<div class="message-label user-label">'
                'You'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(msg["content"])
    else:
        with st.chat_message(
            "assistant",
            avatar=None
        ):

            st.markdown(
                '<div class="message-label assistant-label">'
                'PDFQUERY'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(msg["content"])


            sources = msg.get("sources") or []

            if sources:

                st.markdown(
                    '<div class="source-header">'
                    'Sources'
                    '</div>',
                    unsafe_allow_html=True
                )

                for i, source in enumerate(
                    sources[:5],
                    1
                ):

                    clean_source = (
                        source
                        .replace("\n", " ")
                        .strip()
                    )

                    if len(clean_source) > 180:

                        clean_source = (
                            clean_source[:180]
                            + "..."
                        )

                    st.markdown(
                        textwrap.dedent(
                            f"""
                            <div class="source-card">
                                <span class="source-icon">📄</span>
                                <span>Source {i} · {clean_source}</span>
                            </div>
                            """
                        ),
                        unsafe_allow_html=True
                    )

thinking_slot = st.container()

if active and active["doc_name"]:

    file_chip = st.container(key="file_chip")

    with file_chip:

        st.markdown(
            textwrap.dedent(
                f"""
                <div class="file-chip">
                    <div class="file-chip-icon">📄</div>
                    <div class="file-chip-info">
                        <div class="file-chip-name">{active["doc_name"]}</div>
                        <div class="file-chip-status">● Ready</div>
                    </div>
                </div>
                """
            ),
            unsafe_allow_html=True
        )

composer = st.container(
    key="composer_bar"
)

with composer:

    up_col, input_col = st.columns(
        [58, 1],
        gap="small",
        vertical_alignment="center"
    )

    with up_col:

        uploaded_file = st.file_uploader(
            "Upload PDF",
            type=["pdf"],
            label_visibility="collapsed",
            key=f"uploader_{active_id or 'none'}"
        )

    with input_col:

        if active and active["doc_name"]:

            question_placeholder = (
                "Ask about this PDF..."
            )

        else:

            question_placeholder = (
                "Upload a PDF to get started..."
            )

        question = st.chat_input(
            question_placeholder
        )
if uploaded_file is not None:

    conv_id = active_id or create_conversation()
    st.session_state.active_id = conv_id
    current = get_conversation(conv_id)

    if current["doc_name"] != uploaded_file.name:

        chains = st.session_state.chains

        if conv_id in chains:

            try:
                chains[conv_id][1].delete_collection()
            except Exception:
                pass

            chains.pop(conv_id)

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

        with thinking_slot:

            with st.chat_message("assistant", avatar=None):

                with st.spinner("Thinking..."):

                    chain, _ = get_chain(conv_id)  # rebuilds index for old chats automatically

                    result = chain.invoke(question)

                    docs = result["context"]
                    answer = strip_think(result["answer"])

        add_message(conv_id, "assistant", answer,
                    [d.page_content for d in docs])

        title = " ".join(question.strip().split())

        if len(title) > 32:
            title = title[:32].rstrip() + "..."

        set_title(conv_id, title)

        st.rerun()