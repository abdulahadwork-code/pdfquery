import os
import tempfile
import streamlit as st
import gc
import uuid

from dotenv import load_dotenv
from rag_pipeline import build_chain, strip_think

load_dotenv()

st.set_page_config(
    page_title="PDFQUERY",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed"
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


if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "doc_name" not in st.session_state:
    st.session_state.doc_name = None

st.markdown(
    '<div class="brand">PDFQUERY</div>',
    unsafe_allow_html=True
)

if not st.session_state.messages:

    st.markdown(
        """
        <div class="home-container">
            <div class="greeting">
                What's on your mind today?
            </div>
        """,
        unsafe_allow_html=True
    )

    if st.session_state.chain:

        st.markdown(
            f"""
            <div class="sub-greeting">
                Ready — ask anything about
                <b>{st.session_state.doc_name}</b>
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

for msg in st.session_state.messages:

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

        if st.button(
            "Clear conversation",
            use_container_width=True
        ):

            st.session_state.messages = []

            st.rerun()


with chat_col:

    question = st.chat_input(
        "Ask anything"
    )


if (
    uploaded_file is not None
    and uploaded_file.name != st.session_state.doc_name
):

    if st.session_state.vectorstore is not None:

        try:
            st.session_state.vectorstore.delete_collection()
        except Exception:
            pass

        st.session_state.vectorstore = None
        st.session_state.chain = None

        gc.collect()

    st.session_state.messages = []

    st.session_state.doc_name = uploaded_file.name

    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"pdfquery_{uuid.uuid4().hex}.pdf"
    )

    with open(tmp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    with st.spinner("Reading your PDF..."):

        try:

            (
                st.session_state.chain,
                st.session_state.vectorstore
            ) = build_chain(tmp_path)

        except Exception as e:

            st.error(
                f"Could not process the PDF: {str(e)}"
            )

            st.session_state.chain = None
            st.session_state.vectorstore = None

            st.stop()

    try:
        os.remove(tmp_path)
    except OSError:
        pass

    st.rerun()

if question:

    if not st.session_state.chain:

        st.warning(
            "Please upload a PDF first using the ＋ button."
        )

    else:

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):

            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                result = st.session_state.chain.invoke(question)

                docs = result["context"]
                answer = strip_think(result["answer"])

            st.markdown(answer)

            with st.expander("Sources"):

                for i, doc in enumerate(
                    docs, 1
                ):
                    st.caption(
                        f"[{i}] "
                        f"{doc.page_content[:300]}..."
                    )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": [
                    d.page_content
                    for d in docs
                ],
            }
        )
