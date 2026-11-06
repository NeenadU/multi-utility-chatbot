
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import uuid
import streamlit as st

st.set_page_config(
    page_title="Multi Utility Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("Multi Utility Chatbot")
st.info("Initializing chatbot backend...")

with st.spinner("Loading LLM, embeddings and LangGraph..."):
    from chatbot_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
    generate_thread_title,
    save_thread_title,
    get_thread_title,
    retrieve_thread_titles,
)

st.success("Chatbot backend initialized successfully!")
st.write("Frontend execution started...")

st.write("Loading chatbot backend...")


# =========================== Utilities ===========================
def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    thread_id = str(thread_id)

    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id

    add_thread(thread_id)

    st.session_state["thread_names"][thread_id] = "New Chat"

    st.session_state["message_history"] = []

    st.rerun()

def load_conversation(thread_id):

    thread_id = str(thread_id)

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])

# ======================= Session Initialization ===================
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()
else:
    st.session_state["thread_id"] = str(
        st.session_state["thread_id"]
    )

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = [
        str(t) for t in retrieve_all_threads()
    ]

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

if "thread_names" not in st.session_state:
    st.session_state["thread_names"] = retrieve_thread_titles()

add_thread(st.session_state["thread_id"])

thread_key = str(st.session_state["thread_id"])
thread_docs = st.session_state["ingested_docs"].setdefault(thread_key, {})
threads = st.session_state["chat_threads"][::-1]
selected_thread = None

# ============================ Sidebar ============================
st.sidebar.title("LangGraph PDF Chatbot")
st.sidebar.markdown(f"**Thread ID:** `{thread_key}`")

if st.sidebar.button("New Chat", use_container_width=True):
    reset_chat()
    st.rerun()

if thread_docs:
    latest_doc = list(thread_docs.values())[-1]
    st.sidebar.success(
        f"Using `{latest_doc.get('filename')}` "
        f"({latest_doc.get('chunks')} chunks from {latest_doc.get('documents')} pages)"
    )
else:
    st.sidebar.info("No PDF indexed yet.")

uploaded_pdf = st.sidebar.file_uploader("Upload a PDF for this chat", type=["pdf"])
if uploaded_pdf:

    thread_key = str(st.session_state["thread_id"])

    if uploaded_pdf.name in thread_docs:

        st.sidebar.info(
            f"{uploaded_pdf.name} already processed for this chat."
        )

    else:

        with st.sidebar.status(
            "Indexing PDF...",
            expanded=True
        ) as status_box:

            try:

                summary = ingest_pdf(
                    file_bytes=uploaded_pdf.getvalue(),
                    thread_id=thread_key,
                    filename=uploaded_pdf.name
                )

                thread_docs[uploaded_pdf.name] = summary

                status_box.update(
                    label="PDF indexed successfully",
                    state="complete",
                    expanded=False
                )

                st.rerun()

            except Exception as e:

                status_box.update(
                    label="PDF indexing failed",
                    state="error",
                    expanded=True
                )

                st.sidebar.error(str(e))
st.sidebar.subheader("Past conversations")

if not threads:
    st.sidebar.write("No past conversations yet.")

else:
    for thread_id in threads:

        thread_key_item = str(thread_id)

        thread_name = st.session_state["thread_names"].get(
            thread_key_item
        )

        if not thread_name:
            thread_name = get_thread_title(thread_key_item)

        if st.sidebar.button(
            thread_name,
            key=f"side-thread-{thread_key_item}",
            use_container_width=True
        ):
            selected_thread = thread_id

# ============================ Main Layout ========================
st.title("Multi Utility Chatbot")

# Chat area
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

user_input = st.chat_input("Ask about your document or use tools")

if user_input:

    thread_key = str(st.session_state["thread_id"])

    # Generate a title only for a new conversation
    if get_thread_title(thread_key) == "New Chat":

        with st.spinner("Generating conversation title..."):
            try:
                title = generate_thread_title(user_input)

                if not title:
                    title = user_input[:40]

            except Exception:
                title = user_input[:40]

            save_thread_title(thread_key, title)

            st.session_state["thread_names"][thread_key] = title

    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.text(user_input)

    CONFIG = {
        "configurable": {"thread_id": thread_key},
        "metadata": {"thread_id": thread_key},
        "run_name": "chat_turn",
    }

    with st.chat_message("assistant"):
        status_holder = {"box": None}

    def ai_only_stream():

        for message_chunk, metadata in chatbot.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config=CONFIG,
            stream_mode="messages",
        ):

            if isinstance(message_chunk, ToolMessage):

                tool_name = getattr(
                    message_chunk,
                    "name",
                    "tool"
                )

                if status_holder["box"] is None:

                    status_holder["box"] = st.status(
                        f"Using {tool_name}...",
                        expanded=True
                    )

                else:

                    status_holder["box"].update(
                        label=f"Using {tool_name}...",
                        state="running",
                        expanded=True
                    )

                continue

            if isinstance(message_chunk, AIMessage):

                # Ignore AI messages that are requesting tool execution.
                if message_chunk.tool_calls:
                    continue

                # Ignore empty intermediate messages.
                content = message_chunk.content

                if isinstance(content, str) and content:
                    yield content

                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            text_content = block.get("text")
                            if text_content:
                              yield text_content
    ai_message = st.write_stream(ai_only_stream())

    if status_holder["box"] is not None:
        status_holder["box"].update(
            label="✅ Tool finished", state="complete", expanded=False
        )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    doc_meta = thread_document_metadata(thread_key)
    if doc_meta:
        st.caption(
            f"Document indexed: {doc_meta.get('filename')} "
            f"(chunks: {doc_meta.get('chunks')}, pages: {doc_meta.get('documents')})"
        )

st.divider()

if selected_thread:

    selected_thread = str(selected_thread)

    st.session_state["thread_id"] = selected_thread

    messages = load_conversation(selected_thread)

    temp_messages = []

    for msg in messages:

        if isinstance(msg, HumanMessage):

            temp_messages.append({
                "role": "user",
                "content": msg.content
            })

        elif isinstance(msg, AIMessage):

            # Do not display internal tool-call messages.
            if msg.tool_calls:
                continue

            if isinstance(msg.content, str) and msg.content:

                temp_messages.append({
                    "role": "assistant",
                    "content": msg.content
                })

    st.session_state["message_history"] = temp_messages

    st.session_state["ingested_docs"].setdefault(
        selected_thread,
        {}
    )

    st.rerun()