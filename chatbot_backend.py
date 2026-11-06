from __future__ import annotations

import os
import sqlite3
import tempfile
import requests

from typing import Annotated, Any, Dict, Literal, Optional, TypedDict

from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.vectorstores import FAISS

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# -------------------
# 1. LLM + embeddings
# -------------------
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")
)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------
# 2. PDF retriever store (per thread)
# -------------------
_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}


def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None
) -> dict:

    if not file_bytes:
        raise ValueError("No bytes received for ingestion.")

    thread_id = str(thread_id).strip()

    if not thread_id:
        raise ValueError("Invalid thread ID.")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:

        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        if not docs:
            raise ValueError("No readable pages found in PDF.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=700,
            chunk_overlap=100,
            separators=["\n\n", "\n", " ", ""]
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No text chunks extracted from PDF.")

        vector_store = FAISS.from_documents(
            chunks,
            embeddings
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        _THREAD_RETRIEVERS[thread_id] = retriever

        _THREAD_METADATA[thread_id] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks)
        }

        print("PDF INGESTED")
        print("THREAD ID:", repr(thread_id))
        print("AVAILABLE THREADS:", list(_THREAD_RETRIEVERS.keys()))

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks)
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# 3. Tools
# -------------------
search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: Literal["add", "sub", "mul", "div"],
) -> dict:
    """
    Perform a basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero is not allowed"}
            result = first_num / second_num

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }
    except Exception as e:
        return {"error": str(e)}


@tool
def get_stock_price(symbol: str) -> dict:
    """Fetch the latest stock quote using Alpha Vantage."""

    api_key = os.getenv("ALPHAVANTAGE_API_KEY")

    if not api_key:
        return {"error": "Alpha Vantage API key is missing."}

    url = "https://www.alphavantage.co/query"

    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as e:
        return {"error": str(e)}


@tool
def rag_tool(query: str, config: RunnableConfig) -> str:
    """
    Search the PDF uploaded in the current conversation
    and return relevant text with source page numbers.
    """

    configurable = (config or {}).get("configurable", {})
    thread_id = configurable.get("thread_id")

    print("\n--- RAG DEBUG ---")
    print("RAG THREAD ID:", repr(thread_id))
    print("AVAILABLE THREADS:", list(_THREAD_RETRIEVERS.keys()))

    if not thread_id:
        return (
            "NO_DOCUMENT: Conversation thread ID was not provided. "
            "Check the frontend graph configuration."
        )

    thread_id = str(thread_id).strip()

    retriever = _get_retriever(thread_id)

    print("RETRIEVER FOUND:", retriever is not None)

    if retriever is None:
        return (
            "NO_DOCUMENT: No PDF is indexed for this conversation. "
            "Ask the user to upload a PDF."
        )

    vector_store = getattr(retriever, "vectorstore", None)

    if vector_store is None:
        return "ERROR: PDF vector store is unavailable."

    try:
        results = vector_store.similarity_search_with_relevance_scores(
            query,
            k=3
        )

        print("RETRIEVED CHUNKS:", len(results))

        # Temporarily do not filter by relevance score.
        # We first want to verify that retrieval works.
        relevant_docs = results

        if not relevant_docs:
            return (
                "NOT_FOUND: No relevant information was found "
                "in the uploaded PDF."
            )

        filename = _THREAD_METADATA.get(
            thread_id, {}
        ).get("filename", "Unknown document")

        context = []

        for doc, score in relevant_docs:
            page = doc.metadata.get("page", "Unknown")

            # PyPDFLoader page numbers generally start at 0.
            if isinstance(page, int):
                page += 1

            context.append(
                f"""
Source: {filename}
Page: {page}
Content:
{doc.page_content}
"""
            )

        return "\n\n".join(context)

    except Exception as e:
        print("RAG ERROR:", repr(e))
        return f"ERROR: PDF retrieval failed: {str(e)}"

tools = [search_tool, get_stock_price, calculator, rag_tool]
llm_with_tools = llm.bind_tools(tools)

# -------------------
# 4. State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------
# 5. Nodes
# -------------------
def chat_node(state: ChatState, config=None):

    system_message = SystemMessage(
        content="""
You are an intelligent multi-utility AI assistant.

You have access to four tools:

1. rag_tool - Search uploaded PDF documents.
2. DuckDuckGoSearchRun - Search the web.
3. get_stock_price - Retrieve stock prices.
4. calculator - Perform mathematical calculations.

========================
PDF QUESTION RULES
========================

1. When the user asks a question about an uploaded PDF,
   ALWAYS call rag_tool first.

2. Do not ask the user to upload a PDF merely because
   you cannot see its contents in the conversation.

3. If rag_tool returns NO_DOCUMENT:
   Tell the user that no PDF is indexed for this chat
   and ask them to upload one.

4. If rag_tool returns NOT_FOUND:
   Tell the user that the requested information
   was not found in the uploaded document.

5. If relevant information is retrieved:
   - Read the retrieved context carefully.
   - Understand the information.
   - Generate a proper answer in natural language.
   - Do not simply repeat the retrieved chunks.
   - Mention page numbers when available.
   - Never invent information.

6. Never expose raw tool output, JSON,
   internal execution details or tool arguments.

========================
GENERAL QUESTIONS
========================

For general knowledge questions, answer normally.

Do not force every question through the PDF retriever.

========================
OTHER TOOLS
========================

Calculator:
Use for mathematical calculations.

Stock Price:
Use for stock-related queries.

Web Search:
Use for current information and web research.

========================
RESPONSE STYLE
========================

- Answer directly.
- Use simple, understandable language.
- Provide explanations when necessary.
- Use headings and bullet points when useful.
- Never display internal tool execution.
- Always provide a final user-facing answer.
"""
    )

    messages = [system_message, *state["messages"]]

    response = llm_with_tools.invoke(
        messages,
        config=config
    )

    return {"messages": [response]}

tool_node = ToolNode(tools)

# -------------------
# 6. Checkpointer
# -------------------
conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
conn.execute("""
    CREATE TABLE IF NOT EXISTS thread_titles (
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL
    )
""")

conn.commit()
checkpointer = SqliteSaver(conn=conn)

# -------------------
# 7. Graph
# -------------------
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)

# -------------------
# 8. Helpers
# -------------------
def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


def generate_thread_title(user_message: str) -> str:
    """Generate a short, meaningful conversation title."""

    prompt = f"""
    Generate a concise title for a chatbot conversation.

    User's first message:
    {user_message}

    Rules:
    - Maximum 5 words.
    - Use a meaningful topic-based title.
    - Do not use quotation marks.
    - Do not include phrases like "Chat about".
    - Return only the title.
    """

    response = llm.invoke(prompt)

    title = response.content.strip()

    # Remove accidental quotation marks
    title = title.strip('"').strip("'")

    return title[:60]


def save_thread_title(thread_id: str, title: str):
    conn.execute(
        """
        INSERT OR REPLACE INTO thread_titles (thread_id, title)
        VALUES (?, ?)
        """,
        (str(thread_id), title)
    )
    conn.commit()


def get_thread_title(thread_id: str) -> str:
    cursor = conn.execute(
        "SELECT title FROM thread_titles WHERE thread_id = ?",
        (str(thread_id),)
    )

    result = cursor.fetchone()

    return result[0] if result else "New Chat"


def retrieve_thread_titles():
    cursor = conn.execute(
        "SELECT thread_id, title FROM thread_titles"
    )

    return dict(cursor.fetchall())