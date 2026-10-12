import os
from pathlib import Path

from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# ENVIRONMENT VARIABLES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        f"Gemini API key not found.\nCheck your .env file:\n{ENV_FILE}"
    )

os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ============================================================
# MODELS  (temperature=0 -> factual, repeatable answers)
# ============================================================

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0)


# ============================================================
# PROMPTS
# ============================================================

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a precise document question-answering assistant.

Rules:
1. Answer using ONLY the context below. Never use outside knowledge.
2. If the context does not contain the answer, reply exactly:
   "I don't know based on the uploaded documents."
3. If the context only partly answers the question, give the part that is
   supported and clearly say what is missing.
4. Copy numbers, dates, names and technical terms exactly as written.
5. When you state a fact, cite it like (file.pdf, p.3) or (file.docx),
   using the [Source: ...] labels from the context.
6. If different sources contradict each other, mention both.

Context:
{context}
""",
        ),
        ("human", "{question}"),
    ]
)

condense_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """Rewrite the user's latest question as a single standalone
question, using the chat history to resolve pronouns and references
(e.g. "it", "that section", "the second one").
Do NOT answer it. If it is already standalone, return it unchanged.
Return only the rewritten question.""",
        ),
        ("human", "Chat history:\n{history}\n\nLatest question: {question}"),
    ]
)


# ============================================================
# HELPERS
# ============================================================

def _label(doc) -> str:
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page")
    return f"{source}, p.{page}" if page else source


def _format_docs(docs) -> str:
    return "\n\n---\n\n".join(
        f"[Source: {_label(d)}]\n{d.page_content}" for d in docs
    )


def _format_history(history, max_turns: int = 6) -> str:
    lines = []
    for m in history[-max_turns:]:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content'][:500]}")
    return "\n".join(lines)


# ============================================================
# RAG CHAIN
# ============================================================

class RAGChain:
    def __init__(self, retriever):
        self.retriever = retriever
        self.condense = condense_prompt | llm | StrOutputParser()
        self.answer = answer_prompt | llm | StrOutputParser()

    def invoke(self, question: str, history=None) -> dict:
        history = history or []

        # 1) Make follow-up questions standalone so retrieval works
        search_query = question
        if history:
            try:
                search_query = self.condense.invoke(
                    {"history": _format_history(history), "question": question}
                ).strip() or question
            except Exception:
                search_query = question

        # 2) Retrieve
        docs = self.retriever.invoke(search_query)

        # 3) Answer
        answer = self.answer.invoke(
            {"context": _format_docs(docs), "question": question}
        )

        # 4) Unique sources for display
        sources, seen = [], set()
        for d in docs:
            label = _label(d)
            if label not in seen:
                seen.add(label)
                sources.append(
                    {"label": label, "snippet": d.page_content[:300].strip()}
                )

        return {"answer": answer, "sources": sources}


def create_rag_chain(chunks) -> RAGChain:
    valid_chunks = [c for c in chunks if c.page_content and c.page_content.strip()]

    print("Total chunks:", len(chunks))
    print("Valid chunks:", len(valid_chunks))

    if not valid_chunks:
        raise ValueError(
            "No readable text was extracted from the uploaded files. "
            "PDFs may be scanned/image-based."
        )

    vectorstore = FAISS.from_documents(documents=valid_chunks, embedding=embeddings)

    # MMR = relevant AND diverse chunks (less duplicate overlap text)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 20, "lambda_mult": 0.7},
    )

    return RAGChain(retriever)
