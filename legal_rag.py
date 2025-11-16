# legal_rag.py
"""
Self-contained CLT-RAG module – no UI, no streamlit, no FastAPI.
Only dependency: Postgres + pgvector already populated by ingest_clt.py
"""

from __future__ import annotations
import os
import psycopg2
import psycopg2.extras
from typing import List, TypedDict

from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
# Response contracts
# ------------------------------------------------------------------
class Citation(TypedDict):
    source: str          # always "CLT" today
    article: str         # "Art. 483"
    excerpt: str         # matched chunk
    score: float        # 1 - cosine_distance

class RAGAnswer(TypedDict):
    answer: str
    citations: List[Citation]

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
POSTGRES_DSN = os.getenv("POSTGRES_DSN", "postgresql://clt_user:clt_pass@localhost:5432/clt_db")
EMBED_MODEL = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
EMBED_DIM = 384

TOP_K_VECTOR = 10          # first semantic retrieval
TOP_K_RERANK = 4           # sent to LLM after re-rank

LLM = ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=GROQ_API_KEY,temperature=0)

PROMPT = ChatPromptTemplate.from_template("""
Você é um advogado trabalhista especializado na CLT.

Responda **somente** com base nos trechos abaixo.  
Se a resposta não estiver presente no contexto, diga claramente:
"Não encontrei essa informação na CLT fornecida."

Instruções:
- Cite os artigos exatamente entre aspas.
- Não invente leis, artigos ou jurisprudências.
- Não traga conhecimento externo.

Contexto:
{context}

Pergunta:
{question}

Resposta:
""")


# ------------------------------------------------------------------
# Vector search
# ------------------------------------------------------------------
def _vector_search(question: str, k: int = TOP_K_VECTOR) -> List[dict]:
    """Return list of dicts with keys: id,article_id,title,body,score"""
    emb = EMBED_MODEL.encode(question, normalize_embeddings=True).tolist()
    with psycopg2.connect(POSTGRES_DSN) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id,
                       article_id,
                       title,
                       body,
                       1 - (embedding <=> %(vec)s) AS score
                FROM   clt_chunks
                ORDER  BY embedding <=> %(vec)s
                LIMIT  %(k)s;
                """,
                {"vec": emb, "k": k},
            )
            rows = cur.fetchall()
    return rows

# ------------------------------------------------------------------
# Re-rank (optional but cheap: keep only highest scores)
# ------------------------------------------------------------------
def _rerank(rows: List[dict]) -> List[dict]:
    """Trim to TOP_K_RERANK best scores."""
    return sorted(rows, key=lambda r: r["score"], reverse=True)[:TOP_K_RERANK]

# ------------------------------------------------------------------
# Build context string
# ------------------------------------------------------------------
def _format_context(rows: List[dict]) -> str:
    return "\n\n".join(
        f"Fonte: CLT – {row['article_id']}\nTrecho: {row['body']}"
        for row in rows
    )

# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------
def answer_about_clt(question: str) -> RAGAnswer:
    """
    Performs vector search → re-rank → LLM → structured answer.
    """
    # 1. Retrieve
    hits = _vector_search(question)
    # 2. Re-rank
    top = _rerank(hits)
    # 3. Generate
    context = _format_context(top)
    chain = PROMPT | LLM | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    # 4. Build citations
    citations: List[Citation] = [
        Citation(
            source="CLT",
            article=row["article_id"],
            excerpt=row["body"][:400],  # first 400 chars
            score=round(float(row["score"]), 4),
        )
        for row in top
    ]
    return RAGAnswer(answer=answer, citations=citations)

# ------------------------------------------------------------------
# Quick self-test (only when executed directly)
# ------------------------------------------------------------------
if __name__ == "__main__":
    print(answer_about_clt("Qual o prazo para homologar rescisão?"))