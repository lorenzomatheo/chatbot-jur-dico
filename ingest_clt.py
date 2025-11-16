import os, sqlite3, unicodedata, re
from pathlib import Path
from typing import List

import psycopg2, psycopg2.extras
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Config
# --------------------------------------------------
DATA_DIR   = Path("data/clt")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

DSN = "postgresql://clt_user:clt_pass@localhost:5432/clt_db"

EMBED_MODEL = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBED_DIM = 384

# --------------------------------------------------
# HTML -> (title, body)
# --------------------------------------------------
def extract_from_html(file: Path) -> tuple[str, str]:
    html = file.read_text(encoding="latin-1", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious trashs
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # ---------- TITLE ----------
    # 1) try to get the link of DECRETO-LEI 5.452
    main_link = soup.find("a", href=re.compile(r"DEL%205\.452-1943", re.I))
    if main_link:
        title = main_link.get_text(strip=True)
    else:
        # 2) try to catch the heading "CONSOLIDAÇÃO DAS LEIS DO TRABALHO"
        clt_heading_node = soup.find(
            string=re.compile(r"CONSOLIDA..O DAS LEIS DO TRABALHO", re.I)
        )
        if clt_heading_node:
            title = clt_heading_node.strip()
        elif soup.title:
            title = soup.title.get_text(strip=True)
        else:
            title = file.stem

    # ---------- BODY ----------
    clt_heading_node = soup.find(
        string=re.compile(r"CONSOLIDA..O DAS LEIS DO TRABALHO", re.I)
    )

    if clt_heading_node:
        start_p = clt_heading_node.find_parent("p")
        if not start_p:
            # fallback: climb up to a reasonably sized container
            start_p = clt_heading_node.find_parent(["div", "td", "body"])

        parts = []

        heading_text = clt_heading_node.strip()
        if heading_text:
            parts.append(heading_text)

        # From this <p>, grab ALL the next <p>
        # (TÍTULO I, INTRODUÇÃO, Art. 1º, etc.)
        for sib in start_p.find_all_next("p"):
            text = sib.get_text(" ", strip=True)
            if text:
                parts.append(text)

        body = "\n\n".join(parts)
    else:
        # If it's hard to find the heading, go for the whole body
        body = soup.get_text("\n", strip=True)

    # Normalize accentuation/Unicode
    body = unicodedata.normalize("NFKC", body)
    title = unicodedata.normalize("NFKC", title)

    return title, body

# --------------------------------------------------
# 1. Chunking 
# --------------------------------------------------
def chunk_article(article_id: str, title: str, text: str) -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    docs = splitter.create_documents(
        texts=[text],
        metadatas=[{"article_id": article_id, "title": title}],
    )
    return [
        {
            "id": f"{article_id}#{i}",
            "article_id": article_id,
            "title": title,
            "text": doc.page_content,
            "start_idx": doc.metadata["start_index"],
        }
        for i, doc in enumerate(docs)
    ]

# --------------------------------------------------
# 2. SQLite exact-text index 
# --------------------------------------------------
def build_text_index(chunks: List[dict], db_path: str = "clt_text.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS clt USING fts5(id, title, body)")
    conn.executemany(
        "INSERT INTO clt(id, title, body) VALUES (?, ?, ?)",
        [(chk["id"], chk["title"], chk["text"]) for chk in chunks]
    )
    conn.commit()
    return conn

# --------------------------------------------------
# 3. Postgres + pgvector upsert 
# --------------------------------------------------
def upsert_pgvector(chunks: List[dict]):
    texts = [c["text"] for c in chunks]

    vectors = EMBED_MODEL.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # opcional, but good for cosine similarity
    )

    with psycopg2.connect(DSN) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS clt_chunks (
                    id TEXT PRIMARY KEY,
                    article_id TEXT,
                    title TEXT,
                    body TEXT,
                    embedding vector(384)
                );
            """)

            psycopg2.extras.execute_batch(
                cur,
                """
                INSERT INTO clt_chunks (id, article_id, title, body, embedding)
                VALUES (%(id)s, %(article_id)s, %(title)s, %(body)s, %(embedding)s)
                ON CONFLICT (id) DO UPDATE
                SET body=EXCLUDED.body, embedding=EXCLUDED.embedding;
                """,
                [
                    {
                        "id": c["id"],
                        "article_id": c["article_id"],
                        "title": c["title"],
                        "body": c["text"],
                        "embedding": list(map(float, vec)),
                    }
                    for c, vec in zip(chunks, vectors)
                ],
            )
        conn.commit()
    print(f"Upserted {len(chunks)} chunks to Postgres pgvector.")

# --------------------------------------------------
# 4. Orchestrate
# --------------------------------------------------
def main():
    all_chunks = []
    for file in sorted(DATA_DIR.glob("*.html")):
        article_id = file.stem
        title, body = extract_from_html(file)
        all_chunks.extend(chunk_article(article_id, title, body))

    build_text_index(all_chunks)
    upsert_pgvector(all_chunks)
    print("Ingestion complete.")

if __name__ == "__main__":
    main()
