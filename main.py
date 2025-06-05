import os
import re
from typing import List
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import openai
import pinecone
from langdetect import detect
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import numpy as np
load_dotenv()
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt')  #
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "quickstart"
if index_name not in [i.name for i in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=3072,  # for OpenAI `text-embedding-ada-002`
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws", region="us-east-1"  # or your preferred region
        )
    )
# Connect to an existing index
index = pc.Index(index_name)
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 30

def clean_text(text: str) -> str:
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    sentences = [s.strip() for s in sent_tokenize(text) if len(s.split()) > 4]

    sentences = [s for s in sentences if detect(s) == 'en']
    return '. '.join(sentences)

import numpy as np

def chunk_text(text: str, breakpoint_percentile=80, max_sentences=20) -> List[str]:
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    if len(sentences) <= 1:
        return [text]

    # Use OpenAI embeddings instead of SentenceTransformer
    embeddings = [get_embedding(sentence) for sentence in sentences]

    similarities = cosine_similarity(embeddings[:-1], embeddings[1:])
    distances = 1 - np.diagonal(similarities)

    cutoff = float(np.percentile(distances, breakpoint_percentile))

    chunks, current_chunk = [], [sentences[0]]
    for i, dist in enumerate(distances):
        if dist < cutoff and len(current_chunk) < max_sentences:
            current_chunk.append(sentences[i+1])
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentences[i+1]]
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks
def get_embedding(text: str) -> List[float]:
    
    response = openai.Embedding.create(input=[text], model="text-embedding-3-large")

    return response['data'][0]['embedding']

def upsert_chunks(chunks: List[dict], source_id: str, batch_size=50):
    vectors = []
    for i, chunk_info in enumerate(chunks):
        chunk = chunk_info["text"]
        embedding = get_embedding(chunk)
        vectors.append({
            "id": f"{source_id}_{chunk_info['page']}_{chunk_info['index_in_page']}",
            "values": embedding,
            "metadata": {
                "text": chunk,
                "source": source_id,
                "page": chunk_info["page"],
                "index_in_page": chunk_info["index_in_page"],
                "total_in_page": chunk_info["total_in_page"]
            }
        })

    for i in range(0, len(vectors), batch_size):
        index.upsert(vectors[i:i+batch_size])



def hybrid_search(query: str, top_k=5) -> List[dict]:
    # 1. Embed the query
    embedding = get_embedding(query)

    # 2. Initial dense search
    results = index.query(vector=embedding, top_k=top_k * 5, include_metadata=True)

    # 3. Define keyword match score
    def keyword_score(text: str, query: str):
        q_tokens = set(re.findall(r'\w+', query.lower()))
        t_tokens = set(re.findall(r'\w+', text.lower()))
        return len(q_tokens & t_tokens)

    # 4. Rerank using hybrid score
    reranked = sorted(
        results["matches"],
        key=lambda r: 0.6 * r["score"] + 0.4 * keyword_score(r["metadata"]["text"], query),
        reverse=True
    )

    top_results = reranked[:top_k]

    # 5. Collect all source chunks ONCE to enrich neighbors
    source_chunks_map = {}

    for r in top_results:
        source = r["metadata"]["source"]
        if source in source_chunks_map:
            continue

        # Retrieve all chunks from this source
        all_results = index.query(
            vector=embedding,  # still needs a vector for now
            filter={"source": {"$eq": source}},
            top_k=1000,
            include_metadata=True
        )

        chunks_by_page = {}
        for item in all_results["matches"]:
            meta = item["metadata"]
            key = (meta["page"], meta["index_in_page"])
            chunks_by_page[key] = meta
        source_chunks_map[source] = chunks_by_page

    # 6. Enrich each chunk with neighbors
    enriched = []
    for r in top_results:
        meta = r["metadata"]
        src = meta["source"]
        pg = meta["page"]
        idx = meta["index_in_page"]

        page_chunks = source_chunks_map[src]

        enriched.append({
            "chunk": meta["text"],
            "page": pg,
            "source": src,
            "previous": page_chunks.get((pg, idx - 1), {}).get("text"),
            "next": page_chunks.get((pg, idx + 1), {}).get("text")
        })

    # 7. Optional: print results nicely
    for i, item in enumerate(enriched, 1):
        print(f"{i}. Page {item['page']} | Source: {item['source']}")
        if item["previous"]:
            print(f"   Previous: {item['previous']}\n")
        print(f"   Chunk: {item['chunk']}\n")
        if item["next"]:
            print(f"   Next: {item['next']}\n")

    return enriched


@app.post("/upload_pdf/")
def upload_pdf(file: UploadFile, source_id: str = Form(...)):
    reader = PdfReader(file.file)
    
    all_chunks = []

    for page_num, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if not page_text:
            continue
        cleaned = clean_text(page_text)
        page_chunks = chunk_text(cleaned)
    
        for i, chunk in enumerate(page_chunks):
            all_chunks.append({
            "text": chunk,
            "page": page_num,
            "index_in_page": i,
            "total_in_page": len(page_chunks)
        })

    upsert_chunks(all_chunks, source_id)
    return {"message": f"PDF '{source_id}' processed and stored."}

@app.post("/query/")
def query_docs(req: QueryRequest):
    chunks = hybrid_search(req.question, req.top_k)
    # Extract just the chunk text from each dictionary
    context = "\n\n".join(chunk["chunk"] for chunk in chunks)
    print("contextzz: "+context)
    prompt = f"""Answer the question based on the following context:

{context}

Question: {req.question}
Answer:"""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
            {"role": "user", "content": prompt}
        ]
    )
    return {
        "chunks": chunks,
        "answer": response['choices'][0]['message']['content']
    }
