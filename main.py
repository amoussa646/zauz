import os
import re
from typing import List, Dict, Optional
from fastapi import FastAPI, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PyPDF2 import PdfReader
from pydantic import BaseModel
import requests
import torch
from PIL import Image
import open_clip
from sentence_transformers import SentenceTransformer
import openai
import pinecone
from langdetect import detect
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
import numpy as np
import json
from pathlib import Path
import google.generativeai as genai
from google.generativeai.types import Tool
from google.generativeai.types.generation_types import GenerationConfig
import asyncio
import logging
from livekit import rtc, api
from livekit.rtc.room import Room
import wave
import io
import speech_recognition as sr
load_dotenv()
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.metrics.pairwise import cosine_similarity

# nltk.download('punkt')  #
openai.api_key = os.getenv("OPENAI_API_KEY")
# Initialize Gemini
GOOGLE_API_KEY=os.getenv("GOOGLE_CLOUD_KEY")
GOOGLE_CSE_ID=os.getenv("GOOGLE_CSE_ID")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel(
    model_name='gemini-2.0-flash-001'  # Removed tools configuration
)
# Image handling
IMAGES_DIR = Path("images")
IMAGES_METADATA_FILE = IMAGES_DIR / "metadata.json"

# Initialize CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
clip_model = clip_model.to(device)

class ImageMetadata(BaseModel):
    id: str
    filename: str
    embedding: Optional[List[float]] = None  # Store CLIP embedding

class ImageStore:
    def __init__(self):
        self.images_dir = IMAGES_DIR
        self.metadata_file = IMAGES_METADATA_FILE
        self.images_dir.mkdir(exist_ok=True)
        self.metadata: Dict[str, ImageMetadata] = self._load_metadata()
        # Precompute embeddings for all images
        self._precompute_embeddings()

    def _load_metadata(self) -> Dict[str, ImageMetadata]:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                return {k: ImageMetadata(**v) for k, v in data.items()}
        return {}

    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump({k: v.dict() for k, v in self.metadata.items()}, f, indent=2)

    def _precompute_embeddings(self):
        """Precompute CLIP embeddings for all images"""
        for image_id, metadata in self.metadata.items():
            if metadata.embedding is None:
                image_path = self.images_dir / metadata.filename
                if image_path.exists():
                    try:
                        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                        with torch.no_grad():
                            image_features = clip_model.encode_image(image)
                            metadata.embedding = image_features.cpu().numpy().tolist()[0]
                    except Exception as e:
                        print(f"Error computing embedding for {image_id}: {e}")
        self._save_metadata()

    def add_image(self, file: UploadFile, metadata: ImageMetadata):
        if len(self.metadata) >= 6:
            raise HTTPException(status_code=400, detail="Maximum of 6 images allowed")
        
        # Save image file
        file_path = self.images_dir / file.filename
        with open(file_path, "wb") as f:
            f.write(file.file.read())
        
        # Compute CLIP embedding
        try:
            image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(image)
                metadata.embedding = image_features.cpu().numpy().tolist()[0]
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
        
        # Save metadata with embedding
        self.metadata[metadata.id] = metadata
        self._save_metadata()

    def get_image(self, image_id: str) -> Optional[tuple[Path, ImageMetadata]]:
        if image_id not in self.metadata:
            return None
        metadata = self.metadata[image_id]
        image_path = self.images_dir / metadata.filename
        if not image_path.exists():
            return None
        return image_path, metadata

    def get_relevant_images(self, query: str, top_k: int = 1) -> List[tuple[Path, ImageMetadata]]:
        # Encode the text query using CLIP
        text_tokens = open_clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(text_tokens)
            text_features = text_features.cpu().numpy()[0]

        # Calculate similarities with all images
        image_scores = []
        for metadata in self.metadata.values():
            if metadata.embedding is None:
                continue
            similarity = cosine_similarity([text_features], [metadata.embedding])[0][0]
            image_scores.append((metadata, similarity))
        
        # Sort by similarity score and get top k
        image_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_images = []
        
        for metadata, score in image_scores[:top_k]:
            image_path = self.images_dir / metadata.filename
            if image_path.exists():
                relevant_images.append((image_path, metadata))
        
        return relevant_images

    def get_image_by_filename(self, filename: str) -> Optional[tuple[Path, ImageMetadata]]:
        """Get image by filename"""
        image_path = self.images_dir / filename
        if not image_path.exists():
            return None
        # Find metadata by filename
        for metadata in self.metadata.values():
            if metadata.filename == filename:
                return image_path, metadata
        return None

# Initialize image store
image_store = ImageStore()

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
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Explicitly list allowed methods
    allow_headers=["*"],
    expose_headers=["*"]
)

# LiveKit configuration
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")

# Initialize LiveKit room and API
room = None
lkapi = None

async def initialize_livekit():
    global room, lkapi
    try:
        if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
            raise ValueError("LIVEKIT_API_KEY and LIVEKIT_API_SECRET must be set in environment variables")
            
        # Initialize LiveKit API with credentials
        api_url = LIVEKIT_URL.replace("ws://", "http://").replace("wss://", "https://")
        lkapi = api.LiveKitAPI(
            api_url,
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        )
        
        # Create room if it doesn't exist
        try:
            await lkapi.room.create_room(
                api.CreateRoomRequest(name="voice_room")
            )
        except Exception as e:
            logging.info(f"Room might already exist: {str(e)}")
        
        # Generate access token
        token = api.AccessToken(
            api_key=LIVEKIT_API_KEY,
            api_secret=LIVEKIT_API_SECRET
        ).with_identity("server") \
         .with_name("Server") \
         .with_grants(api.VideoGrants(
            room_join=True,
            room="voice_room"
        )).to_jwt()
        
        # Initialize and connect room
        room = Room()
        
        # Set up event handlers
        @room.on("participant_connected")
        def on_participant_connected(participant: rtc.RemoteParticipant):
            logging.info(f"Participant connected: {participant.sid} {participant.identity}")
        
        @room.on("track_subscribed")
        def on_track_subscribed(track: rtc.Track, publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant):
            logging.info(f"Track subscribed: {publication.sid}")
            if track.kind == rtc.TrackKind.KIND_AUDIO:
                # Handle audio track subscription
                logging.info(f"Audio track subscribed from {participant.identity}")
        
        # Connect to room
        await room.connect(LIVEKIT_URL, token)
        logging.info(f"Connected to room {room.name}")
        
        return room
    except Exception as e:
        logging.error(f"Error initializing LiveKit: {str(e)}")
        raise

# Create startup event handler
@app.on_event("startup")
async def startup_event():
    await initialize_livekit()

# Create shutdown event handler
@app.on_event("shutdown")
async def shutdown_event():
    global lkapi
    if lkapi:
        await lkapi.aclose()

class QueryRequest(BaseModel):
    question: str
    top_k: int = 30
    include_images: bool = True

class VoiceQueryRequest(BaseModel):
    audio_data: bytes
    sample_rate: int = 16000
    sample_width: int = 2

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



def hybrid_search(query: str, top_z=1) -> List[dict]:
    # 1. Embed the query
    top_k=80
    embedding = get_embedding(query)

    # 2. Initial dense search
    results = index.query(vector=embedding, top_k=top_k , include_metadata=True)

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
        pg = meta["page"]
        idx = meta["index_in_page"]
        src = meta["source"]
        hybrid_score = 0.6 * r["score"] + 0.4 * keyword_score(meta["text"], query)

        enriched.append({
        "chunk": meta["text"],
        "page": pg,
        "source": src,
        "score": hybrid_score,  # <-- Add this line
        "previous": source_chunks_map[src].get((pg, idx - 1), {}).get("text"),
        "next": source_chunks_map[src].get((pg, idx + 1), {}).get("text")
    })

    # 7. Optional: print results nicely
    for i, item in enumerate(enriched, 1):
        print(f"{i}. Page {item['page']} | Source: {item['source']} | Score: {item['score']}")
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

@app.post("/upload_image/")
async def upload_image(
    file: UploadFile,
    image_id: str = Form(...)
):
    metadata = ImageMetadata(
        id=image_id,
        filename=file.filename
    )
    image_store.add_image(file, metadata)
    return {"message": f"Image '{image_id}' uploaded successfully"}

@app.get("/images/{filename}")
async def get_image(filename: str):
    result = image_store.get_image_by_filename(filename)
    if not result:
        raise HTTPException(status_code=404, detail="Image not found")
    image_path, metadata = result
    return FileResponse(image_path, media_type="image/jpeg")

@app.post("/query_docs/")
async def query_docs(req: QueryRequest):
    # First try RAG search
    chunks = hybrid_search(req.question, req.top_k)

    # Filter chunks based on a minimum hybrid relevance score
    MIN_SCORE_THRESHOLD = 0.9  # tune this threshold
    relevant_chunks = [c for c in chunks if c.get("score", 0) >= MIN_SCORE_THRESHOLD]

    # Get relevant images using CLIP (do this regardless of text search result)
    relevant_images = []
    if req.include_images:
        relevant_images = image_store.get_relevant_images(req.question, top_k=1)
    
    # Prepare image context
    image_context = ""
    if relevant_images:
        image_context = "\nRelevant images:\n" + "\n".join(
            f"- {metadata.filename}"
            for _, metadata in relevant_images
        )

    # If we have relevant chunks, use them for the answer
    if relevant_chunks:  
        # Only use the first chunk
        context = relevant_chunks[0]["chunk"]
        prompt = f"""Answer the question based on the following context:

{context}
{image_context}
if the context does not contain relative information then perform google search
Question: {req.question}
Answer:"""
        response = model.generate_content(prompt)
        answer = response.text if hasattr(response, 'text') else str(response)
    else:
        # Fallback to direct question
        prompt = f"answer the following question in a summerize way in form of regular sentences : Question: {req.question}\nAnswer:"
        response = model.generate_content(prompt)
        try:
            if hasattr(response, 'text'):
                answer = response.text
            elif hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content.parts:
                    answer = candidate.content.parts[0].text
                else:
                    answer = str(response)
            else:
                answer = str(response)
        except Exception as e:
            print(f"Error processing response: {e}")
            answer = "I apologize, but I encountered an error processing the response."
    
    return {
        "chunks": [relevant_chunks[0]] if relevant_chunks else [],  # Only return the first chunk
        "answer": answer,
        "relevant_images": [
            {
                "filename": metadata.filename
            }
            for _, metadata in relevant_images
        ] if req.include_images else [],
        "source": "rag" if relevant_chunks else "web"  # Indicate the source of the answer
    }

@app.websocket("/ws/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    recognizer = sr.Recognizer()
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Convert bytes to audio file
            audio_io = io.BytesIO(data)
            with wave.open(audio_io, 'rb') as wav_file:
                # Read audio data
                audio_data = wav_file.readframes(wav_file.getnframes())
                
                # Convert to AudioData for speech recognition
                audio = sr.AudioData(
                    audio_data,
                    sample_rate=wav_file.getframerate(),
                    sample_width=wav_file.getsampwidth()
                )
                
                try:
                    # Perform speech recognition
                    text = recognizer.recognize_google(audio)
                    
                    # Process the query using existing query_docs endpoint
                    query_request = QueryRequest(question=text)
                    response = await query_docs(query_request)
                    
                    # Add the transcript to the response
                    response["transcript"] = text
                    
                    # Send back the response
                    await websocket.send_json(response)
                    
                except sr.UnknownValueError:
                    await websocket.send_json({
                        "error": "Could not understand audio",
                        "answer": "I'm sorry, I couldn't understand what you said. Could you please try again?",
                        "transcript": ""
                    })
                except sr.RequestError as e:
                    await websocket.send_json({
                        "error": f"Could not request results; {str(e)}",
                        "answer": "I'm sorry, there was an error processing your voice input. Please try again.",
                        "transcript": ""
                    })
                    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"Error in websocket: {str(e)}")
        await websocket.close()

# Add CORS middleware to allow WebSocket connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)
