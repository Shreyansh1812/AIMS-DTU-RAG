import os
import requests
import uuid
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, SparseVectorParams
from fastembed import TextEmbedding, SparseTextEmbedding

# --- CONFIGURATION ---
PDF_DIR = "./test_pdfs"
QDRANT_PATH = "./qdrant_arxiv_db"
COLLECTION_NAME = "arxiv_phase1_hybrid"
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"

# Exact models specified in your AIMS-DTU report
DENSE_MODEL_NAME = "BAAI/bge-small-en-v1.5"
SPARSE_MODEL_NAME = "prithivida/Splade_PP_en_v1"

def process_pdf_with_grobid(pdf_path):
    """Sends a PDF to local Grobid and extracts sections via TEI XML."""
    filename = os.path.basename(pdf_path)
    print(f"📄 Processing: {filename} via Grobid...")
    
    with open(pdf_path, 'rb') as f:
        files = {'input': (filename, f, 'application/pdf')}
        try:
            response = requests.post(GROBID_URL, files=files, timeout=60)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            print(f"❌ Grobid failed on {filename}: {e}")
            print("Ensure Grobid is running via Docker: docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0")
            return None

    # Parse TEI XML to preserve structural signals (as claimed in your report)
    soup = BeautifulSoup(response.text, "xml")
    title_tag = soup.find("titleStmt")
    paper_title = title_tag.text.strip() if title_tag else "Unknown Title"
    
    chunks = []
    # Extract Abstract
    abstract = soup.find("abstract")
    if abstract:
        chunks.append({
            "source_file": filename,
            "paper_title": paper_title,
            "section_header": "Abstract",
            "text": abstract.text.strip()
        })
    
    # Extract Body Divisions (Sections)
    for div in soup.find_all("div"):
        head = div.find("head")
        section_name = head.text.strip() if head else "Body"
        
        # Simple windowed chunking for the section text
        paragraphs = div.find_all("p")
        section_text = " ".join([p.text.strip() for p in paragraphs])
        
        if len(section_text) > 100: # Filter out empty/tiny noise chunks
            chunks.append({
                "source_file": filename,
                "paper_title": paper_title,
                "section_header": section_name,
                "text": section_text
            })
            
    return chunks

def build_database():
    """Initializes Qdrant and embeds the chunks using Hybrid Search."""
    print("🚀 Initializing Qdrant and Loading Embedding Models...")
    client = QdrantClient(path=QDRANT_PATH)
    
    # Load Models (FastEmbed handles the Splade and BGE-small inference locally)
    dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    
    # Recreate Collection
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": VectorParams(size=384, distance=Distance.COSINE)},
        sparse_vectors_config={"sparse": SparseVectorParams()}
    )
    
    all_chunks = []
    for pdf_file in os.listdir(PDF_DIR):
        if pdf_file.endswith(".pdf"):
            chunks = process_pdf_with_grobid(os.path.join(PDF_DIR, pdf_file))
            if chunks:
                all_chunks.extend(chunks)
                
    if not all_chunks:
        print("⚠️ No valid data extracted. Check your PDFs and Grobid container.")
        return

    print(f"🧠 Embedding {len(all_chunks)} semantic chunks...")
    texts = [chunk["text"] for chunk in all_chunks]
    
    dense_embeddings = list(dense_model.embed(texts))
    sparse_embeddings = list(sparse_model.embed(texts))
    
    points = []
    for i, chunk in enumerate(all_chunks):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_embeddings[i].tolist(),
                    "sparse": sparse_embeddings[i].as_object()
                },
                payload=chunk # Crucial: Contains 'source_file' for your regex citation fix!
            )
        )
        
    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print("✅ Ingestion Complete. Vector database is ready for inference.")

if __name__ == "__main__":
    if not os.path.exists(PDF_DIR):
        os.makedirs(PDF_DIR)
        print(f"Created {PDF_DIR} directory. Please add your PDFs and run again.")
    else:
        build_database()