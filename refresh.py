#if want to  rebuild then run  python refresh.py --mode full
#or if justwant to add new book then run  python refresh.py --mode incremental

import os, argparse, requests, numpy as np, faiss
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

DATA_DIR    = os.getenv("DATA_DIR", "data")
INDEX_PATH  = os.path.join(DATA_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(DATA_DIR, "chunks.npy")
META_PATH   = os.path.join(DATA_DIR, "chunk_meta.npy")

BOOK_API_URL = os.getenv("BOOK_API_URL")
GENRE_API_URL = os.getenv("GENRE_API_URL")  # 🔥 NEW: Genre API
MODEL_NAME   = "all-MiniLM-L6-v2"
MODEL_PATH   = os.getenv("MODEL_PATH", os.path.join(DATA_DIR, "models", MODEL_NAME))
ALLOW_DOWNLOAD = os.getenv("ALLOW_DOWNLOAD", "0") == "1"   # default off (read-only)

CHUNK_SIZE   = 500  # words
EMBED_DIM    = 384  # all-MiniLM-L6-v2

def get_model():
    """Load model strictly from local folder unless ALLOW_DOWNLOAD=1."""
    if os.path.isdir(MODEL_PATH) and os.listdir(MODEL_PATH):
        return SentenceTransformer(MODEL_PATH)
    if not ALLOW_DOWNLOAD:
        raise SystemExit(
            "Model not found locally at: "
            + os.path.abspath(MODEL_PATH)
            + "\nSet MODEL_PATH to your local folder or run the model download script.\n"
            + "To allow one-time download here (not recommended on server), set ALLOW_DOWNLOAD=1."
        )
    return SentenceTransformer(MODEL_NAME)

def atomic_replace(src_path: str, dst_path: str):
    """Replace dst with src atomically."""
    os.replace(src_path, dst_path)

def atomic_save_arrays(chunks, meta):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_chunks = CHUNKS_PATH + ".tmp.npy"
    tmp_meta   = META_PATH   + ".tmp.npy"
    np.save(tmp_chunks, np.array(chunks, dtype=object))
    np.save(tmp_meta,   np.array(meta,   dtype=object))
    atomic_replace(tmp_chunks, CHUNKS_PATH)
    atomic_replace(tmp_meta,   META_PATH)

def atomic_save_index(index: faiss.Index):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp_index = INDEX_PATH + ".tmp"
    faiss.write_index(index, tmp_index)
    atomic_replace(tmp_index, INDEX_PATH)
 
def save_all(chunks, meta, index):
    atomic_save_arrays(chunks, meta)
    atomic_save_index(index)

# 🔥 NEW FUNCTION: Load Genre Mapping
def load_genre_mapping():
    """Fetch all genres and return a dict: {genre_id: genre_name}"""
    if not GENRE_API_URL:
        print("⚠️  GENRE_API_URL not set. Genre names will be empty.")
        return {}
    
    print("Fetching genres from API...")
    try:
        r = requests.get(GENRE_API_URL, timeout=60)
        if not r.ok:
            print(f"⚠️  Genre API failed: {r.status_code}. Genre names will be empty.")
            return {}
        
        genres = r.json()
        if not isinstance(genres, list):
            print("⚠️  Genre API did not return a list. Genre names will be empty.")
            return {}
        
        # Extract only genre_id and genre_name
        mapping = {}
        for g in genres:
            gid = g.get("genre_id")
            gname = g.get("genre_name") or g.get("name") or ""
            if gid and str(gid).strip():
                mapping[str(gid)] = str(gname).strip()
        
        print(f"✓ Loaded {len(mapping)} genres\n")
        return mapping
    
    except Exception as e:
        print(f"⚠️  Error fetching genres: {e}. Genre names will be empty.")
        return {}

def load_books_from_api():
    if not BOOK_API_URL:
        raise SystemExit("BOOK_API_URL not set in environment.")
    r = requests.get(BOOK_API_URL, timeout=60)
    if not r.ok:
        raise SystemExit(f"BOOK_API_URL fetch failed: {r.status_code}")
    data = r.json()
    if not isinstance(data, list):
        raise SystemExit("BOOK_API_URL did not return a JSON list.")
    return data

def get_book_id(b):
    v = b.get("book_id")
    return str(v) if v is not None and str(v).strip() != "" else None

def get_book_title(b):
    return b.get("book_title") or b.get("title") or ""

def get_genre_id(b):
    """Extract genre_id from book data."""
    v = b.get("genre_id")
    return str(v) if v is not None and str(v).strip() != "" else None

def flatten_field(v):
    """Convert any field value to a string."""
    if isinstance(v, list): 
        return " ".join(str(x) for x in v if x)
    if isinstance(v, dict): 
        return " ".join(str(x) for x in v.values() if x)
    return str(v or "")

# 🔥 UPDATED FUNCTION: Now accepts genre_mapping parameter
def chunk_one_book(b, genre_mapping):
    """
    Extract book_summary only and chunk it into pieces.
    Store metadata: book_id, book_title, genre_id, genre_name
    """
    b_id       = get_book_id(b)
    b_title    = get_book_title(b)
    genre_id   = get_genre_id(b)
    
    # 🔥 LOOKUP GENRE NAME FROM MAPPING (not from book data)
    genre_name = genre_mapping.get(genre_id, "") if genre_id else ""

    # Debug print to verify extraction
    print(f"Processing: {b_title} | genre_id={genre_id}, genre_name={genre_name}")

    chunks, metas = [], []

    # Only process book_summary field
    raw = flatten_field(b.get("book_summary", ""))
    words = raw.split()
    
    if not words:
        print(f"  ⚠️  No summary found for book: {b_title}")
        return chunks, metas

    # Chunk the summary into pieces of CHUNK_SIZE words
    for i in range(0, len(words), CHUNK_SIZE):
        piece = " ".join(words[i:i+CHUNK_SIZE])
        chunks.append(piece)
        
        meta_entry = {
            "book_id": b_id,
            "book_title": b_title,
            "genre_id": genre_id,
            "genre_name": genre_name,  # 🔥 NOW FROM LOOKUP
            "chunk_index": i // CHUNK_SIZE,
            "total_words": len(words)
        }
        metas.append(meta_entry)

    print(f"  ✓ Created {len(chunks)} chunk(s)")
    return chunks, metas

def existing_book_ids(meta_list):
    """Get set of book IDs that already exist in the index."""
    return {m.get("book_id") for m in (meta_list or []) if m and m.get("book_id")}

def encode_in_batches(model, texts, batch_size=256):
    """Encode large corpora in small batches to keep RAM stable."""
    if not texts:
        return np.empty((0, EMBED_DIM), dtype=np.float32)
    outs = []
    for i in range(0, len(texts), batch_size):
        part = model.encode(
            texts[i:i+batch_size],
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype(np.float32, copy=False)
        outs.append(part)
    return np.vstack(outs)

# 🔥 UPDATED: Now loads genre mapping first
def full_rebuild(books):
    """Rebuild the entire FAISS index from scratch."""
    print(f"\n{'='*60}")
    print(f"FULL REBUILD MODE")
    print(f"{'='*60}")
    print(f"Total books from API: {len(books)}\n")
    
    # 🔥 LOAD GENRE MAPPING FIRST
    genre_mapping = load_genre_mapping()
    
    all_chunks, all_meta = [], []
    books_processed = 0
    books_skipped = 0
    
    # 🔥 PASS genre_mapping to chunk_one_book
    for b in books:
        cks, mts = chunk_one_book(b, genre_mapping)
        if cks:
            all_chunks.extend(cks)
            all_meta.extend(mts)
            books_processed += 1
        else:
            books_skipped += 1
    
    print(f"\n{'='*60}")
    print(f"Books processed: {books_processed}")
    print(f"Books skipped (no summary): {books_skipped}")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"{'='*60}\n")

    print("Loading embedding model...")
    model = get_model()
    
    print("Encoding chunks...")
    embs = encode_in_batches(model, all_chunks, batch_size=256)

    print("Creating FAISS index...")
    idx = faiss.IndexFlatL2(EMBED_DIM)
    if embs.size:
        idx.add(embs)
    
    print("Saving to disk...")
    save_all(all_chunks, all_meta, idx)
    
    print(f"\n{'='*60}")
    print(f"✓ FULL REBUILD COMPLETE")
    print(f"  Index size: {idx.ntotal} vectors")
    print(f"  Saved to: {os.path.abspath(DATA_DIR)}")
    print(f"{'='*60}\n")
    
    # Show sample metadata
    if all_meta:
        print("Sample metadata (first 3 entries):")
        for i, m in enumerate(all_meta[:3]):
            print(f"  [{i}] {m}")

# 🔥 UPDATED: Now loads genre mapping first
def incremental_add(books):
    """Add only new books to the existing FAISS index."""
    print(f"\n{'='*60}")
    print(f"INCREMENTAL ADD MODE")
    print(f"{'='*60}\n")
    
    if not (os.path.isfile(CHUNKS_PATH) and os.path.isfile(META_PATH) and os.path.isfile(INDEX_PATH)):
        raise SystemExit("[ERROR] Base files missing. Run --mode full first.")
    
    print("Loading existing index...")
    chunks = np.load(CHUNKS_PATH, allow_pickle=True).tolist()
    meta   = np.load(META_PATH,   allow_pickle=True).tolist()
    idx    = faiss.read_index(INDEX_PATH)

    have   = existing_book_ids(meta)
    to_add = [b for b in books if (get_book_id(b) and get_book_id(b) not in have)]
    
    print(f"Existing books in index: {len(have)}")
    print(f"New books to add: {len(to_add)}\n")

    if not to_add:
        print("✓ No new books to add. Index is up to date.\n")
        return

    # 🔥 LOAD GENRE MAPPING
    genre_mapping = load_genre_mapping()

    all_new_chunks, all_new_meta = [], []
    books_processed = 0
    books_skipped = 0
    
    # 🔥 PASS genre_mapping to chunk_one_book
    for b in to_add:
        cks, mts = chunk_one_book(b, genre_mapping)
        if cks:
            all_new_chunks.extend(cks)
            all_new_meta.extend(mts)
            books_processed += 1
        else:
            books_skipped += 1

    if not all_new_chunks:
        print("✓ No new chunks to add.\n")
        return

    print(f"\n{'='*60}")
    print(f"New books processed: {books_processed}")
    print(f"New books skipped (no summary): {books_skipped}")
    print(f"New chunks created: {len(all_new_chunks)}")
    print(f"{'='*60}\n")

    print("Loading embedding model...")
    model = get_model()
    
    print("Encoding new chunks...")
    embs = encode_in_batches(model, all_new_chunks, batch_size=256)

    print("Adding to FAISS index...")
    if idx is None or idx.ntotal == 0:
        idx = faiss.IndexFlatL2(EMBED_DIM)
    idx.add(embs)

    chunks.extend(all_new_chunks)
    meta.extend(all_new_meta)
    
    print("Saving updated index...")
    save_all(chunks, meta, idx)
    
    print(f"\n{'='*60}")
    print(f"✓ INCREMENTAL ADD COMPLETE")
    print(f"  New chunks added: {len(all_new_chunks)}")
    print(f"  Total index size: {idx.ntotal} vectors")
    print(f"  Saved to: {os.path.abspath(DATA_DIR)}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild or increment FAISS index from BOOK_API_URL")
    parser.add_argument("--mode", choices=["full","incremental"], default="incremental",
                        help="full: rebuild entire index | incremental: add only new books")
    args = parser.parse_args()

    print(f"\nDATA_DIR: {os.path.abspath(DATA_DIR)}\n")
    
    print("Fetching books from API...")
    books = load_books_from_api()
    print(f"✓ Fetched {len(books)} books from API\n")
    
    if args.mode == "full":
        full_rebuild(books)
    else:
        incremental_add(books)
    
    print("✓ Done.\n")