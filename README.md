AI_Book_Agent

#  AI-Powered Book Recommendation & Chat Assistant

Ria is a production-grade **AI chatbot** that understands user intent, emotions, and topics to deliver **context-aware book recommendations and concise explanations** using **Retrieval-Augmented Generation (RAG)**.

Built with scalability, accuracy, and real-world UX in mind.

---

##  What This Project Does

- 🔍 Understands **natural language queries**, vague inputs, and follow-ups
- 📖 Recommends books using **semantic search (FAISS + embeddings)**
- 🧠 Uses **LLMs (LLaMA via OpenRouter)** for concise, friendly responses
- 🎯 Detects **intent, emotion, topic, and book titles**
- 🔁 Avoids repeating recommendations using **conversation memory**
- ⚡ Optimized for **low-latency inference** and **large datasets**

---

##  Core Architecture

User Query
↓
Intent + Emotion Detection (spaCy + Rules)
↓
Semantic Retrieval (SentenceTransformers + FAISS)
↓
Context Assembly (Chunks + History)
↓
LLM Response (LLaMA via OpenRouter)
↓
Structured Book Recommendations


---

## 🧩 Key Technical Highlights

### 🔹 Retrieval-Augmented Generation (RAG)
- Text is split into **semantic chunks**
- Chunks are embedded using `all-MiniLM-L6-v2`
- FAISS index enables **fast vector similarity search**
- Context is dynamically injected into LLM prompts

### 🔹 Chunk Management (Scalable Design)
- Chunks and metadata stored as:
  - `chunks.npy`
  - `chunk_meta.npy`
  - `faiss.index`
- Supports:
  - ✅ **Chunk regeneration**
  - ✅ **Partial updates**
  - ✅ **Re-indexing without retraining the model**

> This makes the system scalable for **new books, genres, or datasets**.

### 🔹 Smart Query Understanding
- Handles:
  - Book titles (exact + fuzzy match)
  - Topics & genres
  - Emotional states (stress, anxiety, motivation, etc.)
  - Vague or short inputs like *“mindset”*, *“business”*

### 🔹 Production-Oriented Backend
- Flask-based API
- Session & conversation memory
- Clean separation of:
  - Retrieval
  - NLP logic
  - LLM calls
  - Recommendation logic

---

## 🛠 Tech Stack

**Backend**
- Python, Flask
- FAISS (Vector Search)
- SentenceTransformers
- spaCy (NLP)
- NumPy

**AI / NLP**
- LLaMA 3 via OpenRouter
- Retrieval-Augmented Generation (RAG)
- Emotion & intent classification

**Data**
- Chunk-based document indexing
- Metadata-driven recommendations
- Genre-aware filtering

---

## 📸 Screenshots (Add Your Own)

> Replace the placeholders below with your images

### 🖼️ Image 1
<img width="1903" height="960" alt="final" src="https://github.com/user-attachments/assets/df00cf43-d955-4d86-aeaa-cb6ddc1c467e" />

### 🖼️ Image 2

<img width="1901" height="964" alt="final-2" src="https://github.com/user-attachments/assets/7ec51268-bf28-4d3c-98f4-d5dc34bc25fc" />

### 🖼️ Image 3
<img width="1820" height="838" alt="test" src="https://github.com/user-attachments/assets/f7e0b2df-2b3a-4d3e-b297-241d29e38778" />

## 🎥 Demo Videos (Add Your Own)

### ▶️ Video Demo 1
[[![Demo Video 1](./assets/video1_thumbnail.png)](./assets/video1.mp4)](https://github.com/user-attachments/assets/5ca53237-5c06-4b75-82d9-f1a3f69983a7)

### ▶️ Video Demo 2
[[![Demo Video 2](./assets/video2_thumbnail.png)](./assets/video2.mp4)](https://github.com/user-attachments/assets/f5d90a98-0361-4f66-a5c6-60e12ff62086
)


##  How to Run Locally

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
python app.py


































