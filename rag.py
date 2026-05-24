import io
import os
import re
from functools import lru_cache
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from google import genai
from google.genai import types


def _get_gemini_api_key():
    return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


@lru_cache(maxsize=1)
def get_embeddings_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def _split_text_into_chunks(text_content):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )

    from langchain_core.documents import Document

    base_doc = [Document(page_content=text_content)]
    return text_splitter.split_documents(base_doc)


def _looks_like_reference_chunk(chunk_text):
    normalized_text = chunk_text.lower().strip()
    if not normalized_text:
        return True

    reference_signals = [
        r"\breferences\b",
        r"\bbibliography\b",
        r"\bworks cited\b",
        r"https?://",
        r"\bdoi\b",
        r"\bdownloaded from\b",
        r"\bjournal\b",
        r"\bproceedings\b",
    ]
    signal_hits = sum(1 for pattern in reference_signals if re.search(pattern, normalized_text))
    year_hits = len(re.findall(r"\b(19|20)\d{2}\b", normalized_text))

    return signal_hits >= 2 or (signal_hits >= 1 and year_hits >= 2)


def _clean_document_chunks(chunks):
    cleaned_chunks = []
    for chunk in chunks:
        if not _looks_like_reference_chunk(chunk.page_content):
            cleaned_chunks.append(chunk)
    return cleaned_chunks


def _truncate_preview(text, limit=320):
    compact_text = " ".join(text.split())
    if len(compact_text) <= limit:
        return compact_text
    return compact_text[:limit].rstrip() + "..."


def build_vectorstore_from_text(text_content):
    """Build and return a reusable vector store for one uploaded document."""
    chunks = _clean_document_chunks(_split_text_into_chunks(text_content))

    if not chunks:
        return None, []

    vectorstore = Chroma.from_documents(
        chunks,
        get_embeddings_model(),
        collection_name="uploaded_pdf",
    )

    return vectorstore, chunks


def generate_document_overview(text_content):
    """Generate a structured overview of the uploaded paper."""
    api_key = _get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing. Check your .env file.")

    if not text_content or not text_content.strip():
        return "The uploaded file contains no machine-readable text."

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        config=types.GenerateContentConfig(
            system_instruction=(
                "You are an expert AI Research Assistant. Summarize the uploaded paper clearly and accurately. "
                "Ignore reference lists and citation-only sections. Focus on the paper's topic, objective, method, "
                "important findings, and conclusion. If the document is a review article or a fragmented extract, "
                "say so briefly instead of inventing missing details."
            ),
            temperature=0.2,
        ),
        contents=(
            "Summarize this uploaded paper in 5 short sections: Topic, Objective, Method, Key Findings, Conclusion.\n\n"
            f"Document Text:\n{text_content[:20000]}"
        ),
    )
    return response.text

# =====================================================================
# 1. SAFE PDF TEXT EXTRACTION (FIXES THE '.SEEK' BYTES ERROR)
# =====================================================================
def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Converts raw binary bytes into an in-memory file stream using io.BytesIO 
    so that PdfReader can safely use the .seek() method without crashing.
    """
    if not pdf_bytes:
        return ""
        
    # Wrap the raw bytes into a virtual file stream object
    file_like_object = io.BytesIO(pdf_bytes)
    
    # Initialize the reader with our file-like stream
    reader = PdfReader(file_like_object)
    
    text = ""
    # Loop through all pages and extract readable text content
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
            
    return text

# =====================================================================
# 2. COMPLETE RAG PIPELINE ENGINE
# =====================================================================
def process_text_and_get_answer(text_content, user_query, vectorstore=None):
    """
    Implements a full RAG pipeline: Chunks text, generates semantic embeddings 
    via HuggingFace, retrieves relevant context vectors using ChromaDB, and 
    generates a factual answer using the free Google Gemini API.
    """
    api_key = _get_gemini_api_key()
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is missing. Check your .env file.")

    if vectorstore is None:
        vectorstore, chunks = build_vectorstore_from_text(text_content)
    else:
        chunks = []

    if not vectorstore:
        return "The uploaded file contains no machine-readable text.", []
    
    cleaned_query = user_query.strip().lower()
    is_general_explanation_request = any(
        phrase in cleaned_query
        for phrase in ["explain", "summarize", "summary", "overview", "describe", "what is this paper", "about this document"]
    )

    # Context-Aware Semantic Search: pull more chunks for broad explanation requests.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 6 if is_general_explanation_request else 4})
    relevant_chunks = retriever.invoke(user_query)
    relevant_chunks = [chunk for chunk in relevant_chunks if not _looks_like_reference_chunk(chunk.page_content)] or relevant_chunks
    
    context_payload = "\n---\n".join([chunk.page_content for chunk in relevant_chunks])
    
    # Initialize the modern official Google GenAI Client
    client = genai.Client(api_key=api_key)
    
    # Strict prompt instructions to force factual boundary enforcement
    system_instruction = (
        "You are an expert AI Research Assistant. Your sole task is to explain the uploaded document "
        "using only the provided Context Blocks from the main body of the paper. Ignore reference lists, "
        "bibliographies, citation pages, and unrelated footnotes. If the user asks for a general explanation, "
        "respond with a structured summary covering topic, objective, method, key findings, and conclusion. "
        "If the answer cannot be confidently derived from the context, state clearly that the information is "
        "missing from the document. Do not fabricate or hallucinate details."
    )

    if is_general_explanation_request:
        prompt_payload = (
            "Explain this uploaded paper in a clear, structured way.\n"
            "Use these Context Blocks from the main body of the paper:\n"
            f"{context_payload}\n\n"
            "Write sections for: 1) Topic, 2) Objective, 3) Method / Approach, 4) Key Findings, 5) Conclusion, 6) Limitations if available."
        )
    else:
        prompt_payload = f"Context Blocks:\n{context_payload}\n\nUser Question: {user_query}"
    
    # Query Gemini 2.5 Flash using an enterprise-safe low temperature (0.2)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2
        ),
        contents=prompt_payload,
    )
    
    return response.text, relevant_chunks