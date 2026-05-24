import streamlit as st
from dotenv import load_dotenv
import os
import hashlib
from rag import _truncate_preview, build_vectorstore_from_text, extract_text_from_pdf_bytes, generate_document_overview, process_text_and_get_answer

# Automatically pull credentials from your .env file
load_dotenv()

st.set_page_config(page_title="AI Research Explainer", page_icon="🔬")
st.title("🔬 AI Research Paper Explainer (RAG Pipeline)")

# Verification check for security
if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
    st.error("⚠️ GEMINI_API_KEY not found in your .env file! Please check your configuration.")

# Keep text inside the web app's memory cache cleanly
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = None
if "pdf_vectorstore" not in st.session_state:
    st.session_state.pdf_vectorstore = None
if "pdf_overview" not in st.session_state:
    st.session_state.pdf_overview = None
if "last_uploaded_hash" not in st.session_state:
    st.session_state.last_uploaded_hash = None
if "active_query_key" not in st.session_state:
    st.session_state.active_query_key = None

# Streamlit file uploader component
uploaded_file = st.file_uploader("Upload your Research Paper or Notes (PDF format):", type=["pdf"])

if uploaded_file is not None:
    current_file_name = uploaded_file.name
    raw_bytes = uploaded_file.getvalue()
    current_file_hash = hashlib.sha256(raw_bytes).hexdigest()

    # Reset cached text whenever the actual file content changes.
    if st.session_state.last_uploaded_hash != current_file_hash:
        st.session_state.pdf_text = None
        st.session_state.pdf_vectorstore = None
        st.session_state.pdf_overview = None
        st.session_state.last_uploaded_hash = current_file_hash
        if st.session_state.active_query_key:
            st.session_state.pop(st.session_state.active_query_key, None)
        st.session_state.active_query_key = f"user_query_{current_file_hash[:12]}"

    # Extract the text once and save it safely to the session state
    if st.session_state.pdf_text is None:
        with st.spinner("Extracting layout text components..."):
            try:
                # Calls our safe function wrapped in io.BytesIO inside rag.py
                st.session_state.pdf_text = extract_text_from_pdf_bytes(raw_bytes)
                if st.session_state.pdf_text:
                    with st.spinner("Creating a document summary..."):
                        st.session_state.pdf_overview = generate_document_overview(st.session_state.pdf_text)
                    with st.spinner("Preparing search index for this PDF..."):
                        st.session_state.pdf_vectorstore, _ = build_vectorstore_from_text(st.session_state.pdf_text)
                st.success(f"Successfully loaded and cached: {current_file_name}!")
            except Exception as e:
                st.error(f"Failed to parse PDF file: {str(e)}")

    if st.session_state.pdf_text:
        st.markdown("---")
        if st.session_state.pdf_overview:
            st.markdown("### 📄 Document Summary")
            st.write(st.session_state.pdf_overview)
        user_query = st.text_input(
            "Ask any question about this research document:",
            key=st.session_state.active_query_key,
        )

        if user_query:
            with st.spinner("Searching database and synthesizing factual answer..."):
                try:
                    answer, citations = process_text_and_get_answer(
                        st.session_state.pdf_text,
                        user_query,
                        vectorstore=st.session_state.pdf_vectorstore,
                    )
                    
                    st.markdown("### 🤖 AI Response")
                    st.write(answer)
                    
                    # Show citations to cleanly demonstrate your working RAG architecture
                    with st.expander("🔍 View Retrieved Document Sources"):
                        for idx, chunk in enumerate(citations):
                            st.markdown(f"**Source Chunk {idx+1}:**")
                            st.caption(_truncate_preview(chunk.page_content))
                except Exception as e:
                    st.error(f"Pipeline Error: {str(e)}")