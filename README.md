# AI Research Paper Explainer

AI Research Paper Explainer is a Streamlit app that lets you upload a PDF research paper and ask questions about it. The app extracts text from the PDF, builds a local retrieval index, generates a short document summary, and answers user questions with Gemini.

## Features

- Upload a PDF research paper or notes file
- Extract text from the uploaded document
- Generate a structured document summary
- Ask questions about the uploaded PDF
- Show retrieved source chunks for transparency

## Tech Stack

- Streamlit
- PyPDF2
- LangChain text splitters and Chroma
- Hugging Face sentence embeddings
- Google Gemini API

## Setup

1. Create and activate a virtual environment.

```bash
python -m venv venv
venv\Scripts\activate
```

2. Install the dependencies.

```bash
pip install -r requirements.txt
```

3. Add your Gemini API key to `.env`.

```env
GEMINI_API_KEY=your_api_key_here
```

If you already use `GOOGLE_API_KEY` in `.env`, the app will also accept that variable.

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## How It Works

1. Upload a PDF file.
2. The app extracts text from the PDF.
3. A structured summary is generated for the whole document.
4. The paper text is split into chunks and indexed for retrieval.
5. Your question is matched against the most relevant parts of the uploaded PDF.
6. Gemini returns an answer based on the document context.

## Notes

- The app ignores `.env` and other local environment files when pushing to GitHub.
- For best results, upload a PDF with machine-readable text rather than a scanned image.
- The first upload may take a little longer because the app prepares the summary and search index once per file.
