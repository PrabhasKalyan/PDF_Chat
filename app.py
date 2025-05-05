import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import PyPDF2
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import chromadb

client = OpenAI(
    api_key="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",
    base_url = "https://api.groq.com/openai/v1",
)

chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(name="pdf_chunks")

def load_pdf(file_obj):
    text = ""
    reader = PyPDF2.PdfReader(file_obj)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def load_pdf_with_ocr(file_obj):
    text = ""
    images = convert_from_bytes(file_obj.read(), dpi=300)
    for i, image in enumerate(images):
        page_text = pytesseract.image_to_string(image)
        text += page_text + "\n"
    return text

def load_text(file_path):
    with open(file_path, 'r') as file:
        return file.read()


def get_most_relevant_chunk(text, query, chunk_size=500):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    vectorizer = TfidfVectorizer().fit_transform([query] + chunks).toarray()
    ids = [f"{i}" for i in range(len(chunks))]

    chroma_collection.upsert(
        documents=chunks,
        embeddings=vectorizer[1:],
        ids=ids
    )
    results = chroma_collection.query(
        query_embeddings=[vectorizer[0:1]],
        n_results=1
    )
    # similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()
    # most_relevant_index = similarity.argmax()
    # return chunks[most_relevant_index]
    print(len(chunks), len(vectorizer), len(ids))

    results['documents'][0][0]


def summarize_with_llm(prompt):
    response = client.chat.completions.create(
        model="llama3-8b-8192",
         messages=[
             {"role": "system", "content": "You are a helpful assistant that summarizes relevant content from documents."},
            {"role": "user", "content": prompt}]
        )
    return response.choices[0].message.content.strip()


st.title("Local PDF/Text LLM Assistant")
uploaded_file = st.file_uploader("Upload a PDF or Text file", type=["pdf", "txt"])
query = st.text_input("Ask a question about the document:")
if uploaded_file and query:
    if uploaded_file.type == "application/pdf":
        text = load_pdf(uploaded_file)
    else:
        text = load_text(uploaded_file)

    relevant_chunk = get_most_relevant_chunk(text, query)
    prompt = f"Given this passage:\n\n{relevant_chunk}\n\nAnswer the question: {query}"
    answer = summarize_with_llm(prompt)

    st.subheader("Answer")
    st.write(answer)






