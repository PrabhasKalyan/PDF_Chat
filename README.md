# 🤖 LLM-Powered PDF/Text Assistant with OCR

A simple intelligent assistant built using Python and Streamlit that can search through uploaded **PDFs (including scanned ones)** or **plain text files**, find the most relevant content based on a user query using **TF-IDF + Cosine Similarity**, and optionally **summarize the answer using a language model like OpenAI's GPT**.

---

## 💡 Approach

1. **File Upload**:  
   The user uploads a `.pdf` or `.txt` file through the Streamlit UI.

2. **Text Extraction**:  
   - If the file is a normal PDF, we extract text using `PyPDF2`.
   - If no extractable text is found (indicating a scanned PDF), we apply **OCR** using `pytesseract` and `pdf2image` to convert pages to images.

3. **Chunking**:  
   The extracted text is split into fixed-size chunks (default 500 characters) for efficient search.

4. **Query Matching (Retrieval)**:  
   - We use **TF-IDF vectorization** to convert the query and text chunks into numerical vectors.
   - Then we use **cosine similarity** to compare the query with each chunk and select the most relevant one.

5. **LLM Summarization (Optional)**:  
   The most relevant chunk can optionally be passed to an LLM like OpenAI's GPT to generate a human-readable summary or answer.

---

## 📌 Assumptions

- PDF files may be either:
  - **Text-based**, where content can be read using `PyPDF2`.
  - **Image-based** (e.g. scanned documents), where OCR via `pytesseract` is required.
- Query relevance is measured based on **surface-level keyword overlap**, not deep semantic meaning (unless an LLM is used).
- Summarization or answering is optional and requires an external LLM (e.g. OpenAI).

---

## 🧰 Requirements

Install dependencies:

```bash
pip install streamlit PyPDF2 pdf2image pytesseract scikit-learn openai
