from dotenv import load_dotenv
import streamlit as st

from rag_engine import PDFRAG


load_dotenv()


st.set_page_config(page_title="PDF Assistant", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
        .main {
            background:
                radial-gradient(circle at 15% 20%, #f7efe2 0%, #f7efe2 28%, transparent 29%),
                radial-gradient(circle at 85% 10%, #d7ebe5 0%, #d7ebe5 24%, transparent 25%),
                linear-gradient(120deg, #fffaf4 0%, #f1f7f5 100%);
        }
        .block-container {
            padding-top: 2.0rem;
            max-width: 980px;
        }
        h1, h2, h3 {
            letter-spacing: 0.2px;
            color: #102a26;
        }
        .subtitle {
            color: #2d4a45;
            margin-top: -8px;
            margin-bottom: 16px;
        }
        .panel {
            border: 1px solid #c8dbd6;
            border-radius: 14px;
            padding: 18px;
            background: rgba(255, 255, 255, 0.75);
            backdrop-filter: blur(3px);
        }
    </style>
    """,
    unsafe_allow_html=True,
)

if "rag" not in st.session_state:
    st.session_state.rag = PDFRAG()

st.title("PDF Assistant")
st.markdown('<p class="subtitle">Ask questions grounded only in your uploaded PDF. Groq powers the final answer.</p>', unsafe_allow_html=True)

col_left, col_right = st.columns([1.1, 1.7], gap="large")

with col_left:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("1) Upload PDF Files")
    uploaded_pdfs = st.file_uploader("Choose one or more PDF files", type=["pdf"], accept_multiple_files=True)

    if not st.session_state.rag.api_key:
        st.caption("Add GROQ_API_KEY to your .env file before answering questions.")

    if st.button("Process PDFs", type="primary", use_container_width=True):
        if not uploaded_pdfs:
            st.warning("Please upload at least one PDF first.")
        else:
            try:
                with st.spinner("Reading and indexing documents..."):
                    pdf_bytes_list = [pdf.read() for pdf in uploaded_pdfs]
                    chunk_count = st.session_state.rag.ingest_pdfs(pdf_bytes_list)
                st.success(f"Indexed {len(uploaded_pdfs)} files successfully ({chunk_count} chunks).")
            except Exception as exc:
                st.error(f"Could not process PDF files: {exc}")

    st.info("The assistant answers only using the uploaded PDF context.")
    st.caption(f"Model: {st.session_state.rag.model_name}")
    st.markdown('</div>', unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("2) Ask a Question")

    question = st.text_input("Your question")

    if st.button("Get Answer", use_container_width=True):
        try:
            result = st.session_state.rag.ask(question)
            st.markdown("### Answer")
            st.write(result.answer)

            with st.expander("Retrieved context"):
                if result.context:
                    st.write(result.context)
                else:
                    st.write("No strong supporting context found.")

            with st.expander("Similarity scores"):
                if result.scores:
                    st.write(result.scores)
                else:
                    st.write("No scores available.")
        except Exception as exc:
            st.error(f"Could not answer question: {exc}")

    st.markdown('</div>', unsafe_allow_html=True)
