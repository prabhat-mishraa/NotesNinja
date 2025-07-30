import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from io import StringIO

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# Load environment variable
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",  # Or llama3 model
    api_key=GROQ_API_KEY
)

# Streamlit UI
st.set_page_config(page_title="📄 Q&A with NotesNinja")
st.title("📄 NotesNinja")
st.subheader("Study smarter, not harder — let the Ninja extract what matters!")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("✅ PDF uploaded!")

    try:
        # Load PDF and extract
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        if not chunks:
            st.warning("❌ No content found in the PDF.")
        else:
            embeddings = HuggingFaceEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            retriever = vectorstore.as_retriever()

            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

            st.subheader("🤖 Auto-Generated Q&A")
            question_list = []
            output_buffer = StringIO()  # For downloading

            for chunk in chunks[:3]:
                prompt = f"Generate important questions from this text:\n\n{chunk.page_content}"
                try:
                    response = llm.invoke(prompt)
                    if response and hasattr(response, "content"):
                        questions = response.content.strip().split("\n")
                        cleaned = [q.strip("-•123. ").strip() for q in questions if q.strip()]
                        question_list.extend(cleaned)
                except Exception as e:
                    st.error(f"❌ Failed to generate questions: {str(e)}")

            if not question_list:
                st.warning("⚠️ No questions generated.")
            else:
                for i, question in enumerate(question_list, 1):
                    st.markdown(f"**Q{i}: {question}**")
                    try:
                        answer = qa_chain.run(question)
                        st.write(f"A: {answer}")
                        # Add to output buffer
                        output_buffer.write(f"Q{i}: {question}\n")
                        output_buffer.write(f"A: {answer}\n\n")
                    except Exception as e:
                        st.error(f"❌ Error answering Q{i}: {str(e)}")
                    st.divider()

                # Download button
                st.download_button(
                    label="📥 Download Q&A as TXT",
                    data=output_buffer.getvalue(),
                    file_name="qa_output.txt",
                    mime="text/plain"
                )

    except Exception as e:
        st.error(f"❌ Something went wrong: {str(e)}")
