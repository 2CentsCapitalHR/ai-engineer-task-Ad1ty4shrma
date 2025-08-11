import streamlit as st
import os
import docx
import io
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import core_logic

DB_PATH = "vector_store/"


def reset_analysis_state():
    st.session_state.analysis_complete = False
    st.session_state.all_issues = []
    st.session_state.reviewed_docs_data = []

if 'analysis_complete' not in st.session_state:
    reset_analysis_state()


@st.cache_resource
def load_retriever():
    if not os.path.exists(DB_PATH):
        st.error("Vector store not found. Please run `ingest.py` first.")
        return None
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
    vector_store = Chroma(persist_directory=DB_PATH, embedding_function=embedding_model)
    return vector_store


# --- 2. UI LAYOUT ---
st.set_page_config(page_title="ADGM Corporate Agent", layout="wide")
st.title("📄 ADGM-Compliant Corporate Agent")
st.markdown("An AI assistant to review, validate, and help prepare your ADGM corporate documents.")

vector_store = load_retriever()

# Attach our reset function to the file uploader's on_change event
uploaded_files = st.file_uploader(
    "Upload your .docx company formation documents",
    type=['docx'],
    accept_multiple_files=True,
    on_change=reset_analysis_state
)

if vector_store:
    if st.button("Analyze Documents", disabled=not uploaded_files):
        with st.spinner("Analyzing documents... This may take a few moments."):
            # Use local variables for this specific run
            all_issues_run = []
            classified_docs_run = []
            reviewed_docs_data_run = []

            for uploaded_file in uploaded_files:
                doc = docx.Document(uploaded_file)
                full_text = "\n".join([p.text for p in doc.paragraphs])
                doc_type = core_logic.classify_document(full_text)
                classified_docs_run.append({"name": uploaded_file.name, "type": doc_type})
                
                issues = core_logic.analyze_document_clauses(doc, doc_type, vector_store)
                
                if issues:
                    all_issues_run.extend(issues)
                    core_logic.highlight_issues_in_docx(issues)

                doc_stream = io.BytesIO()
                doc.save(doc_stream)
                doc_stream.seek(0)
                reviewed_docs_data_run.append({
                    "name": f"reviewed_{uploaded_file.name}",
                    "data": doc_stream
                })

            # --- STORE RESULTS IN SESSION STATE ---
            st.session_state.all_issues = all_issues_run
            st.session_state.reviewed_docs_data = reviewed_docs_data_run
            st.session_state.classified_docs = classified_docs_run
            st.session_state.analysis_complete = True

if st.session_state.analysis_complete:
    st.write("---")
    st.header("📋 Analysis Summary")
    
    # Display issue summary
    if st.session_state.all_issues:
        st.write(f"Found a total of {len(st.session_state.all_issues)} potential issues across all documents.")
    else:
        st.success("No major issues detected in the uploaded documents.")

    # Display checklist verification
    checklist_result = core_logic.verify_checklist(st.session_state.classified_docs)
    if checklist_result["missing_documents"]:
        st.warning(f"**Checklist Verification:** Missing: **{', '.join(checklist_result['missing_documents'])}**")
    else:
        st.success("**Checklist Verification:** All required documents for Company Incorporation seem to be present.")

    # Create and display JSON report
    final_report = {
        "process": checklist_result["process"],
        "documents_uploaded_count": checklist_result["documents_uploaded"],
        "required_documents_count": checklist_result["required_documents"],
        "missing_document": checklist_result["missing_documents"],
        "issues_found": [{k: v for k, v in issue.items() if k != 'paragraph_object'} for issue in st.session_state.all_issues]
    }
    
    st.subheader("JSON Summary Report")
    st.json(final_report)
    
    # Create download buttons
    st.subheader("⬇️ Download Reviewed Documents")
    for doc_data in st.session_state.reviewed_docs_data:
        st.download_button(
            label=f"Download {doc_data['name']}",
            data=doc_data['data'],
            file_name=doc_data['name'],
            mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
        )