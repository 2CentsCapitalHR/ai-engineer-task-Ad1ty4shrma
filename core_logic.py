import os
import json
import docx
from docx.enum.text import WD_COLOR_INDEX
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser


def load_credentials_from_env():
    try:
        project_root = os.getcwd()
        env_path = os.path.join(project_root, '.env')
        if not os.path.exists(env_path):
            return
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    clean_value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = clean_value
    except Exception as e:
        print(f"Error reading .env file: {e}")

load_credentials_from_env()

ADGM_PROCESS_CHECKLISTS = { "Company Incorporation": ["Articles of Association", "Memorandum of Association", "Incorporation Application Form", "UBO Declaration Form", "Register of Members and Directors"] }
DOC_TYPE_KEYWORDS = { "Articles of Association": ["articles of association", "aoa"], "Memorandum of Association": ["memorandum of association", "moa"], "Board Resolution": ["board resolution", "resolutions of the board", "written resolutions"], "Incorporation Application Form": ["incorporation application"], "UBO Declaration Form": ["ultimate beneficial owner", "ubo declaration"], "Register of Members and Directors": ["register of members", "register of directors"] }
def classify_document(text_content):
    for doc_type, keywords in DOC_TYPE_KEYWORDS.items():
        if any(keyword in text_content.lower() for keyword in keywords):
            return doc_type
    return "Unknown Document"
def verify_checklist(classified_docs):
    process = "Company Incorporation"
    required_docs = set(ADGM_PROCESS_CHECKLISTS[process])
    uploaded_docs_set = set(doc['type'] for doc in classified_docs)
    missing_docs = list(required_docs - uploaded_docs_set)
    return { "process": process, "documents_uploaded": len(uploaded_docs_set), "required_documents": len(required_docs), "missing_documents": missing_docs }

RAG_PROMPT_TEMPLATE = """
You are an expert AI legal assistant for the Abu Dhabi Global Market (ADGM). Your task is to meticulously review a clause from a legal document and identify any red flags based on the provided ADGM context and general legal best practices. Be very strict and thorough.

**ADGM Regulatory Context:**
{context}

**Clause from a '{document_type}' to Review:**
"{clause_text}"

**Instructions:**
1.  **Analyze the clause against the ADGM Context.** If the context provides a specific rule (e.g., about jurisdiction, required information), check for strict compliance.
2.  **Check for Common Legal Red Flags**, even if not in the context. These include:
    - **Incorrect Jurisdiction:** Any mention of courts other than "ADGM Courts" (e.g., "DIFC Courts", "UAE Federal Courts") is a HIGH severity error.
    - **Ambiguous or Non-binding Language:** Phrases like "will consider," "if possible," "are flexible," or overly vague descriptions are a MEDIUM severity issue.
    - **Missing or Incomplete Information:** Look for placeholder text like "[Leave this blank]", "[Date]", or clauses that refer to information that is clearly missing (e.g., "The bank to be decided later"). This is a HIGH severity issue.
    - **Missing Signatory Sections:** If the clause mentions "Signatories" but is empty, flag it as a HIGH severity issue.
3.  **Respond ONLY in a valid JSON format.**
4.  **If you find NO issues, you MUST respond with `{{ "issue_found": false }}`.** Do not add any other text.
5.  **Be critical.** It is better to flag a potential minor issue than to miss a major one.

**Example of a good response:**
{{
  "issue_found": true,
  "issue": "The jurisdiction clause incorrectly references 'DIFC Courts'. For ADGM-registered companies, this must be 'ADGM Courts'.",
  "suggestion": "Update the clause to state: '...subject to the exclusive jurisdiction of the ADGM Courts.'",
  "citation": "Based on ADGM commercial law principles.",
  "severity": "High"
}}

**Your JSON Output:**
"""

def analyze_document_clauses(doc, doc_type, vector_store):
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("FATAL: GOOGLE_API_KEY not found. Check .env file.")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", google_api_key=api_key, temperature=0, convert_system_message_to_human=True, safety_settings={ HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, })
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    rag_chain = ({"context": retriever, "clause_text": RunnablePassthrough(), "document_type": lambda x: doc_type} | prompt | llm | StrOutputParser())
    issues_found = []
    for para in doc.paragraphs:
        if len(para.text.strip()) > 20:
            response_str = rag_chain.invoke(para.text)
            try:
                if "```json" in response_str: response_str = response_str.split("```json")[1].split("```")[0]
                response_json = json.loads(response_str)
                if response_json.get("issue_found"):
                    issue_details = { "document": doc_type, "section": para.text[:100] + "...", "issue": response_json.get("issue"), "suggestion": response_json.get("suggestion"), "severity": response_json.get("severity"), "paragraph_object": para }
                    issues_found.append(issue_details)
            except (json.JSONDecodeError, IndexError) as e:
                print(f"Warning: Could not decode JSON from Gemini response: {response_str}. Error: {e}")
    return issues_found

# --- 3. Output Generation  ---

def highlight_issues_in_docx(issues):
    """Highlights the paragraphs with issues in the DOCX file."""
    for issue in issues:
        para = issue["paragraph_object"]
        # Highlight based on severity
        severity = issue.get("severity", "Low").lower()
        if severity == "high":
            color = WD_COLOR_INDEX.YELLOW
        elif severity == "medium":
            color = WD_COLOR_INDEX.TURQUOISE
        else:
            color = WD_COLOR_INDEX.GRAY_25
        
        for run in para.runs:
            run.font.highlight_color = color