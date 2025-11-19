
import streamlit as st
import os
import zipfile
import PyPDF2
from pathlib import Path
import json
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import tempfile
import shutil
from datetime import datetime

# run: python -m streamlit run app.py

class CodeReviewer:
    """AI-powered code reviewer using codellama/CodeLlama-7b-Instruct-hf"""
    
    def __init__(self, model_name="codellama/CodeLlama-7b-Instruct-hf"):
        """Initialize the code reviewer with CodeLlama model"""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.15
        )
    
    def extract_zip(self, zip_path: str, extract_to: str) -> str:
        """Extract zip file"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return extract_to
    
    def extract_pdf_rules(self, pdf_path: str) -> str:
        """Extract rules from PDF file"""
        rules_text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                rules_text += page.extract_text() + "\n"
        return rules_text
    
    def get_code_files(self, directory: str, extensions: List[str] = None) -> List[Dict]:
        """Get all code files from directory"""
        if extensions is None:
            extensions = ['.py', '.js', '.java', '.cpp', '.c', '.go', '.rs', 
                         '.ts', '.jsx', '.tsx', '.php', '.rb', '.swift', '.kt', '.cs']
        
        code_files = []
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', 'venv', 'dist', 'build']]
            
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        code_files.append({
                            'path': file_path,
                            'filename': file,
                            'content': content,
                            'extension': Path(file).suffix
                        })
                    except Exception as e:
                        st.warning(f"Error reading {file}: {e}")
        
        return code_files
    
    ##Change Prompt to work around the ai more efficient
    def review_code_file(self, file_info: Dict, rules: str) -> Dict:
        """Review a single code file against rules"""
        prompt = f"""[INST] You are an expert code reviewer. Review the following code against the provided rules and guidelines.

Rules and Guidelines:
{rules[:2000]}

Code File: {file_info['filename']}
```{file_info['extension'][1:]}
{file_info['content'][:1500]}
```

Provide a structured review with:
1. Issues Found (list specific violations)
2. Severity (Critical/High/Medium/Low)
3. Suggestions (actionable improvements)
4. Code Quality Score (1-10)

Be concise and specific. [/INST]"""

        try:
            response = self.pipe(prompt, do_sample=True)[0]['generated_text']
            review = response.split('[/INST]')[-1].strip()
        except Exception as e:
            review = f"Error during review: {str(e)}"
        
        return {
            'file': file_info['path'],
            'filename': file_info['filename'],
            'review': review
        }
    
    def generate_non_functional_requirements(self, code_files: List[Dict], rules: str) -> str:
        """Generate non-functional requirements based on code analysis"""
        total_files = len(code_files)
        total_lines = sum(len(f['content'].split('\n')) for f in code_files)
        languages = set(f['extension'] for f in code_files)
        
        context = f"""Project has {total_files} files with {total_lines} lines of code.
Languages: {', '.join(languages)}
Sample files: {', '.join([f['filename'] for f in code_files[:5]])}"""
        
        prompt = f"""[INST] As a software architect, generate comprehensive non-functional requirements (NFRs) for this codebase.

Project Context:
{context}

Coding Rules Applied:
{rules[:1000]}

Generate NFRs for these categories:
1. Performance (response time, throughput, resource usage)
2. Scalability (concurrent users, data volume, horizontal scaling)
3. Security (authentication, authorization, data protection)
4. Reliability (uptime, fault tolerance, backup/recovery)
5. Maintainability (code quality, documentation, testing)
6. Usability (user experience, accessibility)
7. Compatibility (browsers, devices, platforms)
8. Monitoring (logging, metrics, alerting)

Provide specific, measurable requirements. [/INST]"""

        try:
            response = self.pipe(prompt, do_sample=True)[0]['generated_text']
            nfr = response.split('[/INST]')[-1].strip()
        except Exception as e:
            nfr = f"Error generating NFRs: {str(e)}"
        
        return nfr


def main():
    st.set_page_config(
        page_title="AI Code Reviewer",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("🔍 AI Code Reviewer")
    st.markdown("###")
    st.markdown("Upload your Project (ZIP) and coding rules (PDF) for automated code review")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("**Model:** codellama/CodeLlama-7b-Instruct-hf")
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
        This tool analyzes your code against custom rules and provides:
        - 📋 Detailed code reviews
        - ⚠️ Issue identification
        - 💡 Improvement suggestions
        - 📊 Non-functional requirements
        """)
    
    # Initialize session state
    if 'reviewer' not in st.session_state:
        st.session_state.reviewer = None
    if 'review_complete' not in st.session_state:
        st.session_state.review_complete = False
    if 'report' not in st.session_state:
        st.session_state.report = None
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Upload Project Repository (ZIP)")
        zip_file = st.file_uploader(
            "Choose a ZIP file",
            type=['zip'],
            help="Upload your Project repository as a ZIP file"
        )
        if zip_file:
            st.success(f"✅ Uploaded: {zip_file.name}")
    
    with col2:
        st.subheader("📄 Upload Coding Rules (PDF)")
        pdf_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload your coding standards and rules as a PDF"
        )
        if pdf_file:
            st.success(f"✅ Uploaded: {pdf_file.name}")
    
    st.markdown("---")
    
    # Submit button
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 2])
    with col_btn1:
        submit_button = st.button("🚀 Start Review", type="primary", use_container_width=True)
    with col_btn2:
        if st.session_state.review_complete:
            if st.button("🔄 New Review", use_container_width=True):
                st.session_state.review_complete = False
                st.session_state.report = None
                st.rerun()
    
    # Process review
    if submit_button:
        if not zip_file or not pdf_file:
            st.error("❌ Please upload both ZIP and PDF files!")
        else:
            st.session_state.review_complete = False
            
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files
                zip_path = os.path.join(temp_dir, "repo.zip")
                pdf_path = os.path.join(temp_dir, "rules.pdf")
                extract_dir = os.path.join(temp_dir, "extracted")
                
                with open(zip_path, "wb") as f:
                    f.write(zip_file.getbuffer())
                
                with open(pdf_path, "wb") as f:
                    f.write(pdf_file.getbuffer())
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # Step 1: Load model
                    status_text.text("🔄 Loading CodeLlama model...")
                    progress_bar.progress(10)
                    
                    if st.session_state.reviewer is None:
                        st.session_state.reviewer = CodeReviewer()
                    reviewer = st.session_state.reviewer
                    
                    # Step 2: Extract files
                    status_text.text("📦 Extracting repository...")
                    progress_bar.progress(20)
                    reviewer.extract_zip(zip_path, extract_dir)
                    
                    # Step 3: Extract rules
                    status_text.text("📄 Reading coding rules...")
                    progress_bar.progress(30)
                    rules = reviewer.extract_pdf_rules(pdf_path)
                    
                    # Step 4: Get code files
                    status_text.text("🔍 Scanning code files...")
                    progress_bar.progress(40)
                    code_files = reviewer.get_code_files(extract_dir)
                    
                    if not code_files:
                        st.error("❌ No code files found in the repository!")
                        return
                    
                    st.info(f"📊 Found {len(code_files)} code files to review")
                    
                    # Step 5: Review files
                    reviews = []
                    total_files = len(code_files)
                    
                    for i, file_info in enumerate(code_files):
                        progress = 40 + int((i / total_files) * 40)
                        status_text.text(f"🔍 Reviewing file {i+1}/{total_files}: {file_info['filename']}")
                        progress_bar.progress(progress)
                        
                        review = reviewer.review_code_file(file_info, rules)
                        reviews.append(review)
                    
                    # Step 6: Generate NFRs
                    status_text.text("📋 Generating non-functional requirements...")
                    progress_bar.progress(85)
                    nfr = reviewer.generate_non_functional_requirements(code_files, rules)
                    
                    # Step 7: Prepare report
                    status_text.text("📝 Preparing final report...")
                    progress_bar.progress(95)
                    
                    report = {
                        'summary': {
                            'total_files_reviewed': len(reviews),
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'repository': zip_file.name,
                            'rules_document': pdf_file.name
                        },
                        'file_reviews': reviews,
                        'non_functional_requirements': nfr
                    }
                    
                    st.session_state.report = report
                    st.session_state.review_complete = True
                    
                    # Complete
                    progress_bar.progress(100)
                    status_text.text("✅ Review completed successfully!")
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error during review: {str(e)}")
                    progress_bar.empty()
                    status_text.empty()
    
    # Display results
    if st.session_state.review_complete and st.session_state.report:
        st.markdown("---")
        st.header("📊 Review Results")
        
        report = st.session_state.report
        
        # Summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📁 Files Reviewed", report['summary']['total_files_reviewed'])
        with col2:
            st.metric("📅 Review Date", report['summary']['timestamp'])
        with col3:
            st.metric("📦 Repository", report['summary']['repository'])
        
        # Tabs for different sections
        tab1, tab2, tab3 = st.tabs(["📋 File Reviews", "📊 Non-Functional Requirements", "💾 Download Report"])
        
        with tab1:
            st.subheader("Individual File Reviews")
            for i, review in enumerate(report['file_reviews']):
                with st.expander(f"📄 {review['filename']}", expanded=(i==0)):
                    st.markdown(f"**File Path:** `{review['file']}`")
                    st.markdown("---")
                    st.markdown(review['review'])
        
        with tab2:
            st.subheader("Non-Functional Requirements")
            st.markdown(report['non_functional_requirements'])
        
        with tab3:
            st.subheader("Download Report")
            
            # JSON download
            json_str = json.dumps(report, indent=2)
            st.download_button(
                label="📥 Download JSON Report",
                data=json_str,
                file_name=f"code_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            # Text download
            text_report = f"""CODE REVIEW REPORT
{'='*80}

Summary:
- Total Files Reviewed: {report['summary']['total_files_reviewed']}
- Timestamp: {report['summary']['timestamp']}
- Repository: {report['summary']['repository']}
- Rules Document: {report['summary']['rules_document']}

{'='*80}
FILE REVIEWS
{'='*80}

"""
            for review in report['file_reviews']:
                text_report += f"\nFile: {review['filename']}\nPath: {review['file']}\n{'-'*80}\n{review['review']}\n{'-'*80}\n\n"
            
            text_report += f"\n{'='*80}\nNON-FUNCTIONAL REQUIREMENTS\n{'='*80}\n\n{report['non_functional_requirements']}\n"
            
            st.download_button(
                label="📥 Download Text Report",
                data=text_report,
                file_name=f"code_review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )


if __name__ == "__main__":
    main()
