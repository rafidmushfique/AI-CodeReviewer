import os
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import json

# Required installations:
# pip install transformers torch PyPDF2 accelerate

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import PyPDF2


class PDFRuleExtractor:
    """Extract rules from PDF documents"""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        
    def extract_text(self) -> str:
        """Extract all text from PDF"""
        text = ""
        with open(self.pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    
    def parse_rules(self) -> List[str]:
        """Parse extracted text into individual rules"""
        text = self.extract_text()
        # Split by common rule delimiters
        rules = []
        for line in text.split('\n'):
            line = line.strip()
            if line and len(line) > 10:  # Filter out empty or very short lines
                rules.append(line)
        return rules


class CodeExtractor:
    """Extract code files from GitHub zip"""
    
    SUPPORTED_EXTENSIONS = {
        '.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', 
        '.go', '.rs', '.ts', '.jsx', '.tsx', '.rb', '.php'
    }
    
    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        self.temp_dir = None
        
    def extract_zip(self) -> str:
        """Extract zip file to temporary directory"""
        self.temp_dir = tempfile.mkdtemp()
        with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.temp_dir)
        return self.temp_dir
    
    def get_code_files(self) -> List[Dict[str, str]]:
        """Get all code files with their content"""
        if not self.temp_dir:
            self.extract_zip()
            
        code_files = []
        for root, dirs, files in os.walk(self.temp_dir):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '__pycache__', 'venv', 'env'}]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in self.SUPPORTED_EXTENSIONS:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            code_files.append({
                                'path': str(file_path.relative_to(self.temp_dir)),
                                'name': file,
                                'extension': file_path.suffix,
                                'content': content
                            })
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
        
        return code_files


class CodeLlamaReviewer:
    """AI Code Reviewer using abooze/deepseek-coder-1b-instruct-finetune-cnGen"""
    
    def __init__(self, model_name: str = "abooze/deepseek-coder-1b-instruct-finetune-cnGen"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        
    def load_model(self):
        """Load CodeLlama model from HuggingFace"""
        print("Loading abooze/deepseek-coder-1b-instruct-finetune-cnGen...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            low_cpu_mem_usage=True
        )
        
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True
        )
        
        print("Model loaded successfully!")
        
    def create_review_prompt(self, code: str, filename: str, rules: List[str]) -> str:
        """Create prompt for code review"""
        rules_text = "\n".join([f"- {rule}" for rule in rules[:10]])  # Limit to first 10 rules
        
        prompt = f"""[INST] You are an expert code reviewer. Review the following code against these rules:

{rules_text}

File: {filename}

Code:
```
{code[:2000]}  # Limit code length for context
```

Provide:
1. Issues found (with line references if possible)
2. Suggestions for improvement
3. Compliance status with the rules

Be specific and actionable. [/INST]"""
        
        return prompt
    
    def review_code(self, code_file: Dict[str, str], rules: List[str]) -> Dict[str, Any]:
        """Review a single code file"""
        if not self.pipeline:
            self.load_model()
            
        prompt = self.create_review_prompt(
            code_file['content'],
            code_file['name'],
            rules
        )
        
        try:
            result = self.pipeline(prompt)
            review_text = result[0]['generated_text']
            
            # Extract only the response part (after [/INST])
            if '[/INST]' in review_text:
                review_text = review_text.split('[/INST]')[-1].strip()
            
            return {
                'file': code_file['path'],
                'review': review_text,
                'status': 'completed'
            }
        except Exception as e:
            return {
                'file': code_file['path'],
                'review': f"Error during review: {str(e)}",
                'status': 'error'
            }
    
    def generate_nfr(self, code_files: List[Dict[str, str]], rules: List[str]) -> str:
        """Generate Non-Functional Requirements based on code analysis"""
        if not self.pipeline:
            self.load_model()
            
        # Summarize codebase
        file_summary = "\n".join([f"- {f['path']} ({f['extension']})" for f in code_files[:20]])
        
        prompt = f"""[INST] Based on this codebase structure and the provided rules, generate comprehensive Non-Functional Requirements (NFRs):

Codebase files:
{file_summary}

Rules:
{chr(10).join([f"- {rule}" for rule in rules[:10]])}

Generate NFRs covering:
1. Performance Requirements
2. Security Requirements
3. Scalability Requirements
4. Reliability/Availability
5. Maintainability
6. Usability

Provide specific, measurable requirements. [/INST]"""
        
        try:
            result = self.pipeline(prompt)
            nfr_text = result[0]['generated_text']
            
            if '[/INST]' in nfr_text:
                nfr_text = nfr_text.split('[/INST]')[-1].strip()
            
            return nfr_text
        except Exception as e:
            return f"Error generating NFRs: {str(e)}"


class CodeReviewSystem:
    """Main code review system orchestrator"""
    
    def __init__(self, zip_path: str, pdf_path: str):
        self.zip_path = zip_path
        self.pdf_path = pdf_path
        self.reviewer = CodeLlamaReviewer()
        
    def run_review(self, output_path: str = "review_results.json"):
        """Run complete code review process"""
        print("=" * 60)
        print("AI CODE REVIEWER - Starting Analysis")
        print("=" * 60)
        
        # Step 1: Extract rules from PDF
        print("\n[1/4] Extracting rules from PDF...")
        pdf_extractor = PDFRuleExtractor(self.pdf_path)
        rules = pdf_extractor.parse_rules()
        print(f"✓ Extracted {len(rules)} rules from PDF")
        
        # Step 2: Extract code files from zip
        print("\n[2/4] Extracting code files from zip...")
        code_extractor = CodeExtractor(self.zip_path)
        code_files = code_extractor.get_code_files()
        print(f"✓ Found {len(code_files)} code files")
        
        # Step 3: Load AI model
        print("\n[3/4] Loading CodeLlama model...")
        self.reviewer.load_model()
        
        # Step 4: Review each file
        print("\n[4/4] Reviewing code files...")
        reviews = []
        for i, code_file in enumerate(code_files, 1):
            print(f"  Reviewing {code_file['path']} ({i}/{len(code_files)})...")
            review = self.reviewer.review_code(code_file, rules)
            reviews.append(review)
        
        # Step 5: Generate NFRs
        print("\n[5/5] Generating Non-Functional Requirements...")
        nfr = self.reviewer.generate_nfr(code_files, rules)
        
        # Compile results
        results = {
            'summary': {
                'total_files': len(code_files),
                'total_rules': len(rules),
                'reviewed_files': len([r for r in reviews if r['status'] == 'completed'])
            },
            'rules': rules[:20],  # Include first 20 rules
            'reviews': reviews,
            'non_functional_requirements': nfr
        }
        
        # Save results
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "=" * 60)
        print(f"✓ Review completed! Results saved to {output_path}")
        print("=" * 60)
        
        return results



if __name__ == "__main__":
    # Configuration
    GITHUB_ZIP = "repo.zip"
    RULES_PDF = "Company_Programming_Rules.pdf"
    OUTPUT_FILE = "code_review_results.json"
    
    # Run review
    review_system = CodeReviewSystem(GITHUB_ZIP, RULES_PDF)
    results = review_system.run_review(OUTPUT_FILE)
    
    # Print summary
    print("\n📊 Review Summary:")
    print(f"Total Files Analyzed: {results['summary']['total_files']}")
    print(f"Successfully Reviewed: {results['summary']['reviewed_files']}")
    print(f"Rules Applied: {results['summary']['total_rules']}")
    print(f"\nDetailed results saved to: {OUTPUT_FILE}")
