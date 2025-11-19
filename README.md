# 🔍 AI Code Reviewer

AI Code Reviewer is a **Streamlit web app** that uses **CodeLlama (7B Instruct)** to review your code against a **PDF of coding rules/standards**.

You upload:

1. A **project ZIP file** (your repository)
2. A **PDF file** with coding rules / guidelines

The app will:

- Scan your code files
- Review each file against the PDF rules
- List issues and suggestions
- Generate high-level **Non-Functional Requirements (NFRs)**
- Let you **download a full report** as JSON or text

---

## ✨ Features

- 📦 Upload a **ZIP** of your project repository  
- 📄 Upload a **PDF** with coding standards / rules  
- 🔍 Per-file AI review using **CodeLlama-7b-Instruct-hf**
- 📋 Structured output for each file:
  - Issues found  
  - Severity (Critical / High / Medium / Low)  
  - Suggestions  
  - Code quality score (1–10)
- 📊 Auto-generated **Non-Functional Requirements**:
  - Performance  
  - Scalability  
  - Security  
  - Reliability  
  - Maintainability  
  - Usability  
  - Compatibility  
  - Monitoring
- 💾 Downloadable report (JSON + text)

---

## 🧰 Tech Stack

- **Python**
- **Streamlit** – web UI
- **PyPDF2** – PDF text extraction
- **Hugging Face Transformers**
- **CodeLlama-7b-Instruct-hf** – code review model
- **PyTorch**

---

## 📁 Supported Code File Types

The app looks for common code extensions inside the ZIP:

- `.py`, `.js`, `.ts`, `.jsx`, `.tsx`
- `.java`, `.cpp`, `.c`, `.go`, `.rs`
- `.php`, `.rb`, `.swift`, `.kt`, `.cs`

It **ignores** folders like:

- `.git`
- `node_modules`
- `__pycache__`
- `venv`
- `dist`
- `build`

---

## 🖥️ Requirements

Because this uses a 7B model, a **GPU** is strongly recommended.

- Python 3.9+  
- `pip`  
- GPU with enough VRAM (ideal) or a strong CPU (slower)

Suggested Python packages:

```bash
streamlit
torch
transformers
PyPDF2
