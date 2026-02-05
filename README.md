## Architecture

```
User Question
    ↓
Local Embeddings (sentence-transformers)
    → Finds top 3 most similar Q&A pairs (100ms)
    ↓
Claude AI (Anthropic API)
    → Picks best match + rephrases naturally (1-2s)
    ↓
Natural Answer + Emotion
```

## Prerequisites

- Python 3.11 or 3.12 (Python 3.14 not supported due to PyTorch compatibility)
- macOS, Linux, or Windows
- Anthropic API key ([Get one here](https://console.anthropic.com/))

## Installation

### 1. Clone or Download the Repository

```bash
git clone 
cd Clippy
```

Or download and extract the ZIP file.

### 2. Install Python 3.12 (if needed)

**macOS:**
```bash
brew install python@3.12
```

**Linux:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
```

**Windows:**
Download from [python.org](https://www.python.org/downloads/)

### 3. Create Virtual Environment

```bash
# Create venv with Python 3.12
python3.12 -m venv .venv

# Activate it
# macOS/Linux:
source .venv/bin/activate

# Windows:
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt.

### 4. Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install NumPy 1.x (required for compatibility)
pip install "numpy<2"

# Install all packages
pip install flask anthropic sentence-transformers scikit-learn
```

### 5. Set Up API Key

Get your Anthropic API key from [console.anthropic.com](https://console.anthropic.com/)

**Option A - Environment Variable (Temporary):**
```bash
export ANTHROPIC_API_KEY='your-api-key-here'
```

**Option B - .env File (Recommended):**

Create a `.env` file in the project root:
```bash
echo "ANTHROPIC_API_KEY='your-api-key-here'" > .env
```

Install dotenv:
```bash
pip install python-dotenv
```

Add to top of `app.py`:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Project Structure

```
Clippy/
├── app.py                    # Main Flask application
├── knowledge_base.json       # Q&A pairs database
├── templates/
│   └── index.html           # Web interface
├── static/
│   └── style.css            # Styles
├── .env                     # API keys (create this)
└── README.md                # This file
```

## Usage

### Running the Application

```bash
# Make sure venv is activated (you see (.venv) in prompt)
python app.py
```