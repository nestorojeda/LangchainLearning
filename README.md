# Langchain Learning

This repository contains my journey and projects related to learning Langchain - a framework for developing applications powered by language models.

## Table of Contents

- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Projects](#projects)
- [Resources](#resources)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Langchain is a powerful framework that simplifies the integration of large language models (LLMs) into applications. This repository documents my learning process, experiments, and projects built using Langchain. The projects primarily use local models from Ollama for inference, allowing for privacy-focused and cost-effective LLM applications.

## Getting Started

### Prerequisites

- Python 3.13+
- pipenv
- Ollama (for local model inference)

### Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/LangchainLearning.git
cd LangchainLearning

# Install dependencies using pipenv
pipenv install

# Activate the pipenv environment
pipenv shell

# Make sure Ollama is installed and running for local model inference
```

## Projects
### 1. PDF Processor

A full-stack application to process and summarize academic papers using LLMs.

- **Features**:
    - Downloads PDF files from arXiv links
    - Extracts text content from PDFs
    - Processes large documents in parallel chunks
    - Generates comprehensive summaries focused on technical details
    - Streamlit-based user interface with status feedback

- **Tech Stack**:
    - FastAPI backend
    - Streamlit frontend
    - PyMuPDF for PDF text extraction
    - Ollama with Gemma 3 model for inference
    - Asyncio for parallel processing

- **Running the Project**:
    ```bash
    cd projects/langchain_pdf_processor
    ./run.sh
    ```

## Resources

- [Langchain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Langchain GitHub Repository](https://github.com/hwchase17/langchain)
- [LLM Tutorials and Examples](https://www.pinecone.io/learn/series/langchain/)
- [Ollama - Run LLMs locally](https://ollama.com/)

## Contributing

If you'd like to contribute to this repository or suggest improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
