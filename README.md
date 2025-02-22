<h1 align="center"><b>RAG-Powered-Insomnia-Chatbot</b></h1>

## Table of Contents
- [Quick Start](#quick-start)
- [Introduction](#introduction)
- [Usage](#usage)
    - [Clone the Github Repo](#clone-the-github-repo)
    - [Extracting Information From PDFs](#extracting-information-from-pdfs)
    - [Preparing Data For Creating Vector Store](#preparing-data-for-creating-vector-store)
    - [Create Vector Store And Upsert Points](#create-vector-store-and-upsert-points)
    - ...

## Quick Start
### Video Demo
https://github.com/user-attachments/assets/b9650235-0edc-4827-a8ea-7870b9c789ff

## Introduction
The RAG-Powered Insomnia Chatbot aims to assist users in obtaining reliable information about insomnia by leveraging advanced AI techniques. The chatbot combines data extraction, intelligent retrieval, and dynamic conversation flow to provide accurate responses from research articles. It employs Retrieval-Augmented Generation (RAG) techniques along with hybrid search and agentic workflows to enhance user interaction and improve overall performance.


## Usage
### Clone the Github Repo
```bash
git clone https://github.com/TaiQuach123/RAG-Powered-Insomnia-Chatbot
```

### Extracting information from PDFs

``extract_pdf.py``: Create markdown files from PDF files

```python
# Extract all PDF files from a specific directory
python extract_pdf.py --dir "Data/Database/TÁC HẠI"
```


### Preparing data for creating vector store

```prepare_vector_store.py```: Create JSON files (each JSON file corresponding to a document - contains a list of dictionaries, each dictionary contains the metadata and vector representations of each chunk)

```python
# Create JSON files including text, metadata, dense and sparse vectors for all chunks created from a specific directory
# The result can be viewed in "Data/json_files"
python prepare_vector_store.py --dir "./Data/extracted/TÁC HẠI" --target_dir "./Data/json_output/TÁC HẠI"
```


### Create Vector Store and Upsert Points
```create_vector_store.py```: Create a Qdrant vectorstore (If not exist) and upsert points into this vectorstore.

```python
python create_vector_store.py --dir "./Data/json_output/TÁC HẠI"
```

