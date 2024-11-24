# RAG-Powered-Insomnia-Chatbot

## Extracting information from PDFs

``extract_pdf.py``: Create markdown files from PDF files

```python
# Extract all PDF files from a specific directory
python extract_pdf.py --dir "Data/Database/TÁC HẠI"
```


## Preparing data for creating vector store

```prepare_vector_store.py```: Create JSON files (each JSON file corresponding to a document - contains a list of dictionaries, each dictionary contains the metadata and vector representations of each chunk)

```python
# Create JSON files including text, metadata, dense and sparse vectors for all chunks created from a specific directory
# The result can be viewed in "Data/json_files"
python prepare_vector_store.py --dir "./Data/extracted/TÁC HẠI" --target_dir "./Data/json_output/TÁC HẠI"
```


## Create Vector Store and Upsert Points
```create_vector_store.py```: Create a Qdrant vectorstore (If not exist) and upsert points into this vectorstore.

```python
python create_vector_store.py --dir "./Data/json_output/TÁC HẠI"
```

