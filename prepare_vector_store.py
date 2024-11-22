import json
import os
import argparse
import numpy as np

import torch
from transformers import AutoModel, AutoTokenizer
from FlagEmbedding import BGEM3FlagModel

from RAGModule.chunking import create_semantic_chunks_from_directory_with_overlap
from RAGModule.chunking import long_late_chunking
from RAGModule.chunking import JinaV3Encoder
from RAGModule.utils import convert_defaultdict

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(description="Prepare Vector Store")
parser.add_argument("--dir", default="./Data/extracted/TÁC HẠI")
parser.add_argument("--target_dir", default="./Data/json_files/TÁC HẠI")
args = parser.parse_args()

jina_encoder = JinaV3Encoder()
bge_m3 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)


model_name = 'jinaai/jina-embeddings-v3'
device = 'cuda'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
query_task_id = model._adaptation_map['retrieval.query']
query_adapter_mask = torch.full((1,), query_task_id, dtype=torch.int32, device=model.device)
passage_task_id = model._adaptation_map['retrieval.passage']
passage_adapter_mask = torch.full((1,), passage_task_id, dtype=torch.int32, device=model.device)
model.eval()

if __name__ == "__main__":
    os.makedirs(args.target_dir, exist_ok=True)
    docs_with_chunks = create_semantic_chunks_from_directory_with_overlap(args.dir, encoder=jina_encoder, min_split_tokens=128, max_split_tokens=768, window_size=10)

    jina_encoder._client.to('cpu')
    jina_encoder = None
    torch.cuda.empty_cache()

    for doc_chunks in docs_with_chunks:
        chunks = [chunk.page_content for chunk in doc_chunks]
        sparse_embeddings = bge_m3.encode(chunks, batch_size=8, return_dense=False, return_sparse=True, return_colbert_vecs=False)['lexical_weights']
        dense_embeddings = long_late_chunking(model=model, tokenizer=tokenizer, passage_adapter_mask=passage_adapter_mask, chunks=chunks, max_tokens=8192, overlap_size = 2048)
        for i in range(len(doc_chunks)):
            values_indices = convert_defaultdict(sparse_embeddings[i])
            
            values = values_indices['values'].tolist()
            indices = values_indices['indices'].tolist()
            
            temp = {'metadata': doc_chunks[i].metadata, 'page_content': doc_chunks[i].page_content, 'dense': dense_embeddings[i].tolist(), 'sparse': {'values': values, 'indices': indices}}
            with open(os.path.join(args.target_dir, doc_chunks[i].metadata['doc_id'] + '.json'), 'w') as f:
                json.dump(temp, f)