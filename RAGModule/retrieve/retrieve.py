from typing import List
from qdrant_client import QdrantClient, models

import torch
from transformers import AutoModel
from FlagEmbedding import BGEM3FlagModel

from RAGModule.utils import convert_defaultdict
from langchain_core.documents import Document

torch.set_grad_enabled(False)

def retrieve_relevant_chunks(query: str, jina_embedding: AutoModel, bge_embedding: BGEM3FlagModel, client: QdrantClient) -> List[Document]:
    dense = jina_embedding.encode([query], task = "retrieval.query")
    sparse = bge_embedding.encode([query], return_dense=False, return_sparse=True, return_colbert_vecs=False)
    result = client.query_points(
        "vector_store",
        prefetch=[
            models.Prefetch(
                query = dense[0],
                using = "dense",
                limit = 10
            ),
            models.Prefetch(
                query = models.SparseVector(**convert_defaultdict(sparse['lexical_weights'][0])),
                using = "sparse",
                limit = 10
            )
        ],
        query = models.FusionQuery(
            fusion = models.Fusion.RRF,
        ),
        limit = 3
    )

    relevant_chunks = []
    for point in result.points:
        point_payload = point.payload
        relevant_chunk = Document(page_content=point_payload['page_content'], metadata = {'source': point_payload['source'], 'title': point_payload['title'], 'summary': point_payload['summary'], 'num_chunks': point_payload['num_chunks'], 'doc_id': point_payload['doc_id'], 'chunk_no': point_payload['chunk_no']})
        relevant_chunks.append(relevant_chunk)
    
    return relevant_chunks
