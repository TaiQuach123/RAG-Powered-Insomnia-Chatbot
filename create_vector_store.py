import os
import json
import argparse
from qdrant_client import QdrantClient, models

client = QdrantClient("http://localhost:6333")


if not client.collection_exists(collection_name="vector_store"):
    client.create_collection(
        "vector_store",
        vectors_config={
            "dense": models.VectorParams(
                size=1024,
                distance=models.Distance.COSINE
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams()
        }
    )

parser = argparse.ArgumentParser(description="Create Vector Store")
parser.add_argument("--dir", default="./Data/json_output/TÁC HẠI")
args = parser.parse_args()

if __name__ == "__main__":
    for path in os.listdir(args.dir):
        path = os.path.join(args.dir, path)
        
        with open(path) as f:
            data = json.load(f)

        for chunk in data:
            chunk_id = chunk['metadata']['doc_id']
            chunk_metadata = chunk['metadata']
            chunk_content = chunk['page_content']
            dense_embedding = chunk['dense']
            sparse_embedding = chunk['sparse']

            try:
                client.upload_points(
                    "vector_store",
                    points = [
                        models.PointStruct(
                            id = chunk_id,
                            vector = {
                                "dense": dense_embedding,
                                "sparse": sparse_embedding
                            },
                            payload= {**chunk_metadata, **{"page_content": chunk_content}},
                        )
                    ],
                    batch_size=1
                )
            
            except:
                print(f"Error while uploading - {chunk_id}")
                continue