import os
import argparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue

# --- Environment Variables ---
QDRANT_URL = os.environ.get("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "isc2_toronto_v3")

# --- Qdrant Client ---
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def delete_by_doc_id(doc_id: str):
    """Deletes all data points associated with a specific doc_id."""
    query_filter = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )
    client.delete(collection_name=COLLECTION, filter=query_filter)
    print(f"Deleted all data points associated with doc_id: {doc_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Delete data points by doc_id.")
    parser.add_argument("doc_id", type=str, help="The doc_id to delete.")
    args = parser.parse_args()

    delete_by_doc_id(args.doc_id)