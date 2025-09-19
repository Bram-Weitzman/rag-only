# delete_collection.py
import os
from qdrant_client import QdrantClient

# --- Configuration ---
# Make sure these match your smart_scrape.py settings
QDRANT_URL = os.getenv("QDRANT_URL", "http://10.20.10.30:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "isc2_toronto_v3")

# --- Main script ---
if __name__ == "__main__":
    print(f"Connecting to Qdrant at {QDRANT_URL}...")
    try:
        client = QdrantClient(url=QDRANT_URL)
        
        print(f"Attempting to delete collection: '{COLLECTION}'...")
        delete_result = client.delete_collection(collection_name=COLLECTION)
        
        if delete_result:
            print(f"Collection '{COLLECTION}' deleted successfully.")
        else:
            print(f"Collection '{COLLECTION}' did not exist or could not be deleted.")
            
    except Exception as e:
        print(f"An error occurred: {e}")
