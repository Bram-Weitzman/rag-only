# check_qdrant.py
import os
from qdrant_client import QdrantClient

# --- Configuration ---
# Make sure these match your API and scraper configurations
QDRANT_URL = os.getenv("QDRANT_URL", "http://10.20.10.30:6333")
COLLECTION = os.getenv("QDRANT_COLLECTION", "isc2_toronto_v3")

if __name__ == "__main__":
    try:
        client = QdrantClient(url=QDRANT_URL)
        print(f"Successfully connected to Qdrant at {QDRANT_URL}")
        
        # Get the total count of points in the collection
        count_result = client.count(collection_name=COLLECTION, exact=True)
        point_count = count_result.count
        print(f"\nFound {point_count} points in the collection '{COLLECTION}'.")
        
        # If there are points, fetch and display a few samples
        if point_count > 0:
            print("\n--- Sample of 5 points ---")
            scroll_result = client.scroll(
                collection_name=COLLECTION,
                limit=5,
                with_payload=True,
                with_vectors=False
            )
            for record in scroll_result[0]:
                print(f"ID: {record.id}")
                print(f"  Payload: {record.payload}")
                print("-" * 20)

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check if the Qdrant container is running and the URL is correct.")
