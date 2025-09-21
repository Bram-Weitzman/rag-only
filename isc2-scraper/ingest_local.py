# ingest_local.py
import requests
import os

# --- Configuration ---
API_ENDPOINT = "http://10.20.10.30:8001/ingest"
FILE_PATH = "./data-docs/Event_ISC2-Toronto_2025-09-25.txt"


# --- Main script ---
if __name__ == "__main__":
    try:
        print(f"Reading content from {FILE_PATH}...")
        with open(FILE_PATH, 'r', encoding='utf-8') as f:
            text_content = f.read()

        # Use the filename as the document ID
        doc_id = os.path.basename(FILE_PATH)

        payload = {
            "items": [
                {
                    "text": text_content,
                    "doc_id": doc_id,
                    "source": "curated_text_file"
                }
            ]
        }

        print(f"Sending content to the ingest API at {API_ENDPOINT}...")
        response = requests.post(API_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()  # Raise an exception for bad status codes (like 4xx or 5xx)

        print("\nIngestion successful!")
        print(response.json())

    except FileNotFoundError:
        print(f"\nError: File not found at '{FILE_PATH}'. Make sure you are in the 'isc2-scraper' directory.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
