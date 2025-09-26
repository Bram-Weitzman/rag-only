# ingest_local.py
import requests
import os
import csv

# --- Configuration ---
API_ENDPOINT = "http://10.20.10.30:8001/ingest"
CSV_FILE_PATH = "./data-docs/manual_data.csv"


# --- Helper Functions ---
def ingest_row(row):
    """Send a single row to the ingest API."""
    payload = {
        "items": [
            {
                "text": row["text_content"],
                "doc_id": row["doc_id"],
                "source": row["source_url"]
            }
        ]
    }

    try:
        response = requests.post(API_ENDPOINT, json=payload, timeout=120)
        response.raise_for_status()
        print(f"[SUCCESS] Ingested doc_id: {row['doc_id']}")
    except Exception as e:
        print(f"[ERROR] Failed to ingest doc_id: {row['doc_id']}. Error: {e}")


# --- Main script ---
if __name__ == "__main__":
    try:
        print(f"Reading CSV content from {CSV_FILE_PATH}...")
        with open(CSV_FILE_PATH, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile, delimiter='\t')
            if reader.fieldnames != ["doc_id", "text_content", "source_url", "ingest_keep"]:
                raise ValueError("CSV file must have columns: doc_id, text_content, source_url, ingest_keep")

            for row in reader:
                if row["ingest_keep"].strip().lower() == "true":
                    ingest_row(row)
                else:
                    print(f"[SKIP] Skipped doc_id: {row['doc_id']} (ingest_keep is False)")

    except FileNotFoundError:
        print(f"\nError: File not found at '{CSV_FILE_PATH}'. Make sure the file exists.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
