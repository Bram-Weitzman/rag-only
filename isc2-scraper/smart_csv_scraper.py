#!/usr/bin/env python3
# smart_csv_scraper.py — Scrapes a website and saves chunked content to a CSV file.

import os
import re
import time
import urllib.parse
import itertools
import csv
from typing import List, Dict, Any, Optional

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup, Tag
import tldextract
import xxhash
from tqdm import tqdm

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except ImportError:
    pdf_extract_text = None

# ------------------------- Config (env-overridable) -------------------------
CSV_OUTPUT_FILE  = "scraped_data.csv"
START_URL        = os.getenv("START_URL", "https://isc2chapter-toronto.ca/")
SITEMAP_URL      = os.getenv("SITEMAP_URL", f"{START_URL.rstrip('/')}/sitemap_index.xml")
SITEMAP_STRICT   = os.getenv("SITEMAP_STRICT", "0") == "1"

# Performance / reliability knobs
HTTP_TIMEOUT     = float(os.getenv("HTTP_TIMEOUT", "15"))
HTTP_RETRIES     = int(os.getenv("HTTP_RETRIES", "3"))
USER_AGENT       = os.getenv("UA", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# --- (Regex for allowed/denied paths can remain here if you plan to use the full crawl) ---

# ------------------------- HTTP session w/ retries -------------------------
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT})
retry_cfg = Retry(total=HTTP_RETRIES, backoff_factor=0.8, status_forcelist=(429, 500, 502, 503, 504))
adapter = HTTPAdapter(max_retries=retry_cfg)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)

# ------------------------- URL helpers (and Sitemap) -------------------------
ALLOWED_KEYWORDS = ["index", "events", "leadership", "membership"]  # Keywords to filter URLs

def normalize_url(url: str) -> str:
    u = urllib.parse.urlsplit(url)
    scheme = u.scheme.lower() or "https"
    netloc = u.netloc.lower() or urllib.parse.urlsplit(START_URL).netloc
    path = re.sub(r"/{2,}", "/", u.path or "/")
    return urllib.parse.urlunsplit((scheme, netloc, path, u.query, u.fragment))

def domain_root(url: str) -> str:
    e = tldextract.extract(url)
    return f"{e.domain}.{e.suffix}"

ROOT_DOMAIN = domain_root(START_URL)

def same_site(u: str) -> bool:
    return domain_root(u) == ROOT_DOMAIN

def _fetch_xml(url: str) -> Optional[BeautifulSoup]:
    try:
        r = SESSION.get(url, timeout=HTTP_TIMEOUT)
        return BeautifulSoup(r.text, "xml") if r.ok else None
    except Exception:
        return None

def iter_sitemap_links() -> List[str]:
    urls: List[str] = []
    soup = _fetch_xml(SITEMAP_URL)
    if not soup: return urls

    if soup.find("sitemapindex"):
        submaps = [loc.text.strip() for loc in soup.select("sitemap > loc") if loc.text.strip()]
        for sm_url in submaps:
            sm_soup = _fetch_xml(sm_url)
            if sm_soup:
                urls.extend([loc.text.strip() for loc in sm_soup.select("url > loc") if loc.text.strip()])
    elif soup.find("urlset"):
        urls.extend([loc.text.strip() for loc in soup.select("url > loc") if loc.text.strip()])
    
    # Normalize and filter URLs based on allowed keywords
    filtered_urls = [
        normalize_url(u) for u in urls if same_site(u) and any(keyword in u for keyword in ALLOWED_KEYWORDS)
    ]
    
    return sorted(set(filtered_urls))

# ------------------------- HTML parsing / chunking -------------------------
def text_of(node: Tag) -> str:
    for br in node.find_all(["br"]): br.replace_with("\n")
    txt = node.get_text("\n", strip=True)
    return re.sub(r"\n{3,}", "\n\n", txt)

def chunk_by_sentence(text: str) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

# ------------------------- Indexing to CSV -------------------------
def index_pages(pages: List[str]) -> None:
    print(f"Starting scrape. Output will be saved to {CSV_OUTPUT_FILE}")
    
    with open(CSV_OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "text_content", "source_url"]) # Write header
        
        total_chunks = 0
        
        for url in tqdm(pages, desc="Scraping pages", unit="page"):
            try:
                r = SESSION.get(url, timeout=HTTP_TIMEOUT)
                if not r.ok:
                    print(f"[SKIP] {url} returned status {r.status_code}")
                    continue
                
                content_type = r.headers.get("content-type", "").lower()
                
                if "text/html" in content_type:
                    soup = BeautifulSoup(r.text, "lxml")
                    page_text = text_of(soup.body or soup)
                    chunks = chunk_by_sentence(page_text)
                elif "application/pdf" in content_type and pdf_extract_text:
                    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
                        tmp.write(r.content)
                        tmp.flush()
                        page_text = pdf_extract_text(tmp.name) or ""
                    chunks = chunk_by_sentence(page_text)
                else:
                    print(f"[SKIP] {url} is not a supported content type ({content_type})")
                    continue
                
                for idx, chunk in enumerate(chunks):
                    url_slug = (url.strip('/').split('/')[-1] or "home")[:20]
                    doc_id = f"{url_slug}-{idx}"
                    writer.writerow([doc_id, chunk, url])
                    total_chunks += 1

            except Exception as e:
                print(f"[WARN] Failed to process {url}: {e}")
                continue
    
    print(f"\n[DONE] Scraped {total_chunks} chunks from {len(pages)} pages.")
    print(f"Data saved to {CSV_OUTPUT_FILE}")

# ------------------------- Main -------------------------
def main() -> None:
    print("[BOOT] Running in CSV generation mode.")
    sitemap_urls = iter_sitemap_links()
    if sitemap_urls:
        print(f"[BOOT] Sitemap URLs discovered: {len(sitemap_urls)}")
    else:
        print("[WARN] No sitemap URLs found. Consider checking SITEMAP_URL.")
    
    # Using sitemap URLs as the definitive list of pages to process.
    if not sitemap_urls:
        print("[DONE] No pages found to scrape.")
        return
        
    index_pages(sitemap_urls)

if __name__ == "__main__":
    main()