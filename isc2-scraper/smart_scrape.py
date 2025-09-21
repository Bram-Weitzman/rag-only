
#!/usr/bin/env python3
# smart_scrape.py — robust RAG indexer with progress + resume; spaces only

import os
import re
import time
import math
import tempfile
import urllib.parse
import itertools
from typing import List, Dict, Any, Optional

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup, Tag
import tldextract
from tenacity import retry, stop_after_attempt, wait_exponential
import xxhash
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct,
    OptimizersConfigDiff, PayloadSchemaType,
    TextIndexParams, TextIndexType,
    Filter, FieldCondition, MatchValue,
)

# ------------------------- Config (env-overridable) -------------------------
QDRANT_URL       = os.getenv("QDRANT_URL", "http://10.20.10.30:6333")
OLLAMA_HOST      = os.getenv("OLLAMA_HOST", "http://10.20.10.30:11434")
EMBED_MODEL      = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION       = os.getenv("QDRANT_COLLECTION", "isc2_toronto_v3")
ALIAS            = os.getenv("QDRANT_ALIAS", "isc2_active")

START_URL        = os.getenv("START_URL", "https://isc2chapter-toronto.ca")
SITEMAP_URL      = os.getenv("SITEMAP_URL", f"{START_URL.rstrip('/')}/sitemap.xml")
SITEMAP_STRICT   = os.getenv("SITEMAP_STRICT", "0") == "1"

# Performance / reliability knobs
HTTP_TIMEOUT     = float(os.getenv("HTTP_TIMEOUT", "15"))     # per GET
HTTP_RETRIES     = int(os.getenv("HTTP_RETRIES", "3"))
EMBED_TIMEOUT    = float(os.getenv("EMBED_TIMEOUT", "600"))   # per embed POST
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "8"))    # embed sub-batch size
BATCH_SIZE       = int(os.getenv("BATCH_SIZE", "64"))         # Qdrant upsert batch

#USER_AGENT       = os.getenv("UA", "ISC2-Indexer/1.4 (+self-hosted)")
USER_AGENT       = os.getenv("UA", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36")

# Keep regexes in-code to avoid .env quoting issues
ALLOWED_PATH_REGEX = r"""
^/$
|^/\d{4}/\d{2}/[^/].*                         # date permalinks
|^/(?:post|posts|page|pages)(?:/|$)
|^/(?:blog|news|events?|event)(?:/|$)
|^/(?:category|categories)(?:/|$)
|^/(?:tag|tags)(?:/|$)
|^/(?:author|authors)(?:/|$)
|^/(?:about|team|leadership|board|officers?)(?:/|$)
|^/(?:membership|join|pricing|fees?)(?:/|$)
|^/(?:sponsors?|partners?)(?:/|$)
|^/(?:resources?|faq|polic(?:y|ies)|privacy|terms|contact|volunteer)(?:/|$)
|^/.*\.(?:pdf|ics)$                            # docs & calendars
""".strip()

DENY_PATH_REGEX = r"""
^/(?:wp-admin|wp-login\.php|xmlrpc\.php)(?:/|$)
|/feed/?$
|/comments/?$
|/cart(?:/|$)|/checkout(?:/|$)|/my-account(?:/|$)
|\?replytocom=
|^/category/uncategorized(?:/|$)  # <-- ADD THIS LINE to exclude the problem category
""".strip()

# Optional PDF text extraction
try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

# ------------------------- HTTP session w/ retries -------------------------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en,*;q=0.8",
    "Connection": "keep-alive",
})
retry_cfg = Retry(
    total=HTTP_RETRIES,
    backoff_factor=0.8,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET", "POST"]),
)
adapter = HTTPAdapter(max_retries=retry_cfg, pool_connections=20, pool_maxsize=50)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)

# ------------------------- URL helpers -------------------------
def normalize_url(url: str) -> str:
    u = urllib.parse.urlsplit(url)
    scheme = u.scheme.lower() if u.scheme else "https"
    netloc = u.netloc.lower() if u.netloc else urllib.parse.urlsplit(START_URL).netloc
    path = re.sub(r"/{2,}", "/", u.path or "/")
    return urllib.parse.urlunsplit((scheme, netloc, path, u.query, u.fragment))

def domain_root(url: str) -> str:
    e = tldextract.extract(url)
    return f"{e.domain}.{e.suffix}"

ROOT_DOMAIN = domain_root(START_URL)

def same_site(u: str) -> bool:
    try:
        return domain_root(u) == ROOT_DOMAIN
    except Exception:
        return False

_allowed_re = re.compile(ALLOWED_PATH_REGEX, re.I | re.X)
_deny_re    = re.compile(DENY_PATH_REGEX,    re.I | re.X)

def allowed_path(u: str) -> bool:
    p = urllib.parse.urlsplit(u).path or "/"
    if _deny_re.search(p):
        return False
    return bool(_allowed_re.search(p))

# ------------------------- Health checks -------------------------
def check_qdrant() -> None:
    try:
        r = SESSION.get(f"{QDRANT_URL}/collections", timeout=HTTP_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        print(f"[HEALTH] Qdrant not reachable at {QDRANT_URL}: {e}")

def check_ollama() -> None:
    try:
        r = SESSION.get(f"{OLLAMA_HOST}/api/tags", timeout=HTTP_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        print(f"[HEALTH] Ollama not reachable at {OLLAMA_HOST}: {e}")

# ------------------------- Sitemap handling -------------------------
def _fetch_xml(url: str) -> Optional[BeautifulSoup]:
    try:
        r = SESSION.get(url, timeout=HTTP_TIMEOUT)
        if not r.ok:
            return None
        return BeautifulSoup(r.text, "xml")
    except Exception:
        return None

def iter_sitemap_links() -> List[str]:
    urls: List[str] = []
    soup = _fetch_xml(SITEMAP_URL)
    if not soup:
        return urls

    if soup.find("sitemapindex"):
        submaps = [loc.text.strip() for loc in soup.select("sitemap > loc") if loc.text.strip()]
        for sm in submaps:
            ss = _fetch_xml(sm)
            if not ss:
                continue
            urls.extend([loc.text.strip() for loc in ss.select("url > loc") if loc.text.strip()])
    elif soup.find("urlset"):
        urls.extend([loc.text.strip() for loc in soup.select("url > loc") if loc.text.strip()])

    urls = [normalize_url(u) for u in urls if same_site(u)]
    return sorted(set(urls))

# ------------------------- HTML parsing / chunking -------------------------
def text_of(node: Tag) -> str:
    for br in node.find_all(["br"]):
        br.replace_with("\n")
    for code in node.find_all(["code", "pre"]):
        code.string = (code.get_text("\n") or "").strip()
    txt = node.get_text("\n", strip=True)
    txt = re.sub(r"\n{3,}", "\n\n", txt)
    return txt

def heading_path(h: Optional[Tag]) -> str:
    if not h:
        return ""
    path = []
    cur = h
    while True:
        if cur and isinstance(cur, Tag) and re.fullmatch(r"h[1-6]", cur.name or ""):
            t = cur.get_text(" ", strip=True)
            if t:
                path.append(t)
        cur = cur.find_previous(re.compile(r"^h[1-6]$"))
        if not cur:
            break
    return " > ".join(reversed(path)) if path else ""

def classify_kind(h_txt: str, snippet: str) -> str:
    lower = (h_txt + " " + snippet).lower()
    if re.search(r"\b(member(ship)?|join|fee|price|pricing|dues)\b", lower):
        return "membership"
    if re.search(r"\b(event|meetup|webinar|talk|speaker|rsvp|register)\b", lower):
        return "event"
    if re.search(r"\b(team|board|officer|leadership|director|committee|president|secretary|treasurer)\b", lower):
        return "roles"
    if re.search(r"\b(sponsor|partner)\b", lower):
        return "sponsors"
    if re.search(r"\b(blog|news|post|article)\b", lower):
        return "blog"
    return "generic"

def extract_signals(txt: str) -> Dict[str, Any]:
    prices = re.findall(r"(\$[0-9][0-9,]*)(?:\s*/\s*(year|yr|month|mo|annual))?", txt, flags=re.I)
    roles  = re.findall(r"(President|Vice[-\s]?President|Treasurer|Secretary|Director|Chair|Co[-\s]?Chair|Coordinator|Lead)", txt, flags=re.I)
    dates  = re.findall(r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t\.?|tember)|Oct(?:ober)?|Nov(?:ember)?)\s+\d{1,2},\s+\d{4}\b", txt)
    return {
        "prices": [" ".join(p).strip() for p in prices] if prices else [],
        "roles": list(dict.fromkeys([r.title() for r in roles])) if roles else [],
        "dates": dates or [],
    }

def split_into_semantic_chunks(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    for bad in soup.find_all(["nav", "footer", "aside", "script", "style", "noscript"]):
        bad.decompose()

    roots = soup.select("main") or [soup.body or soup]
    chunks: List[Dict[str, Any]] = []

    for root in roots:
        blocks = root.find_all(["section", "article", "div", "li", "tbody", "table"], recursive=True) or [root]
        for blk in blocks:
            txt = text_of(blk)
            if len(txt) < 120:
                continue
            h = None
            for cand in itertools.chain(
                blk.find_all(re.compile(r"^h[1-6]$"), recursive=True),
                [blk.find_previous(re.compile(r"^h[1-6]$"))]
            ):
                if cand:
                    h = cand
                    break
            h_txt  = h.get_text(" ", strip=True) if h else ""
            h_path = heading_path(h)
            kind   = classify_kind(h_txt, txt[:400])
            chunks.append({
                "heading": h_txt,
                "h_path": h_path,
                "text": txt,
                "kind": kind,
                "signals": extract_signals(txt),
            })

    seen, uniq = set(), []
    for c in chunks:
        hsh = xxhash.xxh64(c["text"]).hexdigest()
        if hsh in seen:
            continue
        seen.add(hsh)
        c["text_hash"] = hsh
        uniq.append(c)
    return uniq

# -----------------------  Chunk by Sentance --------------------------------------------
def chunk_by_sentence(text: str) -> List[Dict[str, Any]]:
    """A simple chunker that splits text by sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    for sentence in sentences:
        # Skip empty strings that can result from the split
        if not sentence.strip():
            continue
            
        # We create a dictionary structure similar to the old chunker for compatibility
        chunk_data = {
            "text": sentence.strip(),
            "text_hash": xxhash.xxh64(sentence.strip()).hexdigest()
        }
        chunks.append(chunk_data)
    return chunks
# ----------------------- End of Chunk by Sentance --------------------------------------

# ------------------------- Embeddings (sub-batches + progress) -------------------------
@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=1, max=10))
def _embed_once(texts: List[str]) -> List[List[float]]:
    r = SESSION.post(
        f"{OLLAMA_HOST}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=EMBED_TIMEOUT
    )
    r.raise_for_status()
    js = r.json()
    if "embeddings" in js:
        return js["embeddings"]
    if "data" in js:
        return [d["embedding"] for d in js["data"]]
    raise RuntimeError("Unexpected embed response from Ollama /api/embed")

def embed_batch(texts: List[str]) -> List[List[float]]:
    vecs: List[List[float]] = []
    if not texts:
        return vecs
    total = len(texts)
    for i in tqdm(range(0, total, EMBED_BATCH_SIZE), desc="  Embedding chunks", unit="chunk", leave=False):
        sub = texts[i:i+EMBED_BATCH_SIZE]
        vecs.extend(_embed_once(sub))
    return vecs

# ------------------------- Qdrant helpers -------------------------
"""def ensure_collection(client: QdrantClient, dim: int) -> None:
    names = [c.name for c in client.get_collections().collections]
    if COLLECTION not in names:
        client.recreate_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            optimizers_config=OptimizersConfigDiff(memmap_threshold=20000, indexing_threshold=20000),
        )
        client.create_payload_index(COLLECTION, "kind",   field_schema=PayloadSchemaType.Keyword)
        client.create_payload_index(COLLECTION, "h_path", field_schema=TextIndexParams(type=TextIndexType.Text))
        client.create_payload_index(COLLECTION, "url",    field_schema=PayloadSchemaType.Keyword)
        # optional indexes for resume fields
        try:
            client.create_payload_index(COLLECTION, "html_hash", field_schema=PayloadSchemaType.Keyword)
            client.create_payload_index(COLLECTION, "doc_hash",  field_schema=PayloadSchemaType.Keyword)
        except Exception:
            pass
"""

# ------------------------- Qdrant helpers -------------------------
def ensure_collection(client: QdrantClient, dim: int) -> None:
    # This function is updated for modern qdrant-client versions
    collection_names = [c.name for c in client.get_collections().collections]
    if COLLECTION not in collection_names:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        # The schema types are now lowercase strings
        client.create_payload_index(COLLECTION, field_name="kind", field_schema="keyword")
        client.create_payload_index(COLLECTION, field_name="url",  field_schema="keyword")
        try:
            client.create_payload_index(COLLECTION, field_name="html_hash", field_schema="keyword")
            client.create_payload_index(COLLECTION, field_name="doc_hash",  field_schema="keyword")
        except Exception:
            pass

def already_indexed(client: QdrantClient, url: str, field: str, value: str) -> bool:
    try:
        cnt = client.count(
            COLLECTION,
            filter=Filter(must=[
                FieldCondition(key="url",  match=MatchValue(value=url)),
                FieldCondition(key=field,  match=MatchValue(value=value)),
            ]),
            exact=True
        )
        return (cnt.count or 0) > 0
    except Exception:
        # collection may not exist yet, or other transient issue
        return False

# ------------------------- Crawl -------------------------
def crawl(start: str, seed_from_sitemap: Optional[List[str]] = None) -> List[str]:
    if SITEMAP_STRICT and seed_from_sitemap:
        return sorted(set(seed_from_sitemap))

    q = [normalize_url(start)]
    seen, out = set(), set(seed_from_sitemap or [])
    if seed_from_sitemap:
        q = list(set(q + seed_from_sitemap))

    while q:
        u = q.pop(0)
        if u in seen:
            continue
        seen.add(u)

        if not allowed_path(u):
            if not (seed_from_sitemap and u in seed_from_sitemap):
                continue

        try:
            r = SESSION.get(u, timeout=HTTP_TIMEOUT)
            if not r.ok:
                continue
            ct = (r.headers.get("content-type", "") or "").lower()

            if u.lower().endswith(".pdf") or "application/pdf" in ct:
                if pdf_extract_text:
                    out.add(u)
                continue

            if u.lower().endswith(".ics") or "text/calendar" in ct:
                out.add(u)
                continue

            if "text/html" in ct:
                out.add(u)
                soup = BeautifulSoup(r.text, "lxml")
                for a in soup.find_all("a", href=True):
                    uu = normalize_url(urllib.parse.urljoin(u, a["href"]))
                    if same_site(uu) and "#" not in uu and uu not in seen and allowed_path(uu):
                        q.append(uu)
        except Exception:
            continue
    return sorted(out)

# ------------------------- Indexing loop with progress + resume -------------------------
def index_pages(pages: List[str]) -> None:
    client = QdrantClient(QDRANT_URL)
    points: List[PointStruct] = []
    total_chunks = 0
    first_dim: Optional[int] = None

    for idx, url in enumerate(tqdm(pages, desc="Indexing pages", unit="page"), 1):
        t0 = time.time()
        try:
            r = SESSION.get(url, timeout=HTTP_TIMEOUT)
            if not r.ok:
                print(f"[SKIP] {url} http={r.status_code}")
                continue
            ct = (r.headers.get("content-type", "") or "").lower()

            chunks: List[Dict[str, Any]] = []
            title = url.rsplit("/", 1)[-1] or url

            # PDF
            if url.lower().endswith(".pdf") or "application/pdf" in ct:
                if not pdf_extract_text:
                    print(f"[PDF] skip (no extractor) {url}")
                else:
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as tmp:
                        tmp.write(r.content)
                        tmp.flush()
                        pdf_txt = pdf_extract_text(tmp.name) or ""
                    if pdf_txt.strip():
                        doc_hash = xxhash.xxh64(pdf_txt[:100000]).hexdigest()
                        if already_indexed(client, url, "doc_hash", doc_hash):
                            print(f"[SKIP] {url} already indexed (doc_hash)")
                            continue
                        chunks = [{
                            "heading": title,
                            "h_path": title,
                            "text": pdf_txt[:100000],
                            "kind": "document",
                            "signals": extract_signals(pdf_txt),
                            "text_hash": xxhash.xxh64(pdf_txt[:100000]).hexdigest(),
                            "doc_hash": doc_hash,
                        }]
                    else:
                        print(f"[PDF] empty text {url}")

            # ICS
            elif url.lower().endswith(".ics") or "text/calendar" in ct:
                ics_txt = r.text
                doc_hash = xxhash.xxh64(ics_txt[:100000]).hexdigest()
                if already_indexed(client, url, "doc_hash", doc_hash):
                    print(f"[SKIP] {url} already indexed (doc_hash)")
                    continue
                chunks = [{
                    "heading": title,
                    "h_path": title,
                    "text": ics_txt[:100000],
                    "kind": "calendar",
                    "signals": extract_signals(ics_txt),
                    "text_hash": xxhash.xxh64(ics_txt[:100000]).hexdigest(),
                    "doc_hash": doc_hash,
                }]

            # HTML
            elif "text/html" in ct:
                soup = BeautifulSoup(r.text, "lxml")
                title = (soup.title.get_text(" ", strip=True) if soup.title else url)
                chunks = split_into_semantic_chunks(soup)
                page_hash = xxhash.xxh64(r.text).hexdigest()
                # SKIP if exact same HTML already indexed
                if already_indexed(client, url, "html_hash", page_hash):
                    print(f"[SKIP] {url} already indexed (html_hash)")
                    continue
                for c in chunks:
                    c["html_hash"] = page_hash

            else:
                print(f"[SKIP] {url} content-type={ct}")
                continue

            found = len(chunks)
            if found == 0:
                print(f"[PARSE] {url} → 0 chunks (took {time.time()-t0:.1f}s)")
                continue

            texts = [c["text"] for c in chunks]
            vecs = embed_batch(texts)
            if not vecs:
                print(f"[EMBED] no vectors for {url}")
                continue

            if first_dim is None:
                first_dim = len(vecs[0])
                ensure_collection(client, first_dim)

            for c, v in zip(chunks, vecs):
                pid = xxhash.xxh64(f"{url}::{c['text_hash']}").intdigest()
                payload = {
                    "url": url,
                    "title": title,
                    "heading": c.get("heading", ""),
                    "h_path": c.get("h_path", ""),
                    "kind": c.get("kind", "generic"),
                    "signals": c.get("signals", {}),
                    "text": c["text"],
                    "ts": int(time.time()),
                }
                if "html_hash" in c:
                    payload["html_hash"] = c["html_hash"]
                if "doc_hash" in c:
                    payload["doc_hash"] = c["doc_hash"]
                points.append(PointStruct(id=pid, vector=v, payload=payload))

            total_chunks += found

            if len(points) >= BATCH_SIZE:
                QdrantClient(QDRANT_URL).upsert(COLLECTION, points=points)
                print(f"[UPSERT] flushed {len(points)} points (total so far: {total_chunks})")
                points.clear()

            print(f"[PAGE] {url} → chunks={found} time={time.time()-t0:.1f}s")

        except Exception as e:
            print(f"[WARN] {url} {e}")
            continue

    if points:
        QdrantClient(QDRANT_URL).upsert(COLLECTION, points=points)
        print(f"[UPSERT] final flush {len(points)} points")

    if first_dim is None:
        print("[DONE] No content indexed.")
        return

    #client.create_alias(collection_name=COLLECTION, alias_name=ALIAS, force=True)
    print(f"[DONE] Indexed {total_chunks} chunks across {len(pages)} pages → collection={COLLECTION}, alias={ALIAS}")
"""
# ------------------------- Main -------------------------
def main() -> None:
    print(f"[BOOT] START_URL={START_URL} | SITEMAP_URL={SITEMAP_URL} | STRICT={SITEMAP_STRICT}")
    check_qdrant()
    check_ollama()
    sitemap_urls = iter_sitemap_links()
    if sitemap_urls:
        print(f"[BOOT] Sitemap URLs discovered: {len(sitemap_urls)}")
    pages = crawl(START_URL, seed_from_sitemap=sitemap_urls) if sitemap_urls else crawl(START_URL)
    print(f"[SCRAPER] pages to fetch: {len(pages)} (seeded from sitemap: {len(sitemap_urls) if sitemap_urls else 0})")
    index_pages(pages)

if __name__ == "__main__":
    main()
"""

# ------------------------- Main -------------------------
def main() -> None:
    print("[BOOT] Running in TARGETED scrape mode.")
    check_qdrant()
    check_ollama()

    # Define the specific pages you want to index from your screenshot
    target_pages = [
        "https://isc2chapter-toronto.ca/",
#        "https://isc2chapter-toronto.ca/blogs/",
#        "https://isc2chapter-toronto.ca/media/",
        "https://isc2chapter-toronto.ca/events/",
#        "https://isc2chapter-toronto.ca/partnerships-and-sponsorships/",
        "https://isc2chapter-toronto.ca/leadership/",
        "https://isc2chapter-toronto.ca/membership/",
    ]

    print(f"[SCRAPER] pages to fetch: {len(target_pages)}")
    index_pages(target_pages)

if __name__ == "__main__":
    main()
