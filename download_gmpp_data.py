"""
01_download_gmpp_data.py  (v2 — discovery-based)
================================================
Downloads UK Government Major Projects Portfolio (GMPP) data 2013-2024 (per-
department files) plus the consolidated 2024-25 NISTA file.

Rather than guess URL patterns (they change every year — departmental renames,
"for X" vs "X-" slugs, MPA-era vs IPA-era publications), this version scrapes
gov.uk's master collection page to discover every publication URL, then visits
each one to find its data file.

Run once. Outputs land in ./data/raw/{year}/.

Source: gov.uk transparency data, Open Government Licence v3.0
Index: https://www.gov.uk/government/collections/major-projects-data
"""

import re
import time
import logging
from pathlib import Path
from urllib.parse import urljoin
import requests
from bs4 import BeautifulSoup

COLLECTION_URL = "https://www.gov.uk/government/collections/major-projects-data"
OUT_DIR = Path("./data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_DELAY_SEC = 0.5
USER_AGENT = "GMPP-research-script/2.0 (educational use)"

PUB_URL_PATTERN = re.compile(
    r"/government/publications/[\w-]*major-projects-portfolio-data[\w-]*",
    re.IGNORECASE,
)
NISTA_URL_PATTERN = re.compile(
    r"/government/publications/nista-annual-report-\d{4}-\d{4}",
    re.IGNORECASE,
)
YEAR_PATTERN = re.compile(r"(\d{4})(?:-\d{4})?$")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def fetch(url):
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
        time.sleep(REQUEST_DELAY_SEC)
        if resp.status_code == 200:
            return resp.content
        log.warning(f"HTTP {resp.status_code} on {url}")
    except requests.RequestException as e:
        log.warning(f"request failed: {url} — {e}")
    return None


def discover_publication_urls():
    log.info(f"Scraping collection index: {COLLECTION_URL}")
    html = fetch(COLLECTION_URL)
    if html is None:
        log.error("Failed to fetch collection index — cannot proceed")
        return []
    soup = BeautifulSoup(html, "html.parser")
    urls = set()
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href.startswith("/government/publications/"):
            continue
        if PUB_URL_PATTERN.search(href) or NISTA_URL_PATTERN.search(href):
            urls.add(urljoin("https://www.gov.uk", href))
    log.info(f"Discovered {len(urls)} GMPP publication URLs")
    return sorted(urls)


def discover_data_file_url(publication_html):
    soup = BeautifulSoup(publication_html, "html.parser")
    for ext in (".csv", ".xlsx", ".xls", ".ods"):
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.endswith(ext) and "assets.publishing.service.gov.uk" in href:
                return href
    return None


def extract_year_and_dept(publication_url):
    slug = publication_url.rstrip("/").rsplit("/", 1)[-1]
    if slug.startswith("nista-annual-report"):
        m = re.search(r"(\d{4})-(\d{4})", slug)
        if m:
            return m.group(2), "nista-consolidated"
    m = YEAR_PATTERN.search(slug)
    year = m.group(1) if m else "unknown"
    label = slug
    label = re.sub(r"-\d{4}$", "", label)
    label = re.sub(r"government-major-projects-portfolio-data", "", label)
    label = re.sub(r"-for-", "-", label)
    label = label.strip("-")
    if not label:
        label = "unknown"
    return year, label


def save(content, year, label, ext):
    year_dir = OUT_DIR / year
    year_dir.mkdir(parents=True, exist_ok=True)
    fp = year_dir / f"{label}_{year}.{ext}"
    fp.write_bytes(content)
    return fp


def main():
    pub_urls = discover_publication_urls()
    if not pub_urls:
        return
    saved = 0
    skipped = 0
    for pub_url in pub_urls:
        year, label = extract_year_and_dept(pub_url)
        log.info(f"[{year}] {label}: visiting publication page")
        page_html = fetch(pub_url)
        if page_html is None:
            skipped += 1
            continue
        data_url = discover_data_file_url(page_html)
        if data_url is None:
            log.info(f"  no data file link found on {pub_url}")
            skipped += 1
            continue
        content = fetch(data_url)
        if content is None:
            skipped += 1
            continue
        ext = data_url.rsplit(".", 1)[-1]
        fp = save(content, year, label, ext)
        log.info(f"  saved {len(content):,} bytes → {fp}")
        saved += 1
    log.info(f"DONE — saved {saved} files, skipped {skipped}")


if __name__ == "__main__":
    main()
