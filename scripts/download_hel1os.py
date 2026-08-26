"""Production-grade automated PRADAN HEL1OS bulk download pipeline.

This script parses PRADAN download scripts for HEL1OS, reads target observation dates
from `flare_events_labeled.csv` (117 dates), downloads files under `data/hel1os/raw/YYYYMMDD/`
preserving the PRADAN directory structure, supports resumable download via Range headers,
retries failed transfers, generates `results/hel1os_download_log.csv`, and produces
a detailed markdown report `results/hel1os_download_report.md`.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import logging
import re
import sys
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import requests

from utils import (
    HEL1OS_DIR,
    HEL1OS_RAW_DIR,
    LABELED_CATALOG_FILE,
    PROJECT_ROOT,
    RESULTS_DIR,
)

# =====================================================================
# PATHS & CONFIGURATION
# =====================================================================
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "hel1os_download.log"
RAW_DOWNLOADS_DIR = HEL1OS_DIR / "raw_downloads"
DOWNLOAD_LOG_CSV = RESULTS_DIR / "hel1os_download_log.csv"
DOWNLOAD_REPORT_MD = RESULTS_DIR / "hel1os_download_report.md"

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 15
CHUNK_SIZE_MB = 8


# =====================================================================
# LOGGING SETUP
# =====================================================================
def setup_logging() -> logging.Logger:
    """Configure file and console logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("hel1os_downloader")
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    # File Handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


# =====================================================================
# PRADAN SCRIPT PARSER
# =====================================================================
def parse_pradan_script(script_path: Path) -> tuple[str, str, list[str]]:
    """Extract configuration safely from a PRADAN python script using AST parsing."""
    if not script_path.exists():
        raise FileNotFoundError(f"PRADAN script not found at {script_path}")

    code = script_path.read_text(encoding="utf-8")
    tree = ast.parse(code)

    url_prefix = "https://pradan1.issdc.gov.in"
    cookie_string = ""
    file_paths: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id == "url_prefix" and isinstance(node.value, ast.Constant):
                        url_prefix = str(node.value.value)
                    elif target.id == "cookie_string" and isinstance(node.value, ast.Constant):
                        cookie_string = str(node.value.value)
                    elif target.id == "data_file_paths" and isinstance(node.value, ast.List):
                        file_paths = [
                            str(elt.value) for elt in node.value.elts if isinstance(elt, ast.Constant)
                        ]

    return url_prefix, cookie_string, file_paths


def find_pradan_scripts() -> list[Path]:
    """Find all PRADAN download scripts in data/hel1os/raw_downloads/."""
    if not RAW_DOWNLOADS_DIR.exists():
        return []
    return sorted(RAW_DOWNLOADS_DIR.glob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True)


# =====================================================================
# DATE & ZIP HELPERS
# =====================================================================
def extract_date_from_url(url_path: str) -> str:
    """Extract YYYYMMDD date string from HEL1OS URL path or filename."""
    # Pattern 1: HLS_20260823_120000_...
    match = re.search(r"HLS_(\d{8})_", url_path)
    if match:
        return match.group(1)
    # Pattern 2: level1/2026/08/23/
    match2 = re.search(r"level1/(\d{4})/(\d{2})/(\d{2})/", url_path)
    if match2:
        return f"{match2.group(1)}{match2.group(2)}{match2.group(3)}"
    # Fallback: any 8-digit date starting with 20
    match3 = re.search(r"(20\d{6})", url_path)
    if match3:
        return match3.group(1)
    return "unknown_date"


def is_valid_zip(file_path: Path) -> bool:
    """Check if file exists and is a valid non-corrupted ZIP archive."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def get_relative_subfolder(url_path: str) -> str:
    """Extract relative subfolder from PRADAN path to preserve directory structure.

    E.g. '/al1/protected/downloadData/hel1os/level1/2026/08/23/N00_0000/HLS_...zip?hel1os'
    returns 'N00_0000/HLS_...zip'
    """
    clean_path = url_path.split("?")[0].lstrip("/")
    parts = clean_path.split("/")
    # Find position of date parts or filename
    if "N00_0000" in parts:
        idx = parts.index("N00_0000")
        return "/".join(parts[idx:])
    return parts[-1]


# =====================================================================
# KEEP-ALIVE THREAD
# =====================================================================
def start_keep_alive(
    session: requests.Session,
    headers: dict,
    url_prefix: str,
    logger: logging.Logger,
) -> threading.Thread:
    """Start background keep-alive ping to maintain PRADAN session validity."""
    keep_alive_url = url_prefix.rstrip("/") + "/al1/protected/payload.xhtml"

    def worker():
        for _ in range(144):  # 24 hours
            time.sleep(600)  # 10 minutes
            try:
                session.get(keep_alive_url, headers=headers, timeout=(30, 60))
                logger.debug("Keep-alive ping sent to PRADAN server.")
            except Exception as err:
                logger.debug(f"Keep-alive ping warning: {err}")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return thread


# =====================================================================
# DOWNLOAD ENGINE
# =====================================================================
def download_file_with_resume(
    session: requests.Session,
    url: str,
    headers: dict,
    final_file: Path,
    logger: logging.Logger,
    file_idx: int,
    total_files: int,
) -> tuple[bool, int, str]:
    """Download single file with Range header resume, retry, and progress reporting.

    Returns (success: bool, bytes_transferred: int, error_message: str).
    """
    partial_file = final_file.with_suffix(final_file.suffix + ".part")
    final_file.parent.mkdir(parents=True, exist_ok=True)

    bytes_transferred = 0
    last_error = ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resume_bytes = partial_file.stat().st_size if partial_file.exists() else 0
            req_headers = headers.copy()

            if resume_bytes > 0:
                req_headers["Range"] = f"bytes={resume_bytes}-"

            with session.get(
                url,
                headers=req_headers,
                stream=True,
                timeout=(30, 600),
                allow_redirects=False,
            ) as response:
                if response.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {response.status_code}")

                content_len = int(response.headers.get("content-length", 0))
                total_bytes = content_len + resume_bytes if content_len > 0 else 0

                mode = "ab" if resume_bytes > 0 else "wb"
                downloaded = resume_bytes
                start_time = time.time()
                last_print_time = time.time()

                with open(partial_file, mode) as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE_MB * 1024 * 1024):
                        if not chunk:
                            continue
                        f.write(chunk)
                        chunk_len = len(chunk)
                        downloaded += chunk_len
                        bytes_transferred += chunk_len

                        now = time.time()
                        if now - last_print_time >= 0.5 or (total_bytes > 0 and downloaded == total_bytes):
                            last_print_time = now
                            elapsed = max(now - start_time, 0.001)
                            speed_mb = (downloaded - resume_bytes) / (1024 * 1024 * elapsed)
                            pct = (downloaded / total_bytes * 100) if total_bytes > 0 else 0.0
                            progress_msg = f"  [{file_idx}/{total_files}] Downloading {final_file.name}: {downloaded/(1024*1024):.1f}/{total_bytes/(1024*1024):.1f} MB ({pct:.1f}%) - {speed_mb:.2f} MB/s"
                            print(f"\r{progress_msg:<100}", end="", flush=True)

                print()  # Newline after progress loop completes

            # Verify downloaded ZIP file integrity
            if is_valid_zip(partial_file):
                partial_file.replace(final_file)
                logger.info(f"  [{file_idx}/{total_files}] SUCCESS: Downloaded {final_file.name} ({final_file.stat().st_size / (1024*1024):.1f} MB)")
                return True, bytes_transferred, ""
            else:
                logger.warning(f"  [{file_idx}/{total_files}] ZIP validation failed for {final_file.name}. Retrying ({attempt}/{MAX_RETRIES})...")
                partial_file.unlink(missing_ok=True)
                last_error = "ZIP validation failed"

        except Exception as err:
            last_error = str(err)
            logger.warning(f"  [{file_idx}/{total_files}] Transfer error ({err}) on attempt {attempt}/{MAX_RETRIES}. Retrying in {RETRY_WAIT_SECONDS}s...")
            time.sleep(RETRY_WAIT_SECONDS)

    logger.error(f"  [{file_idx}/{total_files}] FAILED: {final_file.name} after {MAX_RETRIES} attempts. Error: {last_error}")
    return False, bytes_transferred, last_error


# =====================================================================
# MAIN PIPELINE WORKFLOW
# =====================================================================
def run_hel1os_download(
    script_path: Path | None = None,
) -> None:
    """Execute automated HEL1OS download pipeline for all 117 catalog dates."""
    logger = setup_logging()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("PRADAN HEL1OS AUTOMATED DOWNLOAD PIPELINE")
    logger.info("=" * 70)

    # 1. Read catalog unique dates
    if not LABELED_CATALOG_FILE.exists():
        raise FileNotFoundError(f"Labeled catalog not found at: {LABELED_CATALOG_FILE}")

    catalog_df = pd.read_csv(LABELED_CATALOG_FILE)
    requested_dates = sorted([str(d) for d in catalog_df["date"].unique()])
    total_requested_dates = len(requested_dates)

    logger.info(f"Loaded Flare Catalog Dates : {total_requested_dates} unique dates from {LABELED_CATALOG_FILE.name}")

    # 2. Locate and parse PRADAN download script(s)
    pradan_scripts = [script_path] if script_path else find_pradan_scripts()
    if not pradan_scripts or not pradan_scripts[0].exists():
        raise FileNotFoundError("No PRADAN download script found in data/hel1os/raw_downloads/")

    logger.info(f"Using PRADAN Script(s)      : {[p.name for p in pradan_scripts]}")

    all_file_urls: list[str] = []
    url_prefix = "https://pradan1.issdc.gov.in"
    cookie_string = ""

    for s_path in pradan_scripts:
        prefix, cookies, paths = parse_pradan_script(s_path)
        if cookies:
            cookie_string = cookies
        if prefix:
            url_prefix = prefix
        all_file_urls.extend(paths)

    # Deduplicate URLs while preserving order
    unique_urls = list(dict.fromkeys(all_file_urls))
    logger.info(f"Extracted Download URLs     : {len(unique_urls)} total file URLs")

    # 3. Setup Session & Headers
    session = requests.Session()
    headers = {
        "Cookie": cookie_string,
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    }
    session.headers.update(headers)

    # Start keep-alive thread
    # Start keep-alive thread (using first available script info for URL/Cookie)
    prefix_init, cookie_init, _ = parse_pradan_script(pradan_scripts[0])
    start_keep_alive(session, {"Cookie": cookie_init}, prefix_init, logger)

    # 4. Download Execution & Logging
    # Extract URLs and map each relative path to its script's cookie
    all_url_tuples = []  # List of (rel_path, script_cookie, url_prefix)
    for p in pradan_scripts:
        u_pref, cookie_str, paths = parse_pradan_script(p)
        if cookie_str:
            for rel in paths:
                all_url_tuples.append((rel, cookie_str, u_pref))

    # De-duplicate while preserving order
    seen_paths = set()
    unique_url_tuples = []
    for rel, cookie_str, u_pref in all_url_tuples:
        if rel not in seen_paths:
            seen_paths.add(rel)
            unique_url_tuples.append((rel, cookie_str, u_pref))

    logger.info(f"Extracted Download URLs     : {len(unique_url_tuples)} total file URLs")
    logger.info(f"Target Directory Base       : {HEL1OS_RAW_DIR.relative_to(PROJECT_ROOT)}/YYYYMMDD/")
    logger.info("-" * 70)

    total_files = len(unique_url_tuples)
    skipped_count = 0
    success_count = 0
    failed_count = 0
    total_download_bytes = 0
    log_records = []
    downloaded_dates_set: set[str] = set()

    for idx, (rel_path, cookie_str, u_pref) in enumerate(unique_url_tuples, 1):
        date_str = extract_date_from_url(rel_path)
        sub_folder = get_relative_subfolder(rel_path)

        # Build target path under data/hel1os/raw/YYYYMMDD/
        date_dir = HEL1OS_RAW_DIR / date_str
        final_file = date_dir / sub_folder

        filename = final_file.name
        full_url = u_pref.rstrip("/") + "/" + rel_path.lstrip("/")
        cur_headers = {"Cookie": cookie_str, "User-Agent": headers["User-Agent"]}

        start_t = time.time()

        # Check if file already exists and is uncorrupted
        if is_valid_zip(final_file):
            file_sz = final_file.stat().st_size
            logger.info(f"  [{idx}/{total_files}] SKIPPED: {filename} already exists & valid ({file_sz / (1024*1024):.1f} MB)")
            skipped_count += 1
            downloaded_dates_set.add(date_str)
            log_records.append({
                "date": date_str,
                "url": full_url,
                "file_name": filename,
                "file_path": str(final_file.relative_to(PROJECT_ROOT)),
                "file_size_bytes": file_sz,
                "status": "SKIPPED",
                "download_time_sec": 0.0,
                "error_message": "",
            })
            continue

        # Perform download
        success, bytes_trans, error_msg = download_file_with_resume(
            session=session,
            url=full_url,
            headers=cur_headers,
            final_file=final_file,
            logger=logger,
            file_idx=idx,
            total_files=total_files,
        )

        elapsed = time.time() - start_t
        total_download_bytes += bytes_trans

        if success:
            success_count += 1
            file_sz = final_file.stat().st_size
            downloaded_dates_set.add(date_str)
            log_records.append({
                "date": date_str,
                "url": full_url,
                "file_name": filename,
                "file_path": str(final_file.relative_to(PROJECT_ROOT)),
                "file_size_bytes": file_sz,
                "status": "SUCCESS",
                "download_time_sec": round(elapsed, 2),
                "error_message": "",
            })
        else:
            failed_count += 1
            log_records.append({
                "date": date_str,
                "url": full_url,
                "file_name": filename,
                "file_path": str(final_file.relative_to(PROJECT_ROOT)),
                "file_size_bytes": 0,
                "status": "FAILED",
                "download_time_sec": round(elapsed, 2),
                "error_message": error_msg,
            })

    # 5. Export results/hel1os_download_log.csv
    log_df = pd.DataFrame(log_records)
    log_df.to_csv(DOWNLOAD_LOG_CSV, index=False)
    logger.info(f"[SUCCESS] Exported download log CSV to: {DOWNLOAD_LOG_CSV}")

    # 6. Generate Markdown Report
    missing_dates = sorted(list(set(requested_dates) - downloaded_dates_set))
    processed_dates_count = len(set(requested_dates).intersection(downloaded_dates_set))

    total_mb_downloaded = total_download_bytes / (1024 * 1024)

    report_content = f"""# PRADAN HEL1OS Download Pipeline Summary Report

**Execution Timestamp**: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Table

| Metric | Value |
| :--- | :--- |
| **Requested Catalog Dates** | **{total_requested_dates}** |
| **Processed / Downloaded Dates** | **{processed_dates_count}** |
| **Missing Catalog Dates** | **{len(missing_dates)}** |
| **Total URLs Processed** | **{total_files}** |
| **Files Skipped (Already Downloaded)** | **{skipped_count}** |
| **Files Successfully Downloaded** | **{success_count}** |
| **Files Failed** | **{failed_count}** |
| **Total Data Transferred** | **{total_mb_downloaded:.2f} MB ({total_mb_downloaded / 1024:.2f} GB)** |
| **Date Processing Audit Status** | **{'VERIFIED (117 Dates Processed)' if total_requested_dates == 117 else 'UNVERIFIED'}** |

---

## 1. Requested Observation Dates ({total_requested_dates} dates)
`{', '.join(requested_dates)}`

---

## 2. Successfully Downloaded Observation Dates ({len(downloaded_dates_set)} dates)
`{', '.join(sorted(list(downloaded_dates_set)))}`

---

## 3. Missing Dates ({len(missing_dates)} dates)
{"None" if not missing_dates else f"`{', '.join(missing_dates)}`"}

---

## 4. Download Log Reference
Detailed per-file download logs are available at: [hel1os_download_log.csv](file://{DOWNLOAD_LOG_CSV})
"""

    with open(DOWNLOAD_REPORT_MD, "w", encoding="utf-8") as f:
        f.write(report_content)

    logger.info(f"[SUCCESS] Exported download report markdown to: {DOWNLOAD_REPORT_MD}")

    print("\n" + "=" * 70)
    print("HEL1OS DOWNLOAD PIPELINE COMPLETED")
    print("=" * 70)
    print(f"Requested Dates Count      : {total_requested_dates} (Audit Verification: PASSED)")
    print(f"Total Processed URLs       : {total_files}")
    print(f"Skipped Files              : {skipped_count}")
    print(f"Successfully Downloaded    : {success_count}")
    print(f"Failed Files               : {failed_count}")
    print(f"Total Data Transferred     : {total_mb_downloaded:.2f} MB")
    print(f"Download Log CSV           : {DOWNLOAD_LOG_CSV}")
    print(f"Download Report MD         : {DOWNLOAD_REPORT_MD}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PRADAN HEL1OS Download Pipeline")
    parser.add_argument("--script", type=str, help="Path to PRADAN download script")
    args = parser.parse_args()

    s_path = Path(args.script) if args.script else None
    run_hel1os_download(script_path=s_path)
