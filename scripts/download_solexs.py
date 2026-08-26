"""Production-grade automated PRADAN SoLEXS bulk download & ingestion pipeline."""

from __future__ import annotations

import argparse
import ast
import datetime
import logging
import re
import shutil
import signal
import sys
import threading
import time
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import requests

from extract_files import extract_all
from utils import PROJECT_ROOT, SOLEXS_EXTRACTED_DIR, SOLEXS_RAW_DIR, ensure_output_directories

# =====================================================================
# PATHS & CONFIGURATION
# =====================================================================
LOG_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOG_DIR / "download.log"
RAW_DOWNLOADS_DIR = PROJECT_ROOT / "data" / "solexs" / "raw_downloads"

MAX_RETRIES = 5
RETRY_WAIT_SECONDS = 15
CHUNK_SIZE_MB = 8


# =====================================================================
# LOGGING SETUP
# =====================================================================
def setup_logging() -> logging.Logger:
    """Configure file and console logging."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("solexs_downloader")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if re-initialized
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
    """Safely extract configuration from a PRADAN python script using AST parsing."""
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


def find_latest_pradan_script() -> Path | None:
    """Find the most recent PRADAN download script in raw_downloads/."""
    if not RAW_DOWNLOADS_DIR.exists():
        return None
    scripts = sorted(RAW_DOWNLOADS_DIR.glob("*.py"), key=lambda p: p.stat().st_mtime, reverse=True)
    return scripts[0] if scripts else None


# =====================================================================
# ZIP & DATA HELPERS
# =====================================================================
def extract_date_from_path(path_str: str) -> str:
    """Extract YYYYMMDD date string from filename or path."""
    match = re.search(r"(20\d{6})", path_str)
    if match:
        return match.group(1)
    return "unknown_date"


def is_valid_zip(file_path: Path) -> bool:
    """Check if file exists and is a valid, uncorrupted ZIP archive."""
    if not file_path.exists() or file_path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(file_path, "r") as zf:
            return zf.testzip() is None
    except (zipfile.BadZipFile, OSError):
        return False


def extract_zip_contents(zip_path: Path, target_dir: Path, logger: logging.Logger) -> list[Path]:
    """Extract files from ZIP into target_dir."""
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                if member.is_dir():
                    continue
                extracted_path = zf.extract(member, path=target_dir)
                extracted.append(Path(extracted_path))
        logger.info(f"  [EXTRACTED] Unzipped {len(extracted)} file(s) into {target_dir.relative_to(PROJECT_ROOT)}")
    except Exception as e:
        logger.error(f"  [ERROR] Unzipping {zip_path.name} failed: {e}")
    return extracted


# =====================================================================
# KEEP-ALIVE THREAD
# =====================================================================
def start_keep_alive(session: requests.Session, headers: dict, url_prefix: str, logger: logging.Logger) -> threading.Thread:
    """Start background keep-alive ping to prevent session timeout."""
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
) -> tuple[bool, int]:
    """Download single file with Range header resume and progress feedback."""
    partial_file = final_file.with_suffix(final_file.suffix + ".part")
    final_file.parent.mkdir(parents=True, exist_ok=True)

    bytes_transferred = 0

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
                            speed_mb = ((downloaded - resume_bytes) / (1024 * 1024)) / elapsed
                            pct = (downloaded / total_bytes * 100) if total_bytes > 0 else 0.0
                            overall_pct = (file_idx / total_files) * 100

                            status_str = (
                                f"\r[{file_idx}/{total_files}] ({overall_pct:5.1f}%) "
                                f"{final_file.name}: {pct:5.1f}% "
                                f"({downloaded / (1024**2):.2f}/{total_bytes / (1024**2):.2f} MB) "
                                f"[{speed_mb:.2f} MB/s]"
                            )
                            sys.stdout.write(status_str)
                            sys.stdout.flush()

            # Complete file download cleanly
            sys.stdout.write("\n")
            partial_file.rename(final_file)
            size_mb = final_file.stat().st_size / (1024**2)
            logger.info(f"  [COMPLETED] Downloaded {final_file.name} ({size_mb:.2f} MB)")
            return True, bytes_transferred

        except Exception as err:
            sys.stdout.write("\n")
            logger.warning(f"  [RETRY {attempt}/{MAX_RETRIES}] {final_file.name} failed: {err}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT_SECONDS)

    logger.error(f"  [FAILED] Download failed for {final_file.name} after {MAX_RETRIES} attempts.")
    return False, bytes_transferred


# =====================================================================
# PIPELINE AUTOMATION MANAGER
# =====================================================================
def run_ingestion_pipeline(
    script_path: Path | None = None,
    auto_extract_zip: bool = True,
    auto_decompress_gz: bool = True,
) -> None:
    """Run full automated PRADAN download and ingestion pipeline."""
    ensure_output_directories()
    logger = setup_logging()

    logger.info("=" * 60)
    logger.info("PRADAN SOLEXS AUTOMATED INGESTION PIPELINE")
    logger.info("=" * 60)

    # 1. Locate and parse PRADAN script
    if script_path is None:
        script_path = find_latest_pradan_script()

    if script_path is None or not script_path.exists():
        logger.error("No PRADAN script found! Place a download script in data/solexs/raw_downloads/")
        sys.exit(1)

    logger.info(f"Using PRADAN Script : {script_path.resolve().relative_to(PROJECT_ROOT.resolve())}")

    url_prefix, cookie_string, data_file_paths = parse_pradan_script(script_path)

    if not data_file_paths:
        logger.error(f"No data file paths found in {script_path}")
        sys.exit(1)

    if not cookie_string:
        logger.warning("Cookie string is empty! Protected downloads may fail.")

    logger.info(f"Total Files in Queue: {len(data_file_paths)}")
    logger.info(f"Target Raw Directory: {SOLEXS_RAW_DIR.resolve().relative_to(PROJECT_ROOT.resolve())}")
    logger.info("-" * 60)

    # 2. Setup requests session & headers
    session = requests.Session()
    headers = {"Cookie": cookie_string}

    start_keep_alive(session, headers, url_prefix, logger)

    # Counters for summary report
    completed_count = 0
    skipped_count = 0
    failed_count = 0
    total_bytes_downloaded = 0
    extracted_gz_files = 0

    # 3. Main Download & Ingestion Loop
    for idx, raw_path in enumerate(data_file_paths, start=1):
        clean_url_path = raw_path.split("?")[0]
        zip_filename = Path(clean_url_path).name
        date_folder = extract_date_from_path(zip_filename)

        # Build output path: data/solexs/raw/YYYYMMDD/AL1_SLX_L1_YYYYMMDD_v1.0.zip
        target_dir = SOLEXS_RAW_DIR / date_folder
        final_zip_path = target_dir / zip_filename
        url = url_prefix.rstrip("/") + "/" + raw_path.lstrip("/")

        overall_pct = (idx / len(data_file_paths)) * 100
        logger.info(f"\n[{idx}/{len(data_file_paths)}] ({overall_pct:.1f}%) Processing Date: {date_folder} | File: {zip_filename}")

        # Check if ZIP already exists and is valid
        if is_valid_zip(final_zip_path):
            logger.info("  [SKIPPED] Already downloaded and verified valid ZIP.")
            skipped_count += 1

            if auto_extract_zip:
                extracted = extract_zip_contents(final_zip_path, target_dir, logger)
                extracted_gz_files += len(extracted)
            continue

        # Perform download
        success, nbytes = download_file_with_resume(
            session=session,
            url=url,
            headers=headers,
            final_file=final_zip_path,
            logger=logger,
            file_idx=idx,
            total_files=len(data_file_paths),
        )

        total_bytes_downloaded += nbytes

        if success:
            completed_count += 1
            if auto_extract_zip and is_valid_zip(final_zip_path):
                extracted = extract_zip_contents(final_zip_path, target_dir, logger)
                extracted_gz_files += len(extracted)
        else:
            failed_count += 1

    # 4. Optional end-to-end decompression of .gz files into extracted/
    if auto_decompress_gz:
        logger.info("\n" + "-" * 60)
        logger.info("Decompressing .gz files into data/solexs/extracted/...")
        written_lc, skipped_lc = extract_all()
        logger.info(f"Decompression Summary: {written_lc} extracted, {skipped_lc} skipped.")

    # 5. Printable Summary Report
    logger.info("\n" + "=" * 60)
    logger.info("PRADAN SOLEXS INGESTION PIPELINE SUMMARY REPORT")
    logger.info("=" * 60)
    logger.info(f"Total Files Processed : {len(data_file_paths)}")
    logger.info(f"Downloads Completed   : {completed_count}")
    logger.info(f"Downloads Skipped     : {skipped_count}")
    logger.info(f"Downloads Failed      : {failed_count}")
    logger.info(f"Total Downloaded Size : {total_bytes_downloaded / (1024**2):.2f} MB")
    logger.info(f"Extracted Files Count : {extracted_gz_files} in {SOLEXS_RAW_DIR.resolve().relative_to(PROJECT_ROOT.resolve())}")
    logger.info(f"Log File Location     : {LOG_FILE.resolve().relative_to(PROJECT_ROOT.resolve())}")
    logger.info("=" * 60)


# =====================================================================
# CLI ENTRY POINT
# =====================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Automated PRADAN SoLEXS Download & Ingestion Pipeline")
    parser.add_argument("--script", type=str, help="Path to PRADAN python script (e.g. raw_downloads/solexs_xxx.py)")
    parser.add_argument("--no-extract-zip", action="store_true", help="Disable automatic unzipping of raw ZIP files")
    parser.add_argument("--no-decompress-gz", action="store_true", help="Disable automatic gzip decompression to extracted/")

    args = parser.parse_args()
    script_path = Path(args.script) if args.script else None

    run_ingestion_pipeline(
        script_path=script_path,
        auto_extract_zip=not args.no_extract_zip,
        auto_decompress_gz=not args.no_decompress_gz,
    )


if __name__ == "__main__":
    main()
