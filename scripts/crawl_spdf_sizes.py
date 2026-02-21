#!/usr/bin/env python3
"""
Crawl the SPDF archive to compute the total size of all science data files.

Usage:
    python scripts/crawl_spdf_sizes.py
    python scripts/crawl_spdf_sizes.py --workers 20
    python scripts/crawl_spdf_sizes.py --output spdf_sizes.json

Crawls https://spdf.gsfc.nasa.gov/pub/data/ recursively, parsing Apache
directory listings to find science data files (.cdf, .nc, .fits, .hdf5, etc.)
and sum their sizes. Shows live progress as it goes.
"""

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from html.parser import HTMLParser
from urllib.parse import urljoin

import requests

BASE_URL = "https://spdf.gsfc.nasa.gov/pub/data/"

# Science data file extensions to count
DATA_EXTENSIONS = {
    # CDF / NetCDF
    ".cdf", ".nc",
    # FITS
    ".fits", ".fit", ".fts",
    # HDF
    ".hdf5", ".hdf", ".h5",
    # IDL save files
    ".sav",
    # Plain-text / tabular data
    ".dat", ".csv", ".txt", ".asc", ".tab",
    # VOTable / XML science data
    ".xml", ".vot",
}

# Regex to parse Apache HTML table directory listing rows like:
# <tr><td><a href="file.cdf">file.cdf</a></td><td align="right">2024-01-15 10:30  </td><td align="right">1.2M</td></tr>
# Size field can be: "1.2K", "34M", "5.6G", "  - " (directory)
ROW_PATTERN = re.compile(
    r'<tr><td><a\s+href="([^"]+)">[^<]+</a></td>'  # href
    r'<td[^>]*>[^<]*</td>'                           # date column
    r'<td[^>]*>\s*([\d.]+[KMG]?|-)\s*</td></tr>',   # size column
)


def parse_size(size_str: str) -> int:
    """Convert Apache size string like '1.2M' to bytes."""
    if size_str == "-":
        return 0
    multipliers = {"K": 1024, "M": 1024**2, "G": 1024**3}
    if size_str[-1] in multipliers:
        return int(float(size_str[:-1]) * multipliers[size_str[-1]])
    return int(float(size_str))


class Progress:
    """Thread-safe progress tracker with live terminal output."""

    def __init__(self, output_path=None):
        self.lock = threading.Lock()
        self.dirs_crawled = 0
        self.dirs_queued = 0
        self.data_files = 0
        self.data_bytes = 0
        self.other_files = 0
        self.errors = 0
        self.mission_sizes = defaultdict(int)  # mission -> bytes
        self.mission_counts = defaultdict(int)  # mission -> count
        self.format_sizes = defaultdict(int)    # extension -> bytes
        self.format_counts = defaultdict(int)   # extension -> count
        self.current_dirs = set()
        self.start_time = time.time()
        self._last_print_len = 0
        self._output_path = output_path
        self._last_save_time = 0

    def add_directory(self, count=1):
        with self.lock:
            self.dirs_queued += count

    def finish_directory(self, url):
        with self.lock:
            self.dirs_crawled += 1
            self.current_dirs.discard(url)

    def start_directory(self, url):
        with self.lock:
            self.current_dirs.add(url)

    def add_data_file(self, size_bytes: int, mission: str, ext: str):
        with self.lock:
            self.data_files += 1
            self.data_bytes += size_bytes
            self.mission_sizes[mission] += size_bytes
            self.mission_counts[mission] += 1
            self.format_sizes[ext] += size_bytes
            self.format_counts[ext] += 1

    def add_other(self):
        with self.lock:
            self.other_files += 1

    def add_error(self):
        with self.lock:
            self.errors += 1

    def format_size(self, nbytes: int) -> str:
        for unit in ["B", "KB", "MB", "GB", "TB", "PB"]:
            if nbytes < 1024:
                return f"{nbytes:.2f} {unit}"
            nbytes /= 1024
        return f"{nbytes:.2f} EB"

    def print_status(self):
        with self.lock:
            elapsed = time.time() - self.start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            rate = self.dirs_crawled / elapsed if elapsed > 0 else 0

            # Current directory being crawled (show first one, truncated)
            current = ""
            if self.current_dirs:
                d = next(iter(self.current_dirs))
                # Strip base URL for display
                short = d.replace(BASE_URL, "")
                if len(short) > 50:
                    short = "..." + short[-47:]
                current = f"  crawling: {short}"

            line = (
                f"\r[{elapsed_str}] "
                f"dirs: {self.dirs_crawled}/{self.dirs_queued} ({rate:.1f}/s) | "
                f"data files: {self.data_files:,} ({self.format_size(self.data_bytes)}) | "
                f"errors: {self.errors}"
                f"{current}"
            )
            # Pad to overwrite previous line
            pad = max(0, self._last_print_len - len(line))
            sys.stderr.write(line + " " * pad)
            sys.stderr.flush()
            self._last_print_len = len(line)

            # Auto-save every 30 seconds
            now = time.time()
            if self._output_path and now - self._last_save_time > 30:
                self._last_save_time = now
                self._auto_save()

    def to_dict(self):
        """Build results dict (caller must hold lock or be single-threaded)."""
        return {
            "total_data_files": self.data_files,
            "total_data_bytes": self.data_bytes,
            "total_data_human": self.format_size(self.data_bytes),
            "directories_crawled": self.dirs_crawled,
            "errors": self.errors,
            "formats": {
                ext: {
                    "bytes": self.format_sizes[ext],
                    "human": self.format_size(self.format_sizes[ext]),
                    "files": self.format_counts[ext],
                }
                for ext in sorted(
                    self.format_sizes, key=self.format_sizes.get, reverse=True
                )
            },
            "missions": {
                m: {
                    "bytes": self.mission_sizes[m],
                    "human": self.format_size(self.mission_sizes[m]),
                    "files": self.mission_counts[m],
                }
                for m in sorted(
                    self.mission_sizes, key=self.mission_sizes.get, reverse=True
                )
            },
        }

    def _auto_save(self):
        """Save current results to output file (called under lock)."""
        try:
            result = self.to_dict()
            with open(self._output_path, "w") as f:
                json.dump(result, f, indent=2)
        except Exception:
            pass  # Don't let save errors interrupt crawling

    def spacecraft_snapshot(self) -> str:
        """Return a formatted string of per-spacecraft stats (acquires lock)."""
        with self.lock:
            elapsed = time.time() - self.start_time
            elapsed_str = str(timedelta(seconds=int(elapsed)))
            lines = []
            lines.append(f"=== Spacecraft Data Snapshot [{elapsed_str}] ===")
            lines.append(
                f"Dirs crawled: {self.dirs_crawled}/{self.dirs_queued}  |  "
                f"Total data files: {self.data_files:,}  |  "
                f"Total size: {self.format_size(self.data_bytes)}  |  "
                f"Errors: {self.errors}"
            )
            lines.append(f"{'Spacecraft':<30s} {'Files':>10s} {'Size':>14s}")
            lines.append("-" * 56)
            sorted_missions = sorted(
                self.mission_sizes.items(), key=lambda x: x[1], reverse=True
            )
            for mission, size in sorted_missions:
                count = self.mission_counts[mission]
                lines.append(
                    f"{mission:<30s} {count:>10,} {self.format_size(size):>14s}"
                )
            lines.append("=" * 56)
            return "\n".join(lines)

    def print_summary(self):
        elapsed = time.time() - self.start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))
        print("\n")
        print("=" * 70)
        print("SPDF Data File Size Summary")
        print("=" * 70)
        print(f"Total data files:   {self.data_files:,}")
        print(f"Total data size:    {self.format_size(self.data_bytes)}")
        print(f"Other files:        {self.other_files:,}")
        print(f"Directories:        {self.dirs_crawled:,}")
        print(f"Errors:             {self.errors:,}")
        print(f"Time elapsed:       {elapsed_str}")
        print()
        print("Breakdown by format:")
        print("-" * 50)
        sorted_formats = sorted(
            self.format_sizes.items(), key=lambda x: x[1], reverse=True
        )
        for ext, size in sorted_formats:
            count = self.format_counts[ext]
            print(f"  {ext:<10s} {self.format_size(size):>12s}  ({count:,} files)")
        print()
        print("Top 30 missions by data size:")
        print("-" * 50)
        sorted_missions = sorted(
            self.mission_sizes.items(), key=lambda x: x[1], reverse=True
        )
        for mission, size in sorted_missions[:30]:
            count = self.mission_counts[mission]
            print(f"  {mission:<30s} {self.format_size(size):>12s}  ({count:,} files)")
        print("=" * 70)


def fetch_directory(url: str, session: requests.Session, progress: Progress, retries=3):
    """Fetch a directory listing and return (subdirs, cdf_files_with_sizes)."""
    # Extract mission name (first path component after /pub/data/)
    rel = url.replace(BASE_URL, "")
    mission = rel.split("/")[0] if rel else "root"

    progress.start_directory(url)

    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            break
        except (requests.RequestException, Exception) as e:
            if attempt == retries - 1:
                progress.add_error()
                progress.finish_directory(url)
                return [], mission
            time.sleep(1 * (attempt + 1))

    subdirs = []
    for match in ROW_PATTERN.finditer(resp.text):
        href, size_str = match.groups()
        # Skip parent directory and sorting links
        if href.startswith("?") or href.startswith("/"):
            continue
        full_url = urljoin(url, href)

        if href.endswith("/"):
            subdirs.append(full_url)
        else:
            href_lower = href.lower()
            # Check for any known science data extension
            ext = None
            for data_ext in DATA_EXTENSIONS:
                if href_lower.endswith(data_ext):
                    ext = data_ext
                    break
            if ext:
                progress.add_data_file(parse_size(size_str), mission, ext)
            else:
                progress.add_other()

    progress.finish_directory(url)
    return subdirs, mission


_stop_flag = threading.Event()


def crawl(base_url: str, max_workers: int, progress: Progress):
    """BFS crawl of the SPDF directory tree."""
    session = requests.Session()
    session.headers["User-Agent"] = "spdf-size-crawler/1.0 (research)"

    progress.add_directory(1)
    queue = [base_url]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while queue and not _stop_flag.is_set():
            # Submit a batch of directory fetches
            futures = {}
            for url in queue:
                f = executor.submit(fetch_directory, url, session, progress)
                futures[f] = url
            queue = []

            # Collect results and enqueue subdirectories
            for future in as_completed(futures):
                if _stop_flag.is_set():
                    executor.shutdown(wait=False, cancel_futures=True)
                    return
                subdirs, mission = future.result()
                if subdirs:
                    progress.add_directory(len(subdirs))
                    queue.extend(subdirs)
                progress.print_status()


def _setup_spacecraft_logger(log_path: str) -> logging.Logger:
    """Create a file logger for periodic spacecraft snapshots."""
    logger = logging.getLogger("spdf_spacecraft")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s\n%(message)s\n"))
    logger.addHandler(handler)
    return logger


def _spacecraft_log_loop(progress: Progress, logger: logging.Logger, interval: float = 5.0):
    """Background thread: write per-spacecraft snapshot to log every `interval` seconds."""
    while not _stop_flag.is_set():
        _stop_flag.wait(interval)
        snapshot = progress.spacecraft_snapshot()
        logger.info(snapshot)


def main():
    parser = argparse.ArgumentParser(
        description="Crawl SPDF to compute total CDF file sizes"
    )
    parser.add_argument(
        "--workers", type=int, default=10,
        help="Number of concurrent HTTP requests (default: 10)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--url", type=str, default=BASE_URL,
        help=f"Base URL to crawl (default: {BASE_URL})"
    )
    parser.add_argument(
        "--log", type=str,
        default=f"spdf_spacecraft_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        help="Log file for per-spacecraft snapshots every 5s (default: spdf_spacecraft_<timestamp>.log)"
    )
    args = parser.parse_args()

    progress = Progress(output_path=args.output)

    # Set up the per-spacecraft logger
    sc_logger = _setup_spacecraft_logger(args.log)
    log_thread = threading.Thread(
        target=_spacecraft_log_loop, args=(progress, sc_logger), daemon=True
    )

    print(f"Crawling {args.url}")
    print(f"Using {args.workers} concurrent workers")
    exts = ", ".join(sorted(DATA_EXTENSIONS))
    print(f"Looking for data files: {exts}")
    print(f"Spacecraft log: {os.path.abspath(args.log)}\n")

    # Handle SIGTERM (from TaskStop / kill) — set flag so threads stop gracefully
    def _sigterm_handler(signum, frame):
        _stop_flag.set()
    signal.signal(signal.SIGTERM, _sigterm_handler)

    log_thread.start()

    try:
        crawl(args.url, args.workers, progress)
    except KeyboardInterrupt:
        _stop_flag.set()

    _stop_flag.set()  # ensure log thread exits
    log_thread.join(timeout=2)

    # Write one final snapshot
    sc_logger.info(progress.spacecraft_snapshot())

    if _stop_flag.is_set():
        print("\n\nInterrupted! Showing partial results...")

    progress.print_summary()

    if args.output:
        result = progress.to_dict()
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to {args.output}")

    print(f"Spacecraft log saved to {os.path.abspath(args.log)}")


if __name__ == "__main__":
    main()
