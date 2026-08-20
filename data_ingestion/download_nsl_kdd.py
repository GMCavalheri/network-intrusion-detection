"""
Downloads the NSL-KDD dataset: KDDTrain+.txt and KDDTest+.txt.

NSL-KDD ships as a pre-split train/test pair (no timestamps, 41 connection-
level features + attack-type label + a "difficulty" score). The test set
deliberately includes attack types absent from the training set, which is
the whole point of the benchmark: it measures generalization to unseen
attacks rather than just memorization.

The official UNB download page (https://www.unb.ca/cic/datasets/nsl.html)
currently returns "this dataset is no longer available" with no working
download link, so this script defaults to a long-standing GitHub mirror
that serves the exact same files (verified byte-for-byte row counts against
the canonical dataset: 125,973 train rows / 22,544 test rows). Override
NSL_KDD_BASE_URL if that mirror ever goes away.

Run with:
    python download_nsl_kdd.py
"""

import argparse
import os

from common import DATA_DIR, download_file, get_logger

logger = get_logger("download_nsl_kdd")

DEFAULT_BASE_URL = "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master"
FILES = ["KDDTrain+.txt", "KDDTest+.txt"]

# Sanity-check row counts for the canonical dataset - not a hard assertion
# (a mirror could legitimately reformat line endings etc.) but worth a loud
# warning if a swapped-in mirror serves something else entirely.
EXPECTED_ROW_COUNTS = {"KDDTrain+.txt": 125_973, "KDDTest+.txt": 22_544}


def download_nsl_kdd(out_dir: str, base_url: str = DEFAULT_BASE_URL, force: bool = False) -> dict:
    results = {}
    for filename in FILES:
        dest_path = os.path.join(out_dir, filename)
        if force and os.path.exists(dest_path):
            os.remove(dest_path)
        url = f"{base_url}/{filename.replace('+', '%2B')}"
        written = download_file(url, dest_path, logger)
        results[filename] = dest_path

        with open(dest_path) as f:
            row_count = sum(1 for _ in f)
        expected = EXPECTED_ROW_COUNTS.get(filename)
        if expected and row_count != expected:
            logger.warning(
                "%s has %d rows, expected %d - the mirror may have changed. Proceeding anyway.",
                filename, row_count, expected,
            )
        else:
            logger.info("%s: %d rows (matches expected)", filename, row_count)

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=os.path.join(DATA_DIR, "raw", "nsl_kdd"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist")
    args = parser.parse_args()

    logger.info("Starting NSL-KDD download into %s", args.out_dir)
    download_nsl_kdd(args.out_dir, args.base_url, args.force)
    logger.info("NSL-KDD download complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("download_nsl_kdd failed")
        raise
