"""
Downloads a subset of the CICIDS2017 dataset's flow CSVs (CICFlowMeter
output: 78 flow-level features per row, real per-day timestamps).

The full dataset is 8 CSVs covering Mon-Fri (~885MB total, plus tens of GB
of raw pcaps we don't need). To keep this runnable on a laptop, the default
subset is 3 representative days:

  - Monday-WorkingHours.pcap_ISCX.csv               benign baseline traffic
  - Wednesday-workingHours.pcap_ISCX.csv             DoS GoldenEye/Hulk/Slowloris, Heartbleed
  - Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv DDoS

That's ~480MB and covers the dataset's main attack families (volumetric DoS/
DDoS and a rare high-value exploit) without the full multi-GB download. Pass
--days to pick a different subset of the 8 available days.

The official UNB download page (https://www.unb.ca/cic/datasets/ids-2017.html)
routes through a portal that currently redirects to a gated index page, so
this defaults to a public Hugging Face mirror serving the identical
CICFlowMeter CSVs (verified: file sizes match the official MachineLearningCVE
release exactly). Override CICIDS2017_BASE_URL if that mirror ever moves.

Citation (please keep if you publish results from this data):
  Sharafaldin, Lashkari, Ghorbani, "Toward Generating a New Intrusion
  Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018.

Run with:
    python download_cicids2017.py
    python download_cicids2017.py --days monday tuesday wednesday thursday_web thursday_infiltration friday_morning friday_portscan friday_ddos
"""

import argparse
import os

from common import DATA_DIR, download_file, get_logger

logger = get_logger("download_cicids2017")

DEFAULT_BASE_URL = "https://huggingface.co/datasets/c01dsnap/CIC-IDS2017/resolve/main"

# Maps a short --days key to the exact upstream CSV filename (the upstream
# names are inconsistent in casing/wording, so this table is the one place
# that needs to know the real names).
DAY_FILES = {
    "monday": "Monday-WorkingHours.pcap_ISCX.csv",
    "tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "wednesday": "Wednesday-workingHours.pcap_ISCX.csv",
    "thursday_web": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "thursday_infiltration": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "friday_morning": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "friday_portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "friday_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}

DEFAULT_DAYS = ["monday", "wednesday", "friday_ddos"]


def resolve_files(days: list[str]) -> dict:
    unknown = [d for d in days if d not in DAY_FILES]
    if unknown:
        raise ValueError(f"Unknown day key(s) {unknown}. Valid keys: {sorted(DAY_FILES)}")
    return {d: DAY_FILES[d] for d in days}


def download_cicids2017(out_dir: str, days: list[str], base_url: str = DEFAULT_BASE_URL, force: bool = False) -> dict:
    files = resolve_files(days)
    results = {}
    for day, filename in files.items():
        dest_path = os.path.join(out_dir, filename)
        if force and os.path.exists(dest_path):
            os.remove(dest_path)
        url = f"{base_url}/{filename}"
        download_file(url, dest_path, logger)
        results[day] = dest_path
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=os.path.join(DATA_DIR, "raw", "cicids2017"))
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--days", nargs="+", default=DEFAULT_DAYS, choices=sorted(DAY_FILES))
    parser.add_argument("--force", action="store_true", help="Re-download even if files already exist")
    args = parser.parse_args()

    logger.info("Starting CICIDS2017 download (days=%s) into %s", args.days, args.out_dir)
    download_cicids2017(args.out_dir, args.days, args.base_url, args.force)
    logger.info("CICIDS2017 download complete")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("download_cicids2017 failed")
        raise
