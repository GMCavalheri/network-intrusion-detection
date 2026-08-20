"""Shared configuration and helpers for all Spark jobs in the network
intrusion detection pipeline.

NSL-KDD and CICIDS2017 are two independent pipelines (own raw format, own
clean -> features -> train -> score steps, own trained model) because their
feature spaces don't overlap in any meaningful way - see the README for why
this project doesn't force them into one flat feature table. What they do
share: this module's Spark/Postgres/logging plumbing, and a coarse
cross-dataset attack taxonomy (UNIFIED_CATEGORIES) used only at the serving
layer so the dashboard can show one combined "attacks by category" view.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from pyspark.sql import SparkSession

RAW_NSL_KDD_DIR = os.environ.get("RAW_NSL_KDD_DIR", "/opt/data/raw/nsl_kdd")
RAW_CICIDS2017_DIR = os.environ.get("RAW_CICIDS2017_DIR", "/opt/data/raw/cicids2017")
LOG_DIR = os.environ.get("LOG_DIR", "/opt/logs")


def get_logger(name: str) -> logging.Logger:
    """Console + rotating-file logger shared by every job script.

    Console output is always on (Docker/`docker logs` captures it); the file
    handler is best-effort so tests and local runs without a mounted log
    directory still work.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # already configured (e.g. re-imported within the same process)

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)
    logger.addHandler(console)

    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        file_handler = RotatingFileHandler(
            os.path.join(LOG_DIR, f"{name}.log"), maxBytes=5_000_000, backupCount=3
        )
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        pass  # LOG_DIR not writable/mounted in this environment - console-only is fine

    logger.propagate = False
    return logger


# Small single-file JSON reports (per-dataset data-quality + metrics) are
# written directly by driver-side Python code (plain open()/json.dump), so
# they stay on the shared bind mount - that works fine, only Spark's
# distributed executor writes do not (see get_spark for why).
LOCAL_PROCESSED_DIR = os.environ.get("PROCESSED_DATA_DIR", "/opt/data/processed")
NSL_KDD_DQ_REPORT_PATH = os.path.join(LOCAL_PROCESSED_DIR, "nsl_kdd_data_quality_report.json")
CICIDS2017_DQ_REPORT_PATH = os.path.join(LOCAL_PROCESSED_DIR, "cicids2017_data_quality_report.json")
NSL_KDD_METRICS_PATH = os.path.join(LOCAL_PROCESSED_DIR, "nsl_kdd_metrics.json")
CICIDS2017_METRICS_PATH = os.path.join(LOCAL_PROCESSED_DIR, "cicids2017_metrics.json")

# Bulk Parquet/model artifacts that Spark executors write with the
# distributed FileFormatWriter go to S3-compatible object storage (MinIO)
# instead of the bind-mounted host directory - see get_spark() for why.
S3_BUCKET = os.environ.get("MINIO_BUCKET", "network-intrusion-detection")
NSL_KDD_CLEANED_PATH = f"s3a://{S3_BUCKET}/processed/nsl_kdd/cleaned"
NSL_KDD_FEATURES_PATH = f"s3a://{S3_BUCKET}/processed/nsl_kdd/features"
NSL_KDD_MODEL_PATH = f"s3a://{S3_BUCKET}/models/nsl_kdd_model"
CICIDS2017_CLEANED_PATH = f"s3a://{S3_BUCKET}/processed/cicids2017/cleaned"
CICIDS2017_FEATURES_PATH = f"s3a://{S3_BUCKET}/processed/cicids2017/features"
CICIDS2017_MODEL_PATH = f"s3a://{S3_BUCKET}/models/cicids2017_model"

# ---------------------------------------------------------------------------
# NSL-KDD: 41 pre-engineered connection-level features (KDD'99-style), no
# header row in the raw files, plus a trailing attack-type label and a
# "difficulty" score (how hard existing classifiers found the row - not a
# feature, dropped during ETL).
# ---------------------------------------------------------------------------
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
NSL_KDD_LABEL_COL = "attack_type"
NSL_KDD_DIFFICULTY_COL = "difficulty"
NSL_KDD_NUMERIC_FEATURES = [
    c for c in NSL_KDD_COLUMNS if c not in ("protocol_type", "service", "flag")
]
NSL_KDD_CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

# Cross-dataset taxonomy shown together in the dashboard/API. NSL-KDD's own
# 4-class scheme (DoS/Probe/R2L/U2R) and CICIDS2017's per-attack Label column
# don't line up perfectly - R2L ("remote to local", credential/access abuse)
# maps onto Brute Force, U2R ("user to root") onto Privilege Escalation - so
# this is a deliberate approximation for cross-dataset comparison, not a
# claim that the two datasets model identical attack classes. Each dataset's
# own native label is preserved alongside this in the cleaned output.
UNIFIED_CATEGORIES = [
    "Benign", "DoS", "Probe/Scan", "Brute Force", "Web Attack", "Botnet",
    "Infiltration", "Privilege Escalation", "Other",
]

NSL_KDD_ATTACK_CATEGORY = {
    "normal": "Benign",
    # DoS
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS",
    "smurf": "DoS", "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS",
    "processtable": "DoS", "mailbomb": "DoS", "worm": "DoS",
    # Probe
    "satan": "Probe/Scan", "ipsweep": "Probe/Scan", "nmap": "Probe/Scan",
    "portsweep": "Probe/Scan", "mscan": "Probe/Scan", "saint": "Probe/Scan",
    # R2L -> Brute Force (credential/remote-access abuse; phf/httptunnel are
    # arguably closer to Web Attack but we keep the whole class together
    # since NSL-KDD's own R2L label is preserved in `attack_type` for anyone
    # who wants the precise breakdown)
    "guess_passwd": "Brute Force", "ftp_write": "Brute Force", "imap": "Brute Force",
    "phf": "Brute Force", "multihop": "Brute Force", "warezmaster": "Brute Force",
    "warezclient": "Brute Force", "spy": "Brute Force", "xlock": "Brute Force",
    "xsnoop": "Brute Force", "snmpguess": "Brute Force", "snmpgetattack": "Brute Force",
    "httptunnel": "Brute Force", "sendmail": "Brute Force", "named": "Brute Force",
    # U2R -> Privilege Escalation
    "buffer_overflow": "Privilege Escalation", "loadmodule": "Privilege Escalation",
    "rootkit": "Privilege Escalation", "perl": "Privilege Escalation",
    "sqlattack": "Privilege Escalation", "xterm": "Privilege Escalation",
    "ps": "Privilege Escalation",
}

# ---------------------------------------------------------------------------
# CICIDS2017: CICFlowMeter flow-level features (78 columns after the 5
# identifying columns), one CSV per capture day, real per-day timestamps.
# Column names as shipped have inconsistent leading whitespace - stripped in
# etl_clean_cicids2017.py before anything here is used.
# ---------------------------------------------------------------------------
CICIDS2017_LABEL_COL = "label"

CICIDS2017_LABEL_CATEGORY = {
    "BENIGN": "Benign",
    "DoS GoldenEye": "DoS", "DoS Hulk": "DoS", "DoS Slowhttptest": "DoS",
    "DoS slowloris": "DoS", "DDoS": "DoS",
    "Heartbleed": "Other",  # a single-CVE memory-disclosure exploit, not a flood - doesn't fit any other bucket cleanly
    "PortScan": "Probe/Scan",
    "Bot": "Botnet",
    "Infiltration": "Infiltration",
    "FTP-Patator": "Brute Force", "SSH-Patator": "Brute Force",
    # Thursday's Web Attack labels use an en-dash the CSV encodes inconsistently
    # across mirrors/tools (plain U+2013, or a stray cp1252 0x96 byte read as
    # "\x96") - both variants map here defensively; unmapped labels fall back
    # to "Other" rather than crashing the pipeline either way.
    "Web Attack – Brute Force": "Web Attack", "Web Attack \x96 Brute Force": "Web Attack",
    "Web Attack – XSS": "Web Attack", "Web Attack \x96 XSS": "Web Attack",
    "Web Attack – Sql Injection": "Web Attack", "Web Attack \x96 Sql Injection": "Web Attack",
}

# Raw CICIDS2017 CSVs are read *positionally* (header=False, our own schema)
# rather than by name, because the shipped header has real, verified quirks:
# every column but 4 has a stray leading space, and index 55 ("Fwd Header
# Length") is a byte-for-byte duplicate of index 34 - reading by name would
# make Spark reject the file outright ("Found duplicate column(s)"). Verified
# against a real downloaded CSV (Friday-WorkingHours-Afternoon-DDos), not
# just documentation.
CICIDS2017_RAW_COLUMNS = [
    "destination_port", "flow_duration", "total_fwd_packets", "total_backward_packets",
    "total_length_of_fwd_packets", "total_length_of_bwd_packets",
    "fwd_packet_length_max", "fwd_packet_length_min", "fwd_packet_length_mean", "fwd_packet_length_std",
    "bwd_packet_length_max", "bwd_packet_length_min", "bwd_packet_length_mean", "bwd_packet_length_std",
    "flow_bytes_per_s", "flow_packets_per_s",
    "flow_iat_mean", "flow_iat_std", "flow_iat_max", "flow_iat_min",
    "fwd_iat_total", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max", "fwd_iat_min",
    "bwd_iat_total", "bwd_iat_mean", "bwd_iat_std", "bwd_iat_max", "bwd_iat_min",
    "fwd_psh_flags", "bwd_psh_flags", "fwd_urg_flags", "bwd_urg_flags",
    "fwd_header_length", "bwd_header_length",
    "fwd_packets_per_s", "bwd_packets_per_s",
    "min_packet_length", "max_packet_length", "packet_length_mean", "packet_length_std", "packet_length_variance",
    "fin_flag_count", "syn_flag_count", "rst_flag_count", "psh_flag_count", "ack_flag_count",
    "urg_flag_count", "cwe_flag_count", "ece_flag_count",
    "down_up_ratio", "average_packet_size", "avg_fwd_segment_size", "avg_bwd_segment_size",
    "fwd_header_length_dup",  # duplicate of fwd_header_length (index 34) - dropped in etl_clean
    "fwd_avg_bytes_bulk", "fwd_avg_packets_bulk", "fwd_avg_bulk_rate",
    "bwd_avg_bytes_bulk", "bwd_avg_packets_bulk", "bwd_avg_bulk_rate",
    "subflow_fwd_packets", "subflow_fwd_bytes", "subflow_bwd_packets", "subflow_bwd_bytes",
    "init_win_bytes_forward", "init_win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
    "active_mean", "active_std", "active_max", "active_min",
    "idle_mean", "idle_std", "idle_max", "idle_min",
    "label",
]
# Columns that hold Infinity/NaN in real data (rate = quantity / duration,
# undefined when duration is 0) - the concrete data-quality issue this
# dataset is famous for, cleaned explicitly in etl_clean_cicids2017.py.
CICIDS2017_RATE_COLUMNS = ["flow_bytes_per_s", "flow_packets_per_s", "fwd_packets_per_s", "bwd_packets_per_s"]
CICIDS2017_NUMERIC_FEATURES = [
    c for c in CICIDS2017_RAW_COLUMNS if c not in ("label", "fwd_header_length_dup")
]

# Duplicated (not imported) from data_ingestion/download_cicids2017.py on
# purpose - spark_jobs ships as its own Docker image with no shared
# filesystem with data_ingestion, same reasoning as api/constants.py in the
# sibling fraud-detection-spark project.
CICIDS2017_DAY_FILES = {
    "monday": "Monday-WorkingHours.pcap_ISCX.csv",
    "tuesday": "Tuesday-WorkingHours.pcap_ISCX.csv",
    "wednesday": "Wednesday-workingHours.pcap_ISCX.csv",
    "thursday_web": "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "thursday_infiltration": "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "friday_morning": "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "friday_portscan": "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "friday_ddos": "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}

# CICIDS2017 flows carry no per-row account/session identifier and this
# release of the CSVs strips the Timestamp column entirely (verified - see
# CICIDS2017_RAW_COLUMNS), so there's no way to do a within-day time-based
# split like the fraud project's. Splitting by *capture day* instead achieves
# the same goal (test on data the model never trained on) and is arguably
# cleaner: flows captured minutes apart in the same DDoS burst are highly
# autocorrelated, so a random or even time-sliced split within one file risks
# leaking near-duplicate flows across train/test - a different day can't.
CICIDS2017_TRAIN_DAYS = ["monday", "wednesday"]
CICIDS2017_TEST_DAYS = ["friday_ddos"]


def get_spark(app_name: str) -> SparkSession:
    master = os.environ.get("SPARK_MASTER_URL", "local[*]")
    builder = (
        SparkSession.builder.appName(app_name)
        .master(master)
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.shuffle.partitions", os.environ.get("SPARK_SHUFFLE_PARTITIONS", "16"))
        .config(
            "spark.jars.packages",
            "org.postgresql:postgresql:42.7.3,"
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.262",
        )
        # Bulk Parquet/model output goes to MinIO (S3-compatible) rather than
        # the bind-mounted host directory: when the driver and executors are
        # genuinely separate containers, Hadoop's local file:// commit
        # protocol (mkdirs() on the shared _temporary staging tree) is
        # unreliable across their independent bind-mount views - object
        # storage sidesteps that whole class of problem, which is exactly why
        # real distributed Spark deployments use HDFS/S3 instead of a local
        # path in the first place.
        .config("spark.hadoop.fs.s3a.endpoint", os.environ.get("MINIO_ENDPOINT", "http://minio:9000"))
        .config("spark.hadoop.fs.s3a.access.key", os.environ.get("MINIO_ACCESS_KEY", "minioadmin"))
        .config("spark.hadoop.fs.s3a.secret.key", os.environ.get("MINIO_SECRET_KEY", "minioadmin"))
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.connection.ssl.enabled", "false")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
    )
    return builder.getOrCreate()


def postgres_config():
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ.get("POSTGRES_DB", "network_intrusion_detection")
    user = os.environ.get("POSTGRES_USER", "nids_admin")
    password = os.environ.get("POSTGRES_PASSWORD", "change_me")
    url = f"jdbc:postgresql://{host}:{port}/{db}"
    props = {"user": user, "password": password, "driver": "org.postgresql.Driver"}
    return url, props
